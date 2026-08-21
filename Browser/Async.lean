import Browser.Transition

namespace Browser

/-- Dynamic topology changes are messages too; they are not out-of-band graph
    mutations. Creation carries the epoch of the new child incarnation. -/
inductive GraphPayload where
  | contextCreated (context : ContextId) (epoch : NodeEpoch)
  | contextDestroyed (context : ContextId)
  | pageCreated (page : PageId) (context : ContextId) (epoch : NodeEpoch)
  | pageDestroyed (page : PageId)
  | frameCreated (frame : FrameId) (page : PageId) (parent : Option FrameId) (epoch : NodeEpoch)
  | frameDestroyed (frame : FrameId)
  deriving Repr

inductive AsyncPayload where
  | runtime (event : RuntimeEvent)
  | command (command : PageCommand)
  | graph (event : GraphPayload)
  deriving Repr

/-- One asynchronously delivered message. `source` and `target` are logical
    actor incarnations, not physical threads. -/
structure AsyncEnvelope where
  id : MessageId
  cause : Cause
  source : ActorAddress
  target : ActorAddress
  payload : AsyncPayload
  deriving Repr

structure SeenMessage where
  id : MessageId
  correlation : CorrelationId
  depth : Nat
  deriving Repr, DecidableEq, BEq

/-- Runtime registry around the already-proven browser state model. `slot`
    retains the last epoch even after a node dies. -/
structure AsyncRuntime where
  model : Model
  slot : ActorRef → NodeSlot
  seen : List SeenMessage := []
  nextMessageId : MessageId := 1

inductive DeliveryDisposition where
  | accepted
  | stale
  | duplicate
  | orphan
  | rejected
  deriving Repr, DecidableEq, BEq

structure DeliveryResult where
  runtime : AsyncRuntime
  emitted : List AsyncEnvelope := []
  disposition : DeliveryDisposition

private def contextKnown (m : Model) (context : ContextId) : Bool :=
  m.knownPages.any (fun page => m.graph.contextOf page == context)

private def initialSlot (m : Model) : ActorRef → NodeSlot
  | .browser => { epoch := 1, alive := true }
  | .context context =>
      if contextKnown m context then { epoch := 1, alive := true } else {}
  | .page page =>
      if m.knownPages.contains page then { epoch := 1, alive := true } else {}
  | .frame _ => {}
  | .action page action =>
      if m.knownPages.contains page && ((m.page page).currentAction == some action) then
        { epoch := 1, alive := true }
      else
        {}

/-- Lift an existing synchronous model into the async runtime without changing
    its browser semantics. Existing known Pages/Contexts start at epoch 1. -/
def asyncFromModel (m : Model) : AsyncRuntime := {
  model := m
  slot := initialSlot m
}

/-- Action actors borrow their Page epoch while additionally requiring the
    ActionId to be current. Other actors use the registry directly. -/
def nodeSlot (rt : AsyncRuntime) : ActorRef → NodeSlot
  | .action page action =>
      let pageSlot := rt.slot (.page page)
      if pageSlot.alive && ((rt.model.page page).currentAction == some action) then
        pageSlot
      else
        { epoch := pageSlot.epoch, alive := false }
  | actor => rt.slot actor

def updateSlot (rt : AsyncRuntime) (actor : ActorRef) (slot : NodeSlot) : AsyncRuntime :=
  { rt with slot := fun q => if q = actor then slot else rt.slot q }

private def findSeen (id : MessageId) : List SeenMessage → Option SeenMessage
  | [] => none
  | event :: rest => if event.id = id then some event else findSeen id rest

def hasSeen (rt : AsyncRuntime) (id : MessageId) : Bool :=
  (findSeen id rt.seen).isSome

/-- Causal children may arrive long after their parent, but an accepted child
    must reference an already observed parent with the same correlation and
    exactly one greater depth. -/
def causalValid (rt : AsyncRuntime) (envelope : AsyncEnvelope) : Bool :=
  match envelope.cause.parent with
  | none => envelope.cause.depth == 0
  | some parentId =>
      match findSeen parentId rt.seen with
      | none => false
      | some parent =>
          (parent.correlation == envelope.cause.correlation) &&
          (envelope.cause.depth == parent.depth + 1)

private def seenOf (envelope : AsyncEnvelope) : SeenMessage := {
  id := envelope.id
  correlation := envelope.cause.correlation
  depth := envelope.cause.depth
}

private def markObserved (rt : AsyncRuntime) (envelope : AsyncEnvelope) : AsyncRuntime :=
  { rt with
    seen := seenOf envelope :: rt.seen
    nextMessageId := Nat.max rt.nextMessageId (envelope.id + 1) }

inductive TargetClass where
  | current
  | stale
  | orphan
  deriving Repr, DecidableEq, BEq

/-- Unknown/dead target is orphan. A known actor with a different epoch is
    stale, even if a newer incarnation is currently alive. -/
def classifyTarget (rt : AsyncRuntime) (address : ActorAddress) : TargetClass :=
  let slot := nodeSlot rt address.actor
  if slot.epoch = 0 then
    .orphan
  else if slot.epoch ≠ address.epoch then
    .stale
  else if slot.alive = true then
    .current
  else
    .orphan

private def targetMatches (target : ActorAddress) : AsyncPayload → Bool
  | .runtime (.local event) => target.actor == .page event.page
  | .runtime (.contextUnavailable context) => target.actor == .context context
  | .runtime (.contextAvailable context) => target.actor == .context context
  | .runtime .browserDisconnected => target.actor == .browser
  | .runtime .browserConnected => target.actor == .browser
  | .command command => target.actor == .page command.page
  | .graph (.contextCreated _ _) => target.actor == .browser
  | .graph (.contextDestroyed context) => target.actor == .context context
  | .graph (.pageCreated _ context _) => target.actor == .context context
  | .graph (.pageDestroyed page) => target.actor == .page page
  | .graph (.frameCreated _ page _ _) => target.actor == .page page
  | .graph (.frameDestroyed frame) => target.actor == .frame frame

private def addKnownPage (m : Model) (page : PageId) : List PageId :=
  if m.knownPages.contains page then m.knownPages else page :: m.knownPages

private def removeKnownPage (m : Model) (page : PageId) : List PageId :=
  m.knownPages.filter (fun q => q != page)

private def createSlot
    (rt : AsyncRuntime) (actor : ActorRef) (newEpoch : NodeEpoch) : Except String AsyncRuntime := do
  let old := rt.slot actor
  if old.alive = true then
    throw "node already alive"
  else if newEpoch = old.epoch + 1 then
    pure (updateSlot rt actor { epoch := newEpoch, alive := true })
  else
    throw "node epoch mismatch"

private def destroySlot (rt : AsyncRuntime) (actor : ActorRef) : AsyncRuntime :=
  let old := rt.slot actor
  updateSlot rt actor { old with alive := false }

private def applyGraph (rt : AsyncRuntime) : GraphPayload → Except String AsyncRuntime
  | .contextCreated context epoch => do
      let created ← createSlot rt (.context context) epoch
      pure { created with model := updateContextAvailability created.model context true }
  | .contextDestroyed context =>
      let next := destroySlot rt (.context context)
      pure { next with model := updateContextAvailability next.model context false }
  | .pageCreated page context epoch => do
      let created ← createSlot rt (.page page) epoch
      let m := created.model
      let nextModel : Model := {
        m with
        graph := bindPageContext m.graph page context
        knownPages := addKnownPage m page
        page := fun q => if q = page then {} else m.page q
      }
      pure { created with model := nextModel }
  | .pageDestroyed page =>
      let next := destroySlot rt (.page page)
      let closed := updatePage next.model page (applyLocal .pageClosed)
      pure { next with model := { closed with knownPages := removeKnownPage closed page } }
  | .frameCreated frame page parent epoch => do
      let created ← createSlot rt (.frame frame) epoch
      let m := created.model
      pure { created with model := { m with graph := bindFrame m.graph frame page parent } }
  | .frameDestroyed frame =>
      pure (destroySlot rt (.frame frame))

/-- Local Page messages may reconcile that Page against parent availability,
    but Browser/Context messages do not synchronously rewrite descendant Page
    state. Cross-actor effects are emitted as command messages below. -/
private def applyRuntimeEventAsync (m : Model) (event : RuntimeEvent) : Model :=
  let stepped := step m event
  match event with
  | .local local => reconcileAction stepped (primitiveContract local.page)
  | _ => stepped

private def pageAddress (rt : AsyncRuntime) (page : PageId) : ActorAddress := {
  actor := .page page
  epoch := (nodeSlot rt (.page page)).epoch
}

private def childCauseOf (parent : AsyncEnvelope) : Cause := {
  correlation := parent.cause.correlation
  parent := some parent.id
  depth := parent.cause.depth + 1
}

private def emitCommands :
    AsyncRuntime → AsyncEnvelope → List PageCommand → AsyncRuntime × List AsyncEnvelope
  | rt, _, [] => (rt, [])
  | rt, parent, command :: rest =>
      let id := rt.nextMessageId
      let child : AsyncEnvelope := {
        id := id
        cause := childCauseOf parent
        source := parent.target
        target := pageAddress rt command.page
        payload := .command command
      }
      let advanced := { rt with nextMessageId := id + 1 }
      let (finalRt, tail) := emitCommands advanced parent rest
      (finalRt, child :: tail)

private def acceptedRuntime
    (rt : AsyncRuntime) (envelope : AsyncEnvelope) (event : RuntimeEvent) : DeliveryResult :=
  let commands := commandsFor rt.model event
  let changed := { rt with model := applyRuntimeEventAsync rt.model event }
  let observed := markObserved changed envelope
  let (finalRt, emitted) := emitCommands observed envelope commands
  { runtime := finalRt, emitted := emitted, disposition := .accepted }

private def acceptedCommand
    (rt : AsyncRuntime) (envelope : AsyncEnvelope) (command : PageCommand) : DeliveryResult :=
  let changed := { rt with model := applyCommand rt.model command }
  { runtime := markObserved changed envelope, disposition := .accepted }

private def acceptedGraph
    (rt : AsyncRuntime) (envelope : AsyncEnvelope) (event : GraphPayload) : DeliveryResult :=
  match applyGraph rt event with
  | .error _ => { runtime := rt, disposition := .rejected }
  | .ok changed => { runtime := markObserved changed envelope, disposition := .accepted }

/-- The sole cross-actor delivery boundary. Validation happens before actor-local
    transition logic. Only `accepted` may mutate browser/graph/Page state or emit
    child messages. -/
def deliver (rt : AsyncRuntime) (envelope : AsyncEnvelope) : DeliveryResult :=
  if hasSeen rt envelope.id then
    { runtime := rt, disposition := .duplicate }
  else if !causalValid rt envelope then
    { runtime := rt, disposition := .rejected }
  else
    match classifyTarget rt envelope.target with
    | .stale =>
        { runtime := markObserved rt envelope, disposition := .stale }
    | .orphan =>
        { runtime := rt, disposition := .orphan }
    | .current =>
        if !targetMatches envelope.target envelope.payload then
          { runtime := rt, disposition := .rejected }
        else
          match envelope.payload with
          | .runtime event => acceptedRuntime rt envelope event
          | .command command => acceptedCommand rt envelope command
          | .graph event => acceptedGraph rt envelope event

end Browser
