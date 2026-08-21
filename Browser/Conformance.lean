import Browser.Trace
import Lean.Data.Json

namespace Browser

open Lean

/-- One Page -> BrowserContext ownership record emitted in the trace bootstrap. -/
structure TracePage where
  page : PageId
  context : ContextId
  deriving Repr, FromJson, ToJson

/-- The first JSONL line declares the finite Page/Context universe used to
    reconstruct the runtime model before replaying driver events. -/
structure TraceBootstrap where
  kind : String
  pages : Array TracePage
  deriving Repr, FromJson, ToJson

/-- Minimal tag used to dispatch one JSONL object without making the wire format
    depend on Lean's representation of the semantic sum type. -/
private structure TraceTag where
  kind : String
  deriving FromJson

/-- Raw wire record. Optional fields are validated according to `event`, so a
    malformed driver event is rejected before it reaches runtime `step`. -/
private structure RawDriverTraceRecord where
  kind : String
  eventId : EventId
  correlationId : CorrelationId
  parentId? : Option EventId := none
  depth : Nat := 0
  event : String
  page? : Option PageId := none
  context? : Option ContextId := none
  action? : Option ActionId := none
  call? : Option CdpCallId := none
  deadline? : Option Time := none
  now? : Option Time := none
  reason? : Option String := none
  deriving Repr, FromJson

/-- Semantic event plus wire metadata that is not represented directly by
    `RuntimeEvent`. `declaredAction` is currently used for `actionStarted`: the
    Driver must emit the same ActionId that the formal model allocates. -/
structure DriverTraceRecord where
  envelope : EventEnvelope
  declaredAction : Option ActionId := none
  deriving Repr

inductive TraceLine where
  | bootstrap (value : TraceBootstrap)
  | event (value : DriverTraceRecord)
  deriving Repr

/-- Causal identity retained while replaying the external trace. -/
structure SeenEvent where
  id : EventId
  correlation : CorrelationId
  depth : Nat
  deriving Repr, DecidableEq

private def contextFromPages (pages : Array TracePage) (page : PageId) : ContextId :=
  pages.foldl
    (fun current entry => if entry.page = page then entry.context else current)
    0

/-- Build the executable runtime state from an explicit bootstrap rather than
    hard-coding test-only BrowserContext ownership into the verifier. -/
def modelFromBootstrap (bootstrap : TraceBootstrap) : Model := {
  browserAlive := true
  graph := {
    contextOf := contextFromPages bootstrap.pages
    pageOfFrame := fun frame => frame
    parentOfFrame := fun _ => none
  }
  knownPages := bootstrap.pages.toList.map (fun entry => entry.page)
  contextAvailable := fun _ => true
  page := fun _ => PageState.readyIdle
}

private def required {α : Type} (name : String) : Option α → Except String α
  | some value => pure value
  | none => throw s!"missing {name}"

private def decodeWaitReason : String → Except String WaitReason
  | "action" => pure .action
  | "human" => pure .human
  | "page" => pure .page
  | "network" => pure .network
  | "protocol" => pure .protocol
  | other => throw s!"unknown wait reason: {other}"

private def envelopeOf (raw : RawDriverTraceRecord) (event : RuntimeEvent) : EventEnvelope := {
  id := raw.eventId
  event := event
  cause := {
    correlation := raw.correlationId
    parent := raw.parentId?
    depth := raw.depth
  }
}

private def localRecord
    (raw : RawDriverTraceRecord) (page : PageId) (kind : LocalEventKind)
    (declaredAction : Option ActionId := none) : DriverTraceRecord := {
  envelope := envelopeOf raw (.local { page := page, kind := kind })
  declaredAction := declaredAction
}

private def decodeEvent (raw : RawDriverTraceRecord) : Except String DriverTraceRecord := do
  match raw.event with
  | "actionStarted" =>
      let page ← required "page" raw.page?
      let action ← required "action" raw.action?
      pure (localRecord raw page .automationStarted (some action))
  | "actionFinished" =>
      let page ← required "page" raw.page?
      let action ← required "action" raw.action?
      pure (localRecord raw page (.automationFinished action))
  | "humanInput" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .humanInput)
  | "humanIdle" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .humanIdle)
  | "navigationStarted" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .navigationStarted)
  | "pageReady" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .pageReady)
  | "executionContextDestroyed" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .executionContextDestroyed)
  | "executionContextReady" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .executionContextReady)
  | "deadlineArmed" =>
      let page ← required "page" raw.page?
      let action ← required "action" raw.action?
      let deadline ← required "deadline" raw.deadline?
      let reasonString ← required "reason" raw.reason?
      let reason ← decodeWaitReason reasonString
      pure (localRecord raw page (.deadlineArmed action deadline reason))
  | "deadlineReached" =>
      let page ← required "page" raw.page?
      let action ← required "action" raw.action?
      let now ← required "now" raw.now?
      pure (localRecord raw page (.deadlineReached action now))
  | "protocolWaitStarted" =>
      let page ← required "page" raw.page?
      let action ← required "action" raw.action?
      let call ← required "call" raw.call?
      pure (localRecord raw page (.protocolWaitStarted action call))
  | "protocolResponse" =>
      let page ← required "page" raw.page?
      let action ← required "action" raw.action?
      let call ← required "call" raw.call?
      pure (localRecord raw page (.protocolResponse action call))
  | "pageCrashed" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .pageCrashed)
  | "pageClosed" =>
      let page ← required "page" raw.page?
      pure (localRecord raw page .pageClosed)
  | "contextUnavailable" =>
      let context ← required "context" raw.context?
      pure { envelope := envelopeOf raw (.contextUnavailable context) }
  | "contextAvailable" =>
      let context ← required "context" raw.context?
      pure { envelope := envelopeOf raw (.contextAvailable context) }
  | "browserDisconnected" =>
      pure { envelope := envelopeOf raw .browserDisconnected }
  | "browserConnected" =>
      pure { envelope := envelopeOf raw .browserConnected }
  | other => throw s!"unknown event: {other}"

/-- Decode one JSONL object. The `kind` field is stable wire syntax and keeps
    bootstrap/config records separate from runtime events. -/
def parseTraceLine (line : String) : Except String TraceLine := do
  let json ← Json.parse line
  let tag : TraceTag ← fromJson? json
  match tag.kind with
  | "bootstrap" =>
      let value : TraceBootstrap ← fromJson? json
      pure (.bootstrap value)
  | "event" =>
      let raw : RawDriverTraceRecord ← fromJson? json
      pure (.event (← decodeEvent raw))
  | other => throw s!"unknown trace kind: {other}"

private def findSeen (id : EventId) : List SeenEvent → Option SeenEvent
  | [] => none
  | event :: rest =>
      if event.id = id then some event else findSeen id rest

private def validateCause (seen : List SeenEvent) (envelope : EventEnvelope) : Except String Unit := do
  if (findSeen envelope.id seen).isSome then
    throw "duplicate event id"
  match envelope.cause.parent with
  | none =>
      if envelope.cause.depth = 0 then
        pure ()
      else
        throw "root event must have depth 0"
  | some parentId =>
      match findSeen parentId seen with
      | none => throw "causal parent not seen"
      | some parent =>
          if parent.correlation != envelope.cause.correlation then
            throw "causal correlation mismatch"
          if envelope.cause.depth != parent.depth + 1 then
            throw "causal depth mismatch"
          pure ()

private def seenOf (envelope : EventEnvelope) : SeenEvent := {
  id := envelope.id
  correlation := envelope.cause.correlation
  depth := envelope.cause.depth
}

/-- Verify one Driver record against causality, reaction depth, runtime safety,
    and (for action-start) the ActionId generated by the formal state machine. -/
def verifyRecord
    (budget : ReactionBudget) (m : Model) (seen : List SeenEvent)
    (record : DriverTraceRecord) : Except String Model := do
  validateCause seen record.envelope
  let next ←
    match reactEnvelopeSafe budget m record.envelope with
    | some value => pure value
    | none => throw "reaction budget exceeded"
  match record.envelope.event, record.declaredAction with
  | .local { page := page, kind := .automationStarted }, some declared =>
      if (next.page page).currentAction = some declared then
        pure next
      else
        throw "action generation mismatch"
  | .local { kind := .automationStarted, .. }, none =>
      throw "actionStarted missing action"
  | _, _ => pure next

/-- Replay external Driver records in arrival order. Causal parents may be older
    than the immediately previous line, which is required for late async
    callbacks, but every parent must have already appeared in this trace. -/
def replayDriverRecords
    (budget : ReactionBudget) : Model → List SeenEvent → List DriverTraceRecord → Except String Model
  | m, _, [] => pure m
  | m, seen, record :: rest => do
      let next ← verifyRecord budget m seen record
      if wellFormed? next then
        replayDriverRecords budget next (seenOf record.envelope :: seen) rest
      else
        throw "runtime invariant violated"

private def parseEventLines : List String → Except String (List DriverTraceRecord)
  | [] => pure []
  | line :: rest => do
      match ← parseTraceLine line with
      | .bootstrap _ => throw "bootstrap must be the first line only"
      | .event record => pure (record :: (← parseEventLines rest))

/-- Parse an entire JSONL trace. Empty lines are ignored; the first non-empty
    line must be a bootstrap and every later line must be an event. -/
def parseTraceText (text : String) : Except String (TraceBootstrap × List DriverTraceRecord) := do
  let lines := (text.splitOn "\n").filter (fun line => !line.isEmpty)
  match lines with
  | [] => throw "empty trace"
  | first :: rest =>
      match ← parseTraceLine first with
      | .event _ => throw "first trace line must be bootstrap"
      | .bootstrap bootstrap =>
          pure (bootstrap, ← parseEventLines rest)

/-- End-to-end JSONL conformance entry point used by CI and by a future C/C++
    Driver test harness. -/
def verifyJsonl (budget : ReactionBudget) (text : String) : Except String Model := do
  let (bootstrap, records) ← parseTraceText text
  replayDriverRecords budget (modelFromBootstrap bootstrap) [] records

end Browser
