import Browser.Conformance
import Browser.Async
import Lean.Data.Json

namespace Browser

open Lean

private structure AsyncTraceTag where
  kind : String
  deriving FromJson

private structure RawActorAddress where
  kind : String
  id? : Option Nat := none
  action? : Option ActionId := none
  epoch : NodeEpoch
  deriving FromJson

private structure RawAsyncMessage where
  kind : String
  messageId : MessageId
  correlationId : CorrelationId
  parentId? : Option MessageId := none
  depth : Nat := 0
  source : RawActorAddress
  target : RawActorAddress
  payload : String
  page? : Option PageId := none
  context? : Option ContextId := none
  frame? : Option FrameId := none
  parentFrame? : Option FrameId := none
  action? : Option ActionId := none
  call? : Option CdpCallId := none
  deadline? : Option Time := none
  now? : Option Time := none
  reason? : Option String := none
  nodeEpoch? : Option NodeEpoch := none
  expected : String := "accepted"
  deriving FromJson

structure AsyncDriverRecord where
  envelope : AsyncEnvelope
  expected : DeliveryDisposition
  declaredAction : Option ActionId := none
  deriving Repr

inductive AsyncTraceLine where
  | bootstrap (value : TraceBootstrap)
  | message (value : AsyncDriverRecord)
  deriving Repr

structure AsyncReplayState where
  runtime : AsyncRuntime
  pending : List AsyncEnvelope := []

private def requiredAsync {α : Type} (name : String) : Option α → Except String α
  | some value => pure value
  | none => throw s!"missing {name}"

private def decodeActor (raw : RawActorAddress) : Except String ActorAddress := do
  let actor ←
    match raw.kind with
    | "browser" => pure ActorRef.browser
    | "context" => pure (.context (← requiredAsync "actor id" raw.id?))
    | "page" => pure (.page (← requiredAsync "actor id" raw.id?))
    | "frame" => pure (.frame (← requiredAsync "actor id" raw.id?))
    | "action" =>
        let page ← requiredAsync "actor page id" raw.id?
        let action ← requiredAsync "actor action id" raw.action?
        pure (.action page action)
    | other => throw s!"unknown actor kind: {other}"
  pure { actor := actor, epoch := raw.epoch }

private def decodeDisposition : String → Except String DeliveryDisposition
  | "accepted" => pure .accepted
  | "stale" => pure .stale
  | "duplicate" => pure .duplicate
  | "orphan" => pure .orphan
  | "rejected" => pure .rejected
  | other => throw s!"unknown delivery disposition: {other}"

private def decodeAsyncReason : String → Except String WaitReason
  | "action" => pure .action
  | "human" => pure .human
  | "page" => pure .page
  | "network" => pure .network
  | "protocol" => pure .protocol
  | other => throw s!"unknown wait reason: {other}"

private def localPayload
    (raw : RawAsyncMessage) (page : PageId) (kind : LocalEventKind) : AsyncPayload :=
  .runtime (.local { page := page, kind := kind })

private def decodeAsyncPayload (raw : RawAsyncMessage) : Except String (AsyncPayload × Option ActionId) := do
  match raw.payload with
  | "actionStarted" =>
      let page ← requiredAsync "page" raw.page?
      let action ← requiredAsync "action" raw.action?
      pure (localPayload raw page .automationStarted, some action)
  | "actionFinished" =>
      let page ← requiredAsync "page" raw.page?
      let action ← requiredAsync "action" raw.action?
      pure (localPayload raw page (.automationFinished action), none)
  | "humanInput" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .humanInput, none)
  | "humanIdle" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .humanIdle, none)
  | "navigationStarted" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .navigationStarted, none)
  | "pageReady" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .pageReady, none)
  | "executionContextDestroyed" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .executionContextDestroyed, none)
  | "executionContextReady" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .executionContextReady, none)
  | "deadlineArmed" =>
      let page ← requiredAsync "page" raw.page?
      let action ← requiredAsync "action" raw.action?
      let deadline ← requiredAsync "deadline" raw.deadline?
      let reason ← decodeAsyncReason (← requiredAsync "reason" raw.reason?)
      pure (localPayload raw page (.deadlineArmed action deadline reason), none)
  | "deadlineReached" =>
      let page ← requiredAsync "page" raw.page?
      let action ← requiredAsync "action" raw.action?
      let now ← requiredAsync "now" raw.now?
      pure (localPayload raw page (.deadlineReached action now), none)
  | "protocolWaitStarted" =>
      let page ← requiredAsync "page" raw.page?
      let action ← requiredAsync "action" raw.action?
      let call ← requiredAsync "call" raw.call?
      pure (localPayload raw page (.protocolWaitStarted action call), none)
  | "protocolResponse" =>
      let page ← requiredAsync "page" raw.page?
      let action ← requiredAsync "action" raw.action?
      let call ← requiredAsync "call" raw.call?
      pure (localPayload raw page (.protocolResponse action call), none)
  | "pageCrashed" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .pageCrashed, none)
  | "pageClosed" =>
      let page ← requiredAsync "page" raw.page?
      pure (localPayload raw page .pageClosed, none)
  | "contextUnavailable" =>
      pure (.runtime (.contextUnavailable (← requiredAsync "context" raw.context?)), none)
  | "contextAvailable" =>
      pure (.runtime (.contextAvailable (← requiredAsync "context" raw.context?)), none)
  | "browserDisconnected" => pure (.runtime .browserDisconnected, none)
  | "browserConnected" => pure (.runtime .browserConnected, none)
  | "pauseAutomation" =>
      pure (.command (pauseCommand (← requiredAsync "page" raw.page?)), none)
  | "resumeAutomation" =>
      pure (.command (resumeCommand (← requiredAsync "page" raw.page?)), none)
  | "contextCreated" =>
      let context ← requiredAsync "context" raw.context?
      let epoch ← requiredAsync "nodeEpoch" raw.nodeEpoch?
      pure (.graph (.contextCreated context epoch), none)
  | "contextDestroyed" =>
      pure (.graph (.contextDestroyed (← requiredAsync "context" raw.context?)), none)
  | "pageCreated" =>
      let page ← requiredAsync "page" raw.page?
      let context ← requiredAsync "context" raw.context?
      let epoch ← requiredAsync "nodeEpoch" raw.nodeEpoch?
      pure (.graph (.pageCreated page context epoch), none)
  | "pageDestroyed" =>
      pure (.graph (.pageDestroyed (← requiredAsync "page" raw.page?)), none)
  | "frameCreated" =>
      let frame ← requiredAsync "frame" raw.frame?
      let page ← requiredAsync "page" raw.page?
      let epoch ← requiredAsync "nodeEpoch" raw.nodeEpoch?
      pure (.graph (.frameCreated frame page raw.parentFrame? epoch), none)
  | "frameDestroyed" =>
      pure (.graph (.frameDestroyed (← requiredAsync "frame" raw.frame?)), none)
  | other => throw s!"unknown async payload: {other}"

private def decodeAsyncMessage (raw : RawAsyncMessage) : Except String AsyncDriverRecord := do
  let source ← decodeActor raw.source
  let target ← decodeActor raw.target
  let (payload, declaredAction) ← decodeAsyncPayload raw
  let expected ← decodeDisposition raw.expected
  pure {
    envelope := {
      id := raw.messageId
      cause := {
        correlation := raw.correlationId
        parent := raw.parentId?
        depth := raw.depth
      }
      source := source
      target := target
      payload := payload
    }
    expected := expected
    declaredAction := declaredAction
  }

/-- Parse one line of the async wire format. Bootstrap intentionally reuses the
    existing finite Page/Context declaration. -/
def parseAsyncTraceLine (line : String) : Except String AsyncTraceLine := do
  let json ← Json.parse line
  let tag : AsyncTraceTag ← fromJson? json
  match tag.kind with
  | "bootstrap" =>
      pure (.bootstrap (← fromJson? json))
  | "message" =>
      pure (.message (← decodeAsyncMessage (← fromJson? json)))
  | other => throw s!"unknown async trace kind: {other}"

private def sameCommandEnvelope (expected actual : AsyncEnvelope) : Bool :=
  (expected.id == actual.id) &&
  (expected.cause == actual.cause) &&
  (expected.source == actual.source) &&
  (expected.target == actual.target) &&
  match expected.payload, actual.payload with
  | .command left, .command right => left == right
  | _, _ => false

private def pendingContains (pending : List AsyncEnvelope) (message : AsyncEnvelope) : Bool :=
  pending.any (fun candidate => sameCommandEnvelope candidate message)

private def removePending (pending : List AsyncEnvelope) (id : MessageId) : List AsyncEnvelope :=
  pending.filter (fun candidate => candidate.id != id)

private def isCommand : AsyncPayload → Bool
  | .command _ => true
  | _ => false

private def verifyDeclaredAction
    (record : AsyncDriverRecord) (result : DeliveryResult) : Except String Unit := do
  match record.envelope.payload, record.declaredAction, result.disposition with
  | .runtime (.local { page := page, kind := .automationStarted }), some action, .accepted =>
      if (result.runtime.model.page page).currentAction == some action then
        pure ()
      else
        throw "action generation mismatch"
  | .runtime (.local { kind := .automationStarted, .. }), none, .accepted =>
      throw "actionStarted missing declared action"
  | _, _, _ => pure ()

/-- Verify one externally journaled delivery. An accepted PageCommand must have
    appeared in the pending emitted-message set; this prevents a trace from
    forging synchronous cross-actor policy mutation. -/
def verifyAsyncRecord
    (state : AsyncReplayState) (record : AsyncDriverRecord) : Except String AsyncReplayState := do
  let result := deliver state.runtime record.envelope
  if result.disposition != record.expected then
    throw s!"unexpected delivery disposition"
  if isCommand record.envelope.payload && result.disposition == .accepted &&
      !pendingContains state.pending record.envelope then
    throw "command was not emitted by a prior delivery"
  verifyDeclaredAction record result
  let remaining :=
    if isCommand record.envelope.payload && result.disposition == .accepted then
      removePending state.pending record.envelope.id
    else
      state.pending
  pure {
    runtime := result.runtime
    pending := remaining ++ result.emitted
  }

private def replayAsyncRecords :
    AsyncReplayState → List AsyncDriverRecord → Except String AsyncReplayState
  | state, [] => pure state
  | state, record :: rest => do
      replayAsyncRecords (← verifyAsyncRecord state record) rest

private def parseAsyncMessages : List String → Except String (List AsyncDriverRecord)
  | [] => pure []
  | line :: rest => do
      match ← parseAsyncTraceLine line with
      | .bootstrap _ => throw "bootstrap must be the first line only"
      | .message record => pure (record :: (← parseAsyncMessages rest))

/-- Parse and replay an external asynchronous Driver journal through the exact
    same `deliver` function used by the proofs. -/
def verifyAsyncJsonl (text : String) : Except String AsyncRuntime := do
  let lines := (text.splitOn "\n").filter (fun line => !line.isEmpty)
  match lines with
  | [] => throw "empty async trace"
  | first :: rest =>
      match ← parseAsyncTraceLine first with
      | .message _ => throw "first async trace line must be bootstrap"
      | .bootstrap bootstrap =>
          let initial : AsyncReplayState := {
            runtime := asyncFromModel (modelFromBootstrap bootstrap)
          }
          pure (← replayAsyncRecords initial (← parseAsyncMessages rest)).runtime

end Browser
