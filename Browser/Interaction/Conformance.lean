import Browser.Interaction.Trace
import Lean.Data.Json

namespace Browser.Interaction

open Lean

private structure RawInteractionRecord where
  kind : String
  event : String
  action? : Option ActionId := none
  call? : Option CdpCallId := none
  expiresAt? : Option Time := none
  now? : Option Time := none
  eventId? : Option EventId := none
  correlationId? : Option CorrelationId := none
  depth? : Option Nat := none
  page? : Option PageId := none
  deriving FromJson

private def requiredField {α : Type} (name : String) : Option α → Except String α
  | some value => pure value
  | none => throw s!"missing {name}"

private def decodeInteraction (raw : RawInteractionRecord) : Except String InteractionTraceEvent := do
  match raw.event with
  | "actionStarted" =>
      pure (.actionStarted (← requiredField "action" raw.action?))
  | "actionFinished" =>
      pure (.actionFinished (← requiredField "action" raw.action?))
  | "cdpRequest" =>
      let action ← requiredField "action" raw.action?
      let call ← requiredField "call" raw.call?
      pure (.cdpRequest action call)
  | "cdpResponse" =>
      let action ← requiredField "action" raw.action?
      let call ← requiredField "call" raw.call?
      pure (.cdpResponse action call)
  | "deadlineArmed" =>
      let action ← requiredField "action" raw.action?
      let expiresAt ← requiredField "expiresAt" raw.expiresAt?
      pure (.deadlineArmed action expiresAt)
  | "timerExpired" =>
      let action ← requiredField "action" raw.action?
      let now ← requiredField "now" raw.now?
      pure (.timerExpired action now)
  | "inputDispatch" =>
      pure (.inputDispatch (← requiredField "action" raw.action?))
  | "humanInput" => pure .humanInput
  | "policyIssued" =>
      let id ← requiredField "eventId" raw.eventId?
      let correlation ← requiredField "correlationId" raw.correlationId?
      let page ← requiredField "page" raw.page?
      pure (.policyIssued {
        id := id
        correlation := correlation
        depth := raw.depth?.getD 0
      } page)
  | "policyDelivered" => pure .policyDelivered
  | "epochRecreated" => pure .epochRecreated
  | "actorDestroyed" => pure .actorDestroyed
  | other => throw s!"unknown interaction event: {other}"

/-- Stable line-oriented wire contract for the C/C++ Driver EventJournal. -/
def parseInteractionLine (line : String) : Except String InteractionTraceEvent := do
  let json ← Json.parse line
  let raw : RawInteractionRecord ← fromJson? json
  if raw.kind != "interaction" then
    throw s!"unknown interaction record kind: {raw.kind}"
  decodeInteraction raw

private def verifyInteractionEvents :
    InteractionTraceState → List InteractionTraceEvent → Except String InteractionTraceState
  | state, [] => pure state
  | state, event :: rest => do
      if !interactionEventAllowed state event then
        throw s!"interaction not authorized: {repr event}"
      let next := applyInteractionTraceEvent state event
      if !interactionTraceSafe next then
        throw s!"interaction invariant violated after: {repr event}"
      verifyInteractionEvents next rest

private def parseInteractionLines : List String → Except String (List InteractionTraceEvent)
  | [] => pure []
  | line :: rest => do
      pure ((← parseInteractionLine line) :: (← parseInteractionLines rest))

/-- Verify a compact interaction-only JSONL journal without reconstructing the
    complete Browser.Model or AsyncRuntime. -/
def verifyInteractionJsonl (text : String) : Except String InteractionTraceState := do
  let lines := (text.splitOn "\n").filter (fun line => !line.isEmpty)
  if lines.isEmpty then
    throw "empty interaction trace"
  verifyInteractionEvents {} (← parseInteractionLines lines)

end Browser.Interaction
