import Browser.Interaction.Composition
import Browser.Interaction.Cdp
import Browser.Interaction.Deadline
import Browser.Interaction.Input
import Browser.Interaction.Terminal
import Browser.Interaction.Epoch
import Browser.Interaction.Policy

namespace Browser.Interaction

inductive InteractionTraceEvent where
  | actionStarted (action : ActionId)
  | actionFinished (action : ActionId)
  | cdpRequest (action : ActionId) (call : CdpCallId)
  | cdpResponse (action : ActionId) (call : CdpCallId)
  | deadlineArmed (action : ActionId) (expiresAt : Time)
  | timerExpired (action : ActionId) (now : Time)
  | humanInput
  | policyIssued (trigger : InteractionCause) (page : PageId)
  | policyDelivered
  | epochRecreated
  | actorDestroyed
  deriving Repr, DecidableEq, BEq

/-- One protocol lane, intentionally much smaller than Browser.Model. -/
structure InteractionTraceState where
  cdp : CdpProtocol := {}
  deadline : DeadlineProtocol := {}
  input : InputOwner := .none
  liveness : ActorLiveness := .live
  epoch : EpochProtocol := { epoch := 1, alive := true }
  policy : PolicyProtocol := {}
  pendingPolicy : Option PolicyEmission := none
  deriving Repr, DecidableEq, BEq

private def startAction (state : InteractionTraceState) (action : ActionId) : InteractionTraceState :=
  { state with
    cdp := { currentAction := some action, pending := none }
    deadline := { currentAction := some action, deadline := none, timedOut := false }
    input := .automation action
    liveness := .live }

private def finishAction (state : InteractionTraceState) (action : ActionId) : InteractionTraceState :=
  if state.cdp.currentAction = some action then
    { state with
      cdp := { state.cdp with currentAction := none, pending := none }
      deadline := { state.deadline with currentAction := none, deadline := none }
      input := .none
      liveness := .cancelled }
  else
    state

private def startCdpRequest
    (state : InteractionTraceState) (action : ActionId) (call : CdpCallId) : InteractionTraceState :=
  if state.cdp.currentAction = some action then
    { state with cdp := { state.cdp with pending := some (action, call) } }
  else
    state

private def armTraceDeadline
    (state : InteractionTraceState) (action : ActionId) (expiresAt : Time) : InteractionTraceState :=
  if state.deadline.currentAction = some action then
    { state with deadline := { state.deadline with deadline := some (action, expiresAt) } }
  else
    state

def applyInteractionTraceEvent
    (state : InteractionTraceState) : InteractionTraceEvent → InteractionTraceState
  | .actionStarted action => startAction state action
  | .actionFinished action => finishAction state action
  | .cdpRequest action call => startCdpRequest state action call
  | .cdpResponse action call =>
      { state with cdp := receiveCdpResponse state.cdp action call }
  | .deadlineArmed action expiresAt => armTraceDeadline state action expiresAt
  | .timerExpired action now =>
      { state with deadline := expireDeadline state.deadline action now }
  | .humanInput => { state with input := onHumanInput state.input }
  | .policyIssued trigger page =>
      let (policy, emission) := emitPolicyCommand state.policy trigger page
      { state with policy := policy, pendingPolicy := some emission }
  | .policyDelivered =>
      match state.pendingPolicy with
      | some emission =>
          { state with
            policy := deliverPolicyCommand state.policy emission
            pendingPolicy := none }
      | none => state
  | .epochRecreated => { state with epoch := recreateEpoch state.epoch }
  | .actorDestroyed => { state with liveness := .destroyed }

def replayInteractionTrace : InteractionTraceState → List InteractionTraceEvent → InteractionTraceState
  | state, [] => state
  | state, event :: rest => replayInteractionTrace (applyInteractionTraceEvent state event) rest

/-- Compact executable safety checks over the interaction lane. -/
def interactionTraceSafe (state : InteractionTraceState) : Bool :=
  let terminalOk :=
    match state.liveness with
    | .live => true
    | .cancelled | .timedOut | .destroyed =>
        !mayEmitExternalEffect state.liveness .cdpRequest &&
        !mayEmitExternalEffect state.liveness .inputDispatch
  let inputOk :=
    match state.input with
    | .human => !canAutomationDispatch state.input 0
    | .conflict action => !canAutomationDispatch state.input action
    | _ => true
  terminalOk && inputOk

end Browser.Interaction
