import Browser.Async
import Browser.Interaction.Trace

namespace Browser.Interaction

structure ProjectedInteraction where
  id : MessageId
  correlation : CorrelationId
  parent : Option MessageId
  depth : Nat
  source : InteractionRole
  target : InteractionRole
  event : InteractionTraceEvent
  deriving Repr, DecidableEq, BEq

def roleOfActor : ActorRef → InteractionRole
  | .browser => .browser
  | .context _ => .context
  | .page _ => .page
  | .frame _ => .frame
  | .action _ _ => .action

private def projected
    (envelope : AsyncEnvelope) (event : InteractionTraceEvent) : ProjectedInteraction := {
  id := envelope.id
  correlation := envelope.cause.correlation
  parent := envelope.cause.parent
  depth := envelope.cause.depth
  source := roleOfActor envelope.source.actor
  target := roleOfActor envelope.target.actor
  event := event
}

/-- Thin adapter from the integration model into the small interaction proof
    vocabulary. Unrelated browser lifecycle events are intentionally ignored. -/
def projectAsyncInteraction (envelope : AsyncEnvelope) : Option ProjectedInteraction :=
  match envelope.payload with
  | .runtime (.local { kind := .automationFinished action, .. }) =>
      some (projected envelope (.actionFinished action))
  | .runtime (.local { kind := .protocolWaitStarted action call, .. }) =>
      some (projected envelope (.cdpRequest action call))
  | .runtime (.local { kind := .protocolResponse action call, .. }) =>
      some (projected envelope (.cdpResponse action call))
  | .runtime (.local { kind := .deadlineArmed action expiresAt _, .. }) =>
      some (projected envelope (.deadlineArmed action expiresAt))
  | .runtime (.local { kind := .deadlineReached action now, .. }) =>
      some (projected envelope (.timerExpired action now))
  | .runtime (.local { kind := .humanInput, .. }) =>
      some (projected envelope .humanInput)
  | .command _ =>
      some (projected envelope .policyDelivered)
  | _ => none

end Browser.Interaction
