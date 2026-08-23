import Browser.Interaction.Projection
import Browser.Interaction.Trace

namespace Browser.Interaction

private def trigger : InteractionCause := {
  id := 50
  correlation := 900
  depth := 0
}

private def scenario : List InteractionTraceEvent := [
  .actionStarted 2,
  .cdpRequest 2 17,
  .cdpResponse 1 17,
  .deadlineArmed 2 100,
  .timerExpired 1 999,
  .humanInput,
  .policyIssued trigger 1,
  .policyDelivered,
  .epochRecreated
]

private def finalState : InteractionTraceState := replayInteractionTrace {} scenario

example : finalState.cdp.pending = some (2, 17) := by native_decide
example : finalState.deadline.timedOut = false := by native_decide
example : finalState.input = .conflict 2 := by native_decide
example : finalState.policy.childVersion = 1 := by native_decide
example : finalState.epoch.epoch = 2 := by native_decide

private def asyncResponse : AsyncEnvelope := {
  id := 77
  cause := { correlation := 1000, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .action 1 2, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .protocolResponse 2 17 })
}

example :
    projectAsyncInteraction asyncResponse =
      some {
        id := 77
        correlation := 1000
        parent := none
        depth := 0
        source := .browser
        target := .action
        event := .cdpResponse 2 17
      } := by
  native_decide

end Browser.Interaction
