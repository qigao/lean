import Browser.Interaction.Trace

open Browser
open Browser.Interaction

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

private def scenarioAccepted (state : InteractionTraceState) : Bool :=
  (state.cdp.pending == some (2, 17)) &&
  (state.deadline.timedOut == false) &&
  (state.input == .conflict 2) &&
  (state.policy.childVersion == 1) &&
  (state.epoch.epoch == 2) &&
  interactionTraceSafe state

def main : IO UInt32 := do
  let final := replayInteractionTrace {} scenario
  if scenarioAccepted final then
    IO.println "browser interaction contracts: verified"
    return 0
  else
    IO.eprintln "browser interaction contracts: violation"
    return 1
