import Browser

open Browser

private def graph : RuntimeGraph := {
  contextOf := fun p => if p = 1 then 10 else 20
  pageOfFrame := fun f => f
  parentOfFrame := fun _ => none
}

private def initial : Model := {
  browserAlive := true
  graph := graph
  knownPages := [1, 2]
  contextAvailable := fun _ => true
  page := fun _ => PageState.readyIdle
}

private def afterContextPause : Model :=
  let running := react initial (.local { page := 1, kind := .automationStarted })
  react running (.contextUnavailable 10)

private def contextIsolationOk : Bool :=
  ((afterContextPause.page 1).action == .suspended) &&
  ((afterContextPause.page 2).action == .idle) &&
  (!(afterContextPause.contextAvailable 10)) &&
  afterContextPause.contextAvailable 20

private def scenario : List RuntimeEvent := [
  .local { page := 1, kind := .automationStarted },
  .contextUnavailable 10,
  .contextAvailable 10,
  .local { page := 1, kind := .automationStarted },
  .local { page := 1, kind := .automationFinished },
  .local { page := 2, kind := .automationStarted },
  .browserDisconnected,
  .browserConnected,
  .local { page := 2, kind := .automationStarted },
  .local { page := 2, kind := .automationFinished }
]

private def budget : ReactionBudget := { maxDepth := 3 }

private def causalEnvelope : EventEnvelope := {
  id := 100
  event := .contextUnavailable 10
  cause := { correlation := 9001, parent := none, depth := 0 }
}

private def overBudgetEnvelope : EventEnvelope := {
  id := 101
  event := .contextUnavailable 10
  cause := { correlation := 9001, parent := some 100, depth := 4 }
}

private def causalReplayAccepted : Bool :=
  match replayEnvelopes budget initial [causalEnvelope] with
  | some _ => true
  | none => false

private def overBudgetReplayRejected : Bool :=
  match replayEnvelopes budget initial [overBudgetEnvelope] with
  | none => true
  | some _ => false

private def causalScenarioOk : Bool :=
  causalPolicySafe initial causalEnvelope &&
  causalReplayAccepted &&
  overBudgetReplayRejected

def main : IO Unit := do
  let final := replay initial scenario
  if contextIsolationOk && traceSafe initial scenario && wellFormed? final && causalScenarioOk then
    IO.println "browser-runtime formal model: scoped policy + causal budget verification passed"
  else
    throw <| IO.userError "browser-runtime formal model: runtime invariant violated"
