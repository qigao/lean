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
  let running := reactSafe initial (.local { page := 1, kind := .automationStarted })
  reactSafe running (.contextUnavailable 10)

private def contextIsolationOk : Bool :=
  ((afterContextPause.page 1).action == .suspended) &&
  ((afterContextPause.page 2).action == .idle) &&
  (!(afterContextPause.contextAvailable 10)) &&
  afterContextPause.contextAvailable 20

/-- Regression: a local automation start cannot resurrect execution while the
    Browser parent is disconnected. -/
private def disconnectedStartState : Model :=
  let disconnected := reactSafe initial .browserDisconnected
  reactSafe disconnected (.local { page := 1, kind := .automationStarted })

private def disconnectedStartBlocked : Bool :=
  ((disconnectedStartState.page 1).action == .suspended) &&
  ((disconnectedStartState.page 1).input == .idle)

/-- Regression: a local automation start cannot resurrect execution while its
    BrowserContext parent is unavailable. -/
private def contextUnavailableStartState : Model :=
  let unavailable := reactSafe initial (.contextUnavailable 10)
  reactSafe unavailable (.local { page := 1, kind := .automationStarted })

private def contextUnavailableStartBlocked : Bool :=
  ((contextUnavailableStartState.page 1).action == .suspended) &&
  ((contextUnavailableStartState.page 1).input == .idle) &&
  ((contextUnavailableStartState.page 2).action == .idle)

private def scenario : List RuntimeEvent := [
  .local { page := 1, kind := .automationStarted },
  .contextUnavailable 10,
  .contextAvailable 10,
  .local { page := 1, kind := .automationStarted },
  .local { page := 1, kind := .automationFinished 2 },
  .local { page := 2, kind := .automationStarted },
  .browserDisconnected,
  .browserConnected,
  .local { page := 2, kind := .automationStarted },
  .local { page := 2, kind := .automationFinished 2 }
]

private def humanTimeoutState : Model :=
  let running := reactSafe initial (.local { page := 1, kind := .automationStarted })
  let armed := reactSafe running (.local { page := 1, kind := .deadlineArmed 1 10 .action })
  let human := reactSafe armed (.local { page := 1, kind := .humanInput })
  reactSafe human (.local { page := 1, kind := .deadlineReached 1 10 })

private def pageTimeoutState : Model :=
  let running := reactSafe initial (.local { page := 1, kind := .automationStarted })
  let armed := reactSafe running (.local { page := 1, kind := .deadlineArmed 1 20 .action })
  let navigating := reactSafe armed (.local { page := 1, kind := .navigationStarted })
  reactSafe navigating (.local { page := 1, kind := .deadlineReached 1 20 })

private def networkTimeoutState : Model :=
  let running := reactSafe initial (.local { page := 1, kind := .automationStarted })
  let armed := reactSafe running (.local { page := 1, kind := .deadlineArmed 1 30 .network })
  reactSafe armed (.local { page := 1, kind := .deadlineReached 1 30 })

private def protocolTimeoutState : Model :=
  let running := reactSafe initial (.local { page := 1, kind := .automationStarted })
  let waiting := reactSafe running (.local { page := 1, kind := .protocolWaitStarted 1 400 })
  let armed := reactSafe waiting (.local { page := 1, kind := .deadlineArmed 1 40 .protocol })
  reactSafe armed (.local { page := 1, kind := .deadlineReached 1 40 })

private def restartAfterTimeoutState : Model :=
  reactSafe networkTimeoutState (.local { page := 1, kind := .automationStarted })

private def timeoutScenarioOk : Bool :=
  ((humanTimeoutState.page 1).action == .timedOut) &&
  ((humanTimeoutState.page 1).timeout == some .human) &&
  ((pageTimeoutState.page 1).action == .timedOut) &&
  ((pageTimeoutState.page 1).timeout == some .page) &&
  ((networkTimeoutState.page 1).action == .timedOut) &&
  ((networkTimeoutState.page 1).timeout == some .network) &&
  ((protocolTimeoutState.page 1).action == .timedOut) &&
  ((protocolTimeoutState.page 1).timeout == some .protocol) &&
  ((networkTimeoutState.page 2).action == .idle) &&
  ((restartAfterTimeoutState.page 1).action == .executing) &&
  ((restartAfterTimeoutState.page 1).currentAction == some 2) &&
  ((restartAfterTimeoutState.page 1).timeout == none) &&
  ((restartAfterTimeoutState.page 1).deadline == none)

/-- Action 1 leaves callbacks behind; Action 2 starts and waits on its own call.
    Late Action-1 timer, CDP response and duplicate finish must not mutate Action 2. -/
private def generationRaceState : Model :=
  let action1 := reactSafe initial (.local { page := 1, kind := .automationStarted })
  let wait1 := reactSafe action1 (.local { page := 1, kind := .protocolWaitStarted 1 100 })
  let armed1 := reactSafe wait1 (.local { page := 1, kind := .deadlineArmed 1 10 .protocol })
  let finished1 := reactSafe armed1 (.local { page := 1, kind := .automationFinished 1 })
  let action2 := reactSafe finished1 (.local { page := 1, kind := .automationStarted })
  let wait2 := reactSafe action2 (.local { page := 1, kind := .protocolWaitStarted 2 200 })
  let armed2 := reactSafe wait2 (.local { page := 1, kind := .deadlineArmed 2 100 .protocol })
  let staleTimer := reactSafe armed2 (.local { page := 1, kind := .deadlineReached 1 999 })
  let staleResponse := reactSafe staleTimer (.local { page := 1, kind := .protocolResponse 1 100 })
  reactSafe staleResponse (.local { page := 1, kind := .automationFinished 1 })

private def generationRaceAfterMatchingResponse : Model :=
  reactSafe generationRaceState (.local { page := 1, kind := .protocolResponse 2 200 })

private def generationRaceOk : Bool :=
  ((generationRaceState.page 1).currentAction == some 2) &&
  ((generationRaceState.page 1).actionGeneration == 2) &&
  ((generationRaceState.page 1).action == .waiting) &&
  ((generationRaceState.page 1).timeout == none) &&
  ((generationRaceState.page 1).deadline == some { action := 2, expiresAt := 100 }) &&
  ((generationRaceState.page 1).pendingProtocol == some { action := 2, call := 200 }) &&
  ((generationRaceAfterMatchingResponse.page 1).pendingProtocol == none) &&
  ((generationRaceAfterMatchingResponse.page 1).currentAction == some 2) &&
  ((generationRaceState.page 2).actionGeneration == 0)

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

private def actionDependencyScenarioOk : Bool :=
  disconnectedStartBlocked && contextUnavailableStartBlocked

def main : IO Unit := do
  let final := replay initial scenario
  if contextIsolationOk &&
      actionDependencyScenarioOk &&
      timeoutScenarioOk &&
      generationRaceOk &&
      traceSafe initial scenario &&
      wellFormed? final &&
      causalScenarioOk then
    IO.println "browser-runtime formal model: policy + causality + dependencies + timeout + generation isolation verified"
  else
    throw <| IO.userError "browser-runtime formal model: runtime invariant violated"
