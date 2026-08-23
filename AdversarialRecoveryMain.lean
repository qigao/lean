import Browser.Interaction.AdversarialRecovery

open Browser.Interaction

private def activeAction (supervisor : FaultSupervisor) : Option RecoveryAction :=
  supervisor.active.map (fun recovery => recovery.action)

private def activeBudget (supervisor : FaultSupervisor) : Option Nat :=
  supervisor.active.map (fun recovery => recovery.budget)

private def initial : FaultSupervisor :=
  healthySupervisor 10 5

private def pageFault : TaggedObservation := {
  generation := 10
  event := .targetCrashed .page
}

private def contextFault : TaggedObservation := {
  generation := 10
  event := .contextDestroyed
}

private def sessionFault : TaggedObservation := {
  generation := 10
  event := .targetSessionDetached
}

private def pageRecovering : FaultSupervisor :=
  observeAdversarial initial pageFault

private def duplicatePage : FaultSupervisor :=
  observeAdversarial pageRecovering pageFault

private def pageThenContext : FaultSupervisor :=
  observeAdversarial pageRecovering contextFault

private def contextThenPage : FaultSupervisor :=
  observeAdversarial (observeAdversarial initial contextFault) pageFault

private def pageThenSession : FaultSupervisor :=
  observeAdversarial pageRecovering sessionFault

private def futureIgnored : FaultSupervisor :=
  observeAdversarial pageRecovering {
    generation := 11
    event := .browserProcessExited
  }

private def contextAvailable : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def recovered : FaultSupervisor :=
  resolveSupervisorFuel 4 pageThenContext contextAvailable

private def staleAfterRecovery : FaultSupervisor :=
  observeAdversarial recovered {
    generation := 10
    event := .browserProcessExited
  }

private def poisoned : FaultSupervisor :=
  observeAdversarial pageRecovering {
    generation := 10
    event := .malformed
  }

private def poisonedAfterFakeSuccess : FaultSupervisor :=
  applyRecoveryFeedback poisoned .recovered

private def budgetOne : FaultSupervisor :=
  observeAdversarial (healthySupervisor 30 1) {
    generation := 30
    event := .targetSessionDetached
  }

private def budgetSpent : FaultSupervisor :=
  applyRecoveryFeedback budgetOne .retryableFailure

private def budgetExhausted : FaultSupervisor :=
  applyRecoveryFeedback budgetSpent .retryableFailure

private def scenarioVerified : Bool :=
  (duplicatePage == pageRecovering) &&
  (pageThenContext == contextThenPage) &&
  (activeAction pageThenContext == some .recreateContext) &&
  (activeAction pageThenSession == some .recreatePage) &&
  (activeBudget pageThenSession == some 5) &&
  (futureIgnored == pageRecovering) &&
  (recovered.generation == 11) &&
  (recovered.active == none) &&
  (!recovered.failed) &&
  supervisorResolved recovered &&
  (staleAfterRecovery == recovered) &&
  poisoned.failed &&
  (poisonedAfterFakeSuccess == poisoned) &&
  (activeBudget budgetSpent == some 0) &&
  budgetExhausted.failed &&
  supervisorResolved budgetExhausted

def main : IO UInt32 := do
  if scenarioVerified then
    IO.println "browser adversarial recovery: join + generation isolation + bounded failure verified"
    return 0
  else
    IO.eprintln "browser adversarial recovery: violation"
    return 1
