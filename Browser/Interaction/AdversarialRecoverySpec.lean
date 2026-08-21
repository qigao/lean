import Browser.Interaction.AdversarialRecovery

namespace Browser.Interaction

example : joinRecovery .retry .recreatePage = .recreatePage := by rfl
example : joinRecovery .recreatePage .retry = .recreatePage := by rfl
example : joinRecovery .reattachSession .recreateContext = .recreateContext := by rfl

example (action : RecoveryAction) : joinRecovery action action = action := by
  exact join_recovery_idempotent action

example (a b : RecoveryAction) : joinRecovery a b = joinRecovery b a := by
  exact join_recovery_commutative a b

example (a b c : RecoveryAction) :
    joinRecovery (joinRecovery a b) c = joinRecovery a (joinRecovery b c) := by
  exact join_recovery_associative a b c

example (a b : RecoveryAction) :
    recoveryRank a ≤ recoveryRank (joinRecovery a b) := by
  exact join_recovery_left_upper a b

example (a b : RecoveryAction) :
    recoveryRank b ≤ recoveryRank (joinRecovery a b) := by
  exact join_recovery_right_upper a b

private def initialSupervisor : FaultSupervisor :=
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
  observeAdversarial initialSupervisor pageFault

example : pageRecovering.failed = false := by rfl
example : (pageRecovering.active.map (fun r => r.action)) = some .recreatePage := by rfl

example :
    observeAdversarial pageRecovering { generation := 9, event := .browserProcessExited } =
      pageRecovering := by
  exact stale_observation_noop pageRecovering _ (by decide)

example :
    observeAdversarial pageRecovering { generation := 11, event := .browserProcessExited } =
      pageRecovering := by
  exact future_observation_noop pageRecovering _ (by decide)

example : observeAdversarial pageRecovering pageFault = pageRecovering := by rfl

private def pageThenContext : FaultSupervisor :=
  observeAdversarial pageRecovering contextFault

private def contextThenPage : FaultSupervisor :=
  observeAdversarial (observeAdversarial initialSupervisor contextFault) pageFault

example : pageThenContext = contextThenPage := by rfl
example : (pageThenContext.active.map (fun r => r.action)) = some .recreateContext := by rfl

private def pageThenSession : FaultSupervisor :=
  observeAdversarial pageRecovering sessionFault

example : (pageThenSession.active.map (fun r => r.action)) = some .recreatePage := by rfl
example : (pageThenSession.active.map (fun r => r.budget)) = some 5 := by rfl

example :
    (observeAdversarial initialSupervisor { generation := 10, event := .malformed }).failed = true := by
  rfl

private def retrying : FaultSupervisor :=
  applyRecoveryFeedback pageRecovering .retryableFailure

example : (retrying.active.map (fun r => r.budget)) = some 4 := by rfl
example : (retrying.active.map (fun r => r.action)) = some .recreatePage := by rfl

private def insufficient : FaultSupervisor :=
  applyRecoveryFeedback pageRecovering .insufficientRepair

example : (insufficient.active.map (fun r => r.budget)) = some 4 := by rfl
example : (insufficient.active.map (fun r => r.action)) = some .recreateContext := by rfl

private def recovered : FaultSupervisor :=
  applyRecoveryFeedback pageRecovering .recovered

example : recovered.generation = 11 := by rfl
example : recovered.active = none := by rfl
example : recovered.failed = false := by rfl

example :
    observeAdversarial recovered { generation := 10, event := .browserProcessExited } = recovered := by
  exact stale_observation_noop recovered _ (by decide)

private def poisonedRecovery : FaultSupervisor :=
  observeAdversarial pageRecovering { generation := 10, event := .malformed }

example : poisonedRecovery.failed = true := by rfl
example : applyRecoveryFeedback poisonedRecovery .recovered = poisonedRecovery := by rfl

private def oneBudget : FaultSupervisor :=
  observeAdversarial (healthySupervisor 30 1) { generation := 30, event := .targetSessionDetached }

private def oneBudgetSpent : FaultSupervisor :=
  applyRecoveryFeedback oneBudget .retryableFailure

private def oneBudgetExhausted : FaultSupervisor :=
  applyRecoveryFeedback oneBudgetSpent .retryableFailure

example : (oneBudgetSpent.active.map (fun r => r.budget)) = some 0 := by rfl
example : oneBudgetExhausted.failed = true := by rfl

private def contextAvailable : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def nothingAvailable : RecoveryAction → Bool := fun _ => false

private def resolvedMerged : FaultSupervisor :=
  resolveSupervisorFuel 4 pageThenContext contextAvailable

example : resolvedMerged.generation = 11 := by rfl
example : resolvedMerged.active = none := by rfl
example : resolvedMerged.failed = false := by rfl
example : supervisorResolved resolvedMerged = true := by rfl

private def resolvedFailure : FaultSupervisor :=
  resolveSupervisorFuel 8 oneBudget nothingAvailable

example : resolvedFailure.failed = true := by rfl
example : resolvedFailure.active = none := by rfl
example : supervisorResolved resolvedFailure = true := by rfl

example
    (fuel : Nat) (supervisor : FaultSupervisor)
    (available : RecoveryAction → Bool) :
    supervisorResolved (resolveSupervisorFuel fuel supervisor available) = true := by
  exact resolve_supervisor_fuel_is_resolved fuel supervisor available

end Browser.Interaction
