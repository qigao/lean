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

end Browser.Interaction
