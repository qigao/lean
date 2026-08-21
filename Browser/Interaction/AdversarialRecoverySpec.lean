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

end Browser.Interaction
