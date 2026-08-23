import Browser.Interaction.Recovery

namespace Browser.Interaction

example : minimumRecovery .elementStale = .reResolve := by rfl
example : minimumRecovery .runtimeLost = .rebindRuntime := by rfl
example : minimumRecovery .sessionLost = .reattachSession := by rfl
example : minimumRecovery .pageLost = .recreatePage := by rfl
example : minimumRecovery .contextLost = .recreateContext := by rfl
example : minimumRecovery .browserLost = .restartBrowser := by rfl
example : minimumRecovery .timeoutFault = .fail := by rfl
example : minimumRecovery .unknownFault = .fail := by rfl

example (fault : FaultClass) :
    MinimalSufficient fault (minimumRecovery fault) := by
  exact minimum_recovery_is_minimal fault

private def sessionRecovery : RecoveryState :=
  beginRecovery 7 .sessionLost 3

example : sessionRecovery.status = .recovering := by rfl
example : sessionRecovery.generation = 7 := by rfl
example : sessionRecovery.action = .reattachSession := by rfl
example : sessionRecovery.budget = 3 := by rfl

private def timeoutRecovery : RecoveryState :=
  beginRecovery 9 .timeoutFault 3

private def unknownRecovery : RecoveryState :=
  beginRecovery 9 .unknownFault 3

example : timeoutRecovery.status = .failed := by rfl
example : unknownRecovery.status = .failed := by rfl
example : (stepRecovery timeoutRecovery .recovered).status = .failed := by rfl
example : (stepRecovery unknownRecovery .recovered).status = .failed := by rfl

example : (stepRecovery sessionRecovery .recovered).status = .healthy := by rfl
example : (stepRecovery sessionRecovery .recovered).generation = 8 := by rfl
example : (stepRecovery sessionRecovery .recovered).fault = none := by rfl

example : (stepRecovery sessionRecovery .retryableFailure).status = .recovering := by rfl
example : (stepRecovery sessionRecovery .retryableFailure).budget = 2 := by rfl
example : (stepRecovery sessionRecovery .retryableFailure).action = .reattachSession := by rfl

example : (stepRecovery sessionRecovery .insufficientRepair).budget = 2 := by rfl
example : (stepRecovery sessionRecovery .insufficientRepair).action = .recreatePage := by rfl

private def exhausted : RecoveryState :=
  beginRecovery 4 .pageLost 0

example : (stepRecovery exhausted .retryableFailure).status = .failed := by rfl
example : (stepRecovery exhausted .insufficientRepair).status = .failed := by rfl
example : (stepRecovery sessionRecovery .fatalFailure).status = .failed := by rfl

theorem successful_recovery_is_fresh
    (s : RecoveryState) (h : s.status = .recovering) :
    (stepRecovery s .recovered).generation = s.generation + 1 ∧
    (stepRecovery s .recovered).status = .healthy := by
  simp [stepRecovery, h]

end Browser.Interaction
