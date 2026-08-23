import Browser.Interaction.RecoveryTrace

namespace Browser.Interaction

private def sessionStart : RecoveryState :=
  beginRecovery 10 .sessionLost 3

private def sessionRecovered : RecoveryState :=
  replayRecovery sessionStart [.retryableFailure, .recovered]

example : sessionRecovered.status = .healthy := by rfl
example : sessionRecovered.generation = 11 := by rfl
example : sessionRecovered.generation ≠ sessionStart.generation := by native_decide

private def pageStart : RecoveryState :=
  beginRecovery 20 .pageLost 3

private def pageRecovered : RecoveryState :=
  replayRecovery pageStart [.insufficientRepair, .recovered]

example : pageRecovered.status = .healthy := by rfl
example : pageRecovered.generation = 21 := by rfl

private def exhausted : RecoveryState :=
  replayRecovery (beginRecovery 30 .sessionLost 1)
    [.retryableFailure, .retryableFailure]

example : exhausted.status = .failed := by rfl
example : recoveryTerminal exhausted = true := by rfl

private def sessionAvailable : RecoveryAction → Bool
  | .reattachSession => true
  | _ => false

private def contextAvailable : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def unavailable : RecoveryAction → Bool := fun _ => false

example :
    (runRecoveryFuel 3 (beginRecovery 40 .sessionLost 3) sessionAvailable).status = .healthy := by
  rfl

example :
    (runRecoveryFuel 3 (beginRecovery 50 .pageLost 3) contextAvailable).status = .healthy := by
  rfl

example :
    (runRecoveryFuel 8 (beginRecovery 60 .sessionLost 8) unavailable).status = .failed := by
  rfl

example (fuel : Nat) (state : RecoveryState) (available : RecoveryAction → Bool) :
    recoveryTerminal (runRecoveryFuel fuel state available) = true := by
  exact run_recovery_fuel_is_terminal fuel state available

end Browser.Interaction
