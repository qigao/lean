import Browser.Interaction.FeedbackRecovery

namespace Browser.Interaction

private def contextRepairOnly : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def noRepair : RecoveryAction → Bool := fun _ => false

private def pageRecovered : Option RecoveryState :=
  recoverObservationFuel 4 10 4 contextRepairOnly .pageCrashed

example : pageRecovered.isSome = true := by rfl
example : (pageRecovered.getD {}).status = .healthy := by rfl
example : (pageRecovered.getD {}).generation = 11 := by rfl

private def browserFailed : Option RecoveryState :=
  recoverObservationFuel 4 3 4 noRepair .browserExited

example : browserFailed.isSome = true := by rfl
example : (browserFailed.getD {}).status = .failed := by rfl
example : (browserFailed.getD {}).generation = 3 := by rfl

private def timeoutFailed : Option RecoveryState :=
  recoverObservationFuel 4 8 4 (fun _ => true) .deadlineReached

example : (timeoutFailed.getD {}).status = .failed := by rfl
example : (timeoutFailed.getD {}).generation = 8 := by rfl

private def unknownFailed : Option RecoveryState :=
  recoverObservationFuel 4 9 4 (fun _ => true) .unrecognized

example : (unknownFailed.getD {}).status = .failed := by rfl
example : (unknownFailed.getD {}).generation = 9 := by rfl

example : recoverObservationFuel 4 1 4 noRepair .humanInput = none := by rfl
example : recoverObservationFuel 4 1 4 noRepair .networkTransient = none := by rfl

example
    (fuel generation budget : Nat)
    (available : RecoveryAction → Bool)
    (raw : RawObservation) :
    recoveryResultResolved
      (recoverObservationFuel fuel generation budget available raw) = true := by
  exact recover_observation_fuel_is_resolved fuel generation budget available raw

end Browser.Interaction
