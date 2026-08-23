import Browser.Interaction.DecoderRecovery

namespace Browser.Interaction

private def recreateContextAvailable : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def unavailable : RecoveryAction → Bool := fun _ => false

example : decideAdapterEvent (.targetCrashed .page) = .recover .recreatePage := by rfl
example : decideAdapterEvent .targetSessionDetached = .recover .reattachSession := by rfl
example : decideAdapterEvent (.networkLoadingFailed .transient) = .wait := by rfl
example : decideAdapterEvent (.networkLoadingFailed .cancelled) = .wait := by rfl
example : decideAdapterEvent (.networkLoadingFailed .blocked) = .failSafe := by rfl
example : decideAdapterEvent (.targetCrashed .worker) = .failSafe := by rfl
example : decideAdapterEvent .malformed = .failSafe := by rfl

private def pageCrashRecovery : Option RecoveryState :=
  recoverAdapterEventFuel 4 100 4 recreateContextAvailable (.targetCrashed .page)

example : pageCrashRecovery = some {
    status := .healthy
    generation := 101
    fault := none
    action := .retry
    budget := 3
  } := by rfl

private def malformedRecovery : Option RecoveryState :=
  recoverAdapterEventFuel 4 200 4 unavailable .malformed

example : malformedRecovery = some {
    status := .failed
    generation := 200
    fault := some .unknownFault
    action := .fail
    budget := 4
  } := by rfl

example :
    recoverAdapterEventFuel 4 300 4 unavailable (.networkLoadingFailed .transient) = none := by
  rfl

example (fuel generation budget : Nat)
    (available : RecoveryAction → Bool) (event : AdapterEvent) :
    recoveryResultResolved
      (recoverAdapterEventFuel fuel generation budget available event) = true := by
  exact recover_adapter_event_fuel_is_resolved fuel generation budget available event

end Browser.Interaction
