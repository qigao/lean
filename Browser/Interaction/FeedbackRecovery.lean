import Browser.Interaction.Normalizer
import Browser.Interaction.RecoveryTrace

namespace Browser.Interaction

/-- Compose a raw observation with bounded recovery only when normalization
    identifies a concrete fault. Non-fault feedback remains in the ordinary
    feedback/decision loop and therefore returns `none` here. -/
def recoverObservationFuel
    (fuel generation budget : Nat)
    (available : RecoveryAction → Bool)
    (raw : RawObservation) : Option RecoveryState :=
  match (normalizeObservation raw).fault with
  | none => none
  | some fault =>
      some (runRecoveryFuel fuel (beginRecovery generation fault budget) available)

/-- `none` means recovery was not required. Any actual recovery result must be
    terminal: healthy or failed, never internally stuck in recovering. -/
def recoveryResultResolved : Option RecoveryState → Bool
  | none => true
  | some state => recoveryTerminal state

theorem recover_observation_fuel_is_resolved
    (fuel generation budget : Nat)
    (available : RecoveryAction → Bool)
    (raw : RawObservation) :
    recoveryResultResolved
      (recoverObservationFuel fuel generation budget available raw) = true := by
  cases h : (normalizeObservation raw).fault with
  | none =>
      simp [recoverObservationFuel, recoveryResultResolved, h]
  | some fault =>
      simpa [recoverObservationFuel, recoveryResultResolved, h] using
        run_recovery_fuel_is_terminal
          fuel (beginRecovery generation fault budget) available

theorem no_fault_needs_no_recovery
    (fuel generation budget : Nat)
    (available : RecoveryAction → Bool)
    (raw : RawObservation)
    (h : (normalizeObservation raw).fault = none) :
    recoverObservationFuel fuel generation budget available raw = none := by
  simp [recoverObservationFuel, h]

theorem fault_recovery_uses_normalized_fault
    (fuel generation budget : Nat)
    (available : RecoveryAction → Bool)
    (raw : RawObservation) (fault : FaultClass)
    (h : (normalizeObservation raw).fault = some fault) :
    recoverObservationFuel fuel generation budget available raw =
      some (runRecoveryFuel fuel (beginRecovery generation fault budget) available) := by
  simp [recoverObservationFuel, h]

end Browser.Interaction
