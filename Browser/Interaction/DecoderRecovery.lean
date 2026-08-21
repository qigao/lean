import Browser.Interaction.Decoder
import Browser.Interaction.FeedbackRecovery

namespace Browser.Interaction

/-- Decision composition for the stable adapter event surface. Decoder does not
    choose policy; it only feeds the existing normalizer/decision contract. -/
def decideAdapterEvent (event : AdapterEvent) : Decision :=
  decideNormalized (normalizeObservation (decodeEvent event))

/-- Bounded recovery composition from adapter event through the existing raw
    observation recovery contract. -/
def recoverAdapterEventFuel
    (fuel generation budget : Nat)
    (available : RecoveryAction → Bool)
    (event : AdapterEvent) : Option RecoveryState :=
  recoverObservationFuel fuel generation budget available (decodeEvent event)

theorem recover_adapter_event_fuel_is_resolved
    (fuel generation budget : Nat)
    (available : RecoveryAction → Bool)
    (event : AdapterEvent) :
    recoveryResultResolved
      (recoverAdapterEventFuel fuel generation budget available event) = true := by
  exact recover_observation_fuel_is_resolved
    fuel generation budget available (decodeEvent event)

theorem malformed_adapter_event_fails_safe :
    decideAdapterEvent .malformed = .failSafe := by
  rfl

theorem unknown_adapter_event_fails_safe (name : String) :
    decideAdapterEvent (.unknownSignal name) = .failSafe := by
  rfl

theorem page_crash_selects_page_recovery :
    decideAdapterEvent (.targetCrashed .page) = .recover .recreatePage := by
  rfl

theorem non_page_target_crash_fails_safe :
    decideAdapterEvent (.targetCrashed .worker) = .failSafe := by
  rfl

end Browser.Interaction
