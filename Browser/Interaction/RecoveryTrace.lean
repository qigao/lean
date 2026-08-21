import Browser.Interaction.Recovery

namespace Browser.Interaction

/-- Replay observed recovery feedback without reconstructing Browser runtime state. -/
def replayRecovery : RecoveryState → List RecoveryFeedback → RecoveryState
  | state, [] => state
  | state, feedback :: rest =>
      replayRecovery (stepRecovery state feedback) rest

def recoveryTerminal (state : RecoveryState) : Bool :=
  match state.status with
  | .healthy | .failed => true
  | .recovering => false

/--
`available action = true` abstracts the environment assumption that a repair
at this level succeeds now. `fuel` bounds internal escalation. Fuel exhaustion
turns a still-recovering state into explicit failure, so the Driver cannot
livelock internally even if no external repair ever becomes available.
-/
def runRecoveryFuel :
    Nat → RecoveryState → (RecoveryAction → Bool) → RecoveryState
  | 0, state, _ =>
      match state.status with
      | .recovering => { state with status := .failed }
      | .healthy | .failed => state
  | fuel + 1, state, available =>
      match state.status with
      | .healthy | .failed => state
      | .recovering =>
          if available state.action then
            stepRecovery state .recovered
          else
            runRecoveryFuel fuel (stepRecovery state .insufficientRepair) available

theorem run_recovery_fuel_is_terminal
    (fuel : Nat) (state : RecoveryState)
    (available : RecoveryAction → Bool) :
    recoveryTerminal (runRecoveryFuel fuel state available) = true := by
  induction fuel generalizing state with
  | zero =>
      cases h : state.status <;>
        simp [runRecoveryFuel, recoveryTerminal, h]
  | succ fuel ih =>
      cases h : state.status with
      | healthy =>
          simp [runRecoveryFuel, recoveryTerminal, h]
      | recovering =>
          cases hav : available state.action with
          | false =>
              simpa [runRecoveryFuel, h, hav] using
                ih (stepRecovery state .insufficientRepair)
          | true =>
              simp [runRecoveryFuel, h, hav, stepRecovery, recoveryTerminal]
      | failed =>
          simp [runRecoveryFuel, recoveryTerminal, h]

theorem replay_recovered_is_fresh
    (state : RecoveryState) (h : state.status = .recovering) :
    (replayRecovery state [.recovered]).generation = state.generation + 1 := by
  simp [replayRecovery, stepRecovery, h]

end Browser.Interaction
