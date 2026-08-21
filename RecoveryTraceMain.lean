import Browser.Interaction.RecoveryTrace

open Browser.Interaction

private def contextAvailable : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def unavailable : RecoveryAction → Bool := fun _ => false

private def recoveredScenario : RecoveryState :=
  runRecoveryFuel 4 (beginRecovery 100 .pageLost 4) contextAvailable

private def failedScenario : RecoveryState :=
  runRecoveryFuel 8 (beginRecovery 200 .sessionLost 8) unavailable

private def scenarioVerified : Bool :=
  (recoveredScenario.status == .healthy) &&
  (recoveredScenario.generation == 101) &&
  (failedScenario.status == .failed) &&
  recoveryTerminal recoveredScenario &&
  recoveryTerminal failedScenario

def main : IO UInt32 := do
  if scenarioVerified then
    IO.println "browser recovery contracts: bounded convergence verified"
    return 0
  else
    IO.eprintln "browser recovery contracts: violation"
    return 1
