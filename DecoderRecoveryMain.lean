import Browser.Interaction.DecoderRecovery

open Browser.Interaction

private def recreateContextAvailable : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def reattachSessionAvailable : RecoveryAction → Bool
  | .reattachSession => true
  | _ => false

private def unavailable : RecoveryAction → Bool := fun _ => false

private def pageCrashResult : Option RecoveryState :=
  recoverAdapterEventFuel 4 10 4 recreateContextAvailable (.targetCrashed .page)

private def sessionDetachResult : Option RecoveryState :=
  recoverAdapterEventFuel 2 20 2 reattachSessionAvailable .targetSessionDetached

private def malformedResult : Option RecoveryState :=
  recoverAdapterEventFuel 4 30 4 unavailable .malformed

private def underScopedResult : Option RecoveryState :=
  recoverAdapterEventFuel 4 40 4 unavailable (.targetCrashed .worker)

private def blockedNetworkResult : Option RecoveryState :=
  recoverAdapterEventFuel 4 50 4 unavailable (.networkLoadingFailed .blocked)

private def transientNetworkResult : Option RecoveryState :=
  recoverAdapterEventFuel 4 60 4 unavailable (.networkLoadingFailed .transient)

private def isHealthyGeneration (expected : Nat) : Option RecoveryState → Bool
  | some state => state.status == .healthy && state.generation == expected
  | none => false

private def isFailed : Option RecoveryState → Bool
  | some state => state.status == .failed
  | none => false

private def scenarioVerified : Bool :=
  (decideAdapterEvent (.targetCrashed .page) == .recover .recreatePage) &&
  (decideAdapterEvent .targetSessionDetached == .recover .reattachSession) &&
  (decideAdapterEvent (.networkLoadingFailed .transient) == .wait) &&
  (decideAdapterEvent (.networkLoadingFailed .cancelled) == .wait) &&
  (decideAdapterEvent (.networkLoadingFailed .blocked) == .failSafe) &&
  (decideAdapterEvent (.targetCrashed .worker) == .failSafe) &&
  (decideAdapterEvent .malformed == .failSafe) &&
  isHealthyGeneration 11 pageCrashResult &&
  isHealthyGeneration 21 sessionDetachResult &&
  isFailed malformedResult &&
  isFailed underScopedResult &&
  isFailed blockedNetworkResult &&
  transientNetworkResult == none

def main : IO UInt32 := do
  if scenarioVerified then
    IO.println "browser decoder recovery contracts: observation-to-convergence verified"
    return 0
  else
    IO.eprintln "browser decoder recovery contracts: violation"
    return 1
