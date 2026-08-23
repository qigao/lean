import Browser.Interaction.FeedbackRecovery

open Browser.Interaction

private def contextRepairOnly : RecoveryAction → Bool
  | .recreateContext => true
  | _ => false

private def noRepair : RecoveryAction → Bool := fun _ => false

private def pageRecovery : Option RecoveryState :=
  recoverObservationFuel 4 100 4 contextRepairOnly .pageCrashed

private def browserRecovery : Option RecoveryState :=
  recoverObservationFuel 4 200 4 noRepair .browserExited

private def timeoutRecovery : Option RecoveryState :=
  recoverObservationFuel 4 300 4 (fun _ => true) .deadlineReached

private def unknownRecovery : Option RecoveryState :=
  recoverObservationFuel 4 400 4 (fun _ => true) .unrecognized

private def recoveryStatusIs (result : Option RecoveryState) (status : RecoveryStatus) : Bool :=
  match result with
  | none => false
  | some state => state.status == status

private def recoveryGenerationIs (result : Option RecoveryState) (generation : Nat) : Bool :=
  match result with
  | none => false
  | some state => state.generation == generation

private def noRecovery (result : Option RecoveryState) : Bool :=
  match result with
  | none => true
  | some _ => false

private def scenarioVerified : Bool :=
  (decideNormalized (normalizeObservation .pageCrashed) == .recover .recreatePage) &&
  recoveryStatusIs pageRecovery .healthy &&
  recoveryGenerationIs pageRecovery 101 &&
  (decideNormalized (normalizeObservation .browserExited) == .recover .restartBrowser) &&
  recoveryStatusIs browserRecovery .failed &&
  (decideNormalized (normalizeObservation .deadlineReached) == .failSafe) &&
  recoveryStatusIs timeoutRecovery .failed &&
  (decideNormalized (normalizeObservation .unrecognized) == .failSafe) &&
  recoveryStatusIs unknownRecovery .failed &&
  (decideNormalized (normalizeObservation .humanInput) == .suspend) &&
  noRecovery (recoverObservationFuel 4 500 4 noRepair .humanInput) &&
  (decideNormalized (normalizeObservation .networkTransient) == .wait) &&
  noRecovery (recoverObservationFuel 4 600 4 noRepair .networkTransient) &&
  (decideNormalized (normalizeObservation .operationSucceeded) == .complete)

def main : IO UInt32 := do
  if scenarioVerified then
    IO.println "browser feedback recovery: normalization + decision + bounded convergence verified"
    return 0
  else
    IO.eprintln "browser feedback recovery: violation"
    return 1
