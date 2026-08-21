import Browser.Interaction.Feedback

namespace Browser.Interaction

inductive FaultClass where
  | elementStale
  | runtimeLost
  | sessionLost
  | pageLost
  | contextLost
  | browserLost
  | timeoutFault
  | unknownFault
  deriving Repr, DecidableEq, BEq

/-- Lower rank means less disruptive recovery. -/
def recoveryRank : RecoveryAction → Nat
  | .retry => 0
  | .reResolve => 1
  | .rebindRuntime => 2
  | .reattachSession => 3
  | .recreatePage => 4
  | .recreateContext => 5
  | .restartBrowser => 6
  | .fail => 7

def minimumRecovery : FaultClass → RecoveryAction
  | .elementStale => .reResolve
  | .runtimeLost => .rebindRuntime
  | .sessionLost => .reattachSession
  | .pageLost => .recreatePage
  | .contextLost => .recreateContext
  | .browserLost => .restartBrowser
  | .timeoutFault => .fail
  | .unknownFault => .fail

/-- A recovery is sufficient when it is at least as strong as the minimum
    repair required by the fault. This defines the admissible escalation set. -/
def recoverySufficient (fault : FaultClass) (action : RecoveryAction) : Bool :=
  decide (recoveryRank (minimumRecovery fault) ≤ recoveryRank action)

def MinimalSufficient (fault : FaultClass) (action : RecoveryAction) : Prop :=
  recoverySufficient fault action = true ∧
  ∀ candidate,
    recoverySufficient fault candidate = true →
      recoveryRank action ≤ recoveryRank candidate

theorem sufficient_recovery_has_required_rank
    (fault : FaultClass) (action : RecoveryAction)
    (h : recoverySufficient fault action = true) :
    recoveryRank (minimumRecovery fault) ≤ recoveryRank action := by
  simpa [recoverySufficient] using h

theorem minimum_recovery_is_minimal (fault : FaultClass) :
    MinimalSufficient fault (minimumRecovery fault) := by
  constructor
  · simp [recoverySufficient]
  · intro candidate h
    exact sufficient_recovery_has_required_rank fault candidate h

inductive RecoveryStatus where
  | healthy
  | recovering
  | failed
  deriving Repr, DecidableEq, BEq

structure RecoveryState where
  status : RecoveryStatus := .healthy
  generation : Nat := 0
  fault : Option FaultClass := none
  action : RecoveryAction := .retry
  budget : Nat := 0
  deriving Repr, DecidableEq, BEq

inductive RecoveryFeedback where
  | recovered
  | retryableFailure
  | insufficientRepair
  | fatalFailure
  deriving Repr, DecidableEq, BEq

def beginRecovery
    (generation : Nat) (fault : FaultClass) (budget : Nat) : RecoveryState := {
  status := .recovering
  generation := generation
  fault := some fault
  action := minimumRecovery fault
  budget := budget
}

def nextRecovery : RecoveryAction → RecoveryAction
  | .retry => .reResolve
  | .reResolve => .rebindRuntime
  | .rebindRuntime => .reattachSession
  | .reattachSession => .recreatePage
  | .recreatePage => .recreateContext
  | .recreateContext => .restartBrowser
  | .restartBrowser => .fail
  | .fail => .fail

/-- One recovery observation. Failure consumes finite budget; insufficient
    repair also escalates exactly one disruption level. Success always creates
    a fresh generation instead of reviving the failed incarnation. -/
def stepRecovery (state : RecoveryState) (feedback : RecoveryFeedback) : RecoveryState :=
  match state.status with
  | .healthy | .failed => state
  | .recovering =>
      match feedback with
      | .recovered =>
          { state with
            status := .healthy
            generation := state.generation + 1
            fault := none
            action := .retry }
      | .fatalFailure =>
          { state with status := .failed }
      | .retryableFailure =>
          match state.budget with
          | 0 => { state with status := .failed }
          | remaining + 1 => { state with budget := remaining }
      | .insufficientRepair =>
          match state.budget with
          | 0 => { state with status := .failed }
          | remaining + 1 =>
              let next := nextRecovery state.action
              if next = .fail then
                { state with status := .failed, action := .fail, budget := remaining }
              else
                { state with action := next, budget := remaining }

theorem recovered_increments_generation
    (state : RecoveryState) (h : state.status = .recovering) :
    (stepRecovery state .recovered).generation = state.generation + 1 := by
  simp [stepRecovery, h]

theorem recovered_returns_healthy
    (state : RecoveryState) (h : state.status = .recovering) :
    (stepRecovery state .recovered).status = .healthy := by
  simp [stepRecovery, h]

theorem recovered_rejects_old_generation
    (state : RecoveryState) (h : state.status = .recovering) :
    (stepRecovery state .recovered).generation ≠ state.generation := by
  simp [stepRecovery, h]

end Browser.Interaction
