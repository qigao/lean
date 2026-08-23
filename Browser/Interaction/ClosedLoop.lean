import Browser.Interaction.TraceAggregation

namespace Browser.Interaction

structure ClosedLoopResult where
  runtimeSelected : RecoveryAction
  selected : RecoveryAction
  final : FaultSupervisor
  deriving Repr, DecidableEq, BEq

private def recoveringSupervisor
    (generation budget : Nat) (action : RecoveryAction) : FaultSupervisor := {
  generation := generation
  recoveryBudget := budget
  active := some {
    status := .recovering
    generation := generation
    fault := none
    action := action
    budget := budget
  }
  failed := false
}

private def failedSupervisor (generation budget : Nat) : FaultSupervisor := {
  generation := generation
  recoveryBudget := budget
  active := none
  failed := true
}

/-- Resolve the already-selected combined recovery. `retry` means no recovery is
    required, `fail` is terminal, and all other actions reuse the existing
    bounded recovery algorithm from `runRecoveryFuel` via `resolveSupervisorFuel`. -/
def resolveSelection
    (generation budget fuel : Nat) (selected : RecoveryAction)
    (available : RecoveryAction → Bool) : FaultSupervisor :=
  match selected with
  | .retry => healthySupervisor generation budget
  | .fail => failedSupervisor generation budget
  | action =>
      resolveSupervisorFuel fuel
        (recoveringSupervisor generation budget action)
        available

/-- Closed-loop composition: runtime-style observation folding supplies an
    independently computed recovery requirement, the algebraic path computes the
    minimum combined recovery, and bounded recovery resolves that selected action. -/
def runClosedLoop
    (generation budget fuel : Nat) (trace : List TaggedObservation)
    (available : RecoveryAction → Bool) : ClosedLoopResult :=
  let observed := observeTrace (healthySupervisor generation budget) trace
  let runtimeSelected := supervisorRequirement observed
  let selected := aggregateRecovery (effectiveFaults generation trace)
  {
    runtimeSelected := runtimeSelected
    selected := selected
    final := resolveSelection generation budget fuel selected available
  }

private theorem run_recovery_fuel_failed_stays_failed
    (fuel : Nat) (state : RecoveryState)
    (available : RecoveryAction → Bool)
    (hfailed : state.status = .failed) :
    (runRecoveryFuel fuel state available).status = .failed := by
  cases fuel <;> simp [runRecoveryFuel, hfailed]

private theorem insufficient_generation_unchanged (state : RecoveryState) :
    (stepRecovery state .insufficientRepair).generation = state.generation := by
  cases hstatus : state.status <;>
    cases hbudget : state.budget <;>
      cases haction : state.action <;>
        simp [stepRecovery, hstatus, hbudget, haction, nextRecovery]

private theorem insufficient_from_recovering_not_healthy
    (state : RecoveryState) (hrecovering : state.status = .recovering) :
    (stepRecovery state .insufficientRepair).status ≠ .healthy := by
  cases hbudget : state.budget <;>
    cases haction : state.action <;>
      simp [stepRecovery, hrecovering, hbudget, haction, nextRecovery]

/-- If bounded recovery starts in `recovering` and ends healthy, that healthy
    state is necessarily the fresh incarnation created by `.recovered`. -/
theorem run_recovery_fuel_success_is_fresh
    (fuel : Nat) (state : RecoveryState)
    (available : RecoveryAction → Bool)
    (hrecovering : state.status = .recovering)
    (hhealthy : (runRecoveryFuel fuel state available).status = .healthy) :
    (runRecoveryFuel fuel state available).generation = state.generation + 1 := by
  induction fuel generalizing state with
  | zero =>
      simp [runRecoveryFuel, hrecovering] at hhealthy
  | succ fuel ih =>
      cases havailable : available state.action with
      | true =>
          simp [runRecoveryFuel, hrecovering, havailable, stepRecovery]
      | false =>
          let next := stepRecovery state .insufficientRepair
          have hhealthyNext : (runRecoveryFuel fuel next available).status = .healthy := by
            simpa [runRecoveryFuel, hrecovering, havailable, next] using hhealthy
          have hgeneration : next.generation = state.generation := by
            simpa [next] using insufficient_generation_unchanged state
          have hnotHealthy : next.status ≠ .healthy := by
            simpa [next] using insufficient_from_recovering_not_healthy state hrecovering
          cases hstatus : next.status with
          | healthy =>
              exact False.elim (hnotHealthy hstatus)
          | recovering =>
              have hfresh := ih next hstatus hhealthyNext
              simpa [runRecoveryFuel, hrecovering, havailable, next, hgeneration] using hfresh
          | failed =>
              have hstaysFailed :=
                run_recovery_fuel_failed_stays_failed fuel next available hstatus
              rw [hstaysFailed] at hhealthyNext
              contradiction

private theorem resolve_recovering_success_is_fresh
    (generation budget fuel : Nat) (action : RecoveryAction)
    (available : RecoveryAction → Bool)
    (hsuccess :
      (resolveSupervisorFuel fuel
        (recoveringSupervisor generation budget action)
        available).failed = false) :
    (resolveSupervisorFuel fuel
      (recoveringSupervisor generation budget action)
      available).generation = generation + 1 := by
  let initial : RecoveryState := {
    status := .recovering
    generation := generation
    fault := none
    action := action
    budget := budget
  }
  cases hstatus : (runRecoveryFuel fuel initial available).status with
  | healthy =>
      have hfresh :=
        run_recovery_fuel_success_is_fresh fuel initial available (by rfl) hstatus
      simpa [resolveSupervisorFuel, recoveringSupervisor, initial, hstatus] using hfresh
  | recovering =>
      simp [resolveSupervisorFuel, recoveringSupervisor, initial, hstatus] at hsuccess
  | failed =>
      simp [resolveSupervisorFuel, recoveringSupervisor, initial, hstatus] at hsuccess

private theorem aggregate_cons_ne_retry
    (fault : FaultClass) (rest : List FaultClass) :
    aggregateRecovery (fault :: rest) ≠ .retry := by
  cases fault <;>
    cases hrest : aggregateRecovery rest <;>
      simp [aggregateRecovery, joinRecovery, minimumRecovery, recoveryRank, hrest]

/-- The runtime incremental fold and the algebraic minimum-combined selection are
    identical for the same finite trace. -/
theorem closed_loop_runtime_matches_selected
    (generation budget fuel : Nat) (trace : List TaggedObservation)
    (available : RecoveryAction → Bool) :
    (runClosedLoop generation budget fuel trace available).runtimeSelected =
      (runClosedLoop generation budget fuel trace available).selected := by
  simp [runClosedLoop, healthy_trace_matches_aggregate]

/-- The selected closed-loop recovery is exactly the finite algebraic aggregate. -/
theorem closed_loop_selects_aggregate
    (generation budget fuel : Nat) (trace : List TaggedObservation)
    (available : RecoveryAction → Bool) :
    (runClosedLoop generation budget fuel trace available).selected =
      aggregateRecovery (effectiveFaults generation trace) := by
  rfl

/-- Finite fuel always closes the recovery loop: no active recovery escapes. -/
theorem closed_loop_is_resolved
    (generation budget fuel : Nat) (trace : List TaggedObservation)
    (available : RecoveryAction → Bool) :
    supervisorResolved
      (runClosedLoop generation budget fuel trace available).final = true := by
  unfold runClosedLoop
  dsimp
  cases hselected : aggregateRecovery (effectiveFaults generation trace) with
  | retry =>
      simp [resolveSelection, supervisorResolved, healthySupervisor]
  | reResolve =>
      simpa [hselected, resolveSelection] using
        resolve_supervisor_fuel_is_resolved fuel
          (recoveringSupervisor generation budget .reResolve) available
  | rebindRuntime =>
      simpa [hselected, resolveSelection] using
        resolve_supervisor_fuel_is_resolved fuel
          (recoveringSupervisor generation budget .rebindRuntime) available
  | reattachSession =>
      simpa [hselected, resolveSelection] using
        resolve_supervisor_fuel_is_resolved fuel
          (recoveringSupervisor generation budget .reattachSession) available
  | recreatePage =>
      simpa [hselected, resolveSelection] using
        resolve_supervisor_fuel_is_resolved fuel
          (recoveringSupervisor generation budget .recreatePage) available
  | recreateContext =>
      simpa [hselected, resolveSelection] using
        resolve_supervisor_fuel_is_resolved fuel
          (recoveringSupervisor generation budget .recreateContext) available
  | restartBrowser =>
      simpa [hselected, resolveSelection] using
        resolve_supervisor_fuel_is_resolved fuel
          (recoveringSupervisor generation budget .restartBrowser) available
  | fail =>
      simp [resolveSelection, supervisorResolved, failedSupervisor]

/-- If the trace contains at least one effective fault and bounded recovery
    succeeds rather than failing, the result is the fresh next generation. -/
theorem closed_loop_success_is_fresh
    (generation budget fuel : Nat) (trace : List TaggedObservation)
    (available : RecoveryAction → Bool)
    (hnonempty : effectiveFaults generation trace ≠ [])
    (hsuccess : (runClosedLoop generation budget fuel trace available).final.failed = false) :
    (runClosedLoop generation budget fuel trace available).final.generation =
      generation + 1 := by
  cases heffective : effectiveFaults generation trace with
  | nil =>
      exact False.elim (hnonempty heffective)
  | cons fault rest =>
      have hnotRetry : aggregateRecovery (fault :: rest) ≠ .retry :=
        aggregate_cons_ne_retry fault rest
      cases hselected : aggregateRecovery (fault :: rest) with
      | retry =>
          exact False.elim (hnotRetry hselected)
      | reResolve =>
          simpa [runClosedLoop, heffective, hselected, resolveSelection] using
            resolve_recovering_success_is_fresh generation budget fuel .reResolve available
              (by simpa [runClosedLoop, heffective, hselected, resolveSelection] using hsuccess)
      | rebindRuntime =>
          simpa [runClosedLoop, heffective, hselected, resolveSelection] using
            resolve_recovering_success_is_fresh generation budget fuel .rebindRuntime available
              (by simpa [runClosedLoop, heffective, hselected, resolveSelection] using hsuccess)
      | reattachSession =>
          simpa [runClosedLoop, heffective, hselected, resolveSelection] using
            resolve_recovering_success_is_fresh generation budget fuel .reattachSession available
              (by simpa [runClosedLoop, heffective, hselected, resolveSelection] using hsuccess)
      | recreatePage =>
          simpa [runClosedLoop, heffective, hselected, resolveSelection] using
            resolve_recovering_success_is_fresh generation budget fuel .recreatePage available
              (by simpa [runClosedLoop, heffective, hselected, resolveSelection] using hsuccess)
      | recreateContext =>
          simpa [runClosedLoop, heffective, hselected, resolveSelection] using
            resolve_recovering_success_is_fresh generation budget fuel .recreateContext available
              (by simpa [runClosedLoop, heffective, hselected, resolveSelection] using hsuccess)
      | restartBrowser =>
          simpa [runClosedLoop, heffective, hselected, resolveSelection] using
            resolve_recovering_success_is_fresh generation budget fuel .restartBrowser available
              (by simpa [runClosedLoop, heffective, hselected, resolveSelection] using hsuccess)
      | fail =>
          simp [runClosedLoop, heffective, hselected, resolveSelection,
            failedSupervisor] at hsuccess

/-- Compact top-level closed-loop guarantee: the selected recovery is the least
    sufficient action for all effective faults, and finite recovery is resolved. -/
theorem closed_loop_converges
    (generation budget fuel : Nat) (trace : List TaggedObservation)
    (available : RecoveryAction → Bool) :
    MinimalCombined
        (effectiveFaults generation trace)
        (runClosedLoop generation budget fuel trace available).selected ∧
      supervisorResolved
        (runClosedLoop generation budget fuel trace available).final = true := by
  constructor
  · have hselected :=
      closed_loop_selects_aggregate generation budget fuel trace available
    rw [hselected]
    exact aggregate_is_minimal_combined (effectiveFaults generation trace)
  · exact closed_loop_is_resolved generation budget fuel trace available

end Browser.Interaction
