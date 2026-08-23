import Browser.Interaction.DecoderRecovery

namespace Browser.Interaction

/-- Least recovery action that is at least as strong as both inputs. Recovery
    rank is a total order, so this is the join used to merge concurrent faults. -/
def joinRecovery (a b : RecoveryAction) : RecoveryAction :=
  if recoveryRank a ≤ recoveryRank b then b else a

theorem join_recovery_idempotent (action : RecoveryAction) :
    joinRecovery action action = action := by
  cases action <;> rfl

theorem join_recovery_commutative (a b : RecoveryAction) :
    joinRecovery a b = joinRecovery b a := by
  cases a <;> cases b <;> rfl

theorem join_recovery_associative (a b c : RecoveryAction) :
    joinRecovery (joinRecovery a b) c = joinRecovery a (joinRecovery b c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem join_recovery_left_upper (a b : RecoveryAction) :
    recoveryRank a ≤ recoveryRank (joinRecovery a b) := by
  cases a <;> cases b <;> decide

theorem join_recovery_right_upper (a b : RecoveryAction) :
    recoveryRank b ≤ recoveryRank (joinRecovery a b) := by
  cases a <;> cases b <;> decide

theorem join_recovery_least_upper
    (a b candidate : RecoveryAction)
    (ha : recoveryRank a ≤ recoveryRank candidate)
    (hb : recoveryRank b ≤ recoveryRank candidate) :
    recoveryRank (joinRecovery a b) ≤ recoveryRank candidate := by
  cases a <;> cases b <;> cases candidate <;> simp_all [joinRecovery, recoveryRank]

/-- An observation is owned by exactly one runtime incarnation. -/
structure TaggedObservation where
  generation : Nat
  event : AdapterEvent
  deriving Repr, DecidableEq, BEq

/-- Compact supervisor for adversarial fault arrival. It intentionally stores no
    Browser/Page/Frame state: only generation and at most one active recovery. -/
structure FaultSupervisor where
  generation : Nat
  recoveryBudget : Nat
  active : Option RecoveryState := none
  failed : Bool := false
  deriving Repr, DecidableEq, BEq

def healthySupervisor (generation recoveryBudget : Nat) : FaultSupervisor := {
  generation := generation
  recoveryBudget := recoveryBudget
}

private def mergeActiveRecovery
    (active : RecoveryState) (fault : FaultClass) : RecoveryState :=
  let required := minimumRecovery fault
  let joined := joinRecovery active.action required
  let nextFault :=
    if recoveryRank active.action < recoveryRank required then some fault else active.fault
  { active with action := joined, fault := nextFault }

/-- Only observations owned by the current generation may affect recovery.
    Concurrent current-generation faults merge into the single active recovery
    using the least-upper-bound recovery action; budget is never reset. -/
def observeAdversarial
    (supervisor : FaultSupervisor) (observation : TaggedObservation) : FaultSupervisor :=
  if supervisor.failed then
    supervisor
  else if observation.generation ≠ supervisor.generation then
    supervisor
  else
    match (normalizeObservation (decodeEvent observation.event)).fault with
    | none => supervisor
    | some fault =>
        let required := minimumRecovery fault
        if required = .fail then
          { supervisor with active := none, failed := true }
        else
          match supervisor.active with
          | none =>
              let started := beginRecovery supervisor.generation fault supervisor.recoveryBudget
              { supervisor with active := some started }
          | some active =>
              { supervisor with active := some (mergeActiveRecovery active fault) }

/-- Feed one recovery result back into the single active recovery. `stepRecovery`
    remains the authority for budget/escalation. The supervisor only projects a
    terminal result back into fresh-generation or explicit-failure state. -/
def applyRecoveryFeedback
    (supervisor : FaultSupervisor) (feedback : RecoveryFeedback) : FaultSupervisor :=
  if supervisor.failed then
    supervisor
  else
    match supervisor.active with
    | none => supervisor
    | some active =>
        let next := stepRecovery active feedback
        match next.status with
        | .healthy =>
            { supervisor with generation := next.generation, active := none }
        | .recovering =>
            { supervisor with active := some next }
        | .failed =>
            { supervisor with active := none, failed := true }

/-- A resolved supervisor has no live recovery left. `failed = true` is already
    terminal; otherwise `active = none` means the lane is healthy/idle again. -/
def supervisorResolved (supervisor : FaultSupervisor) : Bool :=
  if supervisor.failed then
    true
  else
    match supervisor.active with
    | none => true
    | some _ => false

/-- Resolve the currently active recovery with finite fuel. The existing
    `runRecoveryFuel` remains the recovery algorithm. This projection never
    exposes a still-recovering result: an impossible/nonterminal result is
    treated fail-safe as explicit supervisor failure. -/
def resolveSupervisorFuel
    (fuel : Nat) (supervisor : FaultSupervisor)
    (available : RecoveryAction → Bool) : FaultSupervisor :=
  if supervisor.failed then
    supervisor
  else
    match supervisor.active with
    | none => supervisor
    | some active =>
        let result := runRecoveryFuel fuel active available
        match result.status with
        | .healthy =>
            { supervisor with generation := result.generation, active := none }
        | .failed =>
            { supervisor with active := none, failed := true }
        | .recovering =>
            { supervisor with active := none, failed := true }

theorem stale_observation_noop
    (supervisor : FaultSupervisor) (observation : TaggedObservation)
    (h : observation.generation < supervisor.generation) :
    observeAdversarial supervisor observation = supervisor := by
  have hne : observation.generation ≠ supervisor.generation := Nat.ne_of_lt h
  cases hfailed : supervisor.failed <;>
    simp [observeAdversarial, hfailed, hne]

theorem future_observation_noop
    (supervisor : FaultSupervisor) (observation : TaggedObservation)
    (h : supervisor.generation < observation.generation) :
    observeAdversarial supervisor observation = supervisor := by
  have hne : observation.generation ≠ supervisor.generation := by
    exact Ne.symm (Nat.ne_of_lt h)
  cases hfailed : supervisor.failed <;>
    simp [observeAdversarial, hfailed, hne]

theorem failed_supervisor_absorbs_feedback
    (supervisor : FaultSupervisor) (h : supervisor.failed = true)
    (feedback : RecoveryFeedback) :
    applyRecoveryFeedback supervisor feedback = supervisor := by
  simp [applyRecoveryFeedback, h]

theorem resolve_supervisor_fuel_is_resolved
    (fuel : Nat) (supervisor : FaultSupervisor)
    (available : RecoveryAction → Bool) :
    supervisorResolved (resolveSupervisorFuel fuel supervisor available) = true := by
  cases hfailed : supervisor.failed with
  | true =>
      simp [resolveSupervisorFuel, supervisorResolved, hfailed]
  | false =>
      cases hactive : supervisor.active with
      | none =>
          simp [resolveSupervisorFuel, supervisorResolved, hfailed, hactive]
      | some active =>
          cases hstatus : (runRecoveryFuel fuel active available).status <;>
            simp [resolveSupervisorFuel, supervisorResolved, hfailed, hactive, hstatus]

end Browser.Interaction
