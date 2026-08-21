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

end Browser.Interaction
