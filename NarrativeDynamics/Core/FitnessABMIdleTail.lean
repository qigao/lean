import NarrativeDynamics.Core.IdleTail
import NarrativeDynamics.Core.FitnessABM

/-!
# Fitness ABM idle-tail adapters

This module exposes repeated idle evolution through the existing `advance`
transition. It does not define replay, parsing, birth, probability, or network
update logic.
-/

namespace NarrativeDynamics.FitnessABM

open NarrativeDynamics

def jointIdleTail (n : Nat) :
    IdleTailModel (JointState n) (JointState n) :=
  { step := advance
    observe := id }

def observeJointWith {n : Nat} {Obs : Type*}
    (project : JointState n → Obs) :
    IdleTailModel (JointState n) Obs :=
  { step := advance
    observe := project }

def idleRunStep (s : RunState) : RunState :=
  ⟨s.nodeCount, s.roundIndex + 1, advance s.state⟩

def runIdleTail : IdleTailModel RunState RunState :=
  { step := idleRunStep
    observe := id }

def runIdleTrajectory (s : RunState) (k : Nat) : RunState :=
  IdleTailModel.trajectory runIdleTail s k

@[simp] theorem idleRunStep_nodeCount (s : RunState) :
    (idleRunStep s).nodeCount = s.nodeCount := rfl

@[simp] theorem idleRunStep_roundIndex (s : RunState) :
    (idleRunStep s).roundIndex = s.roundIndex + 1 := rfl

@[simp] theorem idleRunStep_state (s : RunState) :
    (idleRunStep s).state = advance s.state := rfl

theorem runIdleTrajectory_eq (s : RunState) (k : Nat) :
    runIdleTrajectory s k =
      ⟨s.nodeCount, s.roundIndex + k, (advance^[k]) s.state⟩ := by
  induction k with
  | zero =>
      rcases s with ⟨n, r, state⟩
      rfl
  | succ k ih =>
      rw [show runIdleTrajectory s (Nat.succ k) =
          idleRunStep (runIdleTrajectory s k) by
        simpa [Nat.succ_eq_add_one, runIdleTrajectory] using
          (IdleTailModel.trajectory_succ runIdleTail s k)]
      rw [ih]
      simp [idleRunStep, Function.iterate_succ_apply', Nat.add_assoc]

@[simp] theorem runIdleTrajectory_nodeCount (s : RunState) (k : Nat) :
    (runIdleTrajectory s k).nodeCount = s.nodeCount := by
  rw [runIdleTrajectory_eq]

@[simp] theorem runIdleTrajectory_roundIndex (s : RunState) (k : Nat) :
    (runIdleTrajectory s k).roundIndex = s.roundIndex + k := by
  rw [runIdleTrajectory_eq]

end NarrativeDynamics.FitnessABM
