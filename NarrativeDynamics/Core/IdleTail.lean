import Mathlib

/-!
# Proof-neutral idle-tail iteration

This module packages repeated application of one state transition together with
an observation function. It deliberately contains no model-specific graph,
replay, or probability semantics.
-/

namespace NarrativeDynamics

structure IdleTailModel (State Obs : Type*) where
  step : State → State
  observe : State → Obs

namespace IdleTailModel

variable {State Obs : Type*}

def trajectory (m : IdleTailModel State Obs) (s : State) (k : Nat) : State :=
  (m.step)^[k] s

def observedTrajectory
    (m : IdleTailModel State Obs) (s : State) (k : Nat) : Obs :=
  m.observe (trajectory m s k)

@[simp] theorem trajectory_zero
    (m : IdleTailModel State Obs) (s : State) :
    trajectory m s 0 = s := by
  simp [trajectory]

theorem trajectory_succ
    (m : IdleTailModel State Obs) (s : State) (k : Nat) :
    trajectory m s (k + 1) = m.step (trajectory m s k) := by
  simp [trajectory, Function.iterate_succ_apply']

theorem trajectory_add
    (m : IdleTailModel State Obs) (s : State) (a b : Nat) :
    trajectory m s (a + b) = trajectory m (trajectory m s a) b := by
  unfold trajectory
  rw [add_comm]
  exact Function.iterate_add_apply m.step b a s

@[simp] theorem observedTrajectory_zero
    (m : IdleTailModel State Obs) (s : State) :
    observedTrajectory m s 0 = m.observe s := by
  simp [observedTrajectory]

theorem observedTrajectory_succ
    (m : IdleTailModel State Obs) (s : State) (k : Nat) :
    observedTrajectory m s (k + 1) =
      m.observe (m.step (trajectory m s k)) := by
  simp [observedTrajectory, trajectory_succ]

end IdleTailModel

end NarrativeDynamics
