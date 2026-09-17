import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

/-!
# Exact receptivity schedule classification foundations

This module adds theorem-level schedule classification helpers on top of the
existing exposure-dependent Path2/PathN convergence layer. It does not alter
the executable `FitnessABMPathNExposure.step` semantics.
-/

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology

inductive DecayTarget
  | zero
  | one
  deriving DecidableEq, Repr

def applyDecayTarget (target : DecayTarget) (d : Rat) : Rat :=
  match target with
  | .zero => d
  | .one => 1 - d

noncomputable def mixingMass (a : Rat) : Rat :=
  min a (1 - a)

theorem abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass
    {a : Rat} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    |1 - 2 * a| = 1 - 2 * mixingMass a := by
  by_cases h : a ≤ 1 / 2
  · have hmin : a ≤ 1 - a := by
      linarith
    have habs : 0 ≤ 1 - 2 * a := by
      linarith
    rw [abs_of_nonneg habs]
    simp [mixingMass, min_eq_left hmin]
  · have hhalf : 1 / 2 < a := lt_of_not_ge h
    have hmin : 1 - a ≤ a := by
      linarith
    have habs : 1 - 2 * a ≤ 0 := by
      linarith
    rw [abs_of_nonpos habs]
    simp [mixingMass, min_eq_right hmin]
    ring

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier
