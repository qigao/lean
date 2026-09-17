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

theorem path2_equal_belief_consensus
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hbelief : (s 0).belief = (s 1).belief) :
    ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop (nhds ((s 0).belief : Real)) := by
  have hconst : ∀ k : Nat,
      beliefs ((step p 2)^[k] s) 0 = (s 0).belief ∧
      beliefs ((step p 2)^[k] s) 1 = (s 0).belief := by
    intro k
    have hm := path2_mean_iterate p hvalid s he hb k
    have hd := path2_disagreement_product p hvalid s he hb k
    have hdiff : (s 0).belief - (s 1).belief = 0 := sub_eq_zero.mpr hbelief
    rw [hdiff, zero_mul] at hd
    have hmean : ((s 0).belief + (s 1).belief) / 2 = (s 0).belief := by
      rw [hbelief]
      ring
    rw [hmean] at hm
    constructor <;> linarith
  have hcoord : ∀ (k : Nat) (i : Fin 2),
      beliefs ((step p 2)^[k] s) i = (s 0).belief := by
    intro k i
    have hi : i = (0 : Fin 2) ∨ i = (1 : Fin 2) := by
      have hval : i.val = 0 ∨ i.val = 1 := by
        omega
      rcases hval with h0 | h1
      · left
        apply Fin.ext
        exact h0
      · right
        apply Fin.ext
        exact h1
    rcases hi with rfl | rfl
    · exact (hconst k).1
    · exact (hconst k).2
  intro i
  have hseq :
      (fun k => (beliefs ((step p 2)^[k] s) i : Real)) =
        (fun _ : Nat => ((s 0).belief : Real)) := by
    funext k
    exact congrArg (fun q : Rat => (q : Real)) (hcoord k i)
  rw [hseq]
  exact tendsto_const_nhds

theorem tendsto_zero_const_mul_iff
    {f : Nat → Real} {c : Real} (hc : c ≠ 0) :
    Tendsto (fun k => c * f k) atTop (nhds 0) ↔
      Tendsto f atTop (nhds 0) := by
  constructor
  · intro h
    have hi := Filter.Tendsto.const_mul c⁻¹ h
    simpa [hc, mul_assoc] using hi
  · intro h
    simpa using Filter.Tendsto.const_mul c h

theorem tendsto_const_mul
    {f : Nat → Real} {x c : Real}
    (hf : Tendsto f atTop (nhds x)) :
    Tendsto (fun k => c * f k) atTop (nhds (c * x)) := by
  simpa using Filter.Tendsto.const_mul c hf

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier
