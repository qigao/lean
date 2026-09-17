import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifier
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Summable

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

open Filter Topology
open scoped BigOperators

theorem abs_product_tendsto_zero_of_mixing_sum_tendsto_atTop
    (m : Nat → Real)
    (hm0 : ∀ k, 0 ≤ m k)
    (hmhalf : ∀ k, m k < 1 / 2)
    (hdiv : Tendsto (fun n => ∑ k ∈ Finset.range n, m k) atTop atTop) :
    Tendsto
      (fun n => ∏ k ∈ Finset.range n, (1 - 2 * m k))
      atTop (nhds 0) := by
  let P : Nat → Real := fun n => ∏ k ∈ Finset.range n, (1 - 2 * m k)
  let E : Nat → Real := fun n =>
    Real.exp ((-2 : Real) * (∑ k ∈ Finset.range n, m k))
  have hP0 : ∀ n, 0 ≤ P n := by
    intro n
    exact Finset.prod_nonneg fun k hk => by
      have hkhalf := hmhalf k
      linarith
  have hPE : ∀ n, P n ≤ E n := by
    intro n
    induction n with
    | zero =>
        simp [P, E]
    | succ n ih =>
        have hfactor0 : 0 ≤ 1 - 2 * m n := by
          have hn := hmhalf n
          linarith
        have hfactor_le : 1 - 2 * m n ≤ Real.exp (-(2 * m n)) := by
          exact Real.one_sub_le_exp_neg (2 * m n)
        have hEnonneg : 0 ≤ E n := Real.exp_nonneg _
        calc
          P (n + 1) = P n * (1 - 2 * m n) := by
            simp [P, Finset.prod_range_succ]
          _ ≤ E n * Real.exp (-(2 * m n)) :=
            mul_le_mul ih hfactor_le hfactor0 hEnonneg
          _ = E (n + 1) := by
            rw [← Real.exp_add]
            simp only [E, Finset.sum_range_succ]
            congr 1
            ring
  have hE0 : Tendsto E atTop (nhds 0) := by
    have hneg :
        Tendsto
          (fun n => (-2 : Real) * (∑ k ∈ Finset.range n, m k))
          atTop atBot :=
      hdiv.const_mul_atTop_of_neg (by norm_num)
    simpa [E] using Real.tendsto_exp_atBot.comp hneg
  have hPzero : Tendsto P atTop (nhds 0) :=
    squeeze_zero hP0 hPE hE0
  simpa [P] using hPzero

theorem abs_product_has_nonzero_limit_of_summable_mixing
    (m : Nat → Real)
    (_hm0 : ∀ k, 0 ≤ m k)
    (hmhalf : ∀ k, m k < 1 / 2)
    (hsum : Summable m) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun n => ∏ k ∈ Finset.range n, (1 - 2 * m k))
        atTop (nhds L) := by
  let g : Nat → Real := fun k => (-2 : Real) * m k
  have hgsum : Summable g := by
    simpa [g] using Summable.mul_left (-2 : Real) hsum
  have hpos : ∀ k, 0 < 1 + g k := by
    intro k
    dsimp [g]
    have hk := hmhalf k
    linarith
  have hlog : Summable (fun k => Real.log (1 + g k)) :=
    Real.summable_log_one_add_of_summable hgsum
  let L : Real := Real.exp (∑' k, Real.log (1 + g k))
  have hprod : HasProd (fun k => 1 + g k) L := by
    simpa [L] using Real.hasProd_of_hasSum_log hpos hlog.hasSum
  refine ⟨L, ?_, ?_⟩
  · exact Real.exp_pos _
  · simpa [g, sub_eq_add_neg] using hprod.tendsto_prod_nat

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier