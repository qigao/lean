import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifier
import Mathlib.Analysis.PSeries
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Summable

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
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
    change Tendsto
      (fun n => Real.exp ((-2 : Real) * (∑ k ∈ Finset.range n, m k)))
      atTop (nhds 0)
    exact Real.tendsto_exp_atBot.comp hneg
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

private theorem mixingMass_nonneg_of_bounds
    (a : Rat) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    0 ≤ mixingMass a := by
  exact le_min ha0 (sub_nonneg.mpr ha1)

private theorem mixingMass_lt_half_of_ne
    (a : Rat) (hne : a ≠ 1 / 2) :
    mixingMass a < 1 / 2 := by
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · exact lt_of_le_of_lt (min_le_left _ _) hlt
  · exact lt_of_le_of_lt (min_le_right _ _) (by linarith)

private theorem mixingMass_applyDecayTarget
    (target : DecayTarget) (d : Rat) :
    mixingMass (applyDecayTarget target d) = mixingMass d := by
  cases target <;> simp [applyDecayTarget, mixingMass, min_comm]

private theorem polynomial_p1_decay_div_le_mixingMass
    (c : Rat) (offset e : Nat) (target : DecayTarget)
    (he : 1 ≤ e) (hoffset : 1 ≤ offset) (hc0 : 0 < c)
    (hcvalid : c ≤ offset) :
    polynomialDecay c 1 offset e / (((offset + 1 : Nat) : Rat)) ≤
      mixingMass (polynomialReceptivity c 1 offset target e) := by
  let d : Rat := polynomialDecay c 1 offset e
  let q : Rat := ((offset + 1 : Nat) : Rat)
  have hd0 : 0 < d := by
    simpa [d] using polynomialDecay_pos c 1 offset e (by norm_num) hoffset hc0
  have hd1 : d ≤ 1 := by
    simpa [d] using
      polynomialDecay_le_one c 1 offset e (by norm_num) hoffset hc0 (by simpa using hcvalid)
  have hqpos : 0 < q := by
    dsimp [q]
    positivity
  have hqone : 1 ≤ q := by
    dsimp [q]
    exact_mod_cast Nat.succ_le_succ (Nat.zero_le offset)
  have hdenNat : offset + 1 ≤ e + offset := by omega
  have hden : (((offset + 1 : Nat) : Rat)) ≤ (((e + offset : Nat) : Rat)) := by
    exact_mod_cast hdenNat
  have hdenPos : (0 : Rat) < (((e + offset : Nat) : Rat)) := by
    positivity
  have hoff0 : (0 : Rat) ≤ (offset : Rat) := by positivity
  have hcfrac :
      c / (((e + offset : Nat) : Rat)) ≤
        (offset : Rat) / (((e + offset : Nat) : Rat)) := by
    exact (div_le_div_iff_of_pos_right hdenPos).2 hcvalid
  have hdenfrac :
      (offset : Rat) / (((e + offset : Nat) : Rat)) ≤
        (offset : Rat) / (((offset + 1 : Nat) : Rat)) :=
    div_le_div_of_nonneg_left hoff0 hqpos hden
  have hdmax : d ≤ (offset : Rat) / q := by
    dsimp [d, q]
    simp only [polynomialDecay, pow_one]
    exact hcfrac.trans hdenfrac
  have hleft : d / q ≤ d := div_le_self hd0.le hqone
  have hright0 : d / q ≤ 1 / q :=
    (div_le_div_iff_of_pos_right hqpos).2 hd1
  have hsplit : (1 : Rat) / q + (offset : Rat) / q = 1 := by
    dsimp [q]
    have hne : (((offset + 1 : Nat) : Rat)) ≠ 0 := by positivity
    field_simp [hne]
    push_cast
    ring
  have hright1 : (1 : Rat) / q ≤ 1 - d := by
    linarith
  have hmix : d / q ≤ mixingMass d := by
    exact le_min hleft (hright0.trans hright1)
  calc
    polynomialDecay c 1 offset e / (((offset + 1 : Nat) : Rat)) = d / q := by rfl
    _ ≤ mixingMass d := hmix
    _ = mixingMass (applyDecayTarget target d) :=
      (mixingMass_applyDecayTarget target d).symm
    _ = mixingMass (polynomialReceptivity c 1 offset target e) := by rfl

private theorem shifted_harmonic_sum_tendsto_atTop (shift : Nat) :
    Tendsto
      (fun n => ∑ r ∈ Finset.range n,
        (1 / (((r + shift + 1 : Nat) : Real))))
      atTop atTop := by
  let H : Nat → Real := fun n =>
    ∑ i ∈ Finset.range n, (1 / (((i + 1 : Nat) : Real)))
  let S : Nat → Real := fun n =>
    ∑ r ∈ Finset.range n, (1 / (((r + shift + 1 : Nat) : Real)))
  have hH : Tendsto H atTop atTop := by
    simpa [H] using Real.tendsto_sum_range_one_div_nat_succ_atTop
  have hindex : Tendsto (fun n : Nat => shift + n) atTop atTop := by
    refine Filter.tendsto_atTop.2 ?_
    intro b
    exact Filter.eventually_atTop.2 ⟨b, fun n hn => by omega⟩
  have hHB : Tendsto (fun n => H (shift + n)) atTop atTop := hH.comp hindex
  have htranslated :
      Tendsto (fun n => H (shift + n) + (-H shift)) atTop atTop :=
    tendsto_atTop_add_const_right atTop (-H shift) hHB
  have hdecomp : ∀ n, H (shift + n) = H shift + S n := by
    intro n
    dsimp [H, S]
    rw [Finset.sum_range_add]
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    congr 1
    push_cast
    ring
  have heq : S = (fun n => H (shift + n) + (-H shift)) := by
    funext n
    have hn := hdecomp n
    linarith
  change Tendsto S atTop atTop
  rw [heq]
  exact htranslated

private theorem polynomial_p1_scaled_harmonic_sum_tendsto_atTop
    (c : Rat) (offset e0 : Nat)
    (hoffset : 1 ≤ offset) (hc0 : 0 < c) :
    Tendsto
      (fun n => ∑ r ∈ Finset.range n,
        (((polynomialDecay c 1 offset (e0 + r + 1) /
          (((offset + 1 : Nat) : Rat))) : Rat) : Real))
      atTop atTop := by
  let a : Real := (c : Real) / (((offset + 1 : Nat) : Real))
  have hac : (0 : Real) < (c : Real) := by exact_mod_cast hc0
  have haq : (0 : Real) < (((offset + 1 : Nat) : Real)) := by positivity
  have ha : 0 < a := div_pos hac haq
  have hh := shifted_harmonic_sum_tendsto_atTop (e0 + offset)
  have hscaled :
      Tendsto
        (fun n => a *
          (∑ r ∈ Finset.range n,
            (1 / (((r + (e0 + offset) + 1 : Nat) : Real)))))
        atTop atTop :=
    (tendsto_const_mul_atTop_of_pos ha).2 hh
  have heq :
      (fun n => ∑ r ∈ Finset.range n,
        (((polynomialDecay c 1 offset (e0 + r + 1) /
          (((offset + 1 : Nat) : Rat))) : Rat) : Real)) =
      (fun n => a *
        (∑ r ∈ Finset.range n,
          (1 / (((r + (e0 + offset) + 1 : Nat) : Real))))) := by
    funext n
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro r hr
    dsimp [a]
    push_cast
    simp only [polynomialDecay, pow_one]
    push_cast
    have h1 : (((offset + 1 : Nat) : Real)) ≠ 0 := by positivity
    have h2 : (((e0 + r + 1 + offset : Nat) : Real)) ≠ 0 := by positivity
    field_simp [h1, h2]
    push_cast
    ring
  rw [heq]
  exact hscaled

theorem polynomial_abs_product_tendsto_zero_of_p_eq_one
    (c : Rat) (offset e0 : Nat) (target : DecayTarget)
    (hoffset : 1 ≤ offset)
    (hc0 : 0 < c)
    (hcvalid : c ≤ offset) :
    Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨polynomialReceptivity c 1 offset target, 0⟩ e0 k : Real)|)
      atTop (nhds 0) := by
  let p : ExposureParameters :=
    ⟨polynomialReceptivity c 1 offset target, 0⟩
  by_cases hz : ∃ r : Nat, p.receptivityAt (e0 + r + 1) = 1 / 2
  · rcases hz with ⟨r, hr⟩
    apply piecewise_constant_tail_abs_product_tendsto_zero_of_zero_prefix p e0 (r + 1)
    unfold path2MultiplierProduct
    apply Finset.prod_eq_zero (Finset.mem_range.mpr (Nat.lt_succ_self r))
    rw [hr]
    norm_num
  · have hnozero : ∀ r : Nat, p.receptivityAt (e0 + r + 1) ≠ 1 / 2 := by
      intro r hr
      exact hz ⟨r, hr⟩
    let m : Nat → Real := fun r =>
      ((mixingMass (p.receptivityAt (e0 + r + 1)) : Rat) : Real)
    have hb (r : Nat) :
        0 ≤ p.receptivityAt (e0 + r + 1) ∧
          p.receptivityAt (e0 + r + 1) ≤ 1 := by
      simpa [p] using
        polynomialReceptivity_bounds c 1 offset target (e0 + r + 1)
          (by norm_num) hoffset hc0 (by simpa using hcvalid)
    have hm0 : ∀ r, 0 ≤ m r := by
      intro r
      have h := mixingMass_nonneg_of_bounds
        (p.receptivityAt (e0 + r + 1)) (hb r).1 (hb r).2
      dsimp [m]
      exact_mod_cast h
    have hmhalf : ∀ r, m r < 1 / 2 := by
      intro r
      have h := mixingMass_lt_half_of_ne
        (p.receptivityAt (e0 + r + 1)) (hnozero r)
      dsimp [m]
      have hhalf : (1 / 2 : Real) = ((1 / 2 : Rat) : Real) := by norm_num
      rw [hhalf]
      exact_mod_cast h
    have hlower : ∀ r,
        (((polynomialDecay c 1 offset (e0 + r + 1) /
          (((offset + 1 : Nat) : Rat))) : Rat) : Real) ≤ m r := by
      intro r
      have h := polynomial_p1_decay_div_le_mixingMass
        c offset (e0 + r + 1) target (by omega) hoffset hc0 hcvalid
      dsimp [m, p]
      exact_mod_cast h
    have hdivLower :=
      polynomial_p1_scaled_harmonic_sum_tendsto_atTop c offset e0 hoffset hc0
    have hsumle : ∀ n,
        (∑ r ∈ Finset.range n,
          (((polynomialDecay c 1 offset (e0 + r + 1) /
            (((offset + 1 : Nat) : Rat))) : Rat) : Real)) ≤
          ∑ r ∈ Finset.range n, m r := by
      intro n
      exact Finset.sum_le_sum fun r hr => hlower r
    have hdiv : Tendsto (fun n => ∑ r ∈ Finset.range n, m r) atTop atTop :=
      tendsto_atTop_mono' atTop (Filter.Eventually.of_forall hsumle) hdivLower
    have hgeneric :=
      abs_product_tendsto_zero_of_mixing_sum_tendsto_atTop m hm0 hmhalf hdiv
    have hprodEq :
        (fun k => |(path2MultiplierProduct p e0 k : Real)|) =
          (fun k => ∏ r ∈ Finset.range k, (1 - 2 * m r)) := by
      funext k
      have hRat :
          |path2MultiplierProduct p e0 k| =
            ∏ r ∈ Finset.range k,
              (1 - 2 * mixingMass (p.receptivityAt (e0 + r + 1))) := by
        unfold path2MultiplierProduct
        rw [Finset.abs_prod]
        apply Finset.prod_congr rfl
        intro r hr
        exact abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass
          (hb r).1 (hb r).2
      dsimp [m]
      exact_mod_cast hRat
    change Tendsto (fun k => |(path2MultiplierProduct p e0 k : Real)|) atTop (nhds 0)
    rw [hprodEq]
    exact hgeneric

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier