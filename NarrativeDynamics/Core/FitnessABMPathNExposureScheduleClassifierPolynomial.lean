import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifierInfiniteProduct
import Mathlib.Analysis.PSeries

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology
open scoped BigOperators

private theorem polynomial_mixingMass_nonneg_of_bounds
    (a : Rat) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    0 ≤ mixingMass a := by
  exact le_min ha0 (sub_nonneg.mpr ha1)

private theorem polynomial_mixingMass_lt_half_of_ne
    (a : Rat) (hne : a ≠ 1 / 2) :
    mixingMass a < 1 / 2 := by
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · exact lt_of_le_of_lt (min_le_left _ _) hlt
  · exact lt_of_le_of_lt (min_le_right _ _) (by linarith)

private theorem polynomial_mixingMass_applyDecayTarget
    (target : DecayTarget) (d : Rat) :
    mixingMass (applyDecayTarget target d) = mixingMass d := by
  cases target <;> simp [applyDecayTarget, mixingMass, min_comm]

private theorem polynomial_mixingMass_le_decay
    (c : Rat) (p offset e : Nat) (target : DecayTarget) :
    mixingMass (polynomialReceptivity c p offset target e) ≤
      polynomialDecay c p offset e := by
  rw [polynomialReceptivity, polynomial_mixingMass_applyDecayTarget]
  exact min_le_left _ _

private theorem polynomial_decay_cast_summable_of_two_le_p
    (c : Rat) (p offset e0 : Nat) (hp : 2 ≤ p) :
    Summable
      (fun r : Nat => ((polynomialDecay c p offset (e0 + r + 1) : Rat) : Real)) := by
  have hbase : Summable (fun n : Nat => 1 / (n : Real) ^ p) :=
    Real.summable_one_div_nat_pow.mpr (by omega)
  let shift : Nat := e0 + offset + 1
  have hshift :
      Summable (fun r : Nat => 1 / (((r + shift : Nat) : Real) ^ p)) := by
    simpa using (summable_nat_add_iff shift).mpr hbase
  have hscaled :
      Summable
        (fun r : Nat =>
          (c : Real) * (1 / (((r + shift : Nat) : Real) ^ p))) := by
    exact Summable.mul_left (c : Real) hshift
  refine hscaled.congr ?_
  intro r
  change
    (c : Real) * (1 / (((r + shift : Nat) : Real) ^ p)) =
      ((c / (((e0 + r + 1 + offset : Nat) : Rat) ^ p) : Rat) : Real)
  dsimp [shift]
  push_cast
  have hden :
      (r : Real) + ((e0 : Real) + (offset : Real) + 1) =
        (e0 : Real) + (r : Real) + 1 + (offset : Real) := by
    ring
  rw [hden]
  simp [div_eq_mul_inv]

theorem polynomial_abs_product_has_nonzero_limit_of_two_le_p
    (c : Rat) (p offset e0 : Nat) (target : DecayTarget)
    (hp : 2 ≤ p)
    (hoffset : 1 ≤ offset)
    (hc0 : 0 < c)
    (hcvalid : c ≤ ((offset : Rat) ^ p))
    (hnozero : ∀ r,
      polynomialReceptivity c p offset target (e0 + r + 1) ≠ 1 / 2) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun k => |(path2MultiplierProduct
          ⟨polynomialReceptivity c p offset target, 0⟩ e0 k : Real)|)
        atTop (nhds L) := by
  let params : ExposureParameters :=
    ⟨polynomialReceptivity c p offset target, 0⟩
  let m : Nat → Real := fun r =>
    ((mixingMass (params.receptivityAt (e0 + r + 1)) : Rat) : Real)
  have hp1 : 1 ≤ p := by omega
  have hb (r : Nat) :
      0 ≤ params.receptivityAt (e0 + r + 1) ∧
        params.receptivityAt (e0 + r + 1) ≤ 1 := by
    simpa [params] using
      polynomialReceptivity_bounds c p offset target (e0 + r + 1)
        hp1 hoffset hc0 hcvalid
  have hm0 : ∀ r, 0 ≤ m r := by
    intro r
    have h := polynomial_mixingMass_nonneg_of_bounds
      (params.receptivityAt (e0 + r + 1)) (hb r).1 (hb r).2
    dsimp [m]
    exact_mod_cast h
  have hmhalf : ∀ r, m r < 1 / 2 := by
    intro r
    have hne : params.receptivityAt (e0 + r + 1) ≠ 1 / 2 := by
      simpa [params] using hnozero r
    have h := polynomial_mixingMass_lt_half_of_ne
      (params.receptivityAt (e0 + r + 1)) hne
    dsimp [m]
    have hhalf : (1 / 2 : Real) = ((1 / 2 : Rat) : Real) := by norm_num
    rw [hhalf]
    exact_mod_cast h
  have hmle : ∀ r,
      m r ≤ ((polynomialDecay c p offset (e0 + r + 1) : Rat) : Real) := by
    intro r
    have h := polynomial_mixingMass_le_decay
      c p offset (e0 + r + 1) target
    dsimp [m, params]
    exact_mod_cast h
  have hdecay0 : ∀ r,
      0 ≤ ((polynomialDecay c p offset (e0 + r + 1) : Rat) : Real) := by
    intro r
    have hpos := polynomialDecay_pos c p offset (e0 + r + 1) hp1 hoffset hc0
    exact le_of_lt (by exact_mod_cast hpos)
  have hdecaySum := polynomial_decay_cast_summable_of_two_le_p c p offset e0 hp
  have hmsum : Summable m :=
    Summable.of_nonneg_of_le hm0 hmle hdecaySum
  rcases abs_product_has_nonzero_limit_of_summable_mixing m hm0 hmhalf hmsum with
    ⟨L, hL, hprod⟩
  have hprodEq :
      (fun k => |(path2MultiplierProduct params e0 k : Real)|) =
        (fun k => ∏ r ∈ Finset.range k, (1 - 2 * m r)) := by
    funext k
    have hRat :
        |path2MultiplierProduct params e0 k| =
          ∏ r ∈ Finset.range k,
            (1 - 2 * mixingMass (params.receptivityAt (e0 + r + 1))) := by
      unfold path2MultiplierProduct
      rw [Finset.abs_prod]
      apply Finset.prod_congr rfl
      intro r hr
      exact abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass
        (hb r).1 (hb r).2
    dsimp [m]
    exact_mod_cast hRat
  refine ⟨L, hL, ?_⟩
  change Tendsto
    (fun k => |(path2MultiplierProduct params e0 k : Real)|)
    atTop (nhds L)
  rw [hprodEq]
  exact hprod


theorem polynomial_signed_product_has_nonzero_limit_of_two_le_p_zero
    (c : Rat) (p offset e0 : Nat)
    (hp : 2 ≤ p)
    (hoffset : 1 ≤ offset)
    (hc0 : 0 < c)
    (hcvalid : c ≤ ((offset : Rat) ^ p))
    (hnozero : ∀ r,
      polynomialReceptivity c p offset DecayTarget.zero (e0 + r + 1) ≠ 1 / 2) :
    ∃ L : Real, L ≠ 0 ∧
      Tendsto
        (fun k => (path2MultiplierProduct
          ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩ e0 k : Real))
        atTop (nhds L) := by
  let params : ExposureParameters :=
    ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩
  let f : Nat → Real := fun r =>
    (-2 : Real) *
      ((polynomialDecay c p offset (e0 + r + 1) : Rat) : Real)
  have hp1 : 1 ≤ p := by omega
  have hdecay0 : ∀ r,
      0 ≤ ((polynomialDecay c p offset (e0 + r + 1) : Rat) : Real) := by
    intro r
    have hpos :=
      polynomialDecay_pos c p offset (e0 + r + 1) hp1 hoffset hc0
    exact le_of_lt (by exact_mod_cast hpos)
  have hdecaySum :=
    polynomial_decay_cast_summable_of_two_le_p c p offset e0 hp
  have hscaled :
      Summable
        (fun r =>
          (2 : Real) *
            ((polynomialDecay c p offset (e0 + r + 1) : Rat) : Real)) :=
    Summable.mul_left (2 : Real) hdecaySum
  have hnorm : Summable (fun r => ‖f r‖) := by
    refine hscaled.congr ?_
    intro r
    dsimp [f]
    rw [abs_mul, abs_of_nonneg (hdecay0 r)]
    norm_num
  have hfactor : ∀ r, 1 + f r ≠ 0 := by
    intro r hzero
    apply hnozero r
    have hcast :
        (1 : Real) -
            2 * ((polynomialDecay c p offset (e0 + r + 1) : Rat) : Real) = 0 := by
      dsimp [f] at hzero
      linarith
    have hrat :
        (1 : Rat) - 2 * polynomialDecay c p offset (e0 + r + 1) = 0 := by
      exact_mod_cast hcast
    have hd :
        polynomialDecay c p offset (e0 + r + 1) = (1 / 2 : Rat) := by
      linarith
    simpa [polynomialReceptivity, applyDecayTarget] using hd
  have hmul : Multipliable (fun r => 1 + f r) :=
    multipliable_one_add_of_summable hnorm
  let L : Real := ∏' r, (1 + f r)
  have hL : L ≠ 0 := by
    dsimp [L]
    exact tprod_one_add_ne_zero_of_summable hfactor hnorm
  have hseq :
      (fun k => (path2MultiplierProduct params e0 k : Real)) =
        (fun k => ∏ r ∈ Finset.range k, (1 + f r)) := by
    funext k
    unfold path2MultiplierProduct
    push_cast
    apply Finset.prod_congr rfl
    intro r hr
    simp [params, f, polynomialReceptivity, applyDecayTarget]
    ring
  refine ⟨L, hL, ?_⟩
  rw [hseq]
  simpa [L] using hmul.hasProd.tendsto_prod_nat



theorem polynomial_signed_product_even_odd_limits_of_two_le_p_one
    (c : Rat) (p offset e0 : Nat)
    (hp : 2 ≤ p)
    (hoffset : 1 ≤ offset)
    (hc0 : 0 < c)
    (hcvalid : c ≤ ((offset : Rat) ^ p))
    (hnozero : ∀ r,
      polynomialReceptivity c p offset DecayTarget.one (e0 + r + 1) ≠ 1 / 2) :
    ∃ L : Real, L ≠ 0 ∧
      Tendsto
        (fun k => (path2MultiplierProduct
          ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 (2 * k) : Real))
        atTop (nhds L) ∧
      Tendsto
        (fun k => (path2MultiplierProduct
          ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 (2 * k + 1) : Real))
        atTop (nhds (-L)) := by
  have hnozero0 : ∀ r,
      polynomialReceptivity c p offset DecayTarget.zero (e0 + r + 1) ≠ 1 / 2 := by
    intro r hzero
    apply hnozero r
    simp [polynomialReceptivity, applyDecayTarget] at hzero ⊢
    linarith
  obtain ⟨L, hL, hzeroLim⟩ :=
    polynomial_signed_product_has_nonzero_limit_of_two_le_p_zero
      c p offset e0 hp hoffset hc0 hcvalid hnozero0
  have hfactor (r : Nat) :
      1 - 2 * polynomialReceptivity c p offset DecayTarget.one (e0 + r + 1) =
        -(1 - 2 * polynomialReceptivity c p offset DecayTarget.zero (e0 + r + 1)) := by
    simp [polynomialReceptivity, applyDecayTarget]
    ring
  have hprod : ∀ k,
      path2MultiplierProduct
          ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 k =
        (-1 : Rat) ^ k *
          path2MultiplierProduct
            ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩ e0 k := by
    intro k
    induction k with
    | zero =>
        simp [path2MultiplierProduct]
    | succ k ih =>
        rw [show
          path2MultiplierProduct
              ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0
                (Nat.succ k) =
            path2MultiplierProduct
                ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 k *
              (1 - 2 * polynomialReceptivity c p offset DecayTarget.one
                (e0 + k + 1)) by
          simp [path2MultiplierProduct, Finset.prod_range_succ]]
        rw [show
          path2MultiplierProduct
              ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩ e0
                (Nat.succ k) =
            path2MultiplierProduct
                ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩ e0 k *
              (1 - 2 * polynomialReceptivity c p offset DecayTarget.zero
                (e0 + k + 1)) by
          simp [path2MultiplierProduct, Finset.prod_range_succ]]
        rw [ih, hfactor k, pow_succ]
        ring
  have hevenIndex : Tendsto (fun k : Nat => 2 * k) atTop atTop := by
    refine Filter.tendsto_atTop.2 ?_
    intro b
    exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩
  have hoddIndex : Tendsto (fun k : Nat => 2 * k + 1) atTop atTop := by
    refine Filter.tendsto_atTop.2 ?_
    intro b
    exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩
  have hzeroEven := hzeroLim.comp hevenIndex
  have hzeroOdd := hzeroLim.comp hoddIndex
  have honeEven :
      Tendsto
        (fun k => (path2MultiplierProduct
          ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 (2 * k) : Real))
        atTop (nhds L) := by
    refine hzeroEven.congr' (Filter.Eventually.of_forall ?_)
    intro k
    change
      (path2MultiplierProduct
        ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩ e0 (2 * k) : Real) =
      (path2MultiplierProduct
        ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 (2 * k) : Real)
    rw [hprod]
    push_cast
    simp [pow_mul]
  have hnegZeroOdd :
      Tendsto
        (fun k => -(path2MultiplierProduct
          ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩ e0 (2 * k + 1) : Real))
        atTop (nhds (-L)) := by
    simpa using hzeroOdd.neg
  have honeOdd :
      Tendsto
        (fun k => (path2MultiplierProduct
          ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 (2 * k + 1) : Real))
        atTop (nhds (-L)) := by
    refine hnegZeroOdd.congr' (Filter.Eventually.of_forall ?_)
    intro k
    change
      -(path2MultiplierProduct
        ⟨polynomialReceptivity c p offset DecayTarget.zero, 0⟩ e0 (2 * k + 1) : Real) =
      (path2MultiplierProduct
        ⟨polynomialReceptivity c p offset DecayTarget.one, 0⟩ e0 (2 * k + 1) : Real)
    rw [hprod]
    push_cast
    simp [pow_succ, pow_mul]
  exact ⟨L, hL, honeEven, honeOdd⟩

theorem harmonic_abs_product_tendsto_zero
    (c : Rat) (offset e0 : Nat) (target : DecayTarget)
    (hoffset : 1 ≤ offset)
    (hc0 : 0 < c)
    (hcvalid : c ≤ offset) :
    Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨harmonicReceptivity c offset target, 0⟩ e0 k : Real)|)
      atTop (nhds 0) := by
  simpa [harmonicReceptivity] using
    (polynomial_abs_product_tendsto_zero_of_p_eq_one
      c offset e0 target hoffset hc0 hcvalid)

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier