import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifierPolynomial
import Mathlib.Analysis.SpecificLimits.Normed

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology
open scoped BigOperators

def exponentialDecay (c base : Rat) (offset e : Nat) : Rat :=
  c * base ^ (e + offset)

def exponentialReceptivity
    (c base : Rat) (offset : Nat) (target : DecayTarget) (e : Nat) : Rat :=
  applyDecayTarget target (exponentialDecay c base offset e)

theorem exponentialDecay_pos
    (c base : Rat) (offset e : Nat)
    (hc0 : 0 < c) (hbase0 : 0 < base) :
    0 < exponentialDecay c base offset e := by
  unfold exponentialDecay
  exact mul_pos hc0 (pow_pos hbase0 _)

theorem exponentialDecay_le_one
    (c base : Rat) (offset e : Nat)
    (hc0 : 0 < c)
    (hbase0 : 0 < base) (hbase1 : base < 1)
    (hcvalid : c * base ^ offset ≤ 1) :
    exponentialDecay c base offset e ≤ 1 := by
  have hpow0 : 0 ≤ base ^ e := pow_nonneg hbase0.le _
  have hpow1 : base ^ e ≤ 1 := pow_le_one₀ hbase0.le hbase1.le
  have hcvalid0 : 0 ≤ c * base ^ offset :=
    mul_nonneg hc0.le (pow_nonneg hbase0.le _)
  unfold exponentialDecay
  rw [pow_add]
  calc
    c * (base ^ e * base ^ offset) =
        (c * base ^ offset) * base ^ e := by ring
    _ ≤ 1 * 1 := mul_le_mul hcvalid hpow1 hpow0 (by norm_num)
    _ = 1 := by norm_num

theorem exponentialReceptivity_bounds
    (c base : Rat) (offset : Nat) (target : DecayTarget) (e : Nat)
    (hc0 : 0 < c)
    (hbase0 : 0 < base)
    (hbase1 : base < 1)
    (hcvalid : c * base ^ offset ≤ 1) :
    0 ≤ exponentialReceptivity c base offset target e ∧
      exponentialReceptivity c base offset target e ≤ 1 := by
  have hd0 : 0 < exponentialDecay c base offset e :=
    exponentialDecay_pos c base offset e hc0 hbase0
  have hd1 : exponentialDecay c base offset e ≤ 1 :=
    exponentialDecay_le_one c base offset e hc0 hbase0 hbase1 hcvalid
  cases target with
  | zero =>
      simp only [exponentialReceptivity, applyDecayTarget]
      exact ⟨hd0.le, hd1⟩
  | one =>
      simp only [exponentialReceptivity, applyDecayTarget]
      constructor <;> linarith

private theorem exponential_mixingMass_nonneg_of_bounds
    (a : Rat) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    0 ≤ mixingMass a := by
  exact le_min ha0 (sub_nonneg.mpr ha1)

private theorem exponential_mixingMass_lt_half_of_ne
    (a : Rat) (hne : a ≠ 1 / 2) :
    mixingMass a < 1 / 2 := by
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · exact lt_of_le_of_lt (min_le_left _ _) hlt
  · exact lt_of_le_of_lt (min_le_right _ _) (by linarith)

private theorem exponential_mixingMass_applyDecayTarget
    (target : DecayTarget) (d : Rat) :
    mixingMass (applyDecayTarget target d) = mixingMass d := by
  cases target <;> simp [applyDecayTarget, mixingMass, min_comm]

private theorem exponential_mixingMass_le_decay
    (c base : Rat) (offset e : Nat) (target : DecayTarget) :
    mixingMass (exponentialReceptivity c base offset target e) ≤
      exponentialDecay c base offset e := by
  rw [exponentialReceptivity, exponential_mixingMass_applyDecayTarget]
  exact min_le_left _ _

private theorem exponential_decay_cast_summable
    (c base : Rat) (offset e0 : Nat)
    (hbase0 : 0 < base) (hbase1 : base < 1) :
    Summable
      (fun r : Nat =>
        ((exponentialDecay c base offset (e0 + r + 1) : Rat) : Real)) := by
  have hb0 : (0 : Real) ≤ (base : Real) := by exact_mod_cast hbase0.le
  have hb1 : (base : Real) < 1 := by exact_mod_cast hbase1
  have hgeo : Summable (fun r : Nat => (base : Real) ^ r) :=
    summable_geometric_of_lt_one hb0 hb1
  let K : Real := (c : Real) * (base : Real) ^ (e0 + 1 + offset)
  have hscaled : Summable (fun r : Nat => K * (base : Real) ^ r) :=
    Summable.mul_left K hgeo
  refine hscaled.congr ?_
  intro r
  dsimp [K]
  simp only [exponentialDecay]
  push_cast
  calc
    (c : Real) * (base : Real) ^ (e0 + 1 + offset) * (base : Real) ^ r =
        (c : Real) * ((base : Real) ^ (e0 + 1 + offset) * (base : Real) ^ r) := by
          ring
    _ = (c : Real) * (base : Real) ^ ((e0 + 1 + offset) + r) := by
          rw [← pow_add]
    _ = (c : Real) * (base : Real) ^ (e0 + r + 1 + offset) := by
          congr 2
          omega

theorem exponential_abs_product_has_nonzero_limit
    (c base : Rat) (offset e0 : Nat) (target : DecayTarget)
    (hc0 : 0 < c)
    (hbase0 : 0 < base)
    (hbase1 : base < 1)
    (hcvalid : c * base ^ offset ≤ 1)
    (hnozero : ∀ r,
      exponentialReceptivity c base offset target (e0 + r + 1) ≠ 1 / 2) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun k => |(path2MultiplierProduct
          ⟨exponentialReceptivity c base offset target, 0⟩ e0 k : Real)|)
        atTop (nhds L) := by
  let params : ExposureParameters :=
    ⟨exponentialReceptivity c base offset target, 0⟩
  let m : Nat → Real := fun r =>
    ((mixingMass (params.receptivityAt (e0 + r + 1)) : Rat) : Real)
  have hb (r : Nat) :
      0 ≤ params.receptivityAt (e0 + r + 1) ∧
        params.receptivityAt (e0 + r + 1) ≤ 1 := by
    simpa [params] using
      exponentialReceptivity_bounds
        c base offset target (e0 + r + 1)
        hc0 hbase0 hbase1 hcvalid
  have hm0 : ∀ r, 0 ≤ m r := by
    intro r
    have h := exponential_mixingMass_nonneg_of_bounds
      (params.receptivityAt (e0 + r + 1)) (hb r).1 (hb r).2
    dsimp [m]
    exact_mod_cast h
  have hmhalf : ∀ r, m r < 1 / 2 := by
    intro r
    have hne : params.receptivityAt (e0 + r + 1) ≠ 1 / 2 := by
      simpa [params] using hnozero r
    have h := exponential_mixingMass_lt_half_of_ne
      (params.receptivityAt (e0 + r + 1)) hne
    dsimp [m]
    have hhalf : (1 / 2 : Real) = ((1 / 2 : Rat) : Real) := by norm_num
    rw [hhalf]
    exact_mod_cast h
  have hmle : ∀ r,
      m r ≤
        ((exponentialDecay c base offset (e0 + r + 1) : Rat) : Real) := by
    intro r
    have h := exponential_mixingMass_le_decay
      c base offset (e0 + r + 1) target
    dsimp [m, params]
    exact_mod_cast h
  have hdecay0 : ∀ r,
      0 ≤ ((exponentialDecay c base offset (e0 + r + 1) : Rat) : Real) := by
    intro r
    have h := exponentialDecay_pos c base offset (e0 + r + 1) hc0 hbase0
    exact le_of_lt (by exact_mod_cast h)
  have hdecaySum :=
    exponential_decay_cast_summable c base offset e0 hbase0 hbase1
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


theorem exponential_signed_product_has_nonzero_limit_zero
    (c base : Rat) (offset e0 : Nat)
    (hc0 : 0 < c)
    (hbase0 : 0 < base)
    (hbase1 : base < 1)
    (hcvalid : c * base ^ offset ≤ 1)
    (hnozero : ∀ r,
      exponentialReceptivity c base offset DecayTarget.zero (e0 + r + 1) ≠ 1 / 2) :
    ∃ L : Real, L ≠ 0 ∧
      Tendsto
        (fun k => (path2MultiplierProduct
          ⟨exponentialReceptivity c base offset DecayTarget.zero, 0⟩ e0 k : Real))
        atTop (nhds L) := by
  let params : ExposureParameters :=
    ⟨exponentialReceptivity c base offset DecayTarget.zero, 0⟩
  let f : Nat → Real := fun r =>
    (-2 : Real) *
      ((exponentialDecay c base offset (e0 + r + 1) : Rat) : Real)
  have hdecay0 : ∀ r,
      0 ≤ ((exponentialDecay c base offset (e0 + r + 1) : Rat) : Real) := by
    intro r
    have hpos :=
      exponentialDecay_pos c base offset (e0 + r + 1) hc0 hbase0
    exact le_of_lt (by exact_mod_cast hpos)
  have hdecaySum :=
    exponential_decay_cast_summable c base offset e0 hbase0 hbase1
  have hscaled :
      Summable
        (fun r =>
          (2 : Real) *
            ((exponentialDecay c base offset (e0 + r + 1) : Rat) : Real)) :=
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
            2 * ((exponentialDecay c base offset (e0 + r + 1) : Rat) : Real) = 0 := by
      dsimp [f] at hzero
      linarith
    have hrat :
        (1 : Rat) - 2 * exponentialDecay c base offset (e0 + r + 1) = 0 := by
      exact_mod_cast hcast
    have hd :
        exponentialDecay c base offset (e0 + r + 1) = (1 / 2 : Rat) := by
      linarith
    simpa [exponentialReceptivity, applyDecayTarget] using hd
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
    simp [params, f, exponentialReceptivity, applyDecayTarget]
    ring
  refine ⟨L, hL, ?_⟩
  rw [hseq]
  simpa [L] using hmul.hasProd.tendsto_prod_nat

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier
