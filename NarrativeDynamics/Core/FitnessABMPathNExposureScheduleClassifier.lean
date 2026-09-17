import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence
import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Logic.Equiv.Fin.Rotate
import Mathlib.Order.Filter.AtTopBot.Finite

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
open scoped BigOperators

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

theorem prod_range_add_split
    (f : Nat → Real) (N k : Nat) :
    (∏ r ∈ Finset.range (N + k), f r) =
      (∏ r ∈ Finset.range N, f r) *
        (∏ r ∈ Finset.range k, f (N + r)) := by
  exact Finset.prod_range_add f N k

theorem prod_range_add_eq_zero_of_prefix_zero
    (f : Nat → Real) (N k : Nat)
    (hzero : (∏ r ∈ Finset.range N, f r) = 0) :
    (∏ r ∈ Finset.range (N + k), f r) = 0 := by
  rw [prod_range_add_split f N k, hzero, zero_mul]

theorem tendsto_zero_prod_range_add_iff
    (f : Nat → Real) (N : Nat)
    (hprefix : (∏ r ∈ Finset.range N, f r) ≠ 0) :
    Tendsto
        (fun k => ∏ r ∈ Finset.range (N + k), f r)
        atTop (nhds 0) ↔
      Tendsto
        (fun k => ∏ r ∈ Finset.range k, f (N + r))
        atTop (nhds 0) := by
  have hseq :
      (fun k => ∏ r ∈ Finset.range (N + k), f r) =
        (fun k =>
          (∏ r ∈ Finset.range N, f r) *
            (∏ r ∈ Finset.range k, f (N + r))) := by
    funext k
    exact prod_range_add_split f N k
  rw [hseq]
  exact tendsto_zero_const_mul_iff hprefix

def periodicReceptivity
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat) (e : Nat) : Rat :=
  values ⟨e % period, Nat.mod_lt _ hperiod⟩

def alternatingReceptivity (a b : Rat) : Nat → Rat :=
  periodicReceptivity 2 (by decide) ![a, b]

private theorem periodicReceptivity_add_period
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat) (e : Nat) :
    periodicReceptivity period hperiod values (e + period) =
      periodicReceptivity period hperiod values e := by
  apply congrArg values
  apply Fin.ext
  simp [periodicReceptivity]

private theorem periodic_multiplier_add_period
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat) (e0 r : Nat) :
    1 - 2 * periodicReceptivity period hperiod values (e0 + period + r + 1) =
      1 - 2 * periodicReceptivity period hperiod values (e0 + r + 1) := by
  rw [show e0 + period + r + 1 = (e0 + r + 1) + period by omega]
  rw [periodicReceptivity_add_period period hperiod values (e0 + r + 1)]

private theorem periodic_path2_product_blocks
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat) (e0 n r : Nat) :
    path2MultiplierProduct
        ⟨periodicReceptivity period hperiod values, 0⟩ e0 (period * n + r) =
      (path2MultiplierProduct
        ⟨periodicReceptivity period hperiod values, 0⟩ e0 period) ^ n *
        path2MultiplierProduct
          ⟨periodicReceptivity period hperiod values, 0⟩ e0 r := by
  let f : Nat → Rat := fun x =>
    1 - 2 * periodicReceptivity period hperiod values (e0 + x + 1)
  have hfperiod : ∀ x, f (period + x) = f x := by
    intro x
    simpa [f, Nat.add_assoc] using
      periodic_multiplier_add_period period hperiod values e0 x
  have hblocks : ∀ n r,
      (∏ x ∈ Finset.range (period * n + r), f x) =
        (∏ x ∈ Finset.range period, f x) ^ n *
          (∏ x ∈ Finset.range r, f x) := by
    intro m
    induction m with
    | zero =>
        intro t
        simp
    | succ m ih =>
        intro t
        rw [Nat.mul_succ]
        rw [show period * m + period + t = period + (period * m + t) by omega]
        rw [Finset.prod_range_add]
        have htail :
            (∏ x ∈ Finset.range (period * m + t), f (period + x)) =
              (∏ x ∈ Finset.range (period * m + t), f x) := by
          apply Finset.prod_congr rfl
          intro x hx
          exact hfperiod x
        rw [htail, ih t, pow_succ]
        ring
  simpa [path2MultiplierProduct, f] using hblocks n r

private theorem abs_cast_multiplier_le_one
    (a : Rat) (ha : 0 ≤ a ∧ a ≤ 1) :
    |(((1 - 2 * a : Rat) : Real))| ≤ 1 := by
  have h0 : (0 : Real) ≤ (a : Real) := by exact_mod_cast ha.1
  have h1 : (a : Real) ≤ 1 := by exact_mod_cast ha.2
  rw [abs_le]
  constructor <;> norm_num <;> linarith

private theorem abs_cast_multiplier_lt_one
    (a : Rat) (ha : 0 < a ∧ a < 1) :
    |(((1 - 2 * a : Rat) : Real))| < 1 := by
  have h0 : (0 : Real) < (a : Real) := by exact_mod_cast ha.1
  have h1 : (a : Real) < 1 := by exact_mod_cast ha.2
  rw [abs_lt]
  constructor <;> norm_num <;> linarith

private theorem prod_le_one_of_nonneg_le_one
    (s : Finset Nat) (f : Nat → Real)
    (hnonneg : ∀ x ∈ s, 0 ≤ f x)
    (hle : ∀ x ∈ s, f x ≤ 1) :
    ∏ x ∈ s, f x ≤ 1 := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | @insert a s ha ih =>
      rw [Finset.prod_insert ha]
      have ha0 : 0 ≤ f a := hnonneg a (by simp)
      have ha1 : f a ≤ 1 := hle a (by simp)
      have hs0 : 0 ≤ ∏ x ∈ s, f x := by
        exact Finset.prod_nonneg fun x hx =>
          hnonneg x (Finset.mem_insert_of_mem hx)
      have hs1 : (∏ x ∈ s, f x) ≤ 1 := by
        apply ih
        · intro x hx
          exact hnonneg x (Finset.mem_insert_of_mem hx)
        · intro x hx
          exact hle x (Finset.mem_insert_of_mem hx)
      calc
        f a * ∏ x ∈ s, f x ≤ 1 * ∏ x ∈ s, f x :=
          mul_le_mul_of_nonneg_right ha1 hs0
        _ ≤ 1 * 1 := mul_le_mul_of_nonneg_left hs1 (by norm_num)
        _ = 1 := by norm_num

theorem periodic_abs_product_tendsto_zero_of_contracting_entry
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat)
    (hvalid : ∀ i, 0 ≤ values i ∧ values i ≤ 1)
    (j : Fin period) (hj : 0 < values j ∧ values j < 1)
    (e0 : Nat) :
    Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨periodicReceptivity period hperiod values, 0⟩ e0 k : Real)|)
      atTop (nhds 0) := by
  let p : ExposureParameters :=
    ⟨periodicReceptivity period hperiod values, 0⟩
  let g : Nat → Real := fun r =>
    |(((1 - 2 * periodicReceptivity period hperiod values (e0 + r + 1) : Rat) : Real))|
  let offset : Fin period :=
    ⟨(e0 + 1) % period, Nat.mod_lt _ hperiod⟩
  let r0 : Fin period := (finCycle offset).symm j
  have hphase (r : Fin period) :
      periodicReceptivity period hperiod values (e0 + r.1 + 1) =
        values (finCycle offset r) := by
    apply congrArg values
    apply Fin.ext
    change (e0 + r.1 + 1) % period = (finCycle offset r).1
    rw [finCycle_apply]
    change (e0 + r.1 + 1) % period = (r.1 + offset.1) % period
    rw [show e0 + r.1 + 1 = r.1 + (e0 + 1) by omega]
    simp [offset, Nat.add_mod, Nat.mod_eq_of_lt r.2]
  have hr0phase :
      periodicReceptivity period hperiod values (e0 + r0.1 + 1) = values j := by
    rw [hphase r0]
    simpa [r0] using congrArg values ((finCycle offset).apply_symm_apply j)
  have hgle : ∀ r, g r ≤ 1 := by
    intro r
    let i : Fin period :=
      ⟨(e0 + r + 1) % period, Nat.mod_lt _ hperiod⟩
    have hi := hvalid i
    simpa [g, periodicReceptivity, i] using abs_cast_multiplier_le_one (values i) hi
  have hgr0 : g r0.1 < 1 := by
    change |(((1 - 2 * periodicReceptivity period hperiod values
      (e0 + r0.1 + 1) : Rat) : Real))| < 1
    rw [hr0phase]
    exact abs_cast_multiplier_lt_one (values j) hj
  have hr0mem : r0.1 ∈ Finset.range period :=
    Finset.mem_range.mpr r0.2
  have hrest_nonneg :
      0 ≤ ∏ r ∈ (Finset.range period).erase r0.1, g r := by
    exact Finset.prod_nonneg fun r hr => abs_nonneg _
  have hrest_le :
      (∏ r ∈ (Finset.range period).erase r0.1, g r) ≤ 1 := by
    exact prod_le_one_of_nonneg_le_one
      ((Finset.range period).erase r0.1) g
      (fun r hr => abs_nonneg _)
      (fun r hr => hgle r)
  have hcycle_lt : (∏ r ∈ Finset.range period, g r) < 1 := by
    rw [← (Finset.range period).mul_prod_erase g hr0mem]
    calc
      g r0.1 * (∏ r ∈ (Finset.range period).erase r0.1, g r) ≤
          g r0.1 * 1 :=
        mul_le_mul_of_nonneg_left hrest_le (abs_nonneg _)
      _ = g r0.1 := by ring
      _ < 1 := hgr0
  let B : Rat := path2MultiplierProduct p e0 period
  let q : Real := |(B : Real)|
  have hBabs : q = ∏ r ∈ Finset.range period, g r := by
    simp [q, B, p, g, path2MultiplierProduct, Finset.abs_prod]
  have hq_nonneg : 0 ≤ q := abs_nonneg _
  have hq_lt : q < 1 := by
    rw [hBabs]
    exact hcycle_lt
  have hq_abs_lt : |q| < 1 := by
    simpa [abs_of_nonneg hq_nonneg] using hq_lt
  have hqpow : Tendsto (fun n : Nat => q ^ n) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_abs_lt_one hq_abs_lt
  have hresidue : ∀ r < period,
      Tendsto
        (fun n => |(path2MultiplierProduct p e0 (period * n + r) : Real)|)
        atTop (nhds 0) := by
    intro r hr
    let c : Real := |(path2MultiplierProduct p e0 r : Real)|
    have hmul : Tendsto (fun n : Nat => c * q ^ n) atTop (nhds 0) := by
      simpa using tendsto_const_mul (c := c) hqpow
    have heq :
        (fun n => |(path2MultiplierProduct p e0 (period * n + r) : Real)|) =
          (fun n => c * q ^ n) := by
      funext n
      have hp :
          path2MultiplierProduct p e0 (period * n + r) =
            (path2MultiplierProduct p e0 period) ^ n *
              path2MultiplierProduct p e0 r := by
        simpa [p] using
          periodic_path2_product_blocks period hperiod values e0 n r
      rw [hp]
      simp [c, q, B, p, abs_mul, abs_pow, mul_comm]
    rw [heq]
    exact hmul
  refine Metric.tendsto_atTop.2 ?_
  intro ε hε
  have harith : ∀ r < period,
      ∀ᶠ n in atTop,
        dist |(path2MultiplierProduct p e0 (period * n + r) : Real)| 0 < ε := by
    intro r hr
    rcases (Metric.tendsto_atTop.1 (hresidue r hr)) ε hε with ⟨N, hN⟩
    exact eventually_atTop.2 ⟨N, hN⟩
  have hall : ∀ᶠ k in atTop,
      dist |(path2MultiplierProduct p e0 k : Real)| 0 < ε :=
    Filter.Eventually.atTop_of_arithmetic hperiod.ne' harith
  exact eventually_atTop.1 hall

theorem periodic_abs_product_not_tendsto_zero_of_boundary_values
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat)
    (hboundary : ∀ i, values i = 0 ∨ values i = 1)
    (e0 : Nat) :
    ¬ Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨periodicReceptivity period hperiod values, 0⟩ e0 k : Real)|)
      atTop (nhds 0) := by
  let p : ExposureParameters :=
    ⟨periodicReceptivity period hperiod values, 0⟩
  have hfactor : ∀ r : Nat,
      |1 - 2 * p.receptivityAt (e0 + r + 1)| = 1 := by
    intro r
    let i : Fin period :=
      ⟨(e0 + r + 1) % period, Nat.mod_lt _ hperiod⟩
    change |1 - 2 * values i| = 1
    rcases hboundary i with h0 | h1
    · rw [h0]
      norm_num
    · rw [h1]
      norm_num
  have honeRat : ∀ k : Nat, |path2MultiplierProduct p e0 k| = 1 := by
    intro k
    induction k with
    | zero => simp [path2MultiplierProduct]
    | succ k ih =>
        rw [path2MultiplierProduct, Finset.prod_range_succ]
        rw [abs_mul, hfactor k]
        change |path2MultiplierProduct p e0 k| * 1 = 1
        rw [ih]
        norm_num
  intro hzero
  have honeReal :
      (fun k => |(path2MultiplierProduct p e0 k : Real)|) =
        (fun _ : Nat => (1 : Real)) := by
    funext k
    have hk := honeRat k
    exact_mod_cast hk
  rw [honeReal] at hzero
  have hone : Tendsto (fun _ : Nat => (1 : Real)) atTop (nhds 1) :=
    tendsto_const_nhds
  have h10 : (1 : Real) = 0 := tendsto_nhds_unique hone hzero
  norm_num at h10

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier