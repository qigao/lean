import Mathlib

/-!
# Finite exact consensus helpers

This module is deliberately independent of BB dynamics and graph structure. It
captures only finite rational averaging kernels, normalized stationary weights,
and coordinate-range facts that later path-specific proofs can consume.
-/

namespace NarrativeDynamics.FiniteConsensus

open Filter Topology
open scoped BigOperators

noncomputable section

abbrev Kernel (ι : Type*) := Matrix ι ι Rat

def applyKernel {ι : Type*} [Fintype ι] (K : Kernel ι) (x : ι → Rat) : ι → Rat :=
  K.mulVec x

structure AveragingKernel {ι : Type*} [Fintype ι] (K : Kernel ι) : Prop where
  nonneg : ∀ i j, 0 ≤ K i j
  row_sum : ∀ i, ∑ j, K i j = 1

structure StationaryWeights {ι : Type*} [Fintype ι]
    (K : Kernel ι) (π : ι → Rat) : Prop where
  nonneg : ∀ i, 0 ≤ π i
  sum_one : ∑ i, π i = 1
  stationary : ∀ j, ∑ i, π i * K i j = π j

def weightedMean {ι : Type*} [Fintype ι] (π x : ι → Rat) : Rat :=
  ∑ i, π i * x i

def kernelTrajectory {ι : Type*} [Fintype ι]
    (K : Kernel ι) (x : ι → Rat) (k : Nat) : ι → Rat :=
  (applyKernel K)^[k] x

def CommonColumnMass {ι : Type*} [Fintype ι] (K : Kernel ι) (δ : Rat) : Prop :=
  ∃ c : ι, ∀ i, δ ≤ K i c

private theorem univ_nonempty {ι : Type*} [Fintype ι] [Nonempty ι] :
    (Finset.univ : Finset ι).Nonempty := by
  classical
  let i : ι := Classical.choice (show Nonempty ι from inferInstance)
  exact ⟨i, Finset.mem_univ i⟩

noncomputable def coordMin {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) : Rat :=
  (Finset.univ : Finset ι).inf' (univ_nonempty (ι := ι)) x

noncomputable def coordMax {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) : Rat :=
  (Finset.univ : Finset ι).sup' (univ_nonempty (ι := ι)) x

def coordRange {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) : Rat :=
  coordMax x - coordMin x

theorem coordMin_le {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) (i : ι) : coordMin x ≤ x i := by
  classical
  unfold coordMin
  exact Finset.inf'_le x (Finset.mem_univ i)

theorem le_coordMax {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) (i : ι) : x i ≤ coordMax x := by
  classical
  unfold coordMax
  exact Finset.le_sup' x (Finset.mem_univ i)

private theorem le_coordMin {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) (a : Rat) (h : ∀ i, a ≤ x i) : a ≤ coordMin x := by
  classical
  unfold coordMin
  apply Finset.le_inf'
  intro i _
  exact h i

private theorem coordMax_le {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) (a : Rat) (h : ∀ i, x i ≤ a) : coordMax x ≤ a := by
  classical
  unfold coordMax
  apply Finset.sup'_le
  intro i _
  exact h i

theorem coordRange_nonneg {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) : 0 ≤ coordRange x := by
  classical
  let i : ι := Classical.choice (show Nonempty ι from inferInstance)
  exact sub_nonneg.mpr ((coordMin_le x i).trans (le_coordMax x i))

theorem applyKernel_between {ι : Type*} [Fintype ι] [Nonempty ι]
    {K : Kernel ι} (hK : AveragingKernel K) (x : ι → Rat) (i : ι) :
    coordMin x ≤ applyKernel K x i ∧ applyKernel K x i ≤ coordMax x := by
  classical
  have hlow : 0 ≤ applyKernel K x i - coordMin x := by
    calc
      0 ≤ ∑ j, K i j * (x j - coordMin x) := by
        exact Finset.sum_nonneg fun j _ =>
          mul_nonneg (hK.nonneg i j) (sub_nonneg.mpr (coordMin_le x j))
      _ = applyKernel K x i - coordMin x := by
        simp [applyKernel, Matrix.mulVec_apply, dotProduct, mul_sub,
          Finset.sum_sub_distrib, ← Finset.sum_mul, hK.row_sum i]
  have hupp : 0 ≤ coordMax x - applyKernel K x i := by
    calc
      0 ≤ ∑ j, K i j * (coordMax x - x j) := by
        exact Finset.sum_nonneg fun j _ =>
          mul_nonneg (hK.nonneg i j) (sub_nonneg.mpr (le_coordMax x j))
      _ = coordMax x - applyKernel K x i := by
        simp [applyKernel, Matrix.mulVec_apply, dotProduct, mul_sub,
          Finset.sum_sub_distrib, ← Finset.sum_mul, hK.row_sum i]
  exact ⟨sub_nonneg.mp hlow, sub_nonneg.mp hupp⟩

theorem coordRange_apply_le {ι : Type*} [Fintype ι] [Nonempty ι]
    {K : Kernel ι} (hK : AveragingKernel K) (x : ι → Rat) :
    coordRange (applyKernel K x) ≤ coordRange x := by
  unfold coordRange
  apply sub_le_sub
  · exact coordMax_le (applyKernel K x) (coordMax x) fun i =>
      (applyKernel_between hK x i).2
  · exact le_coordMin (applyKernel K x) (coordMin x) fun i =>
      (applyKernel_between hK x i).1

theorem weightedMean_apply {ι : Type*} [Fintype ι]
    {K : Kernel ι} {π : ι → Rat} (hπ : StationaryWeights K π)
    (x : ι → Rat) :
    weightedMean π (applyKernel K x) = weightedMean π x := by
  classical
  simp only [weightedMean, applyKernel, Matrix.mulVec_apply, dotProduct]
  calc
    (∑ i, π i * ∑ j, K i j * x j) =
        ∑ i, ∑ j, (π i * K i j) * x j := by
      apply Finset.sum_congr rfl
      intro i _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ = ∑ j, ∑ i, (π i * K i j) * x j := by
      rw [Finset.sum_comm]
    _ = ∑ j, (∑ i, π i * K i j) * x j := by
      apply Finset.sum_congr rfl
      intro j _
      rw [Finset.sum_mul]
    _ = ∑ j, π j * x j := by
      apply Finset.sum_congr rfl
      intro j _
      rw [hπ.stationary j]

theorem weightedMean_between {ι : Type*} [Fintype ι] [Nonempty ι]
    {K : Kernel ι} {π : ι → Rat} (hπ : StationaryWeights K π)
    (x : ι → Rat) :
    coordMin x ≤ weightedMean π x ∧ weightedMean π x ≤ coordMax x := by
  classical
  have hconstMin : weightedMean π (fun _ => coordMin x) = coordMin x := by
    simp only [weightedMean]
    calc
      (∑ i, π i * coordMin x) = (∑ i, π i) * coordMin x := by
        rw [Finset.sum_mul]
      _ = coordMin x := by rw [hπ.sum_one]; ring
  have hconstMax : weightedMean π (fun _ => coordMax x) = coordMax x := by
    simp only [weightedMean]
    calc
      (∑ i, π i * coordMax x) = (∑ i, π i) * coordMax x := by
        rw [Finset.sum_mul]
      _ = coordMax x := by rw [hπ.sum_one]; ring
  have hlow : weightedMean π (fun _ => coordMin x) ≤ weightedMean π x := by
    unfold weightedMean
    exact Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_left (coordMin_le x i) (hπ.nonneg i)
  have hupp : weightedMean π x ≤ weightedMean π (fun _ => coordMax x) := by
    unfold weightedMean
    exact Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_left (le_coordMax x i) (hπ.nonneg i)
  exact ⟨hconstMin ▸ hlow, hconstMax ▸ hupp⟩

theorem coordinate_dist_weightedMean_le_range
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {K : Kernel ι} {π : ι → Rat} (hπ : StationaryWeights K π)
    (x : ι → Rat) (i : ι) :
    |x i - weightedMean π x| ≤ coordRange x := by
  have hmean := weightedMean_between hπ x
  have hmin := coordMin_le x i
  have hmax := le_coordMax x i
  apply abs_le.mpr
  unfold coordRange
  constructor <;> linarith

@[simp] theorem kernelTrajectory_zero {ι : Type*} [Fintype ι]
    (K : Kernel ι) (x : ι → Rat) : kernelTrajectory K x 0 = x := by
  simp [kernelTrajectory]

theorem kernelTrajectory_succ {ι : Type*} [Fintype ι]
    (K : Kernel ι) (x : ι → Rat) (k : Nat) :
    kernelTrajectory K x (k + 1) =
      applyKernel K (kernelTrajectory K x k) := by
  simp [kernelTrajectory, Function.iterate_succ_apply']

theorem applyKernel_mul {ι : Type*} [Fintype ι]
    (K L : Kernel ι) (x : ι → Rat) :
    applyKernel (K * L) x = applyKernel K (applyKernel L x) := by
  simpa [applyKernel] using (Matrix.mulVec_mulVec x K L).symm

theorem kernelPow_apply {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : Kernel ι) (x : ι → Rat) (k : Nat) :
    applyKernel (K ^ k) x = kernelTrajectory K x k := by
  induction k with
  | zero => simp [applyKernel, kernelTrajectory]
  | succ k ih =>
      rw [pow_succ']
      rw [applyKernel_mul]
      rw [ih]
      exact (kernelTrajectory_succ K x k).symm

private theorem averagingKernel_one {ι : Type*} [Fintype ι] [DecidableEq ι] :
    AveragingKernel (1 : Kernel ι) := by
  classical
  constructor
  · intro i j
    rw [Matrix.one_apply]
    split_ifs <;> norm_num
  · intro i
    simp only [Matrix.one_apply]
    simp

private theorem averagingKernel_mul {ι : Type*} [Fintype ι]
    {K L : Kernel ι} (hK : AveragingKernel K) (hL : AveragingKernel L) :
    AveragingKernel (K * L) := by
  classical
  constructor
  · intro i j
    simp only [Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ =>
      mul_nonneg (hK.nonneg i k) (hL.nonneg k j)
  · intro i
    simp only [Matrix.mul_apply]
    calc
      (∑ j, ∑ k, K i k * L k j) = ∑ k, ∑ j, K i k * L k j := by
        rw [Finset.sum_comm]
      _ = ∑ k, K i k * (∑ j, L k j) := by
        apply Finset.sum_congr rfl
        intro k _
        rw [Finset.mul_sum]
      _ = ∑ k, K i k := by
        simp [hL.row_sum]
      _ = 1 := hK.row_sum i

private theorem averagingKernel_pow {ι : Type*} [Fintype ι] [DecidableEq ι]
    {K : Kernel ι} (hK : AveragingKernel K) (k : Nat) :
    AveragingKernel (K ^ k) := by
  induction k with
  | zero => simpa using (averagingKernel_one (ι := ι))
  | succ k ih =>
      rw [pow_succ]
      exact averagingKernel_mul ih hK

private def residualKernel {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : Kernel ι) (c : ι) (δ : Rat) : Kernel ι :=
  fun i j => (K i j - if j = c then δ else 0) / (1 - δ)

private theorem residualKernel_averaging
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    {K : Kernel ι} (hK : AveragingKernel K)
    (c : ι) (δ : Rat) (hδ1 : δ < 1)
    (hc : ∀ i, δ ≤ K i c) :
    AveragingKernel (residualKernel K c δ) := by
  classical
  have hden : 0 < 1 - δ := sub_pos.mpr hδ1
  constructor
  · intro i j
    simp only [residualKernel]
    apply div_nonneg
    · by_cases hj : j = c
      · subst j
        simpa using sub_nonneg.mpr (hc i)
      · simp [hj, hK.nonneg i j]
    · exact le_of_lt hden
  · intro i
    simp only [residualKernel]
    rw [← Finset.sum_div]
    have hindicator : (∑ j : ι, if j = c then δ else 0) = δ := by
      simp
    rw [Finset.sum_sub_distrib, hK.row_sum i, hindicator]
    exact div_self (ne_of_gt hden)

private theorem kernel_eq_common_add_residual
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : Kernel ι) (c : ι) (δ : Rat) (hδ1 : δ < 1)
    (i j : ι) :
    K i j = (if j = c then δ else 0) +
      (1 - δ) * residualKernel K c δ i j := by
  simp only [residualKernel]
  have hne : 1 - δ ≠ 0 := ne_of_gt (sub_pos.mpr hδ1)
  field_simp [hne]
  ring

private theorem applyKernel_eq_common_add_residual
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : Kernel ι) (c : ι) (δ : Rat) (hδ1 : δ < 1)
    (x : ι → Rat) (i : ι) :
    applyKernel K x i = δ * x c +
      (1 - δ) * applyKernel (residualKernel K c δ) x i := by
  classical
  simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
  calc
    (∑ j, K i j * x j) =
        ∑ j, ((if j = c then δ else 0) +
          (1 - δ) * residualKernel K c δ i j) * x j := by
      apply Finset.sum_congr rfl
      intro j _
      rw [kernel_eq_common_add_residual K c δ hδ1 i j]
    _ = ∑ j, ((if j = c then δ else 0) * x j +
        (1 - δ) * (residualKernel K c δ i j * x j)) := by
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ = (∑ j, (if j = c then δ else 0) * x j) +
        ∑ j, (1 - δ) * (residualKernel K c δ i j * x j) := by
      rw [Finset.sum_add_distrib]
    _ = (∑ j, (if j = c then δ else 0) * x j) +
        (1 - δ) * (∑ j, residualKernel K c δ i j * x j) := by
      rw [Finset.mul_sum]
    _ = δ * x c +
        (1 - δ) * (∑ j, residualKernel K c δ i j * x j) := by
      simp

theorem coordRange_apply_le_of_commonColumn
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    {K : Kernel ι} (hK : AveragingKernel K)
    (δ : Rat) (_hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hcommon : CommonColumnMass K δ) (x : ι → Rat) :
    coordRange (applyKernel K x) ≤ (1 - δ) * coordRange x := by
  classical
  rcases hcommon with ⟨c, hc⟩
  have hR := residualKernel_averaging hK c δ hδ1 hc
  have hfac : 0 ≤ 1 - δ := le_of_lt (sub_pos.mpr hδ1)
  have hlow : ∀ i,
      δ * x c + (1 - δ) * coordMin x ≤ applyKernel K x i := by
    intro i
    rw [applyKernel_eq_common_add_residual K c δ hδ1 x i]
    exact add_le_add_right
      (mul_le_mul_of_nonneg_left (applyKernel_between hR x i).1 hfac) _
  have hupp : ∀ i,
      applyKernel K x i ≤ δ * x c + (1 - δ) * coordMax x := by
    intro i
    rw [applyKernel_eq_common_add_residual K c δ hδ1 x i]
    exact add_le_add_right
      (mul_le_mul_of_nonneg_left (applyKernel_between hR x i).2 hfac) _
  unfold coordRange
  calc
    coordMax (applyKernel K x) - coordMin (applyKernel K x) ≤
        (δ * x c + (1 - δ) * coordMax x) -
          (δ * x c + (1 - δ) * coordMin x) := by
      exact sub_le_sub
        (coordMax_le (applyKernel K x) _ hupp)
        (le_coordMin (applyKernel K x) _ hlow)
    _ = (1 - δ) * (coordMax x - coordMin x) := by ring

private theorem kernelTrajectory_add {ι : Type*} [Fintype ι]
    (K : Kernel ι) (x : ι → Rat) (a b : Nat) :
    kernelTrajectory K x (a + b) =
      kernelTrajectory K (kernelTrajectory K x a) b := by
  unfold kernelTrajectory
  rw [add_comm]
  exact Function.iterate_add_apply (applyKernel K) b a x

private theorem coordRange_kernelTrajectory_add_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {K : Kernel ι} (hK : AveragingKernel K)
    (x : ι → Rat) (k t : Nat) :
    coordRange (kernelTrajectory K x (k + t)) ≤
      coordRange (kernelTrajectory K x k) := by
  induction t with
  | zero => simp
  | succ t ih =>
      rw [Nat.add_succ, kernelTrajectory_succ]
      exact (coordRange_apply_le hK _).trans ih

private theorem coordRange_kernelTrajectory_tail_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {K : Kernel ι} (hK : AveragingKernel K)
    (x : ι → Rat) {a k : Nat} (hak : a ≤ k) :
    coordRange (kernelTrajectory K x k) ≤
      coordRange (kernelTrajectory K x a) := by
  obtain ⟨t, rfl⟩ := Nat.exists_eq_add_of_le hak
  exact coordRange_kernelTrajectory_add_le hK x a t

theorem coordRange_block_iterate_le
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : Kernel ι) (hK : AveragingKernel K)
    (b : Nat) (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hcommon : CommonColumnMass (K ^ b) δ)
    (x : ι → Rat) :
    coordRange (kernelTrajectory K x b) ≤ (1 - δ) * coordRange x := by
  have h := coordRange_apply_le_of_commonColumn
    (averagingKernel_pow hK b) δ hδ0 hδ1 hcommon x
  rw [kernelPow_apply K x b] at h
  exact h

theorem block_geometric_bound
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : Kernel ι) (hK : AveragingKernel K)
    (b : Nat) (_hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hcommon : CommonColumnMass (K ^ b) δ)
    (x : ι → Rat) (q : Nat) :
    coordRange (kernelTrajectory K x (q * b)) ≤
      (1 - δ) ^ q * coordRange x := by
  induction q with
  | zero => simp
  | succ q ih =>
      have hblock := coordRange_block_iterate_le
        K hK b δ hδ0 hδ1 hcommon (kernelTrajectory K x (q * b))
      have hfac : 0 ≤ 1 - δ := le_of_lt (sub_pos.mpr hδ1)
      calc
        coordRange (kernelTrajectory K x (Nat.succ q * b)) =
            coordRange (kernelTrajectory K (kernelTrajectory K x (q * b)) b) := by
          rw [Nat.succ_mul, kernelTrajectory_add]
        _ ≤ (1 - δ) * coordRange (kernelTrajectory K x (q * b)) := hblock
        _ ≤ (1 - δ) * ((1 - δ) ^ q * coordRange x) :=
          mul_le_mul_of_nonneg_left ih hfac
        _ = (1 - δ) ^ (Nat.succ q) * coordRange x := by
          rw [pow_succ]
          ring

private theorem weightedMean_kernelTrajectory
    {ι : Type*} [Fintype ι]
    {K : Kernel ι} {π : ι → Rat} (hπ : StationaryWeights K π)
    (x : ι → Rat) (k : Nat) :
    weightedMean π (kernelTrajectory K x k) = weightedMean π x := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [show kernelTrajectory K x (Nat.succ k) =
          applyKernel K (kernelTrajectory K x k) by
        simpa [Nat.succ_eq_add_one] using kernelTrajectory_succ K x k]
      rw [weightedMean_apply hπ, ih]

theorem block_contraction_tendsto
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : Kernel ι) (π : ι → Rat)
    (hK : AveragingKernel K)
    (hπ : StationaryWeights K π)
    (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hcommon : CommonColumnMass (K ^ b) δ)
    (x : ι → Rat) (i : ι) :
    Tendsto
      (fun k : Nat => (kernelTrajectory K x k i : Real))
      atTop
      (nhds (weightedMean π x : Real)) := by
  have hfac0Rat : 0 ≤ 1 - δ := le_of_lt (sub_pos.mpr hδ1)
  have hfac1Rat : 1 - δ < 1 := by linarith
  have hfac0 : 0 ≤ ((1 - δ : Rat) : Real) := by exact_mod_cast hfac0Rat
  have hfac1 : ((1 - δ : Rat) : Real) < 1 := by exact_mod_cast hfac1Rat
  have hpow : Tendsto (fun q : Nat => ((1 - δ : Rat) : Real) ^ q)
      atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hfac0 hfac1
  have hgeom : Tendsto
      (fun q : Nat => ((1 - δ : Rat) : Real) ^ q * (coordRange x : Real))
      atTop (nhds 0) := by
    simpa using hpow.mul_const (coordRange x : Real)
  rw [Metric.tendsto_atTop]
  intro ε hε
  obtain ⟨q, hq⟩ := (Metric.tendsto_atTop.1 hgeom) ε hε
  have hbound_nonneg :
      0 ≤ ((1 - δ : Rat) : Real) ^ q * (coordRange x : Real) := by
    have hrange : 0 ≤ (coordRange x : Real) := by
      exact_mod_cast coordRange_nonneg x
    positivity
  have hrange_nonneg : 0 ≤ (coordRange x : Real) := by
    exact_mod_cast coordRange_nonneg x
  have hgeom_lt :
      ((1 - δ : Rat) : Real) ^ q * (coordRange x : Real) < ε := by
    have hqq := hq q le_rfl
    simpa [Real.dist_eq, abs_of_nonneg hbound_nonneg,
      abs_of_nonneg hfac0, Real.norm_eq_abs, abs_of_nonneg hrange_nonneg] using hqq
  refine ⟨q * b, ?_⟩
  intro k hk
  have hrangeTail := coordRange_kernelTrajectory_tail_le hK x hk
  have hblock := block_geometric_bound
    K hK b hb δ hδ0 hδ1 hcommon x q
  have hrangeRat :
      coordRange (kernelTrajectory K x k) ≤
        (1 - δ) ^ q * coordRange x :=
    hrangeTail.trans hblock
  have hrangeReal :
      (coordRange (kernelTrajectory K x k) : Real) ≤
        ((1 - δ : Rat) : Real) ^ q * (coordRange x : Real) := by
    exact_mod_cast hrangeRat
  have hcoord := coordinate_dist_weightedMean_le_range
    hπ (kernelTrajectory K x k) i
  rw [weightedMean_kernelTrajectory hπ x k] at hcoord
  have hcoordReal :
      |(kernelTrajectory K x k i : Real) - (weightedMean π x : Real)| ≤
        (coordRange (kernelTrajectory K x k) : Real) := by
    exact_mod_cast hcoord
  rw [Real.dist_eq]
  exact lt_of_le_of_lt (hcoordReal.trans hrangeReal) hgeom_lt

end

end NarrativeDynamics.FiniteConsensus
