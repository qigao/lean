import NarrativeDynamics.Core.FiniteConsensus

/-!
# Finite time-varying consensus helpers

This module extends the finite exact consensus infrastructure to non-homogeneous
sequences of rational averaging kernels.  It deliberately does not assume a
common stationary distribution: convergence is obtained from uniform block
contraction of the coordinate range.
-/

namespace NarrativeDynamics.FiniteTimeVaryingConsensus

open NarrativeDynamics.FiniteConsensus
open Filter Topology
open scoped BigOperators

abbrev KernelSchedule (ι : Type*) := Nat → Kernel ι

def varyingTrajectory {ι : Type*} [Fintype ι]
    (K : KernelSchedule ι) (x : ι → Rat) : Nat → ι → Rat
  | 0 => x
  | k + 1 => applyKernel (K k) (varyingTrajectory K x k)

@[simp] theorem varyingTrajectory_zero {ι : Type*} [Fintype ι]
    (K : KernelSchedule ι) (x : ι → Rat) :
    varyingTrajectory K x 0 = x := rfl

@[simp] theorem varyingTrajectory_succ {ι : Type*} [Fintype ι]
    (K : KernelSchedule ι) (x : ι → Rat) (k : Nat) :
    varyingTrajectory K x (k + 1) =
      applyKernel (K k) (varyingTrajectory K x k) := rfl

/-- Ordered product for a consecutive window.  The newest kernel is multiplied
on the left, matching `applyKernel_mul`. -/
def windowKernel {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (start : Nat) : Nat → Kernel ι
  | 0 => 1
  | len + 1 => K (start + len) * windowKernel K start len

theorem apply_windowKernel {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (x : ι → Rat) (start len : Nat) :
    applyKernel (windowKernel K start len) (varyingTrajectory K x start) =
      varyingTrajectory K x (start + len) := by
  induction len with
  | zero => simp [windowKernel, applyKernel]
  | succ len ih =>
      rw [windowKernel, applyKernel_mul, ih]
      simpa [Nat.add_assoc] using
        (varyingTrajectory_succ K x (start + len)).symm

private theorem averaging_one {ι : Type*} [Fintype ι] [DecidableEq ι] :
    AveragingKernel (1 : Kernel ι) := by
  classical
  constructor
  · intro i j
    rw [Matrix.one_apply]
    split_ifs <;> norm_num
  · intro i
    simp only [Matrix.one_apply]
    simp

private theorem averaging_mul {ι : Type*} [Fintype ι]
    {A B : Kernel ι} (hA : AveragingKernel A) (hB : AveragingKernel B) :
    AveragingKernel (A * B) := by
  classical
  constructor
  · intro i j
    simp only [Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ =>
      mul_nonneg (hA.nonneg i k) (hB.nonneg k j)
  · intro i
    simp only [Matrix.mul_apply]
    calc
      (∑ j, ∑ k, A i k * B k j) = ∑ k, ∑ j, A i k * B k j := by
        rw [Finset.sum_comm]
      _ = ∑ k, A i k * (∑ j, B k j) := by
        apply Finset.sum_congr rfl
        intro k _
        rw [Finset.mul_sum]
      _ = ∑ k, A i k := by
        simp [hB.row_sum]
      _ = 1 := hA.row_sum i

theorem windowKernel_averaging {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (start len : Nat) : AveragingKernel (windowKernel K start len) := by
  induction len with
  | zero => simpa [windowKernel] using (averaging_one (ι := ι))
  | succ len ih =>
      simp only [windowKernel]
      exact averaging_mul (hK (start + len)) ih

private theorem le_coordMin_of_forall
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) (a : Rat) (h : ∀ i, a ≤ x i) : a ≤ coordMin x := by
  classical
  unfold coordMin
  apply Finset.le_inf'
  intro i _
  exact h i

private theorem coordMax_le_of_forall
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) (a : Rat) (h : ∀ i, x i ≤ a) : coordMax x ≤ a := by
  classical
  unfold coordMax
  apply Finset.sup'_le
  intro i _
  exact h i

theorem coordMin_varying_mono {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) :
    Monotone (fun k => coordMin (varyingTrajectory K x k)) := by
  apply monotone_nat_of_le_succ
  intro k
  rw [varyingTrajectory_succ]
  exact le_coordMin_of_forall _ _ fun i =>
    (applyKernel_between (hK k) (varyingTrajectory K x k) i).1

theorem coordMax_varying_anti {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) :
    Antitone (fun k => coordMax (varyingTrajectory K x k)) := by
  apply antitone_nat_of_succ_le
  intro k
  rw [varyingTrajectory_succ]
  exact coordMax_le_of_forall _ _ fun i =>
    (applyKernel_between (hK k) (varyingTrajectory K x k) i).2

theorem coordRange_varying_le {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (k : Nat) :
    coordRange (varyingTrajectory K x (k + 1)) ≤
      coordRange (varyingTrajectory K x k) := by
  rw [varyingTrajectory_succ]
  exact coordRange_apply_le (hK k) _

def UniformBlockCommonColumn {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (b : Nat) (δ : Rat) : Prop :=
  ∀ start, CommonColumnMass (windowKernel K start b) δ

theorem coordRange_block_le
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (start b : Nat)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    coordRange (varyingTrajectory K x (start + b)) ≤
      (1 - δ) * coordRange (varyingTrajectory K x start) := by
  rw [← apply_windowKernel K x start b]
  exact coordRange_apply_le_of_commonColumn
    (windowKernel_averaging K hK start b) δ hδ0 hδ1 (hc start) _

theorem block_geometric_bound
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (_hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) (q : Nat) :
    coordRange (varyingTrajectory K x (q * b)) ≤
      (1 - δ) ^ q * coordRange x := by
  induction q with
  | zero => simp
  | succ q ih =>
      have hblock := coordRange_block_le
        K hK x (q * b) b δ hδ0 hδ1 hc
      have hfac : 0 ≤ 1 - δ := le_of_lt (sub_pos.mpr hδ1)
      calc
        coordRange (varyingTrajectory K x (Nat.succ q * b)) =
            coordRange (varyingTrajectory K x (q * b + b)) := by
          rw [Nat.succ_mul]
        _ ≤ (1 - δ) * coordRange (varyingTrajectory K x (q * b)) := hblock
        _ ≤ (1 - δ) * ((1 - δ) ^ q * coordRange x) :=
          mul_le_mul_of_nonneg_left ih hfac
        _ = (1 - δ) ^ (Nat.succ q) * coordRange x := by
          rw [pow_succ]
          ring

theorem coordRange_varying_le_of_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) {a k : Nat} (hak : a ≤ k) :
    coordRange (varyingTrajectory K x k) ≤
      coordRange (varyingTrajectory K x a) := by
  unfold coordRange
  exact sub_le_sub
    ((coordMax_varying_anti K hK x) hak)
    ((coordMin_varying_mono K hK x) hak)

theorem coordRange_tendsto_zero
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    Tendsto (fun k => (coordRange (varyingTrajectory K x k) : Real))
      atTop (nhds 0) := by
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
  refine ⟨q * b, ?_⟩
  intro k hk
  have htail := coordRange_varying_le_of_le K hK x hk
  have hblock := block_geometric_bound K hK x b hb δ hδ0 hδ1 hc q
  have hrat :
      coordRange (varyingTrajectory K x k) ≤
        (1 - δ) ^ q * coordRange x := htail.trans hblock
  have hreal :
      (coordRange (varyingTrajectory K x k) : Real) ≤
        ((1 - δ : Rat) : Real) ^ q * (coordRange x : Real) := by
    exact_mod_cast hrat
  have hbound_nonneg :
      0 ≤ ((1 - δ : Rat) : Real) ^ q * (coordRange x : Real) := by
    have hrange : 0 ≤ (coordRange x : Real) := by
      exact_mod_cast coordRange_nonneg x
    positivity
  have hgeom_lt :
      ((1 - δ : Rat) : Real) ^ q * (coordRange x : Real) < ε := by
    have hqq := hq q le_rfl
    rw [Real.dist_eq, sub_zero, abs_of_nonneg hbound_nonneg] at hqq
    exact hqq
  have hrange_nonneg :
      0 ≤ (coordRange (varyingTrajectory K x k) : Real) := by
    exact_mod_cast coordRange_nonneg (varyingTrajectory K x k)
  rw [Real.dist_eq, sub_zero, abs_of_nonneg hrange_nonneg]
  exact lt_of_le_of_lt hreal hgeom_lt

private theorem coordinate_abs_sub_le_range
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → Rat) (i j : ι) :
    |x i - x j| ≤ coordRange x := by
  have hmin_i := coordMin_le x i
  have hmin_j := coordMin_le x j
  have hmax_i := le_coordMax x i
  have hmax_j := le_coordMax x j
  apply abs_le.mpr
  unfold coordRange
  constructor <;> linarith

private theorem later_coordinate_between
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) {a k : Nat} (hak : a ≤ k) (i : ι) :
    coordMin (varyingTrajectory K x a) ≤ varyingTrajectory K x k i ∧
      varyingTrajectory K x k i ≤ coordMax (varyingTrajectory K x a) := by
  constructor
  · exact ((coordMin_varying_mono K hK x) hak).trans
      (coordMin_le (varyingTrajectory K x k) i)
  · exact (le_coordMax (varyingTrajectory K x k) i).trans
      ((coordMax_varying_anti K hK x) hak)

private theorem later_coordinate_abs_sub_le_range
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) {a m n : Nat} (ham : a ≤ m) (han : a ≤ n)
    (i j : ι) :
    |varyingTrajectory K x m i - varyingTrajectory K x n j| ≤
      coordRange (varyingTrajectory K x a) := by
  have hm := later_coordinate_between K hK x ham i
  have hn := later_coordinate_between K hK x han j
  apply abs_le.mpr
  unfold coordRange
  constructor <;> linarith

theorem block_contraction_consensus_exists
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    ∃ c : Real, ∀ i,
      Tendsto (fun k => (varyingTrajectory K x k i : Real)) atTop (nhds c) := by
  classical
  have hrange := coordRange_tendsto_zero K hK x b hb δ hδ0 hδ1 hc
  let i0 : ι := Classical.choice (show Nonempty ι from inferInstance)
  have hrefCauchy :
      CauchySeq (fun k => (varyingTrajectory K x k i0 : Real)) := by
    rw [Metric.cauchySeq_iff]
    intro ε hε
    obtain ⟨N, hN⟩ := (Metric.tendsto_atTop.1 hrange) ε hε
    refine ⟨N, ?_⟩
    intro m hm n hn
    have hdistRat := later_coordinate_abs_sub_le_range K hK x hm hn i0 i0
    have hdistReal :
        |(varyingTrajectory K x m i0 : Real) -
            (varyingTrajectory K x n i0 : Real)| ≤
          (coordRange (varyingTrajectory K x N) : Real) := by
      exact_mod_cast hdistRat
    have hrange_nonneg :
        0 ≤ (coordRange (varyingTrajectory K x N) : Real) := by
      exact_mod_cast coordRange_nonneg (varyingTrajectory K x N)
    have hrange_lt :
        (coordRange (varyingTrajectory K x N) : Real) < ε := by
      have hNN := hN N le_rfl
      rw [Real.dist_eq, sub_zero, abs_of_nonneg hrange_nonneg] at hNN
      exact hNN
    rw [Real.dist_eq]
    exact lt_of_le_of_lt hdistReal hrange_lt
  obtain ⟨c, href⟩ := cauchySeq_tendsto_of_complete hrefCauchy
  refine ⟨c, ?_⟩
  intro i
  rw [Metric.tendsto_atTop]
  intro ε hε
  have heps2 : 0 < ε / 2 := half_pos hε
  obtain ⟨Nref, hNref⟩ := (Metric.tendsto_atTop.1 href) (ε / 2) heps2
  obtain ⟨Nrange, hNrange⟩ := (Metric.tendsto_atTop.1 hrange) (ε / 2) heps2
  refine ⟨max Nref Nrange, ?_⟩
  intro k hk
  have hkref : Nref ≤ k := (Nat.le_max_left Nref Nrange).trans hk
  have hkrange : Nrange ≤ k := (Nat.le_max_right Nref Nrange).trans hk
  have href_lt := hNref k hkref
  have hrange_raw := hNrange k hkrange
  have hrange_nonneg :
      0 ≤ (coordRange (varyingTrajectory K x k) : Real) := by
    exact_mod_cast coordRange_nonneg (varyingTrajectory K x k)
  have hrange_lt :
      (coordRange (varyingTrajectory K x k) : Real) < ε / 2 := by
    rw [Real.dist_eq, sub_zero, abs_of_nonneg hrange_nonneg] at hrange_raw
    exact hrange_raw
  have hsameRat := coordinate_abs_sub_le_range
    (varyingTrajectory K x k) i i0
  have hsame :
      dist (varyingTrajectory K x k i : Real)
        (varyingTrajectory K x k i0 : Real) ≤
        (coordRange (varyingTrajectory K x k) : Real) := by
    rw [Real.dist_eq]
    exact_mod_cast hsameRat
  have hsame_lt :
      dist (varyingTrajectory K x k i : Real)
        (varyingTrajectory K x k i0 : Real) < ε / 2 :=
    lt_of_le_of_lt hsame hrange_lt
  calc
    dist (varyingTrajectory K x k i : Real) c ≤
        dist (varyingTrajectory K x k i : Real)
          (varyingTrajectory K x k i0 : Real) +
        dist (varyingTrajectory K x k i0 : Real) c :=
      dist_triangle _ _ _
    _ < ε / 2 + ε / 2 := add_lt_add hsame_lt href_lt
    _ = ε := by ring

end NarrativeDynamics.FiniteTimeVaryingConsensus
