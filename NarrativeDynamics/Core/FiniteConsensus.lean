import Mathlib

/-!
# Finite exact consensus helpers

This module is deliberately independent of BB dynamics and graph structure. It
captures only finite rational averaging kernels, normalized stationary weights,
and coordinate-range facts that later path-specific proofs can consume.
-/

namespace NarrativeDynamics.FiniteConsensus

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

end

end NarrativeDynamics.FiniteConsensus
