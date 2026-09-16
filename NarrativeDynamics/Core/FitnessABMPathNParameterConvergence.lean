import NarrativeDynamics.Core.FitnessABMPathNParameters

/-!
# Parameterized finite-path convergence helpers

This module proves the stationary-weight and mean-preservation layer needed for
exact convergence of the parameterized finite-path BB belief dynamics.
-/

namespace NarrativeDynamics.FitnessABMPathNParameterConvergence

open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FitnessABMPathNParameters
open Filter Topology
open scoped BigOperators

private theorem pathKernel_averaging
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) :
    AveragingKernel (pathKernel params n) := by
  exact ⟨
    FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn,
    FitnessABMPathNParameters.pathKernel_rowsum params hvalid n hn
  ⟩

private theorem pathAdj_symm {n : Nat} (i j : Fin n) :
    FitnessABMPathN.pathAdj n i j ↔ FitnessABMPathN.pathAdj n j i := by
  unfold FitnessABMPathN.pathAdj
  omega

theorem pathKernel_detailed_balance
    (params : ResponseParameters) (_hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n) :
    FitnessABMPathN.stationaryWeight n i * pathKernel params n i j =
      FitnessABMPathN.stationaryWeight n j * pathKernel params n j i := by
  classical
  by_cases hij : i = j
  · subst j
    rfl
  · by_cases hadj : FitnessABMPathN.pathAdj n j i
    · have hji : FitnessABMPathN.pathAdj n i j := (pathAdj_symm i j).2 hadj
      have hwi : FitnessABMPathN.weightSum n ≠ 0 :=
        ne_of_gt (FitnessABMPathN.weightSum_pos n hn)
      have hdiNat : FitnessABMPathN.degree n i ≠ 0 :=
        Nat.ne_of_gt (FitnessABMPathN.degree_pos n hn i)
      have hdjNat : FitnessABMPathN.degree n j ≠ 0 :=
        Nat.ne_of_gt (FitnessABMPathN.degree_pos n hn j)
      have hdi : (FitnessABMPathN.degree n i : Rat) ≠ 0 := by
        exact_mod_cast hdiNat
      have hdj : (FitnessABMPathN.degree n j : Rat) ≠ 0 := by
        exact_mod_cast hdjNat
      rw [show pathKernel params n i j =
          params.receptivity / (FitnessABMPathN.degree n i : Rat) by
        simp [pathKernel, hij, FitnessABMPathN.mem_neighbors_iff, hadj]]
      rw [show pathKernel params n j i =
          params.receptivity / (FitnessABMPathN.degree n j : Rat) by
        simp [pathKernel, Ne.symm hij, FitnessABMPathN.mem_neighbors_iff, hji]]
      unfold FitnessABMPathN.stationaryWeight
      field_simp [hwi, hdi, hdj]
    · have hji : ¬ FitnessABMPathN.pathAdj n i j := by
        intro h
        exact hadj ((pathAdj_symm i j).1 h)
      simp [pathKernel, hij, Ne.symm hij,
        FitnessABMPathN.mem_neighbors_iff, hadj, hji]

theorem path_stationary_weights
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) :
    StationaryWeights (pathKernel params n)
      (FitnessABMPathN.stationaryWeight n) := by
  refine ⟨FitnessABMPathN.stationaryWeight_nonneg n hn,
    FitnessABMPathN.stationaryWeight_sum_one n hn, ?_⟩
  intro j
  calc
    (∑ i, FitnessABMPathN.stationaryWeight n i * pathKernel params n i j) =
        ∑ i, FitnessABMPathN.stationaryWeight n j * pathKernel params n j i := by
      apply Finset.sum_congr rfl
      intro i _
      exact pathKernel_detailed_balance params hvalid n hn i j
    _ = FitnessABMPathN.stationaryWeight n j *
        (∑ i, pathKernel params n j i) := by
      rw [Finset.mul_sum]
    _ = FitnessABMPathN.stationaryWeight n j := by
      rw [FitnessABMPathNParameters.pathKernel_rowsum params hvalid n hn j]
      ring

theorem mean_kernel_step
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) :
    FitnessABMPathN.mean n (applyKernel (pathKernel params n) x) =
      FitnessABMPathN.mean n x := by
  simpa [FitnessABMPathN.mean] using
    (weightedMean_apply (path_stationary_weights params hvalid n hn) x)

theorem mean_kernel_iterate
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (k : Nat) :
    FitnessABMPathN.mean n (kernelTrajectory (pathKernel params n) x k) =
      FitnessABMPathN.mean n x := by
  induction k with
  | zero => simp [FitnessABMPathN.mean]
  | succ k ih =>
      rw [show kernelTrajectory (pathKernel params n) x (Nat.succ k) =
          applyKernel (pathKernel params n)
            (kernelTrajectory (pathKernel params n) x k) by
        simpa [Nat.succ_eq_add_one] using
          kernelTrajectory_succ (pathKernel params n) x k]
      rw [mean_kernel_step params hvalid n hn, ih]

end NarrativeDynamics.FitnessABMPathNParameterConvergence
