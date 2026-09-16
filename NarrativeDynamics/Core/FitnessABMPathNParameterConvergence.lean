import NarrativeDynamics.Core.FitnessABMPathNParameters

/-!
# Parameterized finite-path convergence helpers

This module proves exact stationary weights and positive block contraction mass
for the parameterized finite-path BB belief dynamics.
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

/-- Conservative positive per-step mass for strict-interior receptivity. -/
def beta (params : ResponseParameters) : Rat :=
  params.receptivity * (1 - params.receptivity) / 2

theorem beta_pos
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1) :
    0 < beta params := by
  unfold beta
  have h1 : 0 < 1 - params.receptivity := sub_pos.mpr ha1
  positivity

theorem beta_lt_one
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1) :
    beta params < 1 := by
  unfold beta
  have hsquare : 0 ≤ (params.receptivity - (1/2 : Rat)) ^ 2 := sq_nonneg _
  nlinarith

theorem pathKernel_self_lower
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (i : Fin n) :
    beta params ≤ pathKernel params n i i := by
  have hself : i ∉ FitnessABMPathN.neighbors n i := by
    intro h
    exact FitnessABMPathN.pathAdj_self i
      ((FitnessABMPathN.mem_neighbors_iff i i).mp h)
  simp [pathKernel, hself, beta]
  nlinarith

theorem pathKernel_adj_lower
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n)
    (h : FitnessABMPathN.pathAdj n j i) :
    beta params ≤ pathKernel params n i j := by
  classical
  have hmem : j ∈ FitnessABMPathN.neighbors n i :=
    (FitnessABMPathN.mem_neighbors_iff i j).mpr h
  have hne : i ≠ j := by
    intro hij
    subst j
    exact FitnessABMPathN.pathAdj_self i h
  have hpos := FitnessABMPathN.degree_pos n hn i
  have hle := FitnessABMPathN.degree_le_two n i
  have hd : FitnessABMPathN.degree n i = 1 ∨
      FitnessABMPathN.degree n i = 2 := by
    omega
  rcases hd with hd | hd <;>
    simp [pathKernel, beta, hne, hmem, hd] <;>
    nlinarith

def delta (params : ResponseParameters) (n : Nat) : Rat :=
  beta params ^ FitnessABMPathN.block n

theorem delta_pos
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (_hn : 2 ≤ n) :
    0 < delta params n := by
  unfold delta
  exact pow_pos (beta_pos params ha0 ha1) _

theorem delta_lt_one
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) :
    delta params n < 1 := by
  unfold delta
  exact pow_lt_one₀
    (le_of_lt (beta_pos params ha0 ha1))
    (beta_lt_one params ha0 ha1)
    (Nat.ne_of_gt (FitnessABMPathN.block_pos n hn))

private def originBasis {n : Nat} (z : Fin n) : Beliefs n :=
  fun j => if j = z then 1 else 0

private theorem kernelTrajectory_origin_nonneg
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (z : Fin n) (k : Nat) (i : Fin n) :
    0 ≤ kernelTrajectory (pathKernel params n) (originBasis z) k i := by
  induction k generalizing i with
  | zero =>
      by_cases h : i = z <;> simp [kernelTrajectory, originBasis, h]
  | succ k ih =>
      rw [kernelTrajectory_succ]
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      exact Finset.sum_nonneg fun j _ =>
        mul_nonneg
          (FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn i j)
          (ih j)

private theorem left_reach_mass
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) (z : Fin n) (hz : z.val = 0)
    (k : Nat) (hk : k < n) :
    beta params ^ k ≤
      kernelTrajectory (pathKernel params n) (originBasis z) k ⟨k, hk⟩ := by
  induction k with
  | zero =>
      have hzero : (⟨0, hk⟩ : Fin n) = z := by
        apply Fin.ext
        simpa using hz.symm
      simp [kernelTrajectory, originBasis, hzero]
  | succ k ih =>
      let p : Fin n := ⟨k, by omega⟩
      let i : Fin n := ⟨k + 1, by omega⟩
      have hreach : beta params ^ k ≤
          kernelTrajectory (pathKernel params n) (originBasis z) k p := by
        exact ih (by omega)
      have hmass_nonneg :
          0 ≤ kernelTrajectory (pathKernel params n) (originBasis z) k p :=
        kernelTrajectory_origin_nonneg params hvalid n hn z k p
      have hadj : FitnessABMPathN.pathAdj n p i := by
        left
        rfl
      have hkernel : beta params ≤ pathKernel params n i p :=
        pathKernel_adj_lower params ha0 ha1 n hn i p hadj
      change beta params ^ (k + 1) ≤
        kernelTrajectory (pathKernel params n) (originBasis z) (k + 1) i
      rw [kernelTrajectory_succ]
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      calc
        beta params ^ (k + 1) = beta params ^ k * beta params := by
          rw [pow_succ]
        _ ≤ kernelTrajectory (pathKernel params n) (originBasis z) k p *
            beta params :=
          mul_le_mul_of_nonneg_right hreach (le_of_lt (beta_pos params ha0 ha1))
        _ ≤ kernelTrajectory (pathKernel params n) (originBasis z) k p *
            pathKernel params n i p :=
          mul_le_mul_of_nonneg_left hkernel hmass_nonneg
        _ = pathKernel params n i p *
            kernelTrajectory (pathKernel params n) (originBasis z) k p := by
          ring
        _ ≤ ∑ j : Fin n,
            pathKernel params n i j *
              kernelTrajectory (pathKernel params n) (originBasis z) k j :=
          Finset.single_le_sum
            (fun j _ => mul_nonneg
              (FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn i j)
              (kernelTrajectory_origin_nonneg params hvalid n hn z k j))
            (Finset.mem_univ p)

private theorem self_pad_mass
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) (z i : Fin n)
    (s t : Nat)
    (hstart : beta params ^ s ≤
      kernelTrajectory (pathKernel params n) (originBasis z) s i) :
    beta params ^ (s + t) ≤
      kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i := by
  induction t with
  | zero => simpa using hstart
  | succ t ih =>
      rw [Nat.add_succ, kernelTrajectory_succ]
      have hx : ∀ j,
          0 ≤ kernelTrajectory (pathKernel params n) (originBasis z) (s + t) j :=
        fun j => kernelTrajectory_origin_nonneg params hvalid n hn z (s + t) j
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      calc
        beta params ^ (s + t + 1) =
            beta params ^ (s + t) * beta params := by
          rw [pow_succ]
        _ ≤ kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i *
            beta params :=
          mul_le_mul_of_nonneg_right ih (le_of_lt (beta_pos params ha0 ha1))
        _ = beta params *
            kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i := by
          ring
        _ ≤ pathKernel params n i i *
            kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i :=
          mul_le_mul_of_nonneg_right
            (pathKernel_self_lower params ha0 ha1 n i) (hx i)
        _ ≤ ∑ j : Fin n,
            pathKernel params n i j *
              kernelTrajectory (pathKernel params n) (originBasis z) (s + t) j :=
          Finset.single_le_sum
            (fun j _ => mul_nonneg
              (FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn i j)
              (hx j))
            (Finset.mem_univ i)

private theorem applyKernel_originBasis
    {n : Nat} (M : Kernel (Fin n)) (z i : Fin n) :
    applyKernel M (originBasis z) i = M i z := by
  classical
  simp [applyKernel, Matrix.mulVec_apply, dotProduct, originBasis]

theorem path_block_common_mass
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass
      ((pathKernel params n) ^ FitnessABMPathN.block n)
      (delta params n) := by
  let z : Fin n := ⟨0, by omega⟩
  refine ⟨z, ?_⟩
  intro i
  have hi : i.val ≤ FitnessABMPathN.block n := by
    simp [FitnessABMPathN.block]
    omega
  have hreach : beta params ^ i.val ≤
      kernelTrajectory (pathKernel params n) (originBasis z) i.val i := by
    have hz : z.val = 0 := by rfl
    simpa using left_reach_mass params hvalid ha0 ha1 n hn z hz i.val i.isLt
  have hpad := self_pad_mass params hvalid ha0 ha1 n hn z i i.val
    (FitnessABMPathN.block n - i.val) hreach
  have htime : i.val + (FitnessABMPathN.block n - i.val) =
      FitnessABMPathN.block n := Nat.add_sub_of_le hi
  have hmass : delta params n ≤
      kernelTrajectory (pathKernel params n) (originBasis z)
        (FitnessABMPathN.block n) i := by
    simpa [delta, htime] using hpad
  have hp := congrFun
    (kernelPow_apply (pathKernel params n) (originBasis z)
      (FitnessABMPathN.block n)) i
  calc
    delta params n ≤
        kernelTrajectory (pathKernel params n) (originBasis z)
          (FitnessABMPathN.block n) i := hmass
    _ = applyKernel ((pathKernel params n) ^ FitnessABMPathN.block n)
        (originBasis z) i := hp.symm
    _ = ((pathKernel params n) ^ FitnessABMPathN.block n) i z :=
      applyKernel_originBasis
        ((pathKernel params n) ^ FitnessABMPathN.block n) z i

theorem trajectory_eq_kernelTrajectory
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n) (hx : allBroadcast params n x)
    (k : Nat) :
    (beliefStep params n)^[k] x =
      kernelTrajectory (pathKernel params n) x k := by
  induction k with
  | zero => simp [kernelTrajectory]
  | succ k ih =>
      have hk := FitnessABMPathNParameters.allBroadcast_iterate
        params hvalid n hn x hx k
      have hbridge :
          beliefStep params n ((beliefStep params n)^[k] x) =
            applyKernel (pathKernel params n) ((beliefStep params n)^[k] x) :=
        FitnessABMPathNParameters.propagate_eq_kernel
          params hvalid n hn ((beliefStep params n)^[k] x) hk
      rw [Function.iterate_succ_apply', hbridge, ih]
      simpa [Nat.succ_eq_add_one] using
        (kernelTrajectory_succ (pathKernel params n) x k).symm

theorem trajectory_tendsto
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n) (hx : allBroadcast params n x)
    (i : Fin n) :
    Tendsto
      (fun k : Nat => ((((beliefStep params n)^[k] x) i : Rat) : Real))
      atTop
      (nhds (FitnessABMPathN.mean n x : Real)) := by
  let i0 : Fin n := ⟨0, by omega⟩
  letI : Nonempty (Fin n) := ⟨i0⟩
  have hkernel := block_contraction_tendsto
    (pathKernel params n) (FitnessABMPathN.stationaryWeight n)
    (pathKernel_averaging params hvalid n hn)
    (path_stationary_weights params hvalid n hn)
    (FitnessABMPathN.block n) (FitnessABMPathN.block_pos n hn)
    (delta params n) (delta_pos params ha0 ha1 n hn)
    (delta_lt_one params ha0 ha1 n hn)
    (path_block_common_mass params hvalid ha0 ha1 n hn) x i
  have htraj :
      (fun k : Nat => ((((beliefStep params n)^[k] x) i : Rat) : Real)) =
        (fun k : Nat => (kernelTrajectory (pathKernel params n) x k i : Real)) := by
    funext k
    rw [trajectory_eq_kernelTrajectory params hvalid n hn x hx k]
  rw [htraj]
  simpa [FitnessABMPathN.mean] using hkernel

end NarrativeDynamics.FitnessABMPathNParameterConvergence
