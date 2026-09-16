import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

namespace NarrativeDynamics.FitnessABMPathNExposureConvergence

open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open NarrativeDynamics.FitnessABMPathNExposure
open Filter Topology
open scoped BigOperators

/-- Uniform interior control only at the receptivity lookup points that the
all-broadcast executable trajectory can actually reach. -/
def ReachableInterior (p : ExposureParameters) (n : Nat)
    (s0 : State n) (eps : Rat) : Prop :=
  ∀ k i,
    eps ≤ p.receptivityAt
      ((s0 i).exposure + (k + 1) * FitnessABMPathN.degree n i) ∧
    p.receptivityAt
      ((s0 i).exposure + (k + 1) * FitnessABMPathN.degree n i) ≤ 1 - eps

def beta (eps : Rat) : Rat := eps / 2

def delta (eps : Rat) (n : Nat) : Rat :=
  beta eps ^ FitnessABMPathN.block n

private theorem beta_pos (eps : Rat) (heps : 0 < eps) :
    0 < beta eps := by
  unfold beta
  linarith

private theorem beta_lt_one
    (p : ExposureParameters) (n : Nat) (hn : 2 ≤ n)
    (s0 : State n) (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    beta eps < 1 := by
  let i0 : Fin n := ⟨0, by omega⟩
  have h := hi 0 i0
  have heps_half : eps ≤ (1/2 : Rat) := by
    linarith [h.1, h.2]
  unfold beta
  linarith

private theorem delta_pos
    (eps : Rat) (n : Nat) (heps : 0 < eps) :
    0 < delta eps n := by
  unfold delta
  exact pow_pos (beta_pos eps heps) _

private theorem delta_lt_one
    (p : ExposureParameters) (n : Nat) (hn : 2 ≤ n)
    (s0 : State n) (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    delta eps n < 1 := by
  unfold delta
  exact pow_lt_one₀
    (le_of_lt (beta_pos eps heps))
    (beta_lt_one p n hn s0 eps heps hi)
    (Nat.ne_of_gt (FitnessABMPathN.block_pos n hn))

/-- Every scheduled kernel keeps at least `eps/2` self mass.  The positivity
hypothesis is explicit: without it the standalone lower-bound statement is
false for arbitrary negative `eps`. -/
theorem exposureKernel_self_lower
    (p : ExposureParameters) (n : Nat) (s0 : State n)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps)
    (k : Nat) (i : Fin n) :
    beta eps ≤ kernelSchedule p n s0 k i i := by
  have hri := hi k i
  have hri' :
      eps ≤ p.receptivityAt
          ((s0 i).exposure + k * FitnessABMPathN.degree n i +
            FitnessABMPathN.degree n i) ∧
      p.receptivityAt
          ((s0 i).exposure + k * FitnessABMPathN.degree n i +
            FitnessABMPathN.degree n i) ≤ 1 - eps := by
    simpa [Nat.succ_mul, Nat.add_assoc] using hri
  simp [kernelSchedule, exposureKernel, beta]
  linarith [hri'.2, heps]

/-- Every scheduled path edge carries at least `eps/2` mass. -/
theorem exposureKernel_adj_lower
    (p : ExposureParameters) (n : Nat) (hn : 2 ≤ n)
    (s0 : State n) (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps)
    (k : Nat) (i j : Fin n)
    (hadj : FitnessABMPathN.pathAdj n j i) :
    beta eps ≤ kernelSchedule p n s0 k i j := by
  have hri := hi k i
  have hri' :
      eps ≤ p.receptivityAt
          ((s0 i).exposure + k * FitnessABMPathN.degree n i +
            FitnessABMPathN.degree n i) ∧
      p.receptivityAt
          ((s0 i).exposure + k * FitnessABMPathN.degree n i +
            FitnessABMPathN.degree n i) ≤ 1 - eps := by
    simpa [Nat.succ_mul, Nat.add_assoc] using hri
  have hmem : j ∈ FitnessABMPathN.neighbors n i :=
    (FitnessABMPathN.mem_neighbors_iff i j).2 hadj
  have hne : i ≠ j := by
    intro hij
    subst j
    exact (FitnessABMPathN.pathAdj_self i) hadj
  have hdpos := FitnessABMPathN.degree_pos n hn i
  have hdle := FitnessABMPathN.degree_le_two n i
  have hd : FitnessABMPathN.degree n i = 1 ∨
      FitnessABMPathN.degree n i = 2 := by
    omega
  rcases hd with hd | hd
  · have hlow :
        eps ≤ p.receptivityAt ((s0 i).exposure + k + 1) := by
      simpa [hd] using hri'.1
    simp [kernelSchedule, exposureKernel, beta, hne, hmem, hd]
    linarith [hlow, heps]
  · have hlow :
        eps ≤ p.receptivityAt ((s0 i).exposure + k * 2 + 2) := by
      simpa [hd] using hri'.1
    simp [kernelSchedule, exposureKernel, beta, hne, hmem, hd]
    linarith [hlow, heps]

private def pathOrigin (n : Nat) (hn : 2 ≤ n) : Fin n :=
  ⟨0, by omega⟩

private theorem window_entry_nonneg
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (start len : Nat) (i j : Fin n) :
    0 ≤ windowKernel (kernelSchedule p n s0) start len i j := by
  exact (windowKernel_averaging
    (kernelSchedule p n s0)
    (kernelSchedule_averaging p hvalid n hn s0)
    start len).nonneg i j

private theorem path_window_left_reach_mass
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps)
    (start k : Nat) (hk : k < n) :
    beta eps ^ k ≤
      windowKernel (kernelSchedule p n s0) start k
        ⟨k, hk⟩ (pathOrigin n hn) := by
  induction k with
  | zero =>
      have hzero : (⟨0, hk⟩ : Fin n) = pathOrigin n hn := by
        apply Fin.ext
        rfl
      simp [windowKernel, Matrix.one_apply, hzero]
  | succ k ih =>
      let pred : Fin n := ⟨k, by omega⟩
      let cur : Fin n := ⟨k + 1, hk⟩
      have hreach :
          beta eps ^ k ≤
            windowKernel (kernelSchedule p n s0) start k pred
              (pathOrigin n hn) := by
        simpa [pred] using ih (by omega)
      have hmass_nonneg :
          0 ≤ windowKernel (kernelSchedule p n s0) start k pred
            (pathOrigin n hn) :=
        window_entry_nonneg p hvalid n hn s0 start k pred (pathOrigin n hn)
      have hadj : FitnessABMPathN.pathAdj n pred cur := by
        left
        rfl
      have hkernel :
          beta eps ≤ kernelSchedule p n s0 (start + k) cur pred :=
        exposureKernel_adj_lower p n hn s0 eps heps hi
          (start + k) cur pred hadj
      have hbeta0 : 0 ≤ beta eps := le_of_lt (beta_pos eps heps)
      change
        beta eps ^ (k + 1) ≤
          windowKernel (kernelSchedule p n s0) start (k + 1) cur
            (pathOrigin n hn)
      rw [show
        windowKernel (kernelSchedule p n s0) start (k + 1) =
          kernelSchedule p n s0 (start + k) *
            windowKernel (kernelSchedule p n s0) start k by rfl]
      simp only [Matrix.mul_apply]
      calc
        beta eps ^ (k + 1) = beta eps ^ k * beta eps := by
          rw [pow_succ]
        _ ≤ windowKernel (kernelSchedule p n s0) start k pred
              (pathOrigin n hn) * beta eps :=
          mul_le_mul_of_nonneg_right hreach hbeta0
        _ ≤ windowKernel (kernelSchedule p n s0) start k pred
              (pathOrigin n hn) * kernelSchedule p n s0 (start + k) cur pred :=
          mul_le_mul_of_nonneg_left hkernel hmass_nonneg
        _ = kernelSchedule p n s0 (start + k) cur pred *
              windowKernel (kernelSchedule p n s0) start k pred
                (pathOrigin n hn) := by
          ring
        _ ≤ ∑ j : Fin n,
              kernelSchedule p n s0 (start + k) cur j *
                windowKernel (kernelSchedule p n s0) start k j
                  (pathOrigin n hn) :=
          Finset.single_le_sum
            (fun j _ => mul_nonneg
              ((kernelSchedule_averaging p hvalid n hn s0 (start + k)).nonneg cur j)
              (window_entry_nonneg p hvalid n hn s0 start k j (pathOrigin n hn)))
            (Finset.mem_univ pred)

private theorem path_window_self_pad_mass
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps)
    (start s t : Nat) (i z : Fin n)
    (hstart : beta eps ^ s ≤
      windowKernel (kernelSchedule p n s0) start s i z) :
    beta eps ^ (s + t) ≤
      windowKernel (kernelSchedule p n s0) start (s + t) i z := by
  induction t with
  | zero => simpa using hstart
  | succ t ih =>
      have hmass_nonneg :
          0 ≤ windowKernel (kernelSchedule p n s0) start (s + t) i z :=
        window_entry_nonneg p hvalid n hn s0 start (s + t) i z
      have hself :
          beta eps ≤ kernelSchedule p n s0 (start + (s + t)) i i :=
        exposureKernel_self_lower p n s0 eps heps hi (start + (s + t)) i
      have hbeta0 : 0 ≤ beta eps := le_of_lt (beta_pos eps heps)
      rw [Nat.add_succ]
      rw [show
        windowKernel (kernelSchedule p n s0) start (s + t + 1) =
          kernelSchedule p n s0 (start + (s + t)) *
            windowKernel (kernelSchedule p n s0) start (s + t) by rfl]
      simp only [Matrix.mul_apply]
      calc
        beta eps ^ (s + t + 1) = beta eps ^ (s + t) * beta eps := by
          rw [pow_succ]
        _ ≤ windowKernel (kernelSchedule p n s0) start (s + t) i z * beta eps :=
          mul_le_mul_of_nonneg_right ih hbeta0
        _ ≤ windowKernel (kernelSchedule p n s0) start (s + t) i z *
              kernelSchedule p n s0 (start + (s + t)) i i :=
          mul_le_mul_of_nonneg_left hself hmass_nonneg
        _ = kernelSchedule p n s0 (start + (s + t)) i i *
              windowKernel (kernelSchedule p n s0) start (s + t) i z := by
          ring
        _ ≤ ∑ j : Fin n,
              kernelSchedule p n s0 (start + (s + t)) i j *
                windowKernel (kernelSchedule p n s0) start (s + t) j z :=
          Finset.single_le_sum
            (fun j _ => mul_nonneg
              ((kernelSchedule_averaging p hvalid n hn s0 (start + (s + t))).nonneg i j)
              (window_entry_nonneg p hvalid n hn s0 start (s + t) j z))
            (Finset.mem_univ i)

/-- Every aligned window of `n-1` scheduled kernels has a common origin
column with at least `(eps/2)^(n-1)` mass in every row. -/
theorem path_window_common_mass
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    UniformBlockCommonColumn
      (kernelSchedule p n s0)
      (FitnessABMPathN.block n)
      (delta eps n) := by
  intro start
  let z : Fin n := pathOrigin n hn
  refine ⟨z, ?_⟩
  intro i
  have hiBlock : i.val ≤ FitnessABMPathN.block n := by
    simp [FitnessABMPathN.block]
    omega
  have hrow : (⟨i.val, i.isLt⟩ : Fin n) = i := by
    apply Fin.ext
    rfl
  have hreach :
      beta eps ^ i.val ≤
        windowKernel (kernelSchedule p n s0) start i.val i z := by
    have h := path_window_left_reach_mass
      p hvalid n hn s0 eps heps hi start i.val i.isLt
    simpa [z, hrow] using h
  have hpad := path_window_self_pad_mass
    p hvalid n hn s0 eps heps hi start i.val
      (FitnessABMPathN.block n - i.val) i z hreach
  have htime :
      i.val + (FitnessABMPathN.block n - i.val) =
        FitnessABMPathN.block n := Nat.add_sub_of_le hiBlock
  simpa [delta, htime] using hpad

theorem varyingTrajectory_consensus_exists
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => (varyingTrajectory (kernelSchedule p n s0)
          (beliefs s0) k i : Real))
        atTop (nhds c) := by
  let i0 : Fin n := ⟨0, by omega⟩
  letI : Nonempty (Fin n) := ⟨i0⟩
  exact block_contraction_consensus_exists
    (kernelSchedule p n s0)
    (kernelSchedule_averaging p hvalid n hn s0)
    (beliefs s0)
    (FitnessABMPathN.block n)
    (FitnessABMPathN.block_pos n hn)
    (delta eps n)
    (delta_pos eps n heps)
    (delta_lt_one p n hn s0 eps heps hi)
    (path_window_common_mass p hvalid n hn s0 eps heps hi)

theorem trajectory_consensus_exists
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (hb : allBroadcast p s0)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => ((((step p n)^[k] s0) i).belief : Real))
        atTop (nhds c) := by
  obtain ⟨c, hc⟩ :=
    varyingTrajectory_consensus_exists p hvalid n hn s0 eps heps hi
  refine ⟨c, ?_⟩
  intro i
  refine (hc i).congr' (Filter.Eventually.of_forall ?_)
  intro k
  have hk := congrFun
    (beliefs_iterate_eq_varyingTrajectory p hvalid n hn s0 hb k) i
  have hkReal := congrArg (fun q : Rat => (q : Real)) hk.symm
  simpa [beliefs] using hkReal

theorem trajectory_consensus_exists_of_global_interior
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (hb : allBroadcast p s0)
    (eps : Rat) (heps : 0 < eps)
    (hi : ∀ e, eps ≤ p.receptivityAt e ∧ p.receptivityAt e ≤ 1 - eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => ((((step p n)^[k] s0) i).belief : Real))
        atTop (nhds c) := by
  apply trajectory_consensus_exists p hvalid n hn s0 hb eps heps
  intro k i
  exact hi _

end NarrativeDynamics.FitnessABMPathNExposureConvergence
