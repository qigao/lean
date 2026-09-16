import NarrativeDynamics.Core.FitnessABMPathNExposure
import NarrativeDynamics.Core.FiniteTimeVaryingConsensus

/-!
# Exposure-dependent finite-path convergence bridge

This module keeps the merged executable exposure semantics unchanged. Inside
an all-broadcast trajectory it derives the exact post-incoming exposure law and
bridges executable beliefs to a genuinely time-varying exact-rational kernel
schedule.
-/

namespace NarrativeDynamics.FitnessABMPathNExposureConvergence

open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open NarrativeDynamics.FitnessABMPathNExposure
open Filter Topology
open scoped BigOperators

def allBroadcast (p : ExposureParameters) (s : State n) : Prop :=
  ∀ i, p.threshold ≤ (s i).belief ∧ (s i).belief ≤ 1

private theorem broadcasters_eq_neighbors
    (p : ExposureParameters) (n : Nat) (s : State n)
    (h : allBroadcast p s) (i : Fin n) :
    broadcasters p n s i = FitnessABMPathN.neighbors n i := by
  ext j
  have hb : broadcasting p s j = true := by
    change decide (p.threshold ≤ (s j).belief) = true
    exact decide_eq_true (h j).1
  simp [broadcasters, hb]

theorem incoming_eq_degree
    (p : ExposureParameters) (n : Nat) (s : State n)
    (h : allBroadcast p s) (i : Fin n) :
    incoming p s i = FitnessABMPathN.degree n i := by
  unfold incoming FitnessABMPathN.degree
  rw [broadcasters_eq_neighbors p n s h i]

private theorem degree_eq_one_or_two
    (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    FitnessABMPathN.degree n i = 1 ∨ FitnessABMPathN.degree n i = 2 := by
  have hpos := FitnessABMPathN.degree_pos n hn i
  have hle := FitnessABMPathN.degree_le_two n i
  omega

/-- The proof kernel uses the exposure after the current round's incoming
broadcasts have been accumulated. -/
def exposureKernel (p : ExposureParameters) (n : Nat)
    (e : Fin n → Nat) : Kernel (Fin n) :=
  fun i j =>
    let alpha := p.receptivityAt (e i + FitnessABMPathN.degree n i)
    (if i = j then 1 - alpha else 0) +
      (if j ∈ FitnessABMPathN.neighbors n i then
        alpha / (FitnessABMPathN.degree n i : Rat)
      else 0)

private theorem exposureKernel_nonneg
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (e : Fin n → Nat) (i j : Fin n) :
    0 ≤ exposureKernel p n e i j := by
  classical
  have ha := hvalid.1 (e i + FitnessABMPathN.degree n i)
  have hd0 : (0 : Rat) ≤ (FitnessABMPathN.degree n i : Rat) := by
    exact_mod_cast Nat.zero_le (FitnessABMPathN.degree n i)
  unfold exposureKernel
  dsimp only
  apply add_nonneg
  · split_ifs <;> linarith
  · split_ifs
    · exact div_nonneg ha.1 hd0
    · norm_num

private theorem exposure_self_mass_sum
    (p : ExposureParameters) (n : Nat) (e : Fin n → Nat) (i : Fin n) :
    (∑ j : Fin n,
      if i = j then
        1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)
      else 0) =
      1 - p.receptivityAt (e i + FitnessABMPathN.degree n i) := by
  classical
  simp

private theorem exposure_neighbor_mass_sum
    (p : ExposureParameters) (n : Nat) (hn : 2 ≤ n)
    (e : Fin n → Nat) (i : Fin n) :
    (∑ j : Fin n,
      if j ∈ FitnessABMPathN.neighbors n i then
        p.receptivityAt (e i + FitnessABMPathN.degree n i) /
          (FitnessABMPathN.degree n i : Rat)
      else 0) =
      p.receptivityAt (e i + FitnessABMPathN.degree n i) := by
  classical
  let alpha := p.receptivityAt (e i + FitnessABMPathN.degree n i)
  let c : Rat := alpha / (FitnessABMPathN.degree n i : Rat)
  have hsum :
      (∑ j : Fin n,
        if j ∈ FitnessABMPathN.neighbors n i then c else 0) =
        ∑ j ∈ FitnessABMPathN.neighbors n i, c := by
    simpa using
      (Finset.sum_ite_mem (Finset.univ : Finset (Fin n))
        (FitnessABMPathN.neighbors n i) (fun _ => c))
  rw [show
      (∑ j : Fin n,
        if j ∈ FitnessABMPathN.neighbors n i then
          p.receptivityAt (e i + FitnessABMPathN.degree n i) /
            (FitnessABMPathN.degree n i : Rat)
        else 0) =
        ∑ j : Fin n,
          if j ∈ FitnessABMPathN.neighbors n i then c else 0 by rfl]
  rw [hsum]
  have hd := degree_eq_one_or_two n hn i
  rcases hd with hd | hd
  · have hcard : (FitnessABMPathN.neighbors n i).card = 1 := by
      simpa [FitnessABMPathN.degree] using hd
    simp [hcard, c, alpha, hd]
  · have hcard : (FitnessABMPathN.neighbors n i).card = 2 := by
      simpa [FitnessABMPathN.degree] using hd
    simp [hcard, c, alpha, hd] <;> ring

private theorem exposureKernel_rowsum
    (p : ExposureParameters) (n : Nat) (hn : 2 ≤ n)
    (e : Fin n → Nat) (i : Fin n) :
    ∑ j, exposureKernel p n e i j = 1 := by
  classical
  rw [show (∑ j : Fin n, exposureKernel p n e i j) =
      (∑ j : Fin n,
        if i = j then
          1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)
        else 0) +
      (∑ j : Fin n,
        if j ∈ FitnessABMPathN.neighbors n i then
          p.receptivityAt (e i + FitnessABMPathN.degree n i) /
            (FitnessABMPathN.degree n i : Rat)
        else 0) by
    simp only [exposureKernel, Finset.sum_add_distrib]]
  rw [exposure_self_mass_sum p n e i,
    exposure_neighbor_mass_sum p n hn e i]
  ring

theorem exposureKernel_averaging
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (e : Fin n → Nat) :
    AveragingKernel (exposureKernel p n e) := by
  exact ⟨exposureKernel_nonneg p hvalid n hn e,
    exposureKernel_rowsum p n hn e⟩

private theorem applyKernel_exposureKernel
    (p : ExposureParameters) (n : Nat) (e : Fin n → Nat)
    (x : Fin n → Rat) (i : Fin n) :
    applyKernel (exposureKernel p n e) x i =
      (1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)) * x i +
        (p.receptivityAt (e i + FitnessABMPathN.degree n i) /
          (FitnessABMPathN.degree n i : Rat)) *
          (∑ j ∈ FitnessABMPathN.neighbors n i, x j) := by
  classical
  simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
  change
    (∑ j : Fin n, exposureKernel p n e i j * x j) =
      (1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)) * x i +
        (p.receptivityAt (e i + FitnessABMPathN.degree n i) /
          (FitnessABMPathN.degree n i : Rat)) *
          (∑ j ∈ FitnessABMPathN.neighbors n i, x j)
  simp only [exposureKernel]
  rw [show
      (∑ j : Fin n,
        ((if i = j then
            1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)
          else 0) +
          if j ∈ FitnessABMPathN.neighbors n i then
            p.receptivityAt (e i + FitnessABMPathN.degree n i) /
              (FitnessABMPathN.degree n i : Rat)
          else 0) * x j) =
        (∑ j : Fin n,
          (if i = j then
            1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)
          else 0) * x j) +
        (∑ j : Fin n,
          (if j ∈ FitnessABMPathN.neighbors n i then
            p.receptivityAt (e i + FitnessABMPathN.degree n i) /
              (FitnessABMPathN.degree n i : Rat)
          else 0) * x j) by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring]
  have hself :
      (∑ j : Fin n,
        (if i = j then
          1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)
        else 0) * x j) =
        (1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)) * x i := by
    simp
  rw [hself]
  let c : Rat :=
    p.receptivityAt (e i + FitnessABMPathN.degree n i) /
      (FitnessABMPathN.degree n i : Rat)
  have hneighbor :
      (∑ j : Fin n,
        (if j ∈ FitnessABMPathN.neighbors n i then c else 0) * x j) =
        c * (∑ j ∈ FitnessABMPathN.neighbors n i, x j) := by
    calc
      (∑ j : Fin n,
        (if j ∈ FitnessABMPathN.neighbors n i then c else 0) * x j) =
          ∑ j ∈ FitnessABMPathN.neighbors n i, c * x j := by
        simpa using
          (Finset.sum_ite_mem (Finset.univ : Finset (Fin n))
            (FitnessABMPathN.neighbors n i) (fun j => c * x j))
      _ = c * (∑ j ∈ FitnessABMPathN.neighbors n i, x j) := by
        rw [Finset.mul_sum]
  change
    (1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)) * x i +
      (∑ j : Fin n,
        (if j ∈ FitnessABMPathN.neighbors n i then c else 0) * x j) =
      (1 - p.receptivityAt (e i + FitnessABMPathN.degree n i)) * x i +
        c * (∑ j ∈ FitnessABMPathN.neighbors n i, x j)
  rw [hneighbor]

/-- Inside the all-broadcast region, one executable belief update is exactly
one row-dependent post-incoming kernel update. -/
theorem step_beliefs_eq_kernel
    (p : ExposureParameters) (_hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) :
    beliefs (step p n s) =
      applyKernel (exposureKernel p n (fun i => (s i).exposure)) (beliefs s) := by
  funext i
  have hbroadcasters := broadcasters_eq_neighbors p n s h i
  have hincoming := incoming_eq_degree p n s h i
  have hd : FitnessABMPathN.degree n i ≠ 0 :=
    Nat.ne_of_gt (FitnessABMPathN.degree_pos n hn i)
  have hdq : (FitnessABMPathN.degree n i : Rat) ≠ 0 := by
    exact_mod_cast hd
  change
    (step p n s i).belief =
      applyKernel (exposureKernel p n (fun i => (s i).exposure)) (beliefs s) i
  rw [applyKernel_exposureKernel]
  simp only [step]
  rw [hincoming]
  simp only [if_neg hd]
  unfold broadcasterMean
  rw [hincoming, hbroadcasters]
  change
    (1 - p.receptivityAt ((s i).exposure + FitnessABMPathN.degree n i)) *
        (s i).belief +
      p.receptivityAt ((s i).exposure + FitnessABMPathN.degree n i) *
        ((∑ j ∈ FitnessABMPathN.neighbors n i, (s j).belief) /
          (FitnessABMPathN.degree n i : Rat)) =
      (1 - p.receptivityAt ((s i).exposure + FitnessABMPathN.degree n i)) *
        (s i).belief +
      (p.receptivityAt ((s i).exposure + FitnessABMPathN.degree n i) /
          (FitnessABMPathN.degree n i : Rat)) *
        (∑ j ∈ FitnessABMPathN.neighbors n i, (s j).belief)
  field_simp [hdq] <;> ring

/-- The all-broadcast region is invariant under one valid executable step. -/
theorem allBroadcast_step
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) : allBroadcast p (step p n s) := by
  let i0 : Fin n := ⟨0, by omega⟩
  letI : Nonempty (Fin n) := ⟨i0⟩
  intro i
  have hbetween :=
    applyKernel_between
      (exposureKernel_averaging p hvalid n hn (fun j => (s j).exposure))
      (beliefs s) i
  have hmin : p.threshold ≤ coordMin (beliefs s) := by
    unfold coordMin
    apply Finset.le_inf'
    intro j _
    exact (h j).1
  have hmax : coordMax (beliefs s) ≤ (1 : Rat) := by
    unfold coordMax
    apply Finset.sup'_le
    intro j _
    exact (h j).2
  have heq := congrFun (step_beliefs_eq_kernel p hvalid n hn s h) i
  change p.threshold ≤ (step p n s i).belief ∧ (step p n s i).belief ≤ 1
  rw [show (step p n s i).belief =
      applyKernel (exposureKernel p n (fun j => (s j).exposure)) (beliefs s) i by
    simpa [beliefs] using heq]
  exact ⟨hmin.trans hbetween.1, hbetween.2.trans hmax⟩

/-- The all-broadcast region remains invariant through every finite iterate. -/
theorem allBroadcast_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) (k : Nat) :
    allBroadcast p ((step p n)^[k] s) := by
  induction k with
  | zero => simpa using h
  | succ k ih =>
      simpa [Function.iterate_succ_apply'] using
        allBroadcast_step p hvalid n hn ((step p n)^[k] s) ih

/-- In the all-broadcast region, exposure increases by the path degree in one
round. -/
theorem exposure_step_eq_add_degree
    (p : ExposureParameters) (n : Nat) (s : State n)
    (h : allBroadcast p s) (i : Fin n) :
    (step p n s i).exposure =
      (s i).exposure + FitnessABMPathN.degree n i := by
  unfold step
  dsimp only
  rw [incoming_eq_degree p n s h i]
  split <;> rfl

/-- Exact cumulative exposure law along an all-broadcast executable path. -/
theorem exposure_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) (k : Nat) (i : Fin n) :
    (((step p n)^[k] s) i).exposure =
      (s i).exposure + k * FitnessABMPathN.degree n i := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Function.iterate_succ_apply']
      have hk := allBroadcast_iterate p hvalid n hn s h k
      rw [exposure_step_eq_add_degree p n ((step p n)^[k] s) hk i]
      rw [ih]
      simp [Nat.succ_mul, Nat.add_assoc]

/-- Kernel schedule derived from the initial exposure vector. The kernel at
round `k` internally queries the post-incoming exposure for round `k+1`. -/
def kernelSchedule (p : ExposureParameters) (n : Nat) (s0 : State n) :
    KernelSchedule (Fin n) :=
  fun k => exposureKernel p n
    (fun i => (s0 i).exposure + k * FitnessABMPathN.degree n i)

theorem kernelSchedule_averaging
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n) :
    ∀ k, AveragingKernel (kernelSchedule p n s0 k) := by
  intro k
  exact exposureKernel_averaging p hvalid n hn _

/-- Every finite executable belief iterate agrees exactly with the corresponding
non-homogeneous kernel trajectory. -/
theorem beliefs_iterate_eq_varyingTrajectory
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (h : allBroadcast p s0) (k : Nat) :
    beliefs ((step p n)^[k] s0) =
      varyingTrajectory (kernelSchedule p n s0) (beliefs s0) k := by
  induction k with
  | zero => rfl
  | succ k ih =>
      rw [Function.iterate_succ_apply']
      have hk := allBroadcast_iterate p hvalid n hn s0 h k
      rw [step_beliefs_eq_kernel p hvalid n hn ((step p n)^[k] s0) hk]
      have hexposure :
          (fun i => (((step p n)^[k] s0) i).exposure) =
            (fun i => (s0 i).exposure + k * FitnessABMPathN.degree n i) := by
        funext i
        exact exposure_iterate p hvalid n hn s0 h k i
      have hkernel :
          exposureKernel p n (fun i => (((step p n)^[k] s0) i).exposure) =
            kernelSchedule p n s0 k := by
        rw [hexposure]
        rfl
      rw [hkernel, ih]
      rfl

end NarrativeDynamics.FitnessABMPathNExposureConvergence
