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

private theorem path2_degree_zero :
    FitnessABMPathN.degree 2 (0 : Fin 2) = 1 := by
  decide_cbv

private theorem path2_degree_one :
    FitnessABMPathN.degree 2 (1 : Fin 2) = 1 := by
  decide_cbv

private theorem path2_neighbors_zero :
    FitnessABMPathN.neighbors 2 (0 : Fin 2) = {(1 : Fin 2)} := by
  decide_cbv

private theorem path2_neighbors_one :
    FitnessABMPathN.neighbors 2 (1 : Fin 2) = {(0 : Fin 2)} := by
  decide_cbv

private theorem path2_broadcasters_eq_neighbors
    (p : ExposureParameters) (s : State 2)
    (hb : allBroadcast p s) (i : Fin 2) :
    broadcasters p 2 s i = FitnessABMPathN.neighbors 2 i := by
  ext j
  have hbj : broadcasting p s j = true := by
    change decide (p.threshold ≤ (s j).belief) = true
    exact decide_eq_true (hb j).1
  simp [broadcasters, hbj]

private theorem path2_step_belief_zero
    (p : ExposureParameters) (s : State 2)
    (hb : allBroadcast p s) :
    (step p 2 s 0).belief =
      (1 - p.receptivityAt ((s 0).exposure + 1)) * (s 0).belief +
        p.receptivityAt ((s 0).exposure + 1) * (s 1).belief := by
  have hi := incoming_eq_degree p 2 s hb (0 : Fin 2)
  have hbr := path2_broadcasters_eq_neighbors p s hb (0 : Fin 2)
  have hmean : broadcasterMean p s (0 : Fin 2) = (s 1).belief := by
    unfold broadcasterMean
    rw [hbr, path2_neighbors_zero, hi, path2_degree_zero]
    simp
  unfold step
  dsimp only
  rw [hi, path2_degree_zero]
  norm_num [hmean]

private theorem path2_step_belief_one
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    (step p 2 s 1).belief =
      (1 - p.receptivityAt ((s 0).exposure + 1)) * (s 1).belief +
        p.receptivityAt ((s 0).exposure + 1) * (s 0).belief := by
  have hi := incoming_eq_degree p 2 s hb (1 : Fin 2)
  have hbr := path2_broadcasters_eq_neighbors p s hb (1 : Fin 2)
  have hmean : broadcasterMean p s (1 : Fin 2) = (s 0).belief := by
    unfold broadcasterMean
    rw [hbr, path2_neighbors_one, hi, path2_degree_one]
    simp
  unfold step
  dsimp only
  rw [hi, path2_degree_one]
  norm_num [hmean, ← he]

theorem path2_equal_exposure_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (((step p 2)^[k] s) 0).exposure =
      (((step p 2)^[k] s) 1).exposure := by
  have h0 := exposure_iterate p hvalid 2 (by omega) s hb k (0 : Fin 2)
  have h1 := exposure_iterate p hvalid 2 (by omega) s hb k (1 : Fin 2)
  rw [path2_degree_zero] at h0
  rw [path2_degree_one] at h1
  omega

theorem path2_disagreement_step
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    (step p 2 s 0).belief - (step p 2 s 1).belief =
      (1 - 2 * p.receptivityAt ((s 0).exposure + 1)) *
        ((s 0).belief - (s 1).belief) := by
  rw [path2_step_belief_zero p s hb, path2_step_belief_one p s he hb]
  ring

def path2MultiplierProduct (p : ExposureParameters) (e0 k : Nat) : Rat :=
  ∏ r ∈ Finset.range k,
    (1 - 2 * p.receptivityAt (e0 + r + 1))

theorem path2_disagreement_product
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 - beliefs ((step p 2)^[k] s) 1) =
      ((s 0).belief - (s 1).belief) *
        path2MultiplierProduct p (s 0).exposure k := by
  induction k with
  | zero =>
      simp [path2MultiplierProduct, beliefs]
  | succ k ih =>
      rw [Function.iterate_succ_apply']
      have hek := path2_equal_exposure_iterate p hvalid s he hb k
      have hbk := allBroadcast_iterate p hvalid 2 (by omega) s hb k
      change
        (step p 2 ((step p 2)^[k] s) 0).belief -
            (step p 2 ((step p 2)^[k] s) 1).belief =
          ((s 0).belief - (s 1).belief) *
            path2MultiplierProduct p (s 0).exposure (k + 1)
      rw [path2_disagreement_step p ((step p 2)^[k] s) hek hbk]
      change
        (1 - 2 * p.receptivityAt ((((step p 2)^[k] s) 0).exposure + 1)) *
            (beliefs ((step p 2)^[k] s) 0 - beliefs ((step p 2)^[k] s) 1) =
          ((s 0).belief - (s 1).belief) *
            path2MultiplierProduct p (s 0).exposure (k + 1)
      rw [ih]
      have hex := exposure_iterate p hvalid 2 (by omega) s hb k (0 : Fin 2)
      rw [path2_degree_zero] at hex
      have hex' : (((step p 2)^[k] s) 0).exposure = (s 0).exposure + k := by
        simpa using hex
      rw [hex']
      simp [path2MultiplierProduct, Finset.prod_range_succ]
      ring

theorem path2_mean_step
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    ((step p 2 s 0).belief + (step p 2 s 1).belief) / 2 =
      ((s 0).belief + (s 1).belief) / 2 := by
  rw [path2_step_belief_zero p s hb, path2_step_belief_one p s he hb]
  ring

theorem path2_mean_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 + beliefs ((step p 2)^[k] s) 1) / 2 =
      ((s 0).belief + (s 1).belief) / 2 := by
  induction k with
  | zero => rfl
  | succ k ih =>
      rw [Function.iterate_succ_apply']
      have hek := path2_equal_exposure_iterate p hvalid s he hb k
      have hbk := allBroadcast_iterate p hvalid 2 (by omega) s hb k
      change
        ((step p 2 ((step p 2)^[k] s) 0).belief +
          (step p 2 ((step p 2)^[k] s) 1).belief) / 2 =
            ((s 0).belief + (s 1).belief) / 2
      rw [path2_mean_step p ((step p 2)^[k] s) hek hbk]
      exact ih

private theorem path2_belief_zero_formula
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    beliefs ((step p 2)^[k] s) 0 =
      ((s 0).belief + (s 1).belief) / 2 +
        (((s 0).belief - (s 1).belief) *
          path2MultiplierProduct p (s 0).exposure k) / 2 := by
  have hm := path2_mean_iterate p hvalid s he hb k
  have hd := path2_disagreement_product p hvalid s he hb k
  linarith

private theorem path2_belief_one_formula
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    beliefs ((step p 2)^[k] s) 1 =
      ((s 0).belief + (s 1).belief) / 2 -
        (((s 0).belief - (s 1).belief) *
          path2MultiplierProduct p (s 0).exposure k) / 2 := by
  have hm := path2_mean_iterate p hvalid s he hb k
  have hd := path2_disagreement_product p hvalid s he hb k
  linarith

private theorem abs_tendsto_zero_iff_raw
    (f : Nat → Real) :
    Tendsto (fun k => |f k|) atTop (nhds 0) ↔
      Tendsto f atTop (nhds 0) := by
  constructor
  · intro habs
    rw [Metric.tendsto_atTop] at habs ⊢
    intro ε hε
    obtain ⟨N, hN⟩ := habs ε hε
    refine ⟨N, ?_⟩
    intro k hk
    have h := hN k hk
    rw [Real.dist_eq, sub_zero] at h ⊢
    simpa [abs_abs] using h
  · intro hraw
    rw [Metric.tendsto_atTop] at hraw ⊢
    intro ε hε
    obtain ⟨N, hN⟩ := hraw ε hε
    refine ⟨N, ?_⟩
    intro k hk
    have h := hN k hk
    rw [Real.dist_eq, sub_zero] at h ⊢
    simpa [abs_abs] using h

theorem path2_consensus_iff_product_tendsto_zero
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hne : (s 0).belief ≠ (s 1).belief) :
    (Tendsto
      (fun k => |((path2MultiplierProduct p (s 0).exposure k : Rat) : Real)|)
      atTop (nhds 0)) ↔
    (∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop
        (nhds ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real))) := by
  let m : Real := ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real)
  let d : Real := (((s 0).belief - (s 1).belief : Rat) : Real)
  let P : Nat → Real :=
    fun k => ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real)
  have hdRat : (s 0).belief - (s 1).belief ≠ 0 := sub_ne_zero.mpr hne
  have hd : d ≠ 0 := by
    dsimp [d]
    exact_mod_cast hdRat
  constructor
  · intro hprod
    have hP : Tendsto P atTop (nhds 0) :=
      (abs_tendsto_zero_iff_raw P).1 (by simpa [P] using hprod)
    have hscaled0 : Tendsto (fun k => d * P k) atTop (nhds 0) := by
      simpa using Filter.Tendsto.const_mul d hP
    have hscaled : Tendsto (fun k => d * P k / 2) atTop (nhds 0) := by
      have h := hscaled0.mul_const ((2 : Real)⁻¹)
      simpa [div_eq_mul_inv] using h
    have hzeroFormula : ∀ k,
        (beliefs ((step p 2)^[k] s) 0 : Real) = m + d * P k / 2 := by
      intro k
      dsimp [m, d, P]
      exact_mod_cast path2_belief_zero_formula p hvalid s he hb k
    have honeFormula : ∀ k,
        (beliefs ((step p 2)^[k] s) 1 : Real) = m - d * P k / 2 := by
      intro k
      dsimp [m, d, P]
      exact_mod_cast path2_belief_one_formula p hvalid s he hb k
    have hz : Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) 0 : Real))
        atTop (nhds m) := by
      have hlim := Filter.Tendsto.const_add m hscaled
      have hlim' : Tendsto (fun k => m + d * P k / 2) atTop (nhds m) := by
        simpa using hlim
      exact hlim'.congr'
        (Filter.Eventually.of_forall fun k => (hzeroFormula k).symm)
    have ho : Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) 1 : Real))
        atTop (nhds m) := by
      have hlim := Filter.Tendsto.const_sub m hscaled
      have hlim' : Tendsto (fun k => m - d * P k / 2) atTop (nhds m) := by
        simpa using hlim
      exact hlim'.congr'
        (Filter.Eventually.of_forall fun k => (honeFormula k).symm)
    intro i
    fin_cases i
    · simpa [m] using hz
    · simpa [m] using ho
  · intro hcons
    have hdiff : Tendsto
        (fun k =>
          (beliefs ((step p 2)^[k] s) 0 : Real) -
            (beliefs ((step p 2)^[k] s) 1 : Real))
        atTop (nhds 0) := by
      have h := (hcons 0).sub (hcons 1)
      simpa using h
    have hscaled : Tendsto (fun k => d * P k) atTop (nhds 0) := by
      refine hdiff.congr' (Filter.Eventually.of_forall ?_)
      intro k
      dsimp [d, P]
      exact_mod_cast path2_disagreement_product p hvalid s he hb k
    have hinv : Tendsto (fun k => (d * P k) * d⁻¹) atTop (nhds 0) := by
      simpa using hscaled.mul_const d⁻¹
    have hP : Tendsto P atTop (nhds 0) := by
      refine hinv.congr' (Filter.Eventually.of_forall ?_)
      intro k
      dsimp [P]
      field_simp [hd]
    have habs : Tendsto (fun k => |P k|) atTop (nhds 0) :=
      (abs_tendsto_zero_iff_raw P).2 hP
    simpa [P] using habs

end NarrativeDynamics.FitnessABMPathNExposureConvergence
