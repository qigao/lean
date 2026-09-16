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

def slowZeroSchedule : ExposureParameters :=
  ⟨fun e => 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)), 0⟩

def nearOneSchedule : ExposureParameters :=
  ⟨fun e => 1 - 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)), 0⟩

def harmonicSchedule : ExposureParameters :=
  ⟨fun e => 1 / (2 * ((e + 1 : Nat) : Rat)), 0⟩

private theorem slowFraction_bounds (e : Nat) :
    0 ≤ (1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) : Rat) ∧
      (1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) : Rat) ≤ 1 := by
  let x : Rat := ((e + 1 : Nat) : Rat)
  have hx : (1 : Rat) ≤ x := by
    dsimp [x]
    exact_mod_cast Nat.succ_le_succ (Nat.zero_le e)
  have hx0 : (0 : Rat) < x := lt_of_lt_of_le (by norm_num) hx
  have hden : (0 : Rat) < 2 * x ^ 2 := by positivity
  constructor
  · exact div_nonneg (by norm_num) hden.le
  · rw [div_le_iff₀ hden]
    have hx2 : (1 : Rat) ≤ x ^ 2 := by
      nlinarith [sq_nonneg (x - 1)]
    nlinarith

private theorem harmonicFraction_bounds (e : Nat) :
    0 ≤ (1 / (2 * ((e + 1 : Nat) : Rat)) : Rat) ∧
      (1 / (2 * ((e + 1 : Nat) : Rat)) : Rat) ≤ 1 := by
  let x : Rat := ((e + 1 : Nat) : Rat)
  have hx : (1 : Rat) ≤ x := by
    dsimp [x]
    exact_mod_cast Nat.succ_le_succ (Nat.zero_le e)
  have hden : (0 : Rat) < 2 * x := by nlinarith
  constructor
  · exact div_nonneg (by norm_num) hden.le
  · rw [div_le_iff₀ hden]
    nlinarith

theorem slowZeroSchedule_valid : slowZeroSchedule.Valid := by
  constructor
  · intro e
    simpa [slowZeroSchedule] using slowFraction_bounds e
  · norm_num [slowZeroSchedule]

theorem nearOneSchedule_valid : nearOneSchedule.Valid := by
  constructor
  · intro e
    have h := slowFraction_bounds e
    change
      0 ≤ (1 : Rat) - 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) ∧
        (1 : Rat) - 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) ≤ 1
    exact ⟨by linarith [h.2], by linarith [h.1]⟩
  · norm_num [nearOneSchedule]

theorem harmonicSchedule_valid : harmonicSchedule.Valid := by
  constructor
  · intro e
    simpa [harmonicSchedule] using harmonicFraction_bounds e
  · norm_num [harmonicSchedule]

private theorem slowZero_factor (r : Nat) :
    1 - 2 * slowZeroSchedule.receptivityAt (r + 1) =
      (((r + 1 : Nat) : Rat) * ((r + 3 : Nat) : Rat)) /
        (((r + 2 : Nat) : Rat) ^ 2) := by
  dsimp [slowZeroSchedule]
  have h : (((r + 2 : Nat) : Rat)) ≠ 0 := by positivity
  field_simp [h]
  push_cast
  ring

private theorem nearOne_factor (r : Nat) :
    1 - 2 * nearOneSchedule.receptivityAt (r + 1) =
      -(1 - 2 * slowZeroSchedule.receptivityAt (r + 1)) := by
  dsimp [nearOneSchedule, slowZeroSchedule]
  ring

private theorem harmonic_factor (r : Nat) :
    1 - 2 * harmonicSchedule.receptivityAt (r + 1) =
      ((r + 1 : Nat) : Rat) / ((r + 2 : Nat) : Rat) := by
  dsimp [harmonicSchedule]
  have h : (((r + 2 : Nat) : Rat)) ≠ 0 := by positivity
  field_simp [h]
  push_cast
  ring

theorem slowZero_product (k : Nat) :
    path2MultiplierProduct slowZeroSchedule 0 k =
      (k + 2 : Rat) / (2 * (k + 1 : Rat)) := by
  induction k with
  | zero => norm_num [path2MultiplierProduct]
  | succ k ih =>
      rw [show path2MultiplierProduct slowZeroSchedule 0 (Nat.succ k) =
          path2MultiplierProduct slowZeroSchedule 0 k *
            (1 - 2 * slowZeroSchedule.receptivityAt (k + 1)) by
        simp [path2MultiplierProduct, Finset.prod_range_succ]]
      rw [ih, slowZero_factor]
      push_cast
      have hk1 : ((k : Rat) + 1) ≠ 0 := by positivity
      have hk2 : ((k : Rat) + 2) ≠ 0 := by positivity
      field_simp [hk1, hk2]
      ring

theorem nearOne_product (k : Nat) :
    path2MultiplierProduct nearOneSchedule 0 k =
      (-1 : Rat) ^ k * ((k + 2 : Rat) / (2 * (k + 1 : Rat))) := by
  induction k with
  | zero => norm_num [path2MultiplierProduct]
  | succ k ih =>
      rw [show path2MultiplierProduct nearOneSchedule 0 (Nat.succ k) =
          path2MultiplierProduct nearOneSchedule 0 k *
            (1 - 2 * nearOneSchedule.receptivityAt (k + 1)) by
        simp [path2MultiplierProduct, Finset.prod_range_succ]]
      rw [ih, nearOne_factor, slowZero_factor, pow_succ]
      push_cast
      have hk1 : ((k : Rat) + 1) ≠ 0 := by positivity
      have hk2 : ((k : Rat) + 2) ≠ 0 := by positivity
      field_simp [hk1, hk2]
      ring

theorem harmonic_product (k : Nat) :
    path2MultiplierProduct harmonicSchedule 0 k =
      1 / (k + 1 : Rat) := by
  induction k with
  | zero => norm_num [path2MultiplierProduct]
  | succ k ih =>
      rw [show path2MultiplierProduct harmonicSchedule 0 (Nat.succ k) =
          path2MultiplierProduct harmonicSchedule 0 k *
            (1 - 2 * harmonicSchedule.receptivityAt (k + 1)) by
        simp [path2MultiplierProduct, Finset.prod_range_succ]]
      rw [ih, harmonic_factor]
      push_cast
      have hk1 : ((k : Rat) + 1) ≠ 0 := by positivity
      have hk2 : ((k : Rat) + 2) ≠ 0 := by positivity
      field_simp [hk1, hk2]
      ring

theorem slowZero_product_tendsto_half :
    Tendsto
      (fun k => (path2MultiplierProduct slowZeroSchedule 0 k : Real))
      atTop (nhds (1/2 : Real)) := by
  have hbase : Tendsto (fun k : Nat => (1 : Real) / ((k : Real) + 1))
      atTop (nhds 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  have hscaled : Tendsto
      (fun k : Nat => ((1 : Real) / ((k : Real) + 1)) * (1/2 : Real))
      atTop (nhds 0) := by
    simpa using hbase.mul_const (1/2 : Real)
  have hlim : Tendsto
      (fun k : Nat => (1/2 : Real) +
        ((1 : Real) / ((k : Real) + 1)) * (1/2 : Real))
      atTop (nhds (1/2 : Real)) := by
    have h := Filter.Tendsto.const_add (1/2 : Real) hscaled
    simpa using h
  refine hlim.congr' (Filter.Eventually.of_forall ?_)
  intro k
  change
    (1/2 : Real) + (1 / ((k : Real) + 1)) * (1/2 : Real) =
      (path2MultiplierProduct slowZeroSchedule 0 k : Real)
  rw [slowZero_product]
  push_cast
  have hk1 : (k : Real) + 1 ≠ 0 := by positivity
  field_simp [hk1]
  ring

private theorem harmonic_product_tendsto_zero :
    Tendsto
      (fun k => (path2MultiplierProduct harmonicSchedule 0 k : Real))
      atTop (nhds 0) := by
  have hbase : Tendsto (fun k : Nat => (1 : Real) / ((k : Real) + 1))
      atTop (nhds 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  refine hbase.congr' (Filter.Eventually.of_forall ?_)
  intro k
  change
    (1 : Real) / ((k : Real) + 1) =
      (path2MultiplierProduct harmonicSchedule 0 k : Real)
  rw [harmonic_product]
  push_cast
  rfl

private theorem evenIndex_tendsto :
    Tendsto (fun k : Nat => 2 * k) atTop atTop := by
  refine Filter.tendsto_atTop.2 ?_
  intro b
  exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩

private theorem oddIndex_tendsto :
    Tendsto (fun k : Nat => 2 * k + 1) atTop atTop := by
  refine Filter.tendsto_atTop.2 ?_
  intro b
  exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩

private theorem nearOne_even_product_tendsto_half :
    Tendsto
      (fun k => (path2MultiplierProduct nearOneSchedule 0 (2 * k) : Real))
      atTop (nhds (1/2 : Real)) := by
  have h := slowZero_product_tendsto_half.comp evenIndex_tendsto
  refine h.congr' (Filter.Eventually.of_forall ?_)
  intro k
  change
    (path2MultiplierProduct slowZeroSchedule 0 (2 * k) : Real) =
      (path2MultiplierProduct nearOneSchedule 0 (2 * k) : Real)
  rw [nearOne_product, slowZero_product]
  have hp : (-1 : Rat) ^ (2 * k) = 1 := by
    simp [pow_mul]
  rw [hp]
  norm_num

private theorem nearOne_odd_product_tendsto_neg_half :
    Tendsto
      (fun k => (path2MultiplierProduct nearOneSchedule 0 (2 * k + 1) : Real))
      atTop (nhds (-1/2 : Real)) := by
  have hslow := slowZero_product_tendsto_half.comp oddIndex_tendsto
  have hneg : Tendsto
      (fun k => -(path2MultiplierProduct slowZeroSchedule 0 (2 * k + 1) : Real))
      atTop (nhds (-(1/2 : Real))) := by
    simpa only [Function.comp_apply] using hslow.neg
  have hnear : Tendsto
      (fun k => (path2MultiplierProduct nearOneSchedule 0 (2 * k + 1) : Real))
      atTop (nhds (-(1/2 : Real))) := by
    refine hneg.congr' (Filter.Eventually.of_forall ?_)
    intro k
    change
      -(path2MultiplierProduct slowZeroSchedule 0 (2 * k + 1) : Real) =
        (path2MultiplierProduct nearOneSchedule 0 (2 * k + 1) : Real)
    rw [nearOne_product, slowZero_product]
    have hp : (-1 : Rat) ^ (2 * k + 1) = -1 := by
      rw [pow_succ]
      simp [pow_mul]
    rw [hp]
    norm_num
  simpa only [neg_div] using hnear

private def split2 : State 2 := ![⟨1, 0⟩, ⟨0, 0⟩]

private theorem slowZero_split2_allBroadcast : allBroadcast slowZeroSchedule split2 := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, slowZeroSchedule, split2]

private theorem nearOne_split2_allBroadcast : allBroadcast nearOneSchedule split2 := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, nearOneSchedule, split2]

private theorem harmonic_split2_allBroadcast : allBroadcast harmonicSchedule split2 := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, harmonicSchedule, split2]

private theorem split2_equal_exposure :
    (split2 0).exposure = (split2 1).exposure := by
  norm_num [split2]

private theorem split2_beliefs_ne :
    (split2 0).belief ≠ (split2 1).belief := by
  norm_num [split2]

theorem slowZero_not_consensus :
    ¬ ∀ i : Fin 2,
      Tendsto
        (fun k =>
          (beliefs ((step slowZeroSchedule 2)^[k]
            (![⟨1, 0⟩, ⟨0, 0⟩] : State 2)) i : Real))
        atTop (nhds (1/2 : Real)) := by
  change ¬ ∀ i : Fin 2,
    Tendsto
      (fun k => (beliefs ((step slowZeroSchedule 2)^[k] split2) i : Real))
      atTop (nhds (1/2 : Real))
  intro hcons
  have habs0 :=
    (path2_consensus_iff_product_tendsto_zero
      slowZeroSchedule slowZeroSchedule_valid split2 split2_equal_exposure
      slowZero_split2_allBroadcast split2_beliefs_ne).2 (by
        simpa [split2] using hcons)
  have habshalf : Tendsto
      (fun k => |(path2MultiplierProduct slowZeroSchedule 0 k : Real)|)
      atTop (nhds (1/2 : Real)) := by
    have h := slowZero_product_tendsto_half.abs
    simpa using h
  have huniq := tendsto_nhds_unique habs0 habshalf
  norm_num at huniq

theorem nearOne_not_convergent :
    ¬ ∃ c : Real, ∀ i : Fin 2,
      Tendsto
        (fun k =>
          (beliefs ((step nearOneSchedule 2)^[k]
            (![⟨1, 0⟩, ⟨0, 0⟩] : State 2)) i : Real))
        atTop (nhds c) := by
  change ¬ ∃ c : Real, ∀ i : Fin 2,
    Tendsto
      (fun k => (beliefs ((step nearOneSchedule 2)^[k] split2) i : Real))
      atTop (nhds c)
  rintro ⟨c, hcons⟩
  have hdiff : Tendsto
      (fun k =>
        (beliefs ((step nearOneSchedule 2)^[k] split2) 0 : Real) -
          (beliefs ((step nearOneSchedule 2)^[k] split2) 1 : Real))
      atTop (nhds 0) := by
    have h := (hcons 0).sub (hcons 1)
    simpa using h
  have hproduct0 : Tendsto
      (fun k => (path2MultiplierProduct nearOneSchedule 0 k : Real))
      atTop (nhds 0) := by
    refine hdiff.congr' (Filter.Eventually.of_forall ?_)
    intro k
    change
      (beliefs ((step nearOneSchedule 2)^[k] split2) 0 : Real) -
          (beliefs ((step nearOneSchedule 2)^[k] split2) 1 : Real) =
        (path2MultiplierProduct nearOneSchedule 0 k : Real)
    have h := path2_disagreement_product nearOneSchedule nearOneSchedule_valid
      split2 split2_equal_exposure nearOne_split2_allBroadcast k
    have h' :
        beliefs ((step nearOneSchedule 2)^[k] split2) 0 -
            beliefs ((step nearOneSchedule 2)^[k] split2) 1 =
          path2MultiplierProduct nearOneSchedule 0 k := by
      simpa [split2] using h
    have hReal := congrArg (fun q : Rat => (q : Real)) h'
    simpa using hReal
  have heven0 := hproduct0.comp evenIndex_tendsto
  have huniq := tendsto_nhds_unique heven0 nearOne_even_product_tendsto_half
  norm_num at huniq

theorem harmonic_consensus :
    ∀ i : Fin 2,
      Tendsto
        (fun k =>
          (beliefs ((step harmonicSchedule 2)^[k]
            (![⟨1, 0⟩, ⟨0, 0⟩] : State 2)) i : Real))
        atTop (nhds (1/2 : Real)) := by
  change ∀ i : Fin 2,
    Tendsto
      (fun k => (beliefs ((step harmonicSchedule 2)^[k] split2) i : Real))
      atTop (nhds (1/2 : Real))
  have habs0 : Tendsto
      (fun k => |(path2MultiplierProduct harmonicSchedule 0 k : Real)|)
      atTop (nhds 0) := by
    have h := harmonic_product_tendsto_zero.abs
    simpa using h
  have hcons :=
    (path2_consensus_iff_product_tendsto_zero
      harmonicSchedule harmonicSchedule_valid split2 split2_equal_exposure
      harmonic_split2_allBroadcast split2_beliefs_ne).1 habs0
  simpa [split2] using hcons

def degreeSplitSchedule : ExposureParameters :=
  ⟨fun e => if e = 1 then 1/4 else if e = 2 then 1/2 else 1/3, 0⟩

def degreeSplitState : State 3 :=
  ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]

theorem degreeSplit_step_beliefs :
    beliefs (step degreeSplitSchedule 3 degreeSplitState) =
      ![3/4, 1/4, 0] := by
  funext i
  fin_cases i <;> decide_cbv

theorem degreeSplit_mean_before :
    FitnessABMPathN.mean 3 (beliefs degreeSplitState) = 1/4 := by
  decide_cbv

theorem degreeSplit_mean_after :
    FitnessABMPathN.mean 3
      (beliefs (step degreeSplitSchedule 3 degreeSplitState)) = 5/16 := by
  rw [degreeSplit_step_beliefs]
  decide_cbv

theorem exposure_degree_weighted_mean_not_invariant :
    FitnessABMPathN.mean 3
        (beliefs (step degreeSplitSchedule 3 degreeSplitState)) ≠
      FitnessABMPathN.mean 3 (beliefs degreeSplitState) := by
  rw [degreeSplit_mean_after, degreeSplit_mean_before]
  norm_num

end NarrativeDynamics.FitnessABMPathNExposureConvergence
