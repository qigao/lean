import NarrativeDynamics.Core.FiniteConsensus
import NarrativeDynamics.Core.NetworkPropagation

/-!
# Generic finite-path BB belief dynamics

The executable belief step remains `NetworkPropagation.propagate`. The rational
kernel in this file is proof-only structure derived from the same finite path.
-/

namespace NarrativeDynamics.FitnessABMPathN

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FiniteConsensus
open scoped BigOperators

abbrev Beliefs (n : Nat) := Fin n → Rat

def pathAdj (n : Nat) (i j : Fin n) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val

instance (n : Nat) : DecidableRel (pathAdj n) := fun i j =>
  show Decidable (i.val + 1 = j.val ∨ j.val + 1 = i.val) from inferInstance

def neighbors (n : Nat) (i : Fin n) : Finset (Fin n) :=
  Finset.univ.filter fun j => pathAdj n j i

def degree (n : Nat) (i : Fin n) : Nat :=
  (neighbors n i).card

def population (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : Population n :=
  ⟨fun _ => ⟨1/2, 1/2⟩, fun i => ⟨x i, e i⟩⟩

def project (n : Nat) (p : Population n) : Beliefs n :=
  fun i => (p.agents i).belief

def allBroadcast (n : Nat) (x : Beliefs n) : Prop :=
  ∀ i, 1/2 ≤ x i ∧ x i ≤ 1

def beliefStep (n : Nat) (x : Beliefs n) : Beliefs n :=
  project n (propagate (pathAdj n) (population n x (fun _ => 0)))

def trajectory (n : Nat) (x : Beliefs n) (k : Nat) : Beliefs n :=
  (beliefStep n)^[k] x

@[simp] theorem mem_neighbors_iff {n : Nat} (i j : Fin n) :
    j ∈ neighbors n i ↔ pathAdj n j i := by
  simp [neighbors]

@[simp] theorem pathAdj_self {n : Nat} (i : Fin n) : ¬ pathAdj n i i := by
  unfold pathAdj
  omega

theorem neighbors_nonempty (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    (neighbors n i).Nonempty := by
  classical
  by_cases hzero : i.val = 0
  · let j : Fin n := ⟨1, by omega⟩
    refine ⟨j, ?_⟩
    simp [neighbors, pathAdj, j, hzero]
  · let j : Fin n := ⟨i.val - 1, by omega⟩
    refine ⟨j, ?_⟩
    simp only [mem_neighbors_iff]
    left
    change (i.val - 1) + 1 = i.val
    omega

theorem degree_pos (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    0 < degree n i := by
  exact Finset.card_pos.mpr (neighbors_nonempty n hn i)

theorem degree_le_two (n : Nat) (i : Fin n) : degree n i ≤ 2 := by
  classical
  let e : Fin n ↪ Nat := Fin.valEmbedding
  have hsubset : (neighbors n i).map e ⊆
      ({i.val - 1, i.val + 1} : Finset Nat) := by
    intro k hk
    rcases Finset.mem_map.mp hk with ⟨j, hj, hjk⟩
    have hkval : j.val = k := by
      simpa [e] using hjk
    subst k
    have hadj : pathAdj n j i := (mem_neighbors_iff i j).mp hj
    simp only [Finset.mem_insert, Finset.mem_singleton]
    rcases hadj with hleft | hright
    · left
      omega
    · right
      omega
  calc
    degree n i = ((neighbors n i).map e).card := by simp [degree]
    _ ≤ ({i.val - 1, i.val + 1} : Finset Nat).card :=
      Finset.card_le_card hsubset
    _ ≤ 2 := Finset.card_le_two

private theorem degree_eq_one_or_two (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    degree n i = 1 ∨ degree n i = 2 := by
  have hpos := degree_pos n hn i
  have hle := degree_le_two n i
  omega

/-- Proof-only lazy averaging kernel for the finite path. -/
def pathKernel (n : Nat) : Kernel (Fin n) := fun i j =>
  (if i = j then 1/2 else 0) +
    (if j ∈ neighbors n i then 1 / (2 * (degree n i : Rat)) else 0)

theorem pathKernel_self (n : Nat) (i : Fin n) :
    pathKernel n i i = 1/2 := by
  simp [pathKernel]

theorem pathKernel_self_lower (n : Nat) (i : Fin n) :
    (1/4 : Rat) ≤ pathKernel n i i := by
  rw [pathKernel_self]
  norm_num

theorem pathKernel_nonneg (n : Nat) (hn : 2 ≤ n) (i j : Fin n) :
    0 ≤ pathKernel n i j := by
  classical
  have hd := degree_eq_one_or_two n hn i
  rcases hd with hd | hd <;>
    by_cases hij : i = j <;>
    by_cases hmem : j ∈ neighbors n i <;>
    simp [pathKernel, hij, hmem, hd] <;> norm_num

theorem pathKernel_adj_lower (n : Nat) (hn : 2 ≤ n) (i j : Fin n)
    (h : pathAdj n j i) :
    (1/4 : Rat) ≤ pathKernel n i j := by
  classical
  have hmem : j ∈ neighbors n i := (mem_neighbors_iff i j).mpr h
  have hne : i ≠ j := by
    intro hij
    subst j
    exact (pathAdj_self i) h
  have hd := degree_eq_one_or_two n hn i
  rcases hd with hd | hd <;>
    simp [pathKernel, hne, hmem, hd] <;> norm_num

private theorem self_mass_sum (n : Nat) (i : Fin n) :
    (∑ j : Fin n, if i = j then (1/2 : Rat) else 0) = 1/2 := by
  classical
  simp

private theorem neighbor_mass_sum (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    (∑ j : Fin n,
      if j ∈ neighbors n i then 1 / (2 * (degree n i : Rat)) else 0) = 1/2 := by
  classical
  let c : Rat := 1 / (2 * (degree n i : Rat))
  have hsum :
      (∑ j : Fin n, if j ∈ neighbors n i then c else 0) =
        ∑ j ∈ neighbors n i, c := by
    simpa using
      (Finset.sum_ite_mem (Finset.univ : Finset (Fin n)) (neighbors n i)
        (fun _ => c))
  rw [hsum]
  have hd := degree_eq_one_or_two n hn i
  rcases hd with hd | hd
  · have hcard : (neighbors n i).card = 1 := by
      simpa [degree] using hd
    simp [hcard, c, hd]
  · have hcard : (neighbors n i).card = 2 := by
      simpa [degree] using hd
    simp [hcard, c, hd]

theorem pathKernel_row_sum (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    ∑ j, pathKernel n i j = 1 := by
  classical
  rw [show (∑ j : Fin n, pathKernel n i j) =
      (∑ j : Fin n, if i = j then (1/2 : Rat) else 0) +
      (∑ j : Fin n,
        if j ∈ neighbors n i then 1 / (2 * (degree n i : Rat)) else 0) by
    simp only [pathKernel, Finset.sum_add_distrib]]
  rw [self_mass_sum n i, neighbor_mass_sum n hn i]
  norm_num

theorem pathKernel_averaging (n : Nat) (hn : 2 ≤ n) :
    AveragingKernel (pathKernel n) := by
  exact ⟨pathKernel_nonneg n hn, pathKernel_row_sum n hn⟩

/-- Broadcasting decisions, received beliefs, and therefore updated beliefs do
not depend on the exposure counters. -/
theorem propagate_independent_exposures
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
    project n (propagate (pathAdj n) (population n x e)) = beliefStep n x := by
  change project n (propagate (pathAdj n) (population n x e)) =
    project n (propagate (pathAdj n) (population n x (fun _ => 0)))
  funext i
  have hin : incoming (pathAdj n) (population n x e) i =
      incoming (pathAdj n) (population n x (fun _ => 0)) i := by
    rfl
  change (nextAgent (pathAdj n) (population n x e) i).belief =
    (nextAgent (pathAdj n) (population n x (fun _ => 0)) i).belief
  simp only [nextAgent, hin]
  split_ifs <;> rfl

/-- In the all-broadcast region the actual incoming set is exactly the path
neighbor set. -/
theorem incoming_eq_neighbors
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat)
    (hx : allBroadcast n x) (i : Fin n) :
    incoming (pathAdj n) (population n x e) i = neighbors n i := by
  ext j
  have hb : broadcasting ((population n x e).profiles j)
      ((population n x e).agents j) = true := by
    change decide ((1 : Rat) / 2 ≤ x j) = true
    exact decide_eq_true (hx j).1
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hb, and_true,
    mem_neighbors_iff]

private theorem applyKernel_pathKernel
    (n : Nat) (x : Beliefs n) (i : Fin n) :
    applyKernel (pathKernel n) x i =
      (1/2 : Rat) * x i +
        (1 / (2 * (degree n i : Rat))) *
          (∑ j ∈ neighbors n i, x j) := by
  classical
  simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
  change
    (∑ j : Fin n, pathKernel n i j * x j) =
      (1/2 : Rat) * x i +
        (1 / (2 * (degree n i : Rat))) *
          (∑ j ∈ neighbors n i, x j)
  simp only [pathKernel]
  rw [show
      (∑ j : Fin n,
        ((if i = j then (1/2 : Rat) else 0) +
          if j ∈ neighbors n i then 1 / (2 * (degree n i : Rat)) else 0) * x j) =
        (∑ j : Fin n, (if i = j then (1/2 : Rat) else 0) * x j) +
        (∑ j : Fin n,
          (if j ∈ neighbors n i then 1 / (2 * (degree n i : Rat)) else 0) * x j) by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring]
  have hself :
      (∑ j : Fin n, (if i = j then (1/2 : Rat) else 0) * x j) =
        (1/2 : Rat) * x i := by
    simp
  rw [hself]
  let c : Rat := 1 / (2 * (degree n i : Rat))
  have hneighbor :
      (∑ j : Fin n, (if j ∈ neighbors n i then c else 0) * x j) =
        c * (∑ j ∈ neighbors n i, x j) := by
    calc
      (∑ j : Fin n, (if j ∈ neighbors n i then c else 0) * x j) =
          ∑ j ∈ neighbors n i, c * x j := by
        simpa using
          (Finset.sum_ite_mem (Finset.univ : Finset (Fin n)) (neighbors n i)
            (fun j => c * x j))
      _ = c * (∑ j ∈ neighbors n i, x j) := by
        rw [Finset.mul_sum]
  change
    (1/2 : Rat) * x i +
      (∑ j : Fin n, (if j ∈ neighbors n i then c else 0) * x j) =
        (1/2 : Rat) * x i + c * (∑ j ∈ neighbors n i, x j)
  rw [hneighbor]

/-- The proof-only kernel is derived from the actual `nextAgent` update in the
all-broadcast region. -/
theorem propagate_eq_kernel
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (e : Fin n → Nat)
    (hx : allBroadcast n x) :
    project n (propagate (pathAdj n) (population n x e)) =
      applyKernel (pathKernel n) x := by
  funext i
  have hd : degree n i ≠ 0 := Nat.ne_of_gt (degree_pos n hn i)
  have hcard : (neighbors n i).card ≠ 0 := by
    simpa [degree] using hd
  have hdq : (degree n i : Rat) ≠ 0 := by
    exact_mod_cast hd
  change (nextAgent (pathAdj n) (population n x e) i).belief =
    applyKernel (pathKernel n) x i
  rw [applyKernel_pathKernel n x i]
  simp only [nextAgent, incoming_eq_neighbors n x e hx i, hcard, if_false]
  change
    (1 - (1/2 : Rat)) * x i +
      (1/2 : Rat) * ((∑ j ∈ neighbors n i, x j) / (degree n i : Rat)) =
        (1/2 : Rat) * x i +
          (1 / (2 * (degree n i : Rat))) * (∑ j ∈ neighbors n i, x j)
  field_simp [hdq]
  ring

private theorem kernel_preserves_allBroadcast
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) :
    allBroadcast n (applyKernel (pathKernel n) x) := by
  let i0 : Fin n := ⟨0, by omega⟩
  letI : Nonempty (Fin n) := ⟨i0⟩
  intro i
  have hbetween := applyKernel_between (pathKernel_averaging n hn) x i
  have hmin : (1/2 : Rat) ≤ coordMin x := by
    unfold coordMin
    apply Finset.le_inf'
    intro j _
    exact (hx j).1
  have hmax : coordMax x ≤ (1 : Rat) := by
    unfold coordMax
    apply Finset.sup'_le
    intro j _
    exact (hx j).2
  exact ⟨hmin.trans hbetween.1, hbetween.2.trans hmax⟩

theorem allBroadcast_step
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) :
    allBroadcast n (beliefStep n x) := by
  have hbridge : beliefStep n x = applyKernel (pathKernel n) x := by
    simpa [beliefStep] using
      propagate_eq_kernel n hn x (fun _ => 0) hx
  rw [hbridge]
  exact kernel_preserves_allBroadcast n hn x hx

theorem allBroadcast_iterate
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) (k : Nat) :
    allBroadcast n (trajectory n x k) := by
  induction k with
  | zero => simpa [trajectory] using hx
  | succ k ih =>
      simpa [trajectory, Function.iterate_succ_apply'] using
        allBroadcast_step n hn (trajectory n x k) ih

theorem trajectory_eq_kernelTrajectory
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) (k : Nat) :
    trajectory n x k = kernelTrajectory (pathKernel n) x k := by
  induction k with
  | zero => simp [trajectory, kernelTrajectory]
  | succ k ih =>
      have hk := allBroadcast_iterate n hn x hx k
      have hbridge : beliefStep n (trajectory n x k) =
          applyKernel (pathKernel n) (trajectory n x k) := by
        simpa [beliefStep] using
          propagate_eq_kernel n hn (trajectory n x k) (fun _ => 0) hk
      rw [show trajectory n x (Nat.succ k) = beliefStep n (trajectory n x k) by
        simp [trajectory, Function.iterate_succ_apply']]
      rw [hbridge, ih]
      simpa [Nat.succ_eq_add_one] using
        (kernelTrajectory_succ (pathKernel n) x k).symm

/-- Sum of path degrees, used to normalize the stationary distribution. -/
def weightSum (n : Nat) : Rat :=
  ∑ i : Fin n, (degree n i : Rat)

def stationaryWeight (n : Nat) (i : Fin n) : Rat :=
  (degree n i : Rat) / weightSum n

def mean (n : Nat) (x : Beliefs n) : Rat :=
  weightedMean (stationaryWeight n) x

private theorem degree_formula (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    degree n i =
      (if i.val = 0 then 0 else 1) +
      (if i.val + 1 < n then 1 else 0) := by
  classical
  by_cases hzero : i.val = 0
  · have hright : i.val + 1 < n := by omega
    have hn1 : 1 < n := by omega
    let r : Fin n := ⟨i.val + 1, hright⟩
    have hneighbors : neighbors n i = {r} := by
      ext j
      simp only [mem_neighbors_iff, Finset.mem_singleton]
      constructor
      · intro h
        apply Fin.ext
        rcases h with h | h
        · dsimp [r]
          omega
        · dsimp [r]
          omega
      · intro h
        subst j
        right
        dsimp [r]
    simp [degree, hneighbors, hzero, hright, hn1]
  · by_cases hright : i.val + 1 < n
    · let l : Fin n := ⟨i.val - 1, by omega⟩
      let r : Fin n := ⟨i.val + 1, hright⟩
      have hl : l ∈ neighbors n i := by
        simp only [mem_neighbors_iff]
        left
        dsimp [l]
        omega
      have hr : r ∈ neighbors n i := by
        simp only [mem_neighbors_iff]
        right
        dsimp [r]
      have hlr : l ≠ r := by
        intro h
        have hv := congrArg Fin.val h
        dsimp [l, r] at hv
        omega
      have hsubset : ({l, r} : Finset (Fin n)) ⊆ neighbors n i := by
        intro j hj
        simp only [Finset.mem_insert, Finset.mem_singleton] at hj
        rcases hj with rfl | rfl
        · exact hl
        · exact hr
      have htwo : 2 ≤ degree n i := by
        have hc := Finset.card_le_card hsubset
        simpa [degree, hlr] using hc
      have hle := degree_le_two n i
      have hd : degree n i = 2 := by omega
      simp [hd, hzero, hright]
    · have hlast : i.val + 1 = n := by omega
      let l : Fin n := ⟨i.val - 1, by omega⟩
      have hneighbors : neighbors n i = {l} := by
        ext j
        simp only [mem_neighbors_iff, Finset.mem_singleton]
        constructor
        · intro h
          apply Fin.ext
          rcases h with h | h
          · dsimp [l]
            omega
          · exfalso
            omega
        · intro h
          subst j
          left
          dsimp [l]
          omega
      simp [degree, hneighbors, hzero, hright]

private theorem sum_has_pred (n : Nat) :
    (∑ i : Fin n, if i.val = 0 then (0 : Nat) else 1) = n - 1 := by
  cases n with
  | zero => simp
  | succ n =>
      rw [Fin.sum_univ_succ]
      simp

private theorem sum_has_succ (n : Nat) :
    (∑ i : Fin n, if i.val + 1 < n then (1 : Nat) else 0) = n - 1 := by
  cases n with
  | zero => simp
  | succ n =>
      rw [Fin.sum_univ_castSucc]
      simp only [Fin.val_last, Fin.val_castSucc]
      have hcast : ∀ i : Fin n, i.val + 1 < n + 1 := by
        intro i
        omega
      simp [hcast]

theorem path_degree_sum (n : Nat) (hn : 2 ≤ n) :
    weightSum n = ((2 * (n - 1) : Nat) : Rat) := by
  have hnat : (∑ i : Fin n, degree n i) = 2 * (n - 1) := by
    calc
      (∑ i : Fin n, degree n i) =
          ∑ i : Fin n,
            ((if i.val = 0 then 0 else 1) +
              (if i.val + 1 < n then 1 else 0)) := by
        apply Finset.sum_congr rfl
        intro i _
        exact degree_formula n hn i
      _ = (∑ i : Fin n, if i.val = 0 then (0 : Nat) else 1) +
          (∑ i : Fin n, if i.val + 1 < n then (1 : Nat) else 0) := by
        rw [Finset.sum_add_distrib]
      _ = (n - 1) + (n - 1) := by
        rw [sum_has_pred, sum_has_succ]
      _ = 2 * (n - 1) := by omega
  unfold weightSum
  exact_mod_cast hnat

theorem weightSum_pos (n : Nat) (hn : 2 ≤ n) : 0 < weightSum n := by
  rw [path_degree_sum n hn]
  exact_mod_cast (show 0 < 2 * (n - 1) by omega)

theorem stationaryWeight_nonneg (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    0 ≤ stationaryWeight n i := by
  exact div_nonneg (by positivity) (le_of_lt (weightSum_pos n hn))

theorem stationaryWeight_sum_one (n : Nat) (hn : 2 ≤ n) :
    ∑ i : Fin n, stationaryWeight n i = 1 := by
  unfold stationaryWeight
  rw [← Finset.sum_div]
  change weightSum n / weightSum n = 1
  exact div_self (ne_of_gt (weightSum_pos n hn))

private theorem pathAdj_symm {n : Nat} (i j : Fin n) :
    pathAdj n i j ↔ pathAdj n j i := by
  unfold pathAdj
  omega

theorem pathKernel_detailed_balance
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n) :
    stationaryWeight n i * pathKernel n i j =
      stationaryWeight n j * pathKernel n j i := by
  classical
  by_cases hij : i = j
  · subst j
    rfl
  · by_cases hadj : pathAdj n j i
    · have hji : pathAdj n i j := (pathAdj_symm i j).2 hadj
      have hwi : weightSum n ≠ 0 := ne_of_gt (weightSum_pos n hn)
      have hdiNat : degree n i ≠ 0 := Nat.ne_of_gt (degree_pos n hn i)
      have hdjNat : degree n j ≠ 0 := Nat.ne_of_gt (degree_pos n hn j)
      have hdi : (degree n i : Rat) ≠ 0 := by exact_mod_cast hdiNat
      have hdj : (degree n j : Rat) ≠ 0 := by exact_mod_cast hdjNat
      rw [show pathKernel n i j = 1 / (2 * (degree n i : Rat)) by
        simp [pathKernel, hij, mem_neighbors_iff, hadj]]
      rw [show pathKernel n j i = 1 / (2 * (degree n j : Rat)) by
        simp [pathKernel, Ne.symm hij, mem_neighbors_iff, hji]]
      unfold stationaryWeight
      field_simp [hwi, hdi, hdj]
    · have hji : ¬ pathAdj n i j := by
        intro h
        exact hadj ((pathAdj_symm i j).1 h)
      simp [pathKernel, hij, Ne.symm hij, mem_neighbors_iff, hadj, hji]

theorem path_stationary_weights (n : Nat) (hn : 2 ≤ n) :
    StationaryWeights (pathKernel n) (stationaryWeight n) := by
  refine ⟨stationaryWeight_nonneg n hn, stationaryWeight_sum_one n hn, ?_⟩
  intro j
  calc
    (∑ i, stationaryWeight n i * pathKernel n i j) =
        ∑ i, stationaryWeight n j * pathKernel n j i := by
      apply Finset.sum_congr rfl
      intro i _
      exact pathKernel_detailed_balance n hn i j
    _ = stationaryWeight n j * (∑ i, pathKernel n j i) := by
      rw [Finset.mul_sum]
    _ = stationaryWeight n j := by
      rw [pathKernel_row_sum n hn j]
      ring

theorem mean_kernel_step (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) :
    mean n (applyKernel (pathKernel n) x) = mean n x := by
  exact weightedMean_apply (path_stationary_weights n hn) x

theorem mean_kernel_iterate
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (k : Nat) :
    mean n (kernelTrajectory (pathKernel n) x k) = mean n x := by
  induction k with
  | zero => simp [mean]
  | succ k ih =>
      rw [show kernelTrajectory (pathKernel n) x (Nat.succ k) =
          applyKernel (pathKernel n) (kernelTrajectory (pathKernel n) x k) by
        simpa [Nat.succ_eq_add_one] using
          kernelTrajectory_succ (pathKernel n) x k]
      rw [mean_kernel_step n hn, ih]

theorem mean_step
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (hx : allBroadcast n x) :
    mean n (beliefStep n x) = mean n x := by
  have hbridge : beliefStep n x = applyKernel (pathKernel n) x := by
    simpa [beliefStep] using
      propagate_eq_kernel n hn x (fun _ => 0) hx
  rw [hbridge]
  exact mean_kernel_step n hn x

theorem mean_iterate
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) (k : Nat) :
    mean n (trajectory n x k) = mean n x := by
  rw [trajectory_eq_kernelTrajectory n hn x hx k]
  exact mean_kernel_iterate n hn x k

theorem mean_between
    (n : Nat) [Nonempty (Fin n)] (hn : 2 ≤ n) (x : Beliefs n) :
    coordMin x ≤ mean n x ∧ mean n x ≤ coordMax x := by
  exact weightedMean_between (path_stationary_weights n hn) x

def block (n : Nat) : Nat := n - 1

def delta (n : Nat) : Rat := (1/4 : Rat) ^ block n

theorem block_pos (n : Nat) (hn : 2 ≤ n) : 0 < block n := by
  simp [block]
  omega

theorem delta_pos (n : Nat) (hn : 2 ≤ n) : 0 < delta n := by
  unfold delta
  exact pow_pos (by norm_num) _

theorem delta_lt_one (n : Nat) (hn : 2 ≤ n) : delta n < 1 := by
  unfold delta
  exact pow_lt_one₀ (by norm_num) (by norm_num) (Nat.ne_of_gt (block_pos n hn))

theorem one_sub_delta_pos (n : Nat) (hn : 2 ≤ n) : 0 < 1 - delta n := by
  linarith [delta_lt_one n hn]

theorem one_sub_delta_lt_one (n : Nat) (hn : 2 ≤ n) : 1 - delta n < 1 := by
  linarith [delta_pos n hn]

private def originBasis {n : Nat} (z : Fin n) : Beliefs n :=
  fun j => if j = z then 1 else 0

private theorem kernelTrajectory_origin_nonneg
    (n : Nat) (hn : 2 ≤ n) (z : Fin n) (k : Nat) (i : Fin n) :
    0 ≤ kernelTrajectory (pathKernel n) (originBasis z) k i := by
  induction k generalizing i with
  | zero =>
      by_cases h : i = z <;> simp [kernelTrajectory, originBasis, h]
  | succ k ih =>
      rw [kernelTrajectory_succ]
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      exact Finset.sum_nonneg fun j _ =>
        mul_nonneg (pathKernel_nonneg n hn i j) (ih j)

private theorem applyKernel_self_mass_lower
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : ∀ j, 0 ≤ x j) (i : Fin n) :
    (1/4 : Rat) * x i ≤ applyKernel (pathKernel n) x i := by
  simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
  calc
    (1/4 : Rat) * x i ≤ pathKernel n i i * x i :=
      mul_le_mul_of_nonneg_right (pathKernel_self_lower n i) (hx i)
    _ ≤ ∑ j : Fin n, pathKernel n i j * x j :=
      Finset.single_le_sum
        (fun j _ => mul_nonneg (pathKernel_nonneg n hn i j) (hx j))
        (Finset.mem_univ i)

private theorem left_reach_mass
    (n : Nat) (hn : 2 ≤ n) (z : Fin n) (hz : z.val = 0)
    (k : Nat) (hk : k < n) :
    (1/4 : Rat) ^ k ≤
      kernelTrajectory (pathKernel n) (originBasis z) k ⟨k, hk⟩ := by
  induction k with
  | zero =>
      have hzero : (⟨0, hk⟩ : Fin n) = z := by
        apply Fin.ext
        simpa using hz.symm
      simp [kernelTrajectory, originBasis, hzero]
  | succ k ih =>
      let p : Fin n := ⟨k, by omega⟩
      let i : Fin n := ⟨k + 1, by omega⟩
      have hreach :
          (1/4 : Rat) ^ k ≤
            kernelTrajectory (pathKernel n) (originBasis z) k p := by
        exact ih (by omega)
      have hmass_nonneg :
          0 ≤ kernelTrajectory (pathKernel n) (originBasis z) k p :=
        kernelTrajectory_origin_nonneg n hn z k p
      have hadj : pathAdj n p i := by
        left
        rfl
      have hkernel : (1/4 : Rat) ≤ pathKernel n i p :=
        pathKernel_adj_lower n hn i p hadj
      change
        (1/4 : Rat) ^ (k + 1) ≤
          kernelTrajectory (pathKernel n) (originBasis z) (k + 1) i
      rw [kernelTrajectory_succ]
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      calc
        (1/4 : Rat) ^ (k + 1) = (1/4 : Rat) ^ k * (1/4 : Rat) := by
          rw [pow_succ]
        _ ≤ kernelTrajectory (pathKernel n) (originBasis z) k p * (1/4 : Rat) :=
          mul_le_mul_of_nonneg_right hreach (by norm_num)
        _ ≤ kernelTrajectory (pathKernel n) (originBasis z) k p * pathKernel n i p :=
          mul_le_mul_of_nonneg_left hkernel hmass_nonneg
        _ = pathKernel n i p * kernelTrajectory (pathKernel n) (originBasis z) k p := by
          ring
        _ ≤ ∑ j : Fin n,
            pathKernel n i j * kernelTrajectory (pathKernel n) (originBasis z) k j :=
          Finset.single_le_sum
            (fun j _ => mul_nonneg (pathKernel_nonneg n hn i j)
              (kernelTrajectory_origin_nonneg n hn z k j))
            (Finset.mem_univ p)

private theorem self_pad_mass
    (n : Nat) (hn : 2 ≤ n) (z i : Fin n)
    (s t : Nat)
    (hstart : (1/4 : Rat) ^ s ≤
      kernelTrajectory (pathKernel n) (originBasis z) s i) :
    (1/4 : Rat) ^ (s + t) ≤
      kernelTrajectory (pathKernel n) (originBasis z) (s + t) i := by
  induction t with
  | zero => simpa using hstart
  | succ t ih =>
      rw [Nat.add_succ, kernelTrajectory_succ]
      calc
        (1/4 : Rat) ^ (s + t + 1) =
            (1/4 : Rat) ^ (s + t) * (1/4 : Rat) := by
          rw [pow_succ]
        _ ≤ kernelTrajectory (pathKernel n) (originBasis z) (s + t) i *
            (1/4 : Rat) :=
          mul_le_mul_of_nonneg_right ih (by norm_num)
        _ = (1/4 : Rat) *
            kernelTrajectory (pathKernel n) (originBasis z) (s + t) i := by
          ring
        _ ≤ applyKernel (pathKernel n)
            (kernelTrajectory (pathKernel n) (originBasis z) (s + t)) i :=
          applyKernel_self_mass_lower n hn _
            (fun j => kernelTrajectory_origin_nonneg n hn z (s + t) j) i

private theorem applyKernel_originBasis
    {n : Nat} (M : Kernel (Fin n)) (z i : Fin n) :
    applyKernel M (originBasis z) i = M i z := by
  classical
  simp [applyKernel, Matrix.mulVec_apply, dotProduct, originBasis]

theorem path_block_common_mass
    (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass ((pathKernel n) ^ block n) (delta n) := by
  let z : Fin n := ⟨0, by omega⟩
  refine ⟨z, ?_⟩
  intro i
  have hi : i.val ≤ block n := by
    simp [block]
    omega
  have hreach :
      (1/4 : Rat) ^ i.val ≤
        kernelTrajectory (pathKernel n) (originBasis z) i.val i := by
    have hz : z.val = 0 := by rfl
    simpa using left_reach_mass n hn z hz i.val i.isLt
  have hpad :=
    self_pad_mass n hn z i i.val (block n - i.val) hreach
  have htime : i.val + (block n - i.val) = block n := Nat.add_sub_of_le hi
  have hmass :
      delta n ≤ kernelTrajectory (pathKernel n) (originBasis z) (block n) i := by
    simpa [delta, htime] using hpad
  have hp := congrFun (kernelPow_apply (pathKernel n) (originBasis z) (block n)) i
  calc
    delta n ≤ kernelTrajectory (pathKernel n) (originBasis z) (block n) i := hmass
    _ = applyKernel ((pathKernel n) ^ block n) (originBasis z) i := hp.symm
    _ = ((pathKernel n) ^ block n) i z :=
      applyKernel_originBasis ((pathKernel n) ^ block n) z i

end NarrativeDynamics.FitnessABMPathN