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

end NarrativeDynamics.FitnessABMPathN
