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

end NarrativeDynamics.FitnessABMPathN
