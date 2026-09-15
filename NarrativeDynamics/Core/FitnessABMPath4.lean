import NarrativeDynamics.Core.NetworkPropagation

/-!
# Belief dynamics on the four-node path

The linear formula below is proved from the actual incoming broadcast sets.
Exposure counters remain part of the population, but do not affect beliefs.
-/

namespace NarrativeDynamics.FitnessABMPath4

open NarrativeDynamics.NetworkPropagation

abbrev Beliefs := Fin 4 → Rat

def pathAdj (i j : Fin 4) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val

instance : DecidableRel pathAdj := fun _ _ => inferInstance

def population (x : Beliefs) (e : Fin 4 → Nat) : Population 4 :=
  ⟨fun _ => ⟨1/2, 1/2⟩, fun i => ⟨x i, e i⟩⟩

def project (p : Population 4) : Beliefs :=
  fun i => (p.agents i).belief

def allBroadcast (x : Beliefs) : Prop := ∀ i, 1/2 ≤ x i ∧ x i ≤ 1

def linearStep (x : Beliefs) : Beliefs :=
  ![(x 0+x 1)/2, x 0/4+x 1/2+x 2/4,
    x 1/4+x 2/2+x 3/4, (x 2+x 3)/2]

def beliefStep (x : Beliefs) : Beliefs :=
  project (propagate pathAdj (population x (fun _ => 0)))

def trajectory (x : Beliefs) (n : Nat) : Beliefs := beliefStep^[n] x

/-- Broadcasting and received beliefs ignore all exposure counters, even
outside the all-broadcast region. -/
theorem propagate_independent_exposures (x : Beliefs) (e : Fin 4 → Nat) :
    project (propagate pathAdj (population x e)) = beliefStep x := by
  funext i
  dsimp [project, propagate, beliefStep, nextAgent, incoming, broadcasting, population]
  split <;> rfl

/-- Enumerate the actual incoming sets using path adjacency and the inclusive
broadcast threshold. -/
private theorem incoming_eq (x : Beliefs) (e : Fin 4 → Nat)
    (hx : allBroadcast x) (i : Fin 4) :
    incoming pathAdj (population x e) i =
      ![({1} : Finset (Fin 4)), {0, 2}, {1, 3}, {2}] i := by
  ext j
  have hb : broadcasting ((population x e).profiles j)
      ((population x e).agents j) = true := by
    simp [broadcasting, population, (hx j).1]
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hb, and_true]
  fin_cases i <;> fin_cases j <;> decide

/-- The rational averaging matrix is the belief projection of propagation for
arbitrary exposures whenever every coordinate broadcasts. -/
theorem propagate_eq_linear (x : Beliefs) (e : Fin 4 → Nat)
    (hx : allBroadcast x) :
    project (propagate pathAdj (population x e)) = linearStep x := by
  funext i
  change (nextAgent pathAdj (population x e) i).belief = linearStep x i
  simp only [nextAgent, incoming_eq x e hx i]
  fin_cases i <;> norm_num [linearStep, population] <;> ring

theorem allBroadcast_step (x : Beliefs) (hx : allBroadcast x) :
    allBroadcast (beliefStep x) := by
  rw [show beliefStep x = linearStep x from propagate_eq_linear x _ hx]
  intro i
  have h0 := hx 0
  have h1 := hx 1
  have h2 := hx 2
  have h3 := hx 3
  fin_cases i <;> dsimp [linearStep] <;> constructor <;> linarith

theorem allBroadcast_iterate (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    allBroadcast (trajectory x n) := by
  induction n with
  | zero => simpa [trajectory] using hx
  | succ n ih =>
      simpa [trajectory, Function.iterate_succ_apply'] using
        allBroadcast_step (trajectory x n) ih

end NarrativeDynamics.FitnessABMPath4
