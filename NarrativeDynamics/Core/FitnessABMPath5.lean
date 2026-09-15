import NarrativeDynamics.Core.NetworkPropagation

/-!
# Belief dynamics on the five-node path

This module is the concrete Path5 vertical slice for issue #83. The update is
always defined through `NetworkPropagation.propagate`; the incoming-set theorem
below only exposes what that operator receives in the all-broadcast region.
-/

namespace NarrativeDynamics.FitnessABMPath5

open NarrativeDynamics.NetworkPropagation

abbrev Beliefs := Fin 5 → Rat

def path5Adj (i j : Fin 5) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val

instance : DecidableRel path5Adj := fun i j =>
  show Decidable (i.val + 1 = j.val ∨ j.val + 1 = i.val) from inferInstance

def population (x : Beliefs) (e : Fin 5 → Nat) : Population 5 :=
  ⟨fun _ => ⟨1/2, 1/2⟩, fun i => ⟨x i, e i⟩⟩

def project (p : Population 5) : Beliefs :=
  fun i => (p.agents i).belief

def allBroadcast (x : Beliefs) : Prop := ∀ i, 1/2 ≤ x i ∧ x i ≤ 1

def beliefStep (x : Beliefs) : Beliefs :=
  project (propagate path5Adj (population x (fun _ => 0)))

def trajectory (x : Beliefs) (n : Nat) : Beliefs := beliefStep^[n] x

/-- Broadcasting and received beliefs ignore all exposure counters, even
outside the all-broadcast region. -/
theorem propagate_independent_exposures (x : Beliefs) (e : Fin 5 → Nat) :
    project (propagate path5Adj (population x e)) =
      project (propagate path5Adj (population x (fun _ => 0))) := by
  funext i
  have hin : incoming path5Adj (population x e) i =
      incoming path5Adj (population x (fun _ => 0)) i := by
    rfl
  change (nextAgent path5Adj (population x e) i).belief =
    (nextAgent path5Adj (population x (fun _ => 0)) i).belief
  simp only [nextAgent, hin]
  split_ifs <;> rfl

/-- Enumerate the actual Path5 incoming sets from adjacency and the inclusive
broadcast threshold. This is a bridge theorem, not the implementation of the
step. -/
theorem incoming_eq (x : Beliefs) (e : Fin 5 → Nat)
    (hx : allBroadcast x) (i : Fin 5) :
    incoming path5Adj (population x e) i =
      ![({1} : Finset (Fin 5)), {0, 2}, {1, 3}, {2, 4}, {3}] i := by
  ext j
  have hb : broadcasting ((population x e).profiles j)
      ((population x e).agents j) = true := by
    change decide ((1 : Rat) / 2 ≤ x j) = true
    exact decide_eq_true (hx j).1
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hb, and_true]
  fin_cases i <;> fin_cases j <;> decide

end NarrativeDynamics.FitnessABMPath5
