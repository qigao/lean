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

private def linearStep (x : Beliefs) : Beliefs :=
  ![(x 0 + x 1) / 2,
    x 0 / 4 + x 1 / 2 + x 2 / 4,
    x 1 / 4 + x 2 / 2 + x 3 / 4,
    x 2 / 4 + x 3 / 2 + x 4 / 4,
    (x 3 + x 4) / 2]

/-- The local linear form is derived from the actual incoming sets; it is not
used to define the dynamics. -/
private theorem beliefStep_eq_linear (x : Beliefs) (hx : allBroadcast x) :
    beliefStep x = linearStep x := by
  have h02 : (0 : Fin 5) ≠ 2 := by decide
  have h13 : (1 : Fin 5) ≠ 3 := by decide
  have h24 : (2 : Fin 5) ≠ 4 := by decide
  funext i
  fin_cases i
  · change (nextAgent path5Adj (population x (fun _ => 0)) 0).belief = linearStep x 0
    simp only [nextAgent, incoming_eq x _ hx 0]
    norm_num [linearStep, population]
    ring
  · change (nextAgent path5Adj (population x (fun _ => 0)) 1).belief = linearStep x 1
    simp only [nextAgent, incoming_eq x _ hx 1]
    norm_num [linearStep, population,
      Finset.sum_pair h02, Finset.card_pair h02]
    ring
  · change (nextAgent path5Adj (population x (fun _ => 0)) 2).belief = linearStep x 2
    simp only [nextAgent, incoming_eq x _ hx 2]
    norm_num [linearStep, population,
      Finset.sum_pair h13, Finset.card_pair h13]
    ring
  · change (nextAgent path5Adj (population x (fun _ => 0)) 3).belief = linearStep x 3
    simp only [nextAgent, incoming_eq x _ hx 3]
    norm_num [linearStep, population,
      Finset.sum_pair h24, Finset.card_pair h24]
    ring
  · change (nextAgent path5Adj (population x (fun _ => 0)) 4).belief = linearStep x 4
    simp only [nextAgent, incoming_eq x _ hx 4]
    norm_num [linearStep, population]
    ring

theorem allBroadcast_step (x : Beliefs) (hx : allBroadcast x) :
    allBroadcast (beliefStep x) := by
  rw [beliefStep_eq_linear x hx]
  intro i
  have h0 := hx 0
  have h1 := hx 1
  have h2 := hx 2
  have h3 := hx 3
  have h4 := hx 4
  fin_cases i <;> dsimp [linearStep] <;> constructor <;> linarith

theorem allBroadcast_iterate (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    allBroadcast (trajectory x n) := by
  induction n with
  | zero => simpa [trajectory] using hx
  | succ n ih =>
      simpa [trajectory, Function.iterate_succ_apply'] using
        allBroadcast_step (trajectory x n) ih

/-- The invariant mean weights path vertices by their degrees `1,2,2,2,1`. -/
def mean (x : Beliefs) : Rat :=
  (x 0 + 2 * x 1 + 2 * x 2 + 2 * x 3 + x 4) / 8

theorem mean_step (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := by
  rw [beliefStep_eq_linear x hx]
  dsimp [mean, linearStep]
  ring

end NarrativeDynamics.FitnessABMPath5
