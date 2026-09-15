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

instance : DecidableRel pathAdj := fun i j =>
  show Decidable (i.val + 1 = j.val ∨ j.val + 1 = i.val) from inferInstance

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
  have hin : incoming pathAdj (population x e) i =
      incoming pathAdj (population x (fun _ => 0)) i := by
    rfl
  change (nextAgent pathAdj (population x e) i).belief =
    (nextAgent pathAdj (population x (fun _ => 0)) i).belief
  simp only [nextAgent, hin]
  split_ifs <;> rfl

/-- Enumerate the actual incoming sets using path adjacency and the inclusive
broadcast threshold. -/
private theorem incoming_eq (x : Beliefs) (e : Fin 4 → Nat)
    (hx : allBroadcast x) (i : Fin 4) :
    incoming pathAdj (population x e) i =
      ![({1} : Finset (Fin 4)), {0, 2}, {1, 3}, {2}] i := by
  ext j
  have hb : broadcasting ((population x e).profiles j)
      ((population x e).agents j) = true := by
    change decide ((1 : Rat) / 2 ≤ x j) = true
    exact decide_eq_true (hx j).1
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hb, and_true]
  fin_cases i <;> fin_cases j <;> decide

/-- The rational averaging matrix is the belief projection of propagation for
arbitrary exposures whenever every coordinate broadcasts. -/
theorem propagate_eq_linear (x : Beliefs) (e : Fin 4 → Nat)
    (hx : allBroadcast x) :
    project (propagate pathAdj (population x e)) = linearStep x := by
  have h02 : (0 : Fin 4) ≠ 2 := by decide
  have h13 : (1 : Fin 4) ≠ 3 := by decide
  funext i
  fin_cases i
  · change (nextAgent pathAdj (population x e) 0).belief = linearStep x 0
    simp only [nextAgent, incoming_eq x e hx 0]
    norm_num [linearStep, population]
    ring
  · change (nextAgent pathAdj (population x e) 1).belief = linearStep x 1
    simp only [nextAgent, incoming_eq x e hx 1]
    norm_num [linearStep, population,
      Finset.sum_pair h02, Finset.card_pair h02]
    ring
  · change (nextAgent pathAdj (population x e) 2).belief = linearStep x 2
    simp only [nextAgent, incoming_eq x e hx 2]
    dsimp only [linearStep, Matrix.cons_val]
    norm_num [population,
      Finset.sum_pair h13, Finset.card_pair h13]
    ring
  · change (nextAgent pathAdj (population x e) 3).belief = linearStep x 3
    simp only [nextAgent, incoming_eq x e hx 3]
    dsimp only [linearStep, Matrix.cons_val]
    norm_num [population]
    ring

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

/-- The invariant mean weights the path vertices by their degrees. -/
def mean (x : Beliefs) : Rat := (x 0+2*x 1+2*x 2+x 3)/6

def coeffA (x : Beliefs) : Rat := (x 0+x 1-x 2-x 3)/3

def coeffB (x : Beliefs) : Rat := (x 0-x 1-x 2+x 3)/3

def coeffC (x : Beliefs) : Rat := (x 0-2*x 1+2*x 2-x 3)/6

def modeV : Beliefs := ![1,1/2,-1/2,-1]

def modeW : Beliefs := ![1,-1/2,-1/2,1]

def modeZ : Beliefs := ![1,-1,1,-1]

/-- All four modes are retained, including the zero mode at time zero. -/
def closedForm (x : Beliefs) (n : Nat) : Beliefs := fun i =>
  mean x + coeffA x*(3/4)^n*modeV i +
    coeffB x*(1/4)^n*modeW i + coeffC x*(0:Rat)^n*modeZ i

theorem mean_step (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := by
  rw [show beliefStep x = linearStep x from propagate_eq_linear x _ hx]
  dsimp [mean, linearStep]
  ring

theorem decompose (x : Beliefs) : closedForm x 0 = x := by
  funext i
  fin_cases i <;>
    dsimp [closedForm, mean, coeffA, coeffB, coeffC, modeV, modeW, modeZ] <;>
    ring

theorem linear_closedForm_succ (x : Beliefs) (n : Nat) :
    linearStep (closedForm x n) = closedForm x (n+1) := by
  funext i
  fin_cases i <;>
    dsimp [linearStep, closedForm, modeV, modeW, modeZ] <;>
    simp only [pow_succ] <;>
    ring

/-- The formula follows the actual broadcast dynamics at every time. -/
theorem iterate_closedForm (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    trajectory x n = closedForm x n := by
  induction n with
  | zero => simpa [trajectory] using (decompose x).symm
  | succ n ih =>
      have hregion := allBroadcast_iterate x hx n
      calc
        trajectory x (n+1) = beliefStep (trajectory x n) := by
          simp only [trajectory, Function.iterate_succ_apply']
        _ = linearStep (trajectory x n) := propagate_eq_linear _ _ hregion
        _ = linearStep (closedForm x n) := congrArg linearStep ih
        _ = closedForm x (n+1) := linear_closedForm_succ x n

end NarrativeDynamics.FitnessABMPath4
