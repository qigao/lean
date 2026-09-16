import NarrativeDynamics.Core.FitnessABMPathN

/-!
# Parameterized finite-path BB belief dynamics

This module keeps the executable step on the existing
`NetworkPropagation.propagate` path while making receptivity and threshold
explicit exact-rational parameters.
-/

namespace NarrativeDynamics.FitnessABMPathNParameters

open NarrativeDynamics.NetworkPropagation

abbrev Beliefs (n : Nat) := FitnessABMPathN.Beliefs n

structure ResponseParameters where
  receptivity : Rat
  threshold : Rat
  deriving Repr, DecidableEq

namespace ResponseParameters

def Valid (p : ResponseParameters) : Prop :=
  0 ≤ p.receptivity ∧ p.receptivity ≤ 1 ∧
    0 ≤ p.threshold ∧ p.threshold ≤ 1

end ResponseParameters

def population (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : Population n :=
  ⟨fun _ => ⟨params.receptivity, params.threshold⟩,
    fun i => ⟨x i, e i⟩⟩

def project (n : Nat) (p : Population n) : Beliefs n :=
  fun i => (p.agents i).belief

def allBroadcast (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) : Prop :=
  ∀ i, params.threshold ≤ x i ∧ x i ≤ 1

def beliefStep (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) : Beliefs n :=
  project n
    (propagate (FitnessABMPathN.pathAdj n)
      (population params n x (fun _ => 0)))

def half : ResponseParameters :=
  ⟨1/2, 1/2⟩

/-- The canonical half-response parameterization is exactly the existing fixed
finite-path population, including supplied exposure counters. -/
theorem population_half (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
    population half n x e = FitnessABMPathN.population n x e := by
  rfl

/-- The canonical half-response executable step is exactly the existing fixed
finite-path step. No second propagation implementation is introduced. -/
theorem beliefStep_half (n : Nat) (x : Beliefs n) :
    beliefStep half n x = FitnessABMPathN.beliefStep n x := by
  rfl

/-- The canonical half threshold has the same inclusive all-broadcast region
as the existing fixed finite-path model. -/
theorem allBroadcast_half (n : Nat) (x : Beliefs n) :
    allBroadcast half n x ↔ FitnessABMPathN.allBroadcast n x := by
  rfl

end NarrativeDynamics.FitnessABMPathNParameters
