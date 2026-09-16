import NarrativeDynamics.Core.FitnessABMPathN

/-!
# Parameterized finite-path BB belief dynamics

This module keeps the executable step on the existing
`NetworkPropagation.propagate` path while making receptivity and threshold
explicit exact-rational parameters.
-/

namespace NarrativeDynamics.FitnessABMPathNParameters

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FiniteConsensus

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

/-- Broadcasting decisions, received beliefs, and therefore projected updated
beliefs do not depend on the stored exposure counters. -/
theorem propagate_independent_exposures
    (params : ResponseParameters) (n : Nat) (x : Beliefs n)
    (e : Fin n → Nat) :
    project n
        (propagate (FitnessABMPathN.pathAdj n) (population params n x e)) =
      beliefStep params n x := by
  change project n
      (propagate (FitnessABMPathN.pathAdj n) (population params n x e)) =
    project n
      (propagate (FitnessABMPathN.pathAdj n)
        (population params n x (fun _ => 0)))
  funext i
  have hin :
      incoming (FitnessABMPathN.pathAdj n) (population params n x e) i =
        incoming (FitnessABMPathN.pathAdj n)
          (population params n x (fun _ => 0)) i := by
    rfl
  change
    (nextAgent (FitnessABMPathN.pathAdj n) (population params n x e) i).belief =
      (nextAgent (FitnessABMPathN.pathAdj n)
        (population params n x (fun _ => 0)) i).belief
  simp only [nextAgent, hin]
  split_ifs <;> rfl

/-- Proof-only lazy averaging kernel for arbitrary response parameters.
The self mass is `1 - α`; broadcasting-neighbor mass is `α / degree`. -/
def pathKernel (params : ResponseParameters) (n : Nat) : Kernel (Fin n) :=
  fun i j =>
    (if i = j then 1 - params.receptivity else 0) +
      (if j ∈ FitnessABMPathN.neighbors n i then
        params.receptivity / (FitnessABMPathN.degree n i : Rat)
      else 0)

/-- The parameter kernel at `α = 1/2` is exactly the existing fixed PathN
kernel. -/
theorem pathKernel_half (n : Nat) :
    pathKernel half n = FitnessABMPathN.pathKernel n := by
  have hdiv : ∀ d : Rat, (1/2 : Rat) / d = 1 / (2 * d) := by
    intro d
    by_cases hd : d = 0
    · simp [hd]
    · field_simp [hd]
  funext i j
  by_cases hij : i = j <;>
    by_cases hmem : j ∈ FitnessABMPathN.neighbors n i <;>
    simp [pathKernel, FitnessABMPathN.pathKernel, half, hij, hmem, hdiv]

end NarrativeDynamics.FitnessABMPathNParameters
