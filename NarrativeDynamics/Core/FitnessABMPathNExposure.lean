import NarrativeDynamics.Core.FitnessABMPathNParameters

/-!
# Exposure-dependent finite-path belief dynamics

This is a separate exact-rational model. It reuses only the canonical finite
path topology. Current-round incoming broadcasts are accumulated before the
learning-rate schedule is queried for the same belief update.
-/

namespace NarrativeDynamics.FitnessABMPathNExposure

open scoped BigOperators

structure ExposureParameters where
  receptivityAt : Nat → Rat
  threshold : Rat

namespace ExposureParameters

def Valid (p : ExposureParameters) : Prop :=
  (∀ e, 0 ≤ p.receptivityAt e ∧ p.receptivityAt e ≤ 1) ∧
    0 ≤ p.threshold ∧ p.threshold ≤ 1

end ExposureParameters

structure AgentState where
  belief : Rat
  exposure : Nat
  deriving Repr, DecidableEq

abbrev State (n : Nat) := Fin n → AgentState

def beliefs (s : State n) : NarrativeDynamics.FitnessABMPathN.Beliefs n :=
  fun i => (s i).belief

def broadcasting (p : ExposureParameters) (s : State n) (i : Fin n) : Bool :=
  decide (p.threshold ≤ (s i).belief)

def broadcasters (p : ExposureParameters) (n : Nat) (s : State n)
    (i : Fin n) : Finset (Fin n) :=
  (NarrativeDynamics.FitnessABMPathN.neighbors n i).filter
    (fun j => broadcasting p s j = true)

def incoming (p : ExposureParameters) (s : State n) (i : Fin n) : Nat :=
  (broadcasters p n s i).card

def broadcasterMean (p : ExposureParameters) (s : State n) (i : Fin n) : Rat :=
  (∑ j ∈ broadcasters p n s i, (s j).belief) /
    (incoming p s i : Rat)

def step (p : ExposureParameters) (n : Nat) (s : State n) : State n :=
  fun i =>
    let received := incoming p s i
    let exposure' := (s i).exposure + received
    if received = 0 then
      ⟨(s i).belief, exposure'⟩
    else
      let alpha := p.receptivityAt exposure'
      let mean := broadcasterMean p s i
      ⟨(1 - alpha) * (s i).belief + alpha * mean, exposure'⟩

/-- A constant schedule recovers an exposure-independent response parameter. -/
def constant (alpha tau : Rat) : ExposureParameters :=
  ⟨fun _ => alpha, tau⟩

private theorem broadcasters_constant_eq_incoming
    (alpha tau : Rat) (n : Nat) (s : State n) (e : Fin n → Nat) (i : Fin n) :
    broadcasters (constant alpha tau) n s i =
      NarrativeDynamics.NetworkPropagation.incoming
        (NarrativeDynamics.FitnessABMPathN.pathAdj n)
        (NarrativeDynamics.FitnessABMPathNParameters.population
          ⟨alpha, tau⟩ n (beliefs s) e) i := by
  ext j
  simp [broadcasters, NarrativeDynamics.FitnessABMPathN.neighbors,
    NarrativeDynamics.NetworkPropagation.incoming,
    NarrativeDynamics.NetworkPropagation.broadcasting,
    NarrativeDynamics.FitnessABMPathNParameters.population,
    broadcasting, beliefs, constant]

/-- Under a constant schedule, the exposure-dependent model has exactly the
same projected belief step as the parameterized baseline. Exposure bookkeeping
is intentionally not part of this equality. -/
theorem constant_beliefStep (alpha tau : Rat) (n : Nat) (s : State n) :
    beliefs (step (constant alpha tau) n s) =
      NarrativeDynamics.FitnessABMPathNParameters.beliefStep
        ⟨alpha, tau⟩ n (beliefs s) := by
  funext i
  have hreceived :=
    broadcasters_constant_eq_incoming alpha tau n s (fun _ => 0) i
  change
    (step (constant alpha tau) n s i).belief =
      (NarrativeDynamics.NetworkPropagation.nextAgent
        (NarrativeDynamics.FitnessABMPathN.pathAdj n)
        (NarrativeDynamics.FitnessABMPathNParameters.population
          ⟨alpha, tau⟩ n (beliefs s) (fun _ => 0)) i).belief
  simp only [step]
  simp only [incoming, broadcasterMean]
  rw [hreceived]
  simp [NarrativeDynamics.NetworkPropagation.nextAgent,
    NarrativeDynamics.FitnessABMPathNParameters.population,
    beliefs, constant]

/-- The canonical half schedule therefore recovers the existing fixed PathN
belief step through the #82 exact specialization. -/
theorem half_beliefStep (n : Nat) (s : State n) :
    beliefs (step (constant (1/2) (1/2)) n s) =
      NarrativeDynamics.FitnessABMPathN.beliefStep n (beliefs s) := by
  rw [constant_beliefStep]
  exact NarrativeDynamics.FitnessABMPathNParameters.beliefStep_half n (beliefs s)

end NarrativeDynamics.FitnessABMPathNExposure
