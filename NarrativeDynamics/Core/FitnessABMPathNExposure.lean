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

end NarrativeDynamics.FitnessABMPathNExposure
