import NarrativeDynamics.Core.FitnessDistributionInvariance
import NarrativeDynamics.Core.SmallWorldMetrics

open NarrativeDynamics
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

namespace NarrativeDynamics.FitnessAttachment.DistributionNetworkFixtures

private def unitFitness : PosFitness := ⟨1, by norm_num⟩

/-! ## Two births from a unit-fitness edge -/

def edgeSeed : RawSeed := ⟨2, #[1, 1], #[(0, 1)]⟩

private theorem edgeSeedConnected : (seedGraph edgeSeed).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v
      exact ⟨.nil⟩
    · have huv : (seedGraph edgeSeed).Adj u v := by
        fin_cases u <;> fin_cases v <;>
          simp_all [seedGraph, edgeSeed, canonicalEdge]
      exact ⟨.cons huv .nil⟩
  nonempty := ⟨⟨0, by decide⟩⟩

def edge2State : State 2 :=
  ⟨seedSnapshot edgeSeed rfl, by
    exact ⟨by decide, edgeSeedConnected, by decide⟩⟩

private def twoBirthSchedule : List PosFitness := [unitFitness, unitFitness]

/-- Executable diameter-at-most-two predicate using the already-proved finite
    reachability computation. This is not a second distance semantics. -/
def diameterAtMostTwo : RunState → Prop
  | ⟨n, s⟩ => ∀ a b : Fin n, b ∈ reached s.snapshot a 2

instance diameterAtMostTwoDecidable : DecidablePred diameterAtMostTwo := by
  intro out
  rcases out with ⟨n, s⟩
  change Decidable (∀ a b : Fin n, b ∈ reached s.snapshot a 2)
  infer_instance

/-- The executable finite predicate is exactly the existing mesh-diameter event. -/
theorem executableDiameterAtMostTwo_iff_meshDiameter_le_two {n : Nat} (s : State n) :
    diameterAtMostTwo ⟨n, s⟩ ↔
      meshDiameter s.snapshot.graph.Adj ⟨n - 1, state_bounded s⟩ ≤ 2 := by
  constructor
  · intro h
    change ∀ a b : Fin n, b ∈ reached s.snapshot a 2 at h
    apply meshDiameter_minimal s.snapshot.graph.Adj ⟨n - 1, state_bounded s⟩
    intro a b
    exact (reached_iff s.snapshot a b 2).mp (h a b)
  · intro h
    change ∀ a b : Fin n, b ∈ reached s.snapshot a 2
    intro a b
    apply (reached_iff s.snapshot a b 2).mpr
    exact reachWithin_mono
      (meshDiameter_spec s.snapshot.graph.Adj ⟨n - 1, state_bounded s⟩ a b) h

/-- A cheap topology signature for this four-vertex tree fixture: one of the two
    stable seed vertices is adjacent to all other vertices. -/
def oldStarCenter : RunState → Prop
  | ⟨0, _⟩ => False
  | ⟨1, _⟩ => False
  | ⟨n + 2, s⟩ =>
      degree s.snapshot (0 : Fin (n + 2)) = 3 ∨
        degree s.snapshot (1 : Fin (n + 2)) = 3

instance oldStarCenterDecidable : DecidablePred oldStarCenter := by
  intro out
  rcases out with ⟨n, s⟩
  cases n with
  | zero => exact isFalse id
  | succ n =>
      cases n with
      | zero => exact isFalse id
      | succ n =>
          change Decidable
            (degree s.snapshot (0 : Fin (n + 3)) = 3 ∨
              degree s.snapshot (1 : Fin (n + 3)) = 3)
          infer_instance

/-- Events that agree on every final state reached by this typed experiment have
    identical exact probability. -/
private theorem eventProbability_congr_on_traceFinal {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event₁ event₂ : RunState → Prop) [DecidablePred event₁] [DecidablePred event₂]
    (h : ∀ trace : TargetTrace n m schedule.length,
      event₁ (traceFinal s m hm hb schedule trace) ↔
        event₂ (traceFinal s m hm hb schedule trace)) :
    eventProbability s m hm hb schedule event₁ =
      eventProbability s m hm hb schedule event₂ := by
  unfold eventProbability
  apply Finset.sum_congr rfl
  intro trace _
  by_cases h₁ : event₁ (traceFinal s m hm hb schedule trace)
  · have h₂ := (h trace).mp h₁
    simp [h₁, h₂]
  · have h₂ : ¬ event₂ (traceFinal s m hm hb schedule trace) := by
      intro h₂
      exact h₁ ((h trace).mpr h₂)
    simp [h₁, h₂]

/-- For each of the six concrete traces, the cheap star signature is equivalent
    to the executable form of the official diameter-at-most-two event. -/
private theorem oldStarCenter_iff_diameterAtMostTwo_trace
    (trace : TargetTrace 2 1 twoBirthSchedule.length) :
    oldStarCenter
        (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule trace) ↔
      diameterAtMostTwo
        (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule trace) := by
  fin_cases trace <;> decide_cbv

/-- Exhaustive six-trace classification: the two diameter-two outcomes have mass
    1/4 each; the other four outcomes have mass 1/8 each. -/
example :
    ∀ trace : TargetTrace 2 1 twoBirthSchedule.length,
      (diameterAtMostTwo
          (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule trace) ∧
        traceProbability edge2State 1 (by decide) (by decide)
          twoBirthSchedule trace = 1 / 4) ∨
      (¬ diameterAtMostTwo
          (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule trace) ∧
        traceProbability edge2State 1 (by decide) (by decide)
          twoBirthSchedule trace = 1 / 8) := by
  intro trace
  fin_cases trace <;> decide_cbv

/-- Therefore the exact finite BB probability of final mesh diameter at most two
    is one half. The expensive reachability predicate is transferred tracewise
    to the equivalent star signature before the finite probability sum is reduced. -/
example :
    eventProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule diameterAtMostTwo = 1 / 2 := by
  calc
    eventProbability edge2State 1 (by decide) (by decide)
        twoBirthSchedule diameterAtMostTwo =
      eventProbability edge2State 1 (by decide) (by decide)
        twoBirthSchedule oldStarCenter :=
      eventProbability_congr_on_traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule diameterAtMostTwo oldStarCenter
        (fun trace => (oldStarCenter_iff_diameterAtMostTwo_trace trace).symm)
    _ = 1 / 2 := by decide_cbv

/-! ## One birth from a unit-fitness triangle, m = 2 -/

def triangleSeed : RawSeed :=
  ⟨3, #[1, 1, 1], #[(0, 1), (1, 2), (0, 2)]⟩

private theorem triangleSeedConnected : (seedGraph triangleSeed).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v
      exact ⟨.nil⟩
    · have huv : (seedGraph triangleSeed).Adj u v := by
        fin_cases u <;> fin_cases v <;>
          simp_all [seedGraph, triangleSeed, canonicalEdge]
      exact ⟨.cons huv .nil⟩
  nonempty := ⟨⟨0, by decide⟩⟩

def triangle3State : State 3 :=
  ⟨seedSnapshot triangleSeed rfl, by
    exact ⟨by decide, triangleSeedConnected, by decide⟩⟩

private def oneBirthSchedule : List PosFitness := [unitFitness]

/-- Adjacency between stable numeric IDs, guarded by the run-state node count. -/
def adjacentById : RunState → Nat → Nat → Prop
  | ⟨n, s⟩, u, v =>
      if hu : u < n then
        if hv : v < n then s.snapshot.graph.Adj ⟨u, hu⟩ ⟨v, hv⟩
        else False
      else False

instance adjacentByIdDecidable (out : RunState) (u v : Nat) :
    Decidable (adjacentById out u v) := by
  rcases out with ⟨n, s⟩
  unfold adjacentById
  letI := s.snapshot.adjDec
  infer_instance

/-- In this one-birth fixture the newborn has stable ID 3. -/
def newbornAdjacentToOneTwo (out : RunState) : Prop :=
  adjacentById out 3 1 ∧ adjacentById out 3 2

instance newbornAdjacentToOneTwoDecidable : DecidablePred newbornAdjacentToOneTwo := by
  intro out
  unfold newbornAdjacentToOneTwo
  infer_instance

/-- All six ordered target traces from the symmetric unit-fitness triangle have
    exact mass 1/6. Each trace is reduced separately to keep normal proof limits. -/
example :
    ∀ trace : TargetTrace 3 2 oneBirthSchedule.length,
      traceProbability triangle3State 2 (by decide) (by decide)
        oneBirthSchedule trace = 1 / 6 := by
  intro trace
  fin_cases trace <;> decide_cbv

/-- The unordered target-set event {1,2} receives both orders (1,2) and (2,1),
    hence exact mass 2/6 = 1/3. -/
example :
    eventProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule newbornAdjacentToOneTwo = 1 / 3 := by
  decide_cbv

#print axioms NarrativeDynamics.FitnessAttachment.DistributionNetworkFixtures.executableDiameterAtMostTwo_iff_meshDiameter_le_two

end NarrativeDynamics.FitnessAttachment.DistributionNetworkFixtures
