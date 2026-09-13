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

/-- Stable numeric-ID degree lookup without dependent `Fin` casts in callers. -/
def degreeById : RunState → Nat → Nat
  | ⟨n, s⟩, u => if hu : u < n then degree s.snapshot ⟨u, hu⟩ else 0

/-- A cheap topology signature for this four-vertex tree fixture: one of the two
    stable seed vertices is adjacent to all other vertices. -/
def oldStarCenter (out : RunState) : Prop :=
  degreeById out 0 = 3 ∨ degreeById out 1 = 3

instance oldStarCenterDecidable : DecidablePred oldStarCenter := by
  intro out
  unfold oldStarCenter degreeById
  split <;> split <;> infer_instance

/-- The only embedding from a singleton target-index type selecting `i`. -/
private def singletonTarget {n : Nat} (i : Fin n) : Targets n 1 :=
  ⟨fun _ => i, fun a b _ => Subsingleton.elim a b⟩

private theorem targetsOne_eq_singleton {n : Nat} (T : Targets n 1) :
    T = singletonTarget (T (0 : Fin 1)) := by
  ext i
  have hi : i = (0 : Fin 1) := Subsingleton.elim _ _
  subst i
  rfl

private def edgeTrace (i : Fin 2) (j : Fin 3) :
    TargetTrace 2 1 twoBirthSchedule.length :=
  (singletonTarget i, (singletonTarget j, PUnit.unit))

private theorem edgeTrace_cases (trace : TargetTrace 2 1 twoBirthSchedule.length) :
    ∃ i : Fin 2, ∃ j : Fin 3, trace = edgeTrace i j := by
  change Targets 2 1 × (Targets 3 1 × PUnit) at trace
  rcases trace with ⟨T, U, z⟩
  refine ⟨T (0 : Fin 1), U (0 : Fin 1), ?_⟩
  cases z
  rw [targetsOne_eq_singleton T, targetsOne_eq_singleton U]
  rfl

private def edgeTrace00 := edgeTrace (0 : Fin 2) (0 : Fin 3)
private def edgeTrace01 := edgeTrace (0 : Fin 2) (1 : Fin 3)
private def edgeTrace02 := edgeTrace (0 : Fin 2) (2 : Fin 3)
private def edgeTrace10 := edgeTrace (1 : Fin 2) (0 : Fin 3)
private def edgeTrace11 := edgeTrace (1 : Fin 2) (1 : Fin 3)
private def edgeTrace12 := edgeTrace (1 : Fin 2) (2 : Fin 3)

private theorem edgeTrace00_mass :
    traceProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace00 = 1 / 4 := by decide_cbv
private theorem edgeTrace01_mass :
    traceProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace01 = 1 / 8 := by decide_cbv
private theorem edgeTrace02_mass :
    traceProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace02 = 1 / 8 := by decide_cbv
private theorem edgeTrace10_mass :
    traceProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace10 = 1 / 8 := by decide_cbv
private theorem edgeTrace11_mass :
    traceProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace11 = 1 / 4 := by decide_cbv
private theorem edgeTrace12_mass :
    traceProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace12 = 1 / 8 := by decide_cbv

private theorem edgeTrace00_diameter :
    diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace00) := by
  decide_cbv
private theorem edgeTrace01_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace01) := by
  decide_cbv
private theorem edgeTrace02_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace02) := by
  decide_cbv
private theorem edgeTrace10_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace10) := by
  decide_cbv
private theorem edgeTrace11_diameter :
    diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace11) := by
  decide_cbv
private theorem edgeTrace12_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace12) := by
  decide_cbv

private theorem edgeTrace00_event :
    oldStarCenter
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace00) := by
  decide_cbv
private theorem edgeTrace01_event :
    ¬ oldStarCenter
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace01) := by
  decide_cbv
private theorem edgeTrace02_event :
    ¬ oldStarCenter
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace02) := by
  decide_cbv
private theorem edgeTrace10_event :
    ¬ oldStarCenter
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace10) := by
  decide_cbv
private theorem edgeTrace11_event :
    oldStarCenter
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace11) := by
  decide_cbv
private theorem edgeTrace12_event :
    ¬ oldStarCenter
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace12) := by
  decide_cbv

/-- The six concrete traces are now explicit proof commands, so each exact
    mass/outcome check stays within the ordinary deterministic heartbeat bound. -/
example :
    traceProbability edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace00 = 1 / 4 ∧
    traceProbability edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace01 = 1 / 8 ∧
    traceProbability edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace02 = 1 / 8 ∧
    traceProbability edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace10 = 1 / 8 ∧
    traceProbability edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace11 = 1 / 4 ∧
    traceProbability edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace12 = 1 / 8 :=
  ⟨edgeTrace00_mass, edgeTrace01_mass, edgeTrace02_mass,
    edgeTrace10_mass, edgeTrace11_mass, edgeTrace12_mass⟩

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

private theorem oldStarCenter_iff_diameterAtMostTwo_trace
    (trace : TargetTrace 2 1 twoBirthSchedule.length) :
    oldStarCenter
        (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule trace) ↔
      diameterAtMostTwo
        (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule trace) := by
  obtain ⟨i, j, rfl⟩ := edgeTrace_cases trace
  fin_cases i <;> fin_cases j
  · exact ⟨fun _ => edgeTrace00_diameter, fun _ => edgeTrace00_event⟩
  · exact ⟨False.elim ∘ edgeTrace01_event, False.elim ∘ edgeTrace01_diameter⟩
  · exact ⟨False.elim ∘ edgeTrace02_event, False.elim ∘ edgeTrace02_diameter⟩
  · exact ⟨False.elim ∘ edgeTrace10_event, False.elim ∘ edgeTrace10_diameter⟩
  · exact ⟨fun _ => edgeTrace11_diameter, fun _ => edgeTrace11_event⟩
  · exact ⟨False.elim ∘ edgeTrace12_event, False.elim ∘ edgeTrace12_diameter⟩

/-- Therefore the exact finite BB probability of final mesh diameter at most two
    is one half. The official metric event is transferred tracewise to the cheap
    but equivalent star signature before the finite probability sum is reduced. -/
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

private def target01 : Targets 3 2 := ⟨![0, 1], by decide⟩
private def target02 : Targets 3 2 := ⟨![0, 2], by decide⟩
private def target10 : Targets 3 2 := ⟨![1, 0], by decide⟩
private def target12 : Targets 3 2 := ⟨![1, 2], by decide⟩
private def target20 : Targets 3 2 := ⟨![2, 0], by decide⟩
private def target21 : Targets 3 2 := ⟨![2, 1], by decide⟩

private def triangleTrace01 : TargetTrace 3 2 oneBirthSchedule.length := (target01, PUnit.unit)
private def triangleTrace02 : TargetTrace 3 2 oneBirthSchedule.length := (target02, PUnit.unit)
private def triangleTrace10 : TargetTrace 3 2 oneBirthSchedule.length := (target10, PUnit.unit)
private def triangleTrace12 : TargetTrace 3 2 oneBirthSchedule.length := (target12, PUnit.unit)
private def triangleTrace20 : TargetTrace 3 2 oneBirthSchedule.length := (target20, PUnit.unit)
private def triangleTrace21 : TargetTrace 3 2 oneBirthSchedule.length := (target21, PUnit.unit)

private theorem triangleTrace01_mass :
    traceProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule triangleTrace01 = 1 / 6 := by decide_cbv
private theorem triangleTrace02_mass :
    traceProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule triangleTrace02 = 1 / 6 := by decide_cbv
private theorem triangleTrace10_mass :
    traceProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule triangleTrace10 = 1 / 6 := by decide_cbv
private theorem triangleTrace12_mass :
    traceProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule triangleTrace12 = 1 / 6 := by decide_cbv
private theorem triangleTrace20_mass :
    traceProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule triangleTrace20 = 1 / 6 := by decide_cbv
private theorem triangleTrace21_mass :
    traceProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule triangleTrace21 = 1 / 6 := by decide_cbv

private theorem triangleTrace12_event :
    newbornAdjacentToOneTwo
      (traceFinal triangle3State 2 (by decide) (by decide)
        oneBirthSchedule triangleTrace12) := by decide_cbv
private theorem triangleTrace21_event :
    newbornAdjacentToOneTwo
      (traceFinal triangle3State 2 (by decide) (by decide)
        oneBirthSchedule triangleTrace21) := by decide_cbv

/-- All six ordered traces are present separately, and each carries exact mass 1/6. -/
example :
    traceProbability triangle3State 2 (by decide) (by decide) oneBirthSchedule triangleTrace01 = 1 / 6 ∧
    traceProbability triangle3State 2 (by decide) (by decide) oneBirthSchedule triangleTrace02 = 1 / 6 ∧
    traceProbability triangle3State 2 (by decide) (by decide) oneBirthSchedule triangleTrace10 = 1 / 6 ∧
    traceProbability triangle3State 2 (by decide) (by decide) oneBirthSchedule triangleTrace12 = 1 / 6 ∧
    traceProbability triangle3State 2 (by decide) (by decide) oneBirthSchedule triangleTrace20 = 1 / 6 ∧
    traceProbability triangle3State 2 (by decide) (by decide) oneBirthSchedule triangleTrace21 = 1 / 6 :=
  ⟨triangleTrace01_mass, triangleTrace02_mass, triangleTrace10_mass,
    triangleTrace12_mass, triangleTrace20_mass, triangleTrace21_mass⟩

/-- Both orders selecting stable IDs 1 and 2 satisfy the same final-graph event. -/
example :
    newbornAdjacentToOneTwo
        (traceFinal triangle3State 2 (by decide) (by decide)
          oneBirthSchedule triangleTrace12) ∧
      newbornAdjacentToOneTwo
        (traceFinal triangle3State 2 (by decide) (by decide)
          oneBirthSchedule triangleTrace21) :=
  ⟨triangleTrace12_event, triangleTrace21_event⟩

/-- The unordered target-set event {1,2} receives both orders (1,2) and (2,1),
    hence exact mass 2/6 = 1/3. -/
example :
    eventProbability triangle3State 2 (by decide) (by decide)
      oneBirthSchedule newbornAdjacentToOneTwo = 1 / 3 := by
  decide_cbv

#print axioms NarrativeDynamics.FitnessAttachment.DistributionNetworkFixtures.executableDiameterAtMostTwo_iff_meshDiameter_le_two

end NarrativeDynamics.FitnessAttachment.DistributionNetworkFixtures
