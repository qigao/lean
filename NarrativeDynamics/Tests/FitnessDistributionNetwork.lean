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

private abbrev twoBirthSchedule : List PosFitness := [unitFitness, unitFitness]

/-- Executable finite closure for the already-existing bounded-walk semantics. -/
def diameterAtMostTwo : RunState → Prop
  | ⟨n, s⟩ => ∀ a b : Fin n, b ∈ reached s.snapshot a 2

instance diameterAtMostTwoDecidable : DecidablePred diameterAtMostTwo := by
  intro out
  rcases out with ⟨n, s⟩
  change Decidable (∀ a b : Fin n, b ∈ reached s.snapshot a 2)
  infer_instance

/-- The executable predicate is exactly the existing `meshDiameter ≤ 2` event. -/
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

/-- Degree vectors are derived from the authoritative one-birth theorem. -/
private theorem birthDegreeFn {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    degree (applyBirth s T hm eta).snapshot =
      Fin.lastCases m (fun u => degree s.snapshot u + if u ∈ T.selected then 1 else 0) := by
  funext v
  refine Fin.lastCases ?_ (fun u => ?_) v
  · rw [Fin.lastCases_last]
    exact birth_degree_new s T hm eta
  · rw [Fin.lastCases_castSucc]
    exact birth_degree_old s T hm eta u

private theorem edge2Degrees :
    degree edge2State.snapshot = (![1, 1] : Fin 2 → Nat) := by
  funext i
  fin_cases i <;> decide_cbv

private def edge3State0 : State 3 :=
  applyBirth edge2State (singletonTarget (0 : Fin 2)) (by decide) unitFitness
private def edge3State1 : State 3 :=
  applyBirth edge2State (singletonTarget (1 : Fin 2)) (by decide) unitFitness
private def edge4State00 : State 4 :=
  applyBirth edge3State0 (singletonTarget (0 : Fin 3)) (by decide) unitFitness
private def edge4State11 : State 4 :=
  applyBirth edge3State1 (singletonTarget (1 : Fin 3)) (by decide) unitFitness

private theorem edge3State0Degrees :
    degree edge3State0.snapshot = (![2, 1, 1] : Fin 3 → Nat) := by
  rw [edge3State0, birthDegreeFn, edge2Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem edge3State1Degrees :
    degree edge3State1.snapshot = (![1, 2, 1] : Fin 3 → Nat) := by
  rw [edge3State1, birthDegreeFn, edge2Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem edge4State00Degrees :
    degree edge4State00.snapshot = (![3, 1, 1, 1] : Fin 4 → Nat) := by
  rw [edge4State00, birthDegreeFn, edge3State0Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem edge4State11Degrees :
    degree edge4State11.snapshot = (![1, 3, 1, 1] : Fin 4 → Nat) := by
  rw [edge4State11, birthDegreeFn, edge3State1Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem edgeTrace00_center_degree :
    degree (traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace00).state.snapshot (0 : Fin 4) = 3 := by
  change degree edge4State00.snapshot (0 : Fin 4) = 3
  simpa using congrFun edge4State00Degrees (0 : Fin 4)

private theorem edgeTrace11_center_degree :
    degree (traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace11).state.snapshot (1 : Fin 4) = 3 := by
  change degree edge4State11.snapshot (1 : Fin 4) = 3
  simpa using congrFun edge4State11Degrees (1 : Fin 4)

/-- On four vertices, degree three forces adjacency to every other vertex. -/
private theorem adj_from_center_of_degree_three (s : Snapshot 4)
    (center v : Fin 4) (hdeg : degree s center = 3) (hv : v ≠ center) :
    s.graph.Adj center v := by
  letI := s.adjDec
  have hset : NarrativeDynamics.neighborSet s.graph.Adj center =
      (Finset.univ : Finset (Fin 4)).erase center := by
    apply Finset.eq_of_subset_of_card_le
    · intro w hw
      have hadj : s.graph.Adj center w := by
        simpa [NarrativeDynamics.neighborSet] using hw
      exact Finset.mem_erase.mpr ⟨by
        intro h
        subst w
        exact s.graph.irrefl hadj,
        Finset.mem_univ _⟩
    · have hcard : (NarrativeDynamics.neighborSet s.graph.Adj center).card = 3 := by
        simpa [degree] using hdeg
      rw [hcard]
      simp
  have hm : v ∈ NarrativeDynamics.neighborSet s.graph.Adj center := by
    rw [hset]
    exact Finset.mem_erase.mpr ⟨hv, Finset.mem_univ _⟩
  simpa [NarrativeDynamics.neighborSet] using hm

private theorem globalBoundTwo_of_degree_three (s : Snapshot 4)
    (center : Fin 4) (hdeg : degree s center = 3) :
    GlobalHopBound s.graph.Adj 2 := by
  intro a b
  by_cases hab : a = b
  · subst b
    exact ⟨0, by omega, MeshWalk.refl _⟩
  by_cases ha : a = center
  · subst a
    have hb : b ≠ center := by
      intro h
      exact hab h.symm
    exact ⟨1, by omega,
      MeshWalk.single (adj_from_center_of_degree_three s center b hdeg hb)⟩
  by_cases hb : b = center
  · subst b
    exact ⟨1, by omega,
      MeshWalk.single (s.graph.symm.symm center a
        (adj_from_center_of_degree_three s center a hdeg ha))⟩
  · exact ⟨2, by omega,
      MeshWalk.step (s.graph.symm.symm center a
        (adj_from_center_of_degree_three s center a hdeg ha))
        (MeshWalk.single (adj_from_center_of_degree_three s center b hdeg hb))⟩

/-- A non-edge with no common one-hop intermediate cannot be reached within two hops. -/
private theorem noReachWithinTwo {g : MeshGraph (Fin 4)}
    (source target : Fin 4) (hne : source ≠ target)
    (hnotAdj : ¬ g source target)
    (hcommon : ∀ middle, ¬ (g source middle ∧ g middle target)) :
    ¬ ReachWithin g 2 source target := by
  rintro ⟨length, hlength, walk⟩
  interval_cases length
  · cases walk
    exact hne rfl
  · cases walk with
    | step edge rest =>
      cases rest
      exact hnotAdj edge
  · cases walk with
    | step edge rest =>
      cases rest with
      | step edge' rest' =>
        cases rest'
        exact (hcommon _) ⟨edge, edge'⟩

private theorem edgeTrace01_no_adj :
    ¬ (traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace01).state.snapshot.graph.Adj (2 : Fin 4) (3 : Fin 4) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace01).state.snapshot.adjDec
  decide_cbv

private theorem edgeTrace01_no_common (middle : Fin 4) :
    ¬ ((traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace01).state.snapshot.graph.Adj (2 : Fin 4) middle ∧
      (traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace01).state.snapshot.graph.Adj middle (3 : Fin 4)) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace01).state.snapshot.adjDec
  fin_cases middle <;> decide_cbv

private theorem edgeTrace02_no_adj :
    ¬ (traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace02).state.snapshot.graph.Adj (1 : Fin 4) (3 : Fin 4) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace02).state.snapshot.adjDec
  decide_cbv

private theorem edgeTrace02_no_common (middle : Fin 4) :
    ¬ ((traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace02).state.snapshot.graph.Adj (1 : Fin 4) middle ∧
      (traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace02).state.snapshot.graph.Adj middle (3 : Fin 4)) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace02).state.snapshot.adjDec
  fin_cases middle <;> decide_cbv

private theorem edgeTrace10_no_adj :
    ¬ (traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace10).state.snapshot.graph.Adj (2 : Fin 4) (3 : Fin 4) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace10).state.snapshot.adjDec
  decide_cbv

private theorem edgeTrace10_no_common (middle : Fin 4) :
    ¬ ((traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace10).state.snapshot.graph.Adj (2 : Fin 4) middle ∧
      (traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace10).state.snapshot.graph.Adj middle (3 : Fin 4)) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace10).state.snapshot.adjDec
  fin_cases middle <;> decide_cbv

private theorem edgeTrace12_no_adj :
    ¬ (traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule edgeTrace12).state.snapshot.graph.Adj (0 : Fin 4) (3 : Fin 4) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace12).state.snapshot.adjDec
  decide_cbv

private theorem edgeTrace12_no_common (middle : Fin 4) :
    ¬ ((traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace12).state.snapshot.graph.Adj (0 : Fin 4) middle ∧
      (traceFinal edge2State 1 (by decide) (by decide)
        twoBirthSchedule edgeTrace12).state.snapshot.graph.Adj middle (3 : Fin 4)) := by
  letI := (traceFinal edge2State 1 (by decide) (by decide)
    twoBirthSchedule edgeTrace12).state.snapshot.adjDec
  fin_cases middle <;> decide_cbv

private theorem edgeTrace00_diameter :
    diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace00) := by
  change ∀ a b : Fin 4, b ∈ reached
    (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace00).state.snapshot a 2
  intro a b
  apply (reached_iff _ a b 2).mpr
  exact globalBoundTwo_of_degree_three _ (0 : Fin 4) edgeTrace00_center_degree a b

private theorem edgeTrace01_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace01) := by
  intro h
  change ∀ a b : Fin 4, b ∈ reached
    (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace01).state.snapshot a 2 at h
  have hreach := (reached_iff _ (2 : Fin 4) (3 : Fin 4) 2).mp
    (h (2 : Fin 4) (3 : Fin 4))
  exact (noReachWithinTwo (2 : Fin 4) (3 : Fin 4) (by decide)
    edgeTrace01_no_adj edgeTrace01_no_common) hreach

private theorem edgeTrace02_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace02) := by
  intro h
  change ∀ a b : Fin 4, b ∈ reached
    (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace02).state.snapshot a 2 at h
  have hreach := (reached_iff _ (1 : Fin 4) (3 : Fin 4) 2).mp
    (h (1 : Fin 4) (3 : Fin 4))
  exact (noReachWithinTwo (1 : Fin 4) (3 : Fin 4) (by decide)
    edgeTrace02_no_adj edgeTrace02_no_common) hreach

private theorem edgeTrace10_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace10) := by
  intro h
  change ∀ a b : Fin 4, b ∈ reached
    (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace10).state.snapshot a 2 at h
  have hreach := (reached_iff _ (2 : Fin 4) (3 : Fin 4) 2).mp
    (h (2 : Fin 4) (3 : Fin 4))
  exact (noReachWithinTwo (2 : Fin 4) (3 : Fin 4) (by decide)
    edgeTrace10_no_adj edgeTrace10_no_common) hreach

private theorem edgeTrace11_diameter :
    diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace11) := by
  change ∀ a b : Fin 4, b ∈ reached
    (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace11).state.snapshot a 2
  intro a b
  apply (reached_iff _ a b 2).mpr
  exact globalBoundTwo_of_degree_three _ (1 : Fin 4) edgeTrace11_center_degree a b

private theorem edgeTrace12_diameter :
    ¬ diameterAtMostTwo
      (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace12) := by
  intro h
  change ∀ a b : Fin 4, b ∈ reached
    (traceFinal edge2State 1 (by decide) (by decide) twoBirthSchedule edgeTrace12).state.snapshot a 2 at h
  have hreach := (reached_iff _ (0 : Fin 4) (3 : Fin 4) 2).mp
    (h (0 : Fin 4) (3 : Fin 4))
  exact (noReachWithinTwo (0 : Fin 4) (3 : Fin 4) (by decide)
    edgeTrace12_no_adj edgeTrace12_no_common) hreach

private theorem edgeTrace_univ :
    (Finset.univ : Finset (TargetTrace 2 1 twoBirthSchedule.length)) =
      {edgeTrace00, edgeTrace01, edgeTrace02, edgeTrace10, edgeTrace11, edgeTrace12} := by
  ext trace
  simp only [Finset.mem_univ, true_iff, Finset.mem_insert, Finset.mem_singleton]
  obtain ⟨i, j, rfl⟩ := edgeTrace_cases trace
  fin_cases i <;> fin_cases j <;>
    simp [edgeTrace00, edgeTrace01, edgeTrace02, edgeTrace10, edgeTrace11, edgeTrace12]

/-- Exact finite probability of final mesh diameter at most two is one half. -/
example :
    eventProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule diameterAtMostTwo = 1 / 2 := by
  unfold eventProbability
  rw [edgeTrace_univ]
  rw [Finset.sum_insert (by decide)]
  rw [Finset.sum_insert (by decide)]
  rw [Finset.sum_insert (by decide)]
  rw [Finset.sum_insert (by decide)]
  rw [Finset.sum_insert (by decide)]
  rw [Finset.sum_singleton]
  rw [if_pos edgeTrace00_diameter, if_neg edgeTrace01_diameter,
    if_neg edgeTrace02_diameter, if_neg edgeTrace10_diameter,
    if_pos edgeTrace11_diameter, if_neg edgeTrace12_diameter]
  rw [edgeTrace00_mass, edgeTrace11_mass]
  norm_num

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

private abbrev oneBirthSchedule : List PosFitness := [unitFitness]
private theorem triangleHm : 0 < 2 := by decide
private theorem triangleHb : 2 ≤ 3 := by decide

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
    traceProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule triangleTrace01 = 1 / 6 := by decide_cbv
private theorem triangleTrace02_mass :
    traceProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule triangleTrace02 = 1 / 6 := by decide_cbv
private theorem triangleTrace10_mass :
    traceProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule triangleTrace10 = 1 / 6 := by decide_cbv
private theorem triangleTrace12_mass :
    traceProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule triangleTrace12 = 1 / 6 := by decide_cbv
private theorem triangleTrace20_mass :
    traceProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule triangleTrace20 = 1 / 6 := by decide_cbv
private theorem triangleTrace21_mass :
    traceProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule triangleTrace21 = 1 / 6 := by decide_cbv

private theorem oneBirth_traceProbability (T : Targets 3 2) :
    traceProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule (T, PUnit.unit) = orderedMass triangle3State T := by
  change orderedMass triangle3State T * 1 = orderedMass triangle3State T
  simp

/-- For one birth, newborn adjacency to IDs 1 and 2 is exactly selection of
    the unordered target set `{1,2}`. -/
private theorem newbornAdjacentToOneTwo_iff_selected (T : Targets 3 2) :
    newbornAdjacentToOneTwo
        (traceFinal triangle3State 2 triangleHm triangleHb
          oneBirthSchedule (T, PUnit.unit)) ↔
      T.selected = ({1, 2} : Finset (Fin 3)) := by
  change
    ((1 : Fin 3) ∈ T.selected ∧ (2 : Fin 3) ∈ T.selected) ↔
      T.selected = ({1, 2} : Finset (Fin 3))
  constructor
  · rintro ⟨h1, h2⟩
    have hsub : ({1, 2} : Finset (Fin 3)) ⊆ T.selected := by
      intro x hx
      rcases Finset.mem_insert.mp hx with h | h
      · subst x
        exact h1
      · have hx2 := Finset.mem_singleton.mp h
        subst x
        exact h2
    have hcard : T.selected.card ≤ ({1, 2} : Finset (Fin 3)).card := by
      rw [selected_card]
      decide
    exact (Finset.eq_of_subset_of_card_le hsub hcard).symm
  · intro h
    rw [h]
    simp

private theorem triangle_eventProbability_eq_setMass :
    eventProbability triangle3State 2 triangleHm triangleHb
        oneBirthSchedule newbornAdjacentToOneTwo =
      setMass triangle3State 2 ({1, 2} : Finset (Fin 3)) := by
  unfold eventProbability setMass
  change
    (∑ trace : Targets 3 2 × PUnit,
      if newbornAdjacentToOneTwo
          (traceFinal triangle3State 2 triangleHm triangleHb
            oneBirthSchedule trace)
      then traceProbability triangle3State 2 triangleHm triangleHb
        oneBirthSchedule trace
      else 0) = _
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro T _
  simp only [Fintype.sum_unique]
  have hunit : (default : PUnit) = PUnit.unit := Subsingleton.elim _ _
  rw [hunit, oneBirth_traceProbability T]
  have he := newbornAdjacentToOneTwo_iff_selected T
  by_cases h : T.selected = ({1, 2} : Finset (Fin 3))
  · rw [if_pos (he.mpr h), if_pos h]
  · rw [if_neg (fun hev => h (he.mp hev)), if_neg h]

/-- Both target orders `(1,2)` and `(2,1)` contribute, so the final-graph event
    has exact probability `2/6 = 1/3`. -/
example :
    eventProbability triangle3State 2 triangleHm triangleHb
      oneBirthSchedule newbornAdjacentToOneTwo = 1 / 3 := by
  rw [triangle_eventProbability_eq_setMass]
  decide_cbv

#print axioms NarrativeDynamics.FitnessAttachment.DistributionNetworkFixtures.executableDiameterAtMostTwo_iff_meshDiameter_le_two

end NarrativeDynamics.FitnessAttachment.DistributionNetworkFixtures
