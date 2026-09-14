import NarrativeDynamics
import NarrativeDynamics.Core.SmallWorldMetrics

#check NarrativeDynamics.FitnessABM.replay_projection

namespace NarrativeDynamics.FitnessABM.DistributionFixtures

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open scoped BigOperators

private theorem seed2_connected : (⊤ : SimpleGraph (Fin 2)).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v; exact ⟨.nil⟩
    · exact ⟨.cons h .nil⟩
  nonempty := inferInstance

def seed2 : JointState 2 where
  network :=
    { snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![1, 1] }
      valid := ⟨by decide, seed2_connected, by
        intro i
        fin_cases i <;> norm_num⟩ }
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨0, 0⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

def birthOne : BirthData :=
  { fitness := ⟨1, by norm_num⟩
    agent :=
      { profile := ⟨1, 1/2⟩, initialBelief := 0
        profileValid := by norm_num [AgentProfile.Valid]
        beliefValid := by norm_num } }

def newbornBroadcasts (out : RunState) : Bool :=
  if h : 2 < out.nodeCount then
    NetworkPropagation.broadcasting
      (out.state.population.profiles ⟨2, h⟩)
      (out.state.population.agents ⟨2, h⟩)
  else false

private def seedWeighted : JointState 2 :=
  { seed2 with network :=
      { snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![1, 3] }
        valid := ⟨by decide, seed2_connected, by
          intro i
          fin_cases i <;> norm_num⟩ } }

private def seedFor (weighted : Bool) : JointState 2 :=
  if weighted then seedWeighted else seed2

private def target {n : Nat} (i : Fin n) : Targets n 1 :=
  ⟨fun _ => i, fun _ _ _ => Subsingleton.elim _ _⟩

private theorem target_selected {n : Nat} (i : Fin n) :
    (target i).selected = {i} := by
  ext j
  simp only [Targets.selected, Finset.mem_image, Finset.mem_univ, true_and,
    Finset.mem_singleton]
  constructor
  · rintro ⟨a, h⟩
    exact h.symm
  · intro h
    exact ⟨0, h.symm⟩

private def oneTrace (i : Fin 2) : TargetTrace 2 1 1 := (target i, PUnit.unit)

private theorem oneTrace_univ :
    (Finset.univ : Finset (TargetTrace 2 1 1)) = {oneTrace 0, oneTrace 1} := by
  ext trace
  simp only [Finset.mem_univ, true_iff, Finset.mem_insert, Finset.mem_singleton]
  rcases trace with ⟨T, z⟩
  have ht : T = target (T 0) := by
    ext i
    have hi : i = (0 : Fin 1) := Subsingleton.elim _ _
    subst i
    rfl
  cases z
  rw [ht]
  generalize T (0 : Fin 1) = i
  fin_cases i <;> simp [oneTrace]

private theorem joint_eq {n : Nat} (s t : JointState n)
    (hn : s.network = t.network)
    (hp : s.population.profiles = t.population.profiles)
    (ha : s.population.agents = t.population.agents) : s = t := by
  rcases s with ⟨sn, ⟨sp, sa⟩, sv⟩
  rcases t with ⟨tn, ⟨tp, ta⟩, tv⟩
  dsimp only at hn hp ha
  cases hn
  cases hp
  cases ha
  rfl

private theorem advance_agents {n : Nat} (s : JointState n)
    (received : Fin n → Finset (Fin n))
    (h : letI := s.network.snapshot.adjDec
      incoming s.network.snapshot.graph.Adj s.population = received) :
    (advance s).population.agents = fun i =>
      if (received i).card = 0 then s.population.agents i
      else
        { belief := (1 - (s.population.profiles i).receptivity) *
              (s.population.agents i).belief +
            (s.population.profiles i).receptivity *
              ((∑ j ∈ received i, (s.population.agents j).belief) /
                ((received i).card : Rat))
          exposures := (s.population.agents i).exposures + (received i).card } := by
  letI := s.network.snapshot.adjDec
  change nextAgent s.network.snapshot.graph.Adj s.population = _
  funext i
  simp only [nextAgent, h]

private theorem lastCases_eq_if {α : Sort u} {n : Nat}
    (last : α) (old : Fin n → α) (i : Fin (n + 1)) :
    Fin.lastCases last old i =
      if h : i.val < n then old ⟨i.val, h⟩ else last := by
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simp
  · simp [j.isLt]

section FiniteFixtures
set_option maxRecDepth 4096

-- Flat round states make agent observations independent of graph proof reduction.
private def firstRound (weighted : Bool) (i : Fin 2) : JointState 3 where
  network := (grow (seedFor weighted) (target i) (by decide) birthOne).network
  population := ⟨fun _ => ⟨1, 1/2⟩,
    ![⟨1,0⟩, ⟨1,1⟩, if i = 0 then ⟨1,1⟩ else ⟨0,0⟩]⟩
  populationValid := by
    constructor
    · intro j
      norm_num [AgentProfile.Valid]
    · intro j
      fin_cases i <;> fin_cases j <;> norm_num [AgentState.Valid]

private theorem first_incoming (weighted : Bool) (i : Fin 2) :
    let s := grow (seedFor weighted) (target i) (by decide) birthOne
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, if i = 0 then {0} else ∅] : Fin 3 → Finset (Fin 3)) := by
  letI := (grow (seedFor weighted) (target i) (by decide) birthOne).network.snapshot.adjDec
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext a
  ext b
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  cases weighted <;> fin_cases i <;> fin_cases a <;> fin_cases b <;>
    (simp only [seedFor, seedWeighted, seed2, grow, applyBirth, birthSnapshot,
       birthGraph, birthAdj, lastCases_eq_if, extendPopulation,
       birthOne, target_selected, broadcasting]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem first_step (weighted : Bool) (i : Fin 2) :
    advance (grow (seedFor weighted) (target i) (by decide) birthOne) =
      firstRound weighted i := by
  apply joint_eq
  · rfl
  · funext j
    cases weighted <;> fin_cases i <;> fin_cases j <;> decide_cbv
  · rw [advance_agents _ _ (first_incoming weighted i)]
    funext j
    cases weighted <;> fin_cases i <;> fin_cases j <;> decide_cbv

private theorem one_final (weighted : Bool) (i : Fin 2) :
    jointFinal (seedFor weighted) 1 (by decide) (by decide) [some birthOne]
      (oneTrace i) = ⟨3, 1, firstRound weighted i⟩ := by
  change (runTyped (seedFor weighted) (by decide) (by decide)
    (.birth (target i) birthOne .nil) 0).final = _
  simp only [runTyped]
  rw [first_step]

private theorem one_broadcasts (weighted : Bool) (i : Fin 2) :
    newbornBroadcasts (jointFinal (seedFor weighted) 1 (by decide) (by decide)
      [some birthOne] (oneTrace i)) = decide (i = 0) := by
  rw [one_final]
  fin_cases i <;> decide_cbv

private theorem unit_mass (i : Fin 2) :
    jointProbability seed2 1 (by decide) (by decide) [some birthOne]
      (oneTrace i) = 1/2 := by
  fin_cases i <;> decide_cbv

private theorem weighted_mass_zero :
    jointProbability seedWeighted 1 (by decide) (by decide) [some birthOne]
      (oneTrace 0) = 1/4 := by decide_cbv

private theorem weighted_mass_one :
    jointProbability seedWeighted 1 (by decide) (by decide) [some birthOne]
      (oneTrace 1) = 3/4 := by decide_cbv

-- Catches altered attachment probabilities or a newborn forwarding within its birth round.
theorem newborn_mass_unit : eventProbability seed2 1 (by decide) (by decide)
    [some birthOne] (fun out => newbornBroadcasts out = true) = 1/2 := by
  unfold eventProbability
  change (∑ trace : TargetTrace 2 1 1, _) = _
  rw [oneTrace_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
  have h0 := one_broadcasts false 0
  have h1 := one_broadcasts false 1
  change newbornBroadcasts (jointFinal seed2 1 (by decide) (by decide) [some birthOne] (oneTrace 0)) = true at h0
  change newbornBroadcasts (jointFinal seed2 1 (by decide) (by decide) [some birthOne] (oneTrace 1)) = false at h1
  simp only [h0, h1, Bool.false_eq_true, if_false, if_true]
  simp [unit_mass]

-- Only fitness changed: the chosen-graph propagation result above is shared.
theorem newborn_mass_weighted : eventProbability seedWeighted 1 (by decide) (by decide)
    [some birthOne] (fun out => newbornBroadcasts out = true) = 1/4 := by
  unfold eventProbability
  change (∑ trace : TargetTrace 2 1 1, _) = _
  rw [oneTrace_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
  have h0 := one_broadcasts true 0
  have h1 := one_broadcasts true 1
  change newbornBroadcasts (jointFinal seedWeighted 1 (by decide) (by decide) [some birthOne] (oneTrace 0)) = true at h0
  change newbornBroadcasts (jointFinal seedWeighted 1 (by decide) (by decide) [some birthOne] (oneTrace 1)) = false at h1
  simp only [h0, h1, Bool.false_eq_true, if_false, if_true]
  simp [weighted_mass_zero]

private def idleRound (i : Fin 2) : JointState 3 where
  network := (firstRound false i).network
  population := ⟨fun _ => ⟨1, 1/2⟩,
    ![⟨1, if i = 0 then 2 else 1⟩, ⟨1,2⟩, ⟨1, if i = 0 then 2 else 1⟩]⟩
  populationValid := by
    constructor
    · intro j
      norm_num [AgentProfile.Valid]
    · intro j
      fin_cases j <;> norm_num [AgentState.Valid]

private theorem idle_incoming (i : Fin 2) :
    letI := (firstRound false i).network.snapshot.adjDec
    incoming (firstRound false i).network.snapshot.graph.Adj
      (firstRound false i).population =
      (![if i = 0 then {1,2} else {1}, {0}, if i = 0 then {0} else {1}] :
        Fin 3 → Finset (Fin 3)) := by
  letI := (firstRound false i).network.snapshot.adjDec
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext a
  ext b
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases a <;> fin_cases b <;>
    (simp only [firstRound, seedFor, seed2, grow, applyBirth, birthSnapshot,
       birthGraph, birthAdj, lastCases_eq_if, birthOne,
       target_selected, broadcasting]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem idle_step (i : Fin 2) : advance (firstRound false i) = idleRound i := by
  apply joint_eq
  · rfl
  · rfl
  · rw [advance_agents _ _ (idle_incoming i)]
    funext j
    fin_cases i <;> fin_cases j <;> decide_cbv

private theorem idle_broadcasts (i : Fin 2) :
    newbornBroadcasts (jointFinal seed2 1 (by decide) (by decide)
      [some birthOne, none] (oneTrace i)) = true := by
  change newbornBroadcasts
    ⟨3, 2, advance (advance (grow (seedFor false) (target i) (by decide) birthOne))⟩ = true
  rw [first_step, idle_step]
  fin_cases i <;> decide_cbv

theorem newborn_mass_after_idle : eventProbability seed2 1 (by decide) (by decide)
    [some birthOne, none] (fun out => newbornBroadcasts out = true) = 1 := by
  unfold eventProbability
  change (∑ trace : TargetTrace 2 1 1, _) = _
  rw [oneTrace_univ, Finset.sum_insert (by decide), Finset.sum_singleton]
  simp only [idle_broadcasts, if_true]
  change jointProbability seed2 1 (by decide) (by decide) [some birthOne] (oneTrace 0) +
    jointProbability seed2 1 (by decide) (by decide) [some birthOne] (oneTrace 1) = 1
  rw [unit_mass, unit_mass]
  norm_num

-- The chosen second target is the first newborn, whose frozen weight is one.
private def twoTrace : TargetTrace 2 1 2 :=
  (target 1, (target (2 : Fin 3), PUnit.unit))

private theorem first_newborn_mass :
    orderedMass (firstRound false 1).network (target (2 : Fin 3)) = 1/4 := by
  decide_cbv

example : (runTyped seed2 (by decide) (by decide)
    (scheduleOfTrace [some birthOne, some birthOne] twoTrace)).probability = 1/8 := by
  rw [← jointProbability_eq_runTyped]
  change orderedMass seed2.network (target 1) *
    (orderedMass (firstRound false 1).network (target (2 : Fin 3)) * 1) = 1/8
  have h : orderedMass seed2.network (target 1) = 1/2 := by decide_cbv
  rw [h, first_newborn_mass]
  norm_num

-- Empty and idle-only calendars share the existing one-inhabitant trace carrier.
example : Fintype.card (TargetTrace 2 1 (fitnessSchedule []).length) = 1 := by decide_cbv
example : Fintype.card (TargetTrace 2 1 (fitnessSchedule [none, none]).length) = 1 := by
  decide_cbv
example (trace : TargetTrace 2 1 (fitnessSchedule [none, none]).length) :
    trace = PUnit.unit := by cases trace; rfl
example : (jointFinal seed2 1 (by decide) (by decide) [] PUnit.unit).roundIndex = 0 := rfl
example : jointProbability seed2 1 (by decide) (by decide) [] PUnit.unit = 1 := rfl
example : jointProbability seed2 1 (by decide) (by decide) [none, none] PUnit.unit = 1 := rfl
example (event : RunState → Prop) [DecidablePred event] :
    eventProbability seed2 1 (by decide) (by decide) [none, none] event = 0 ∨
      eventProbability seed2 1 (by decide) (by decide) [none, none] event = 1 := by
  simp only [eventProbability, fitnessSchedule, List.length_nil]
  change (∑ trace : PUnit, _) = 0 ∨ (∑ trace : PUnit, _) = 1
  simp only [Fintype.sum_unique, jointProbability, fitnessSchedule, traceProbability]
  split <;> simp

-- Both orderings survive even though their selected sets (and final graphs) agree.
private def forward : Targets 2 2 := ⟨![0,1], by decide⟩
private def backward : Targets 2 2 := ⟨![1,0], by decide⟩
private def forwardTrace : TargetTrace 2 2 1 := (forward, PUnit.unit)
private def backwardTrace : TargetTrace 2 2 1 := (backward, PUnit.unit)
example : forward.selected = backward.selected := by decide_cbv
example : forwardTrace ≠ backwardTrace := by decide_cbv
example : Fintype.card (TargetTrace 2 2 1) = 2 := by decide_cbv
example : jointProbability seedWeighted 2 (by decide) (by decide) [some birthOne]
    forwardTrace = 1/4 := by decide_cbv
example : jointProbability seedWeighted 2 (by decide) (by decide) [some birthOne]
    backwardTrace = 3/4 := by decide_cbv
example : eventProbability seedWeighted 2 (by decide) (by decide) [some birthOne]
    (fun _ => True) = 1 := eventProbability_true _ _ _ _ _

private theorem walk_zero_eq {Node : Type*} {g : MeshGraph Node} {a b : Node}
    (walk : MeshWalk g 0 a b) : a = b := by
  cases walk
  rfl

private theorem walk_one_adj {Node : Type*} {g : MeshGraph Node} {a b : Node}
    (walk : MeshWalk g 1 a b) : g a b := by
  cases walk with
  | step edge rest =>
      have h := walk_zero_eq rest
      cases h
      exact edge

-- Actual final graphs both have diameter two, but the newborn states differ.
private theorem one_diameter (i : Fin 2) :
    meshDiameter (firstRound false i).network.snapshot.graph.Adj
      ⟨2, by simpa using state_bounded (firstRound false i).network⟩ = 2 := by
  let s := (firstRound false i).network
  have upper : meshDiameter s.snapshot.graph.Adj
      ⟨2, by simpa using state_bounded s⟩ ≤ 2 :=
    meshDiameter_minimal _ _ (by simpa using state_bounded s)
  have noEdge : ¬ s.snapshot.graph.Adj
      (if i = 0 then (1 : Fin 3) else 0) 2 := by
    fin_cases i
    · change ¬ (applyBirth seed2.network (target 0) (by decide) birthOne.fitness).snapshot.graph.Adj
        (oldId 2 (1 : Fin 2)) (newId 2)
      rw [birth_new_adj_iff, target_selected]
      decide
    · change ¬ (applyBirth seed2.network (target 1) (by decide) birthOne.fitness).snapshot.graph.Adj
        (oldId 2 (0 : Fin 2)) (newId 2)
      rw [birth_new_adj_iff, target_selected]
      decide
  have different : (if i = 0 then (1 : Fin 3) else 0) ≠ 2 := by
    fin_cases i <;> decide
  have noOne : ¬ ReachWithin s.snapshot.graph.Adj 1
      (if i = 0 then (1 : Fin 3) else 0) 2 := by
    rintro ⟨length, hlength, walk⟩
    have casesLength : length = 0 ∨ length = 1 := by omega
    rcases casesLength with h | h
    · subst length
      exact different (walk_zero_eq walk)
    · subst length
      exact noEdge (walk_one_adj walk)
  have lower : 2 ≤ meshDiameter s.snapshot.graph.Adj
      ⟨2, by simpa using state_bounded s⟩ := by
    by_contra h
    apply noOne
    exact reachWithin_mono
      (meshDiameter_spec s.snapshot.graph.Adj
        ⟨2, by simpa using state_bounded s⟩ _ _) (by omega)
  exact Nat.le_antisymm upper lower

example (i : Fin 2) :
    let out := jointFinal seed2 1 (by decide) (by decide) [some birthOne] (oneTrace i)
    meshDiameter out.state.network.snapshot.graph.Adj
      ⟨out.nodeCount - 1, state_bounded out.state.network⟩ = 2 := by
  have h := one_final false i
  change jointFinal seed2 1 (by decide) (by decide) [some birthOne] (oneTrace i) = _ at h
  dsimp only
  rw [h]
  exact one_diameter i

example :
    newbornBroadcasts (jointFinal seed2 1 (by decide) (by decide) [some birthOne]
      (oneTrace 0)) = true ∧
    newbornBroadcasts (jointFinal seed2 1 (by decide) (by decide) [some birthOne]
      (oneTrace 1)) = false := ⟨one_broadcasts false 0, one_broadcasts false 1⟩


example :
    List.ofFn (jointFinal seed2 1 (by decide) (by decide) [some birthOne]
      (oneTrace 0)).state.population.agents = [⟨1,0⟩, ⟨1,1⟩, ⟨1,1⟩] ∧
    List.ofFn (jointFinal seed2 1 (by decide) (by decide) [some birthOne]
      (oneTrace 1)).state.population.agents = [⟨1,0⟩, ⟨1,1⟩, ⟨0,0⟩] := by
  change List.ofFn (jointFinal (seedFor false) 1 (by decide) (by decide) [some birthOne] (oneTrace 0)).state.population.agents = _ ∧
    List.ofFn (jointFinal (seedFor false) 1 (by decide) (by decide) [some birthOne] (oneTrace 1)).state.population.agents = _
  rw [one_final, one_final]
  decide_cbv

end FiniteFixtures

#print axioms NarrativeDynamics.FitnessABM.jointProbability_eq_runTyped
#print axioms NarrativeDynamics.FitnessABM.jointFinal_projection
#print axioms NarrativeDynamics.FitnessABM.jointProbability_sum_one
#print axioms NarrativeDynamics.FitnessABM.eventProbability_true
#print axioms NarrativeDynamics.FitnessABM.eventProbability_false
#print axioms NarrativeDynamics.FitnessABM.eventProbability_nonneg
#print axioms NarrativeDynamics.FitnessABM.eventProbability_le_one
#print axioms newborn_mass_unit
#print axioms newborn_mass_weighted
#print axioms newborn_mass_after_idle

end NarrativeDynamics.FitnessABM.DistributionFixtures
