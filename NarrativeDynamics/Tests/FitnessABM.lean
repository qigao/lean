import NarrativeDynamics.Core.FitnessABM

namespace NarrativeDynamics.Tests.FitnessABM

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open NarrativeDynamics.FitnessABM
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


def newbornZero : NewAgent :=
  { profile := ⟨1, 1/2⟩, initialBelief := 0
    profileValid := by norm_num [AgentProfile.Valid]
    beliefValid := by norm_num }
def birthOne : BirthData := ⟨⟨1, by norm_num⟩, newbornZero⟩
def target1 : Targets 2 1 := ⟨![1], by decide⟩
private theorem one_pos : 0 < (1 : Nat) := by decide
def joined := grow seed2 target1 one_pos birthOne

-- Equality of data fields transports the already checked validity proofs.
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

-- Substitute a proved incoming table before computing its finite mean.
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

section FiniteJointFixtures
-- Existing finite BB fixture budget; the core and heartbeat limit are unchanged.
set_option maxRecDepth 4096

private def round1 : JointState 3 where
  network := (joined).network
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,0⟩, ⟨1,1⟩, ⟨0,0⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

private theorem round1_incoming :
    letI := (joined).network.snapshot.adjDec
    incoming (joined).network.snapshot.graph.Adj (joined).population =
      (![∅, {0}, ∅] : Fin 3 → Finset (Fin 3)) := by
  letI := (joined).network.snapshot.adjDec
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    norm_num [joined, grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      Fin.lastCases, seed2, extendPopulation, birthOne, newbornZero,
      target1, Targets.selected, broadcasting]

private theorem round1_step : advance (joined) = round1 := by
  apply joint_eq
  · rfl
  · funext i
    fin_cases i <;> decide_cbv
  · rw [advance_agents (joined) _ round1_incoming]
    funext i
    fin_cases i <;> decide_cbv

private theorem round1_birth_step :
    advance (grow seed2 target1 one_pos birthOne) = round1 := round1_step

private def target2 : Targets 3 1 := ⟨![2], by decide⟩

private def round2 : JointState 4 where
  network := (grow round1 target2 one_pos birthOne).network
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,1⟩, ⟨1,2⟩, ⟨1,1⟩, ⟨0,0⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

private theorem round2_incoming :
    letI := (grow round1 target2 one_pos birthOne).network.snapshot.adjDec
    incoming (grow round1 target2 one_pos birthOne).network.snapshot.graph.Adj (grow round1 target2 one_pos birthOne).population =
      (![{1}, {0}, {1}, ∅] : Fin 4 → Finset (Fin 4)) := by
  letI := (grow round1 target2 one_pos birthOne).network.snapshot.adjDec
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    norm_num [round1, joined, grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      Fin.lastCases, seed2, extendPopulation, birthOne, newbornZero,
      target1, target2, Targets.selected, broadcasting]

private theorem round2_step : advance (grow round1 target2 one_pos birthOne) = round2 := by
  apply joint_eq
  · rfl
  · funext i
    fin_cases i <;> decide_cbv
  · rw [advance_agents (grow round1 target2 one_pos birthOne) _ round2_incoming]
    funext i
    fin_cases i <;> decide_cbv

private def round3 : JointState 4 where
  network := (round2).network
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,2⟩, ⟨1,4⟩, ⟨1,2⟩, ⟨1,1⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

private theorem round3_incoming :
    letI := (round2).network.snapshot.adjDec
    incoming (round2).network.snapshot.graph.Adj (round2).population =
      (![{1}, {0,2}, {1}, {2}] : Fin 4 → Finset (Fin 4)) := by
  letI := (round2).network.snapshot.adjDec
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    norm_num [round2, round1, joined, grow, applyBirth, birthSnapshot, birthGraph,
      birthAdj, Fin.lastCases, seed2, extendPopulation, birthOne, newbornZero,
      target1, target2, Targets.selected, broadcasting]

private theorem round3_step : advance (round2) = round3 := by
  apply joint_eq
  · rfl
  · funext i
    fin_cases i <;> decide_cbv
  · rw [advance_agents (round2) _ round3_incoming]
    funext i
    fin_cases i <;> decide_cbv

-- Catches loss of old IDs, nonzero newborn exposure, and in-round relay cascades.
example : joined.population.agents (oldId 2 0) = seed2.population.agents 0 := by
  exact grow_old_state seed2 target1 (by decide) birthOne 0
example : joined.population.agents (newId 2) = ⟨0,0⟩ := by decide_cbv
example : List.ofFn (fun i => ((advance joined).population.agents i).belief) =
    [1,1,0] := by
  rw [round1_step]
  decide_cbv


-- Profiles survive embedding, independently of their agents' changing beliefs.
example : joined.population.profiles (oldId 2 0) = seed2.population.profiles 0 := by
  exact grow_old_profile seed2 target1 (by decide) birthOne 0

def successiveBirths : Schedule 2 1 :=
  .birth target1 birthOne (.birth ⟨![2], by decide⟩ birthOne .nil)
def twoBirths := runTyped seed2 (by decide) (by decide) successiveBirths

private theorem seed_mass : orderedMass seed2.network target1 = 1/2 := by decide_cbv
private theorem round1_mass : orderedMass round1.network target2 = 1/4 := by decide_cbv

private theorem twoBirths_result : twoBirths = ⟨⟨4, 2, round2⟩, 1/8⟩ := by
  change runTyped seed2 one_pos (by decide)
    (.birth target1 birthOne (.birth target2 birthOne .nil)) 0 = _
  simp only [runTyped]
  rw [round1_birth_step, round2_step, seed_mass, round1_mass]
  norm_num

-- Catches stale attachment weights, omitted rounds, and asynchronous forwarding.
example : List.ofFn (fun i => (twoBirths.final.state.population.agents i).belief) =
    [1,1,1,0] := by
  rw [twoBirths_result]
  decide_cbv

example : List.ofFn (fun i => (twoBirths.final.state.population.agents i).exposures) =
    [1,2,1,0] := by
  rw [twoBirths_result]
  decide_cbv

example : twoBirths.final.nodeCount = 4 := by rw [twoBirths_result]
example : actualEdgeCount twoBirths.final.state.network.snapshot = 3 := by
  have h := (runTyped_counts seed2 (by decide) (by decide) successiveBirths).2.2
  have hseed : actualEdgeCount seed2.network.snapshot = 1 := by decide_cbv
  change actualEdgeCount twoBirths.final.state.network.snapshot = _ at h
  simpa only [successiveBirths, Schedule.birthCount, hseed,
    Nat.zero_add, Nat.add_zero, Nat.one_mul] using h
example : twoBirths.final.roundIndex = 2 := by rw [twoBirths_result]
example : twoBirths.probability = 1/8 := by rw [twoBirths_result]

def birthThenIdle : Schedule 2 1 :=
  .birth target1 birthOne (.birth ⟨![2], by decide⟩ birthOne (.idle .nil))
def thirdRound := runTyped seed2 (by decide) (by decide) birthThenIdle

private theorem thirdRound_result : thirdRound = ⟨⟨4, 3, round3⟩, 1/8⟩ := by
  change runTyped seed2 one_pos (by decide)
    (.birth target1 birthOne (.birth target2 birthOne (.idle .nil))) 0 = _
  simp only [runTyped]
  rw [round1_birth_step, round2_step, round3_step, seed_mass, round1_mass]
  norm_num

-- The last newborn receives on the following round; idle adds no probability.
example : List.ofFn (fun i => (thirdRound.final.state.population.agents i).belief) =
    [1,1,1,1] := by
  rw [thirdRound_result]
  decide_cbv

example : List.ofFn (fun i => (thirdRound.final.state.population.agents i).exposures) =
    [2,4,2,1] := by
  rw [thirdRound_result]
  decide_cbv

example : thirdRound.final.roundIndex = 3 := by rw [thirdRound_result]
example : thirdRound.probability = 1/8 := by rw [thirdRound_result]

def newbornOne : NewAgent :=
  { newbornZero with initialBelief := 1, beliefValid := by norm_num }
def broadcastingBirth : BirthData := ⟨⟨1, by norm_num⟩, newbornOne⟩
def immediate := tick seed2 (by decide) (by decide)
  (.birth target1 broadcastingBirth)

private def immediateState : JointState 3 where
  network := (grow seed2 target1 one_pos broadcastingBirth).network
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,0⟩, ⟨1,2⟩, ⟨1,0⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

private theorem immediateState_incoming :
    letI := (grow seed2 target1 one_pos broadcastingBirth).network.snapshot.adjDec
    incoming (grow seed2 target1 one_pos broadcastingBirth).network.snapshot.graph.Adj (grow seed2 target1 one_pos broadcastingBirth).population =
      (![∅, {0,2}, ∅] : Fin 3 → Finset (Fin 3)) := by
  letI := (grow seed2 target1 one_pos broadcastingBirth).network.snapshot.adjDec
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    norm_num [grow, applyBirth, birthSnapshot, birthGraph, birthAdj, Fin.lastCases,
      seed2, extendPopulation, broadcastingBirth, newbornOne, newbornZero,
      target1, Targets.selected, broadcasting]

private theorem immediateState_step : advance (grow seed2 target1 one_pos broadcastingBirth) = immediateState := by
  apply joint_eq
  · rfl
  · funext i
    fin_cases i <;> decide_cbv
  · rw [advance_agents (grow seed2 target1 one_pos broadcastingBirth) _ immediateState_incoming]
    funext i
    fin_cases i <;> decide_cbv

private theorem immediate_result : immediate = ⟨⟨3, 1, immediateState⟩, 1/2⟩ := by
  unfold immediate
  simp only [tick, runTyped]
  rw [immediateState_step, seed_mass]
  norm_num

-- Birth precedes propagation: a newborn above threshold broadcasts immediately.
example : List.ofFn (fun i => (immediate.final.state.population.agents i).belief) =
    [1,1,1] := by
  rw [immediate_result]
  decide_cbv
example : List.ofFn (fun i => (immediate.final.state.population.agents i).exposures) =
    [0,2,0] := by
  rw [immediate_result]
  decide_cbv

private def idle1 : JointState 2 where
  network := (seed2).network
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,0⟩, ⟨1,1⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

private theorem idle1_incoming :
    letI := (seed2).network.snapshot.adjDec
    incoming (seed2).network.snapshot.graph.Adj (seed2).population =
      (![∅, {0}] : Fin 2 → Finset (Fin 2)) := by
  letI := (seed2).network.snapshot.adjDec
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem idle1_step : advance (seed2) = idle1 := by
  apply joint_eq
  · rfl
  · funext i
    fin_cases i <;> decide_cbv
  · rw [advance_agents (seed2) _ idle1_incoming]
    funext i
    fin_cases i <;> decide_cbv

private def idle2 : JointState 2 where
  network := (idle1).network
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,1⟩, ⟨1,2⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

private theorem idle2_incoming :
    letI := (idle1).network.snapshot.adjDec
    incoming (idle1).network.snapshot.graph.Adj (idle1).population =
      (![{1}, {0}] : Fin 2 → Finset (Fin 2)) := by
  letI := (idle1).network.snapshot.adjDec
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem idle2_step : advance (idle1) = idle2 := by
  apply joint_eq
  · rfl
  · funext i
    fin_cases i <;> decide_cbv
  · rw [advance_agents (idle1) _ idle2_incoming]
    funext i
    fin_cases i <;> decide_cbv

def idleOnly := runTyped (m := 1) seed2 (by decide) (by decide)
  (.idle (.idle .nil)) 7
private theorem idleOnly_result : idleOnly = ⟨⟨2, 9, idle2⟩, 1⟩ := by
  unfold idleOnly
  simp only [runTyped]
  rw [idle1_step, idle2_step]

example : idleOnly.probability = 1 := by rw [idleOnly_result]
example : idleOnly.final.roundIndex = 9 := by rw [idleOnly_result]
example : idleOnly.final.nodeCount = 2 := by rw [idleOnly_result]
example : List.ofFn (fun i => (idleOnly.final.state.population.agents i).belief) =
    [1,1] := by
  rw [idleOnly_result]
  decide_cbv
example : (tick (m := 1) seed2 (by decide) (by decide) .idle 7).final.roundIndex =
    8 := rfl
example : (runTyped (m := 1) seed2 (by decide) (by decide) .nil 7).final.roundIndex =
    7 := rfl

-- Reordering a full target set changes neither topology nor the observed round.
def target01 : Targets 2 2 := ⟨![0,1], by decide⟩
def target10 : Targets 2 2 := ⟨![1,0], by decide⟩
example :
    (grow seed2 target01 (by decide) birthOne).network.snapshot.graph =
      (grow seed2 target10 (by decide) birthOne).network.snapshot.graph ∧
    ∀ i,
      (advance (grow seed2 target01 (by decide) birthOne)).population.profiles i =
        (advance (grow seed2 target10 (by decide) birthOne)).population.profiles i ∧
      ((advance (grow seed2 target01 (by decide) birthOne)).population.agents i).belief =
        ((advance (grow seed2 target10 (by decide) birthOne)).population.agents i).belief ∧
      ((advance (grow seed2 target01 (by decide) birthOne)).population.agents i).exposures =
        ((advance (grow seed2 target10 (by decide) birthOne)).population.agents i).exposures :=
  grow_order_irrelevant seed2 target01 target10 (by decide) birthOne (by decide_cbv)

#print axioms NarrativeDynamics.FitnessABM.extendPopulation_valid
#print axioms NarrativeDynamics.FitnessABM.grow_order_irrelevant
#print axioms NarrativeDynamics.FitnessABM.runTyped_counts
#print axioms NarrativeDynamics.FitnessABM.runTyped_probability_pos

#print axioms twoBirths_result
#print axioms thirdRound_result
#print axioms immediate_result
#print axioms idleOnly_result

end FiniteJointFixtures

end NarrativeDynamics.Tests.FitnessABM
