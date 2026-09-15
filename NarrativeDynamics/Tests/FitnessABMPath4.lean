import NarrativeDynamics.Core.FitnessABMPath4
import NarrativeDynamics.Tests.FitnessABMReplay
import NarrativeDynamics.Core.FitnessABMPath4Convergence

open NarrativeDynamics.NetworkPropagation NarrativeDynamics.FitnessABMPath4

example (x : Beliefs) (e : Fin 4 → Nat) (hx : allBroadcast x) :
    project (propagate pathAdj (population x e)) = linearStep x :=
  propagate_eq_linear x e hx

example (x : Beliefs) (e : Fin 4 → Nat) :
    project (propagate pathAdj (population x e)) = beliefStep x :=
  propagate_independent_exposures x e

example (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    allBroadcast (trajectory x n) :=
  allBroadcast_iterate x hx n

-- Node 3 is below the threshold: its belief must not enter node 2's mean.
example : beliefStep ![3/4, 11/16, 5/8, 1/4] =
    ![23/32, 11/16, 21/32, 7/16] := by
  decide_cbv

-- The symbolic contract catches losing any mode or applying the averaging
-- formula without maintaining the actual trajectory's broadcast region.
example (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    trajectory x n = closedForm x n := iterate_closedForm x hx n

example (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := mean_step x hx

-- The zero-eigenvalue mode contributes at n = 0, and the invariant is weighted.
example : closedForm ![45/64,44/64,43/64,35/64] 0 =
    ![45/64,44/64,43/64,35/64] := by decide_cbv

example : mean ![45/64,44/64,43/64,35/64] = 127/192 := by decide_cbv

namespace NarrativeDynamics.Tests.FitnessABMPath4
open NarrativeDynamics.NetworkPropagation NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal NarrativeDynamics.FitnessABM

inductive History where
  | bbii | bibi | iibb
  deriving DecidableEq

def rawSeed : RawSeed := ⟨2, #[1,1], #[(0,1)]⟩
def rawAgents : Array RawAgent := #[⟨1/2,1/2,1,0⟩, ⟨1/2,1/2,0,0⟩]
def rawBirth (target : Nat) : RawBirthInput := ⟨⟨1, #[target]⟩, 1/2, 1/2, 0⟩
def ticks : History → List RawTick
  | .bbii => [some (rawBirth 1), some (rawBirth 2), none, none]
  | .bibi => [some (rawBirth 1), none, some (rawBirth 2), none]
  | .iibb => [none, none, some (rawBirth 1), some (rawBirth 2)]
def rawTail (h : History) (k : Nat) :=
  FitnessABM.replay rawSeed 1 rawAgents (ticks h ++ List.replicate k none)

section FiniteReplayFixtures
-- Finite proof-bearing BB fixtures use the same recursion budget as FitnessABMReplay.
set_option maxRecDepth 4096
private def one : PosFitness := ⟨1, by norm_num⟩
private theorem positiveM : 0 < (1 : Nat) := by decide

private theorem top2_connected : (⊤ : SimpleGraph (Fin 2)).Connected where
  preconnected := by
    intro i j
    by_cases h : i = j
    · subst j; exact ⟨.nil⟩
    · exact ⟨.cons h .nil⟩
  nonempty := inferInstance

private def seedNetwork (eta : PosFitness) : FitnessAttachment.State 2 :=
  { snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![1, eta.val] }
    valid := ⟨by decide, top2_connected, by
      intro i
      fin_cases i
      · norm_num
      · exact eta.property⟩ }

private def initial (eta : PosFitness) : JointState 2 :=
  { network := seedNetwork eta
    population := ⟨fun _ => ⟨1/2, 1/2⟩, ![⟨1, 0⟩, ⟨0, 0⟩]⟩
    populationValid := by
      constructor
      · intro i; fin_cases i <;> norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private theorem network_eq {n : Nat} (s t : FitnessAttachment.State n)
    (hg : s.snapshot.graph = t.snapshot.graph)
    (hf : s.snapshot.fitness = t.snapshot.fitness) : s = t := by
  rcases s with ⟨⟨sg, sd, sf⟩, sv⟩
  rcases t with ⟨⟨tg, td, tf⟩, tv⟩
  dsimp only at hg hf
  cases hg
  cases hf
  have hd : sd = td := Subsingleton.elim _ _
  cases hd
  rfl

private theorem raw_graph (eta : PosFitness) :
    seedGraph ⟨2, #[1, eta.val], #[(0, 1)]⟩ = (⊤ : SimpleGraph (Fin 2)) := by
  ext i j
  change (i ≠ j ∧ canonicalEdge (i.val, j.val) ∈ [(0, 1)]) ↔ i ≠ j
  constructor
  · exact And.left
  · intro h
    refine ⟨h, ?_⟩
    fin_cases i <;> fin_cases j <;>
      first | exact False.elim (h rfl) | decide_cbv

private theorem parsed_seed (eta : PosFitness) :
    parseSeed ⟨2, #[1, eta.val], #[(0, 1)]⟩ = .ok (seedNetwork eta) := by
  let raw : RawSeed := ⟨2, #[1, eta.val], #[(0, 1)]⟩
  have valid : raw.Valid :=
    { nodes := by change 2 ≤ (2 : Nat); decide
      size := rfl
      fitness := by
        intro i
        fin_cases i
        · norm_num [raw]
        · exact eta.property
      edges := by
        dsimp only [validSeedEdges, raw]
        decide_cbv
      distinct := by
        dsimp only [raw]
        decide_cbv
      connected := by rw [raw_graph]; exact top2_connected }
  obtain ⟨s, hs⟩ := parseSeed_complete raw valid
  have he : s = seedNetwork eta := by
    apply network_eq
    · rw [parseSeed_graph raw s hs]
      exact raw_graph eta
    · obtain ⟨hsize, hsn⟩ := parseSeed_snapshot raw s hs
      rw [hsn]
      funext i
      fin_cases i <;> rfl
  exact hs.trans (congrArg Except.ok he)

private theorem parsed_rawSeed : parseSeed rawSeed = .ok (seedNetwork one) :=
  parsed_seed one

private theorem parsed_agents :
    parseAgents 2 rawAgents = .ok
      ⟨(initial one).population, (initial one).populationValid⟩ := by
  have valid : ∀ a ∈ rawAgents.toList, a.Valid := by
    intro a ha
    simp [rawAgents] at ha
    rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]
  obtain ⟨p, hp⟩ := parseAgents_complete 2 rawAgents rfl valid
  have values := (parseAgents_sound 2 rawAgents p hp).2.2
  have he : p.val = (initial one).population := by
    rcases p with ⟨⟨profiles, agents⟩, hv⟩
    have hprofiles : profiles = (initial one).population.profiles := by
      funext i
      have h := (values i (by have := i.isLt; simpa [rawAgents] using this)).1
      fin_cases i <;> simpa [rawAgents, initial] using h
    have hagents : agents = (initial one).population.agents := by
      funext i
      have h := (values i (by have := i.isLt; simpa [rawAgents] using this)).2
      fin_cases i <;> simpa [rawAgents, initial] using h
    cases hprofiles
    cases hagents
    rfl
  have hp' : p = ⟨(initial one).population, (initial one).populationValid⟩ :=
    Subtype.ext he
  exact hp.trans (congrArg Except.ok hp')

private theorem replay_start (eta : PosFitness) (m : Nat) (hm : 0 < m ∧ m ≤ 2)
    (ticks : List RawTick) :
    FitnessABM.replay ⟨2, #[1, eta.val], #[(0, 1)]⟩ m rawAgents ticks =
      runInputs m 0 0 ⟨2, 0, initial eta⟩ ticks := by
  simp only [FitnessABM.replay, parsed_seed, if_pos hm, parsed_agents]
  rfl

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


private def targets (n : Nat) (i : Fin n) : Targets n 1 :=
  ⟨fun _ => i, by intro a b h; exact Subsingleton.elim _ _⟩
private def birthData : BirthData :=
  ⟨one, ⟨⟨1/2,1/2⟩, 0, by norm_num [AgentProfile.Valid], by norm_num⟩⟩

private theorem checked {n : Nat} (s : JointState n) (i : Fin n) :
    checkedBirth s 1 (rawBirth i.val) = .ok
      (grow s (targets n i) positiveM birthData, orderedMass s.network (targets n i)) := by
  apply (checkedBirth_spec _ _ _).mpr
  refine ⟨⟨targets n i, one, positiveM⟩,
    ⟨(⟨1/2,1/2⟩, ⟨0,0⟩), by norm_num [AgentProfile.Valid, AgentState.Valid]⟩,
    ?_, ?_, rfl⟩
  · have hm : 0 < (1 : Nat) ∧ 1 ≤ n := ⟨by decide, by have := i.isLt; omega⟩
    have hf : 0 < (rawBirth i.val).birth.fitness := by norm_num [rawBirth]
    have hs : (rawBirth i.val).birth.targets.size = 1 := rfl
    have hb : targetsBounded n (rawBirth i.val).birth.targets := by
      simp [targetsBounded, rawBirth, i.isLt]
    have hd : targetsDistinct (rawBirth i.val).birth.targets := by
      simp only [targetsDistinct, rawBirth]
      intro a b h
      apply Fin.ext
      have ha := a.isLt
      have hb := b.isLt
      change a.val < 1 at ha
      change b.val < 1 at hb
      omega
    have he : (⟨hb, hd⟩ : CheckedTargets n (rawBirth i.val).birth.targets).embedding =
        targets n i := by
      apply Function.Embedding.ext
      intro j
      fin_cases j
      rfl
    simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
      checkTargets, dif_pos hb, dif_pos hd]
    rw [he]
    rfl
  · change parseAgent ⟨1/2,1/2,0,0⟩ = _
    norm_num [parseAgent]

private theorem checked_first (s : JointState 2) :
    checkedBirth s 1 (rawBirth 1) = .ok
      (grow s (targets 2 1) positiveM birthData, orderedMass s.network (targets 2 1)) :=
  checked s 1
private theorem checked_second (s : JointState 3) :
    checkedBirth s 1 (rawBirth 2) = .ok
      (grow s (targets 3 2) positiveM birthData, orderedMass s.network (targets 3 2)) :=
  checked s 2

private def pathGraph : SimpleGraph (Fin 4) where
  Adj := NarrativeDynamics.FitnessABMPath4.pathAdj
  symm := ⟨by intro i j h; exact h.symm⟩
  loopless := ⟨by intro i h; rcases h with h | h <;> omega⟩

private instance : DecidableRel pathGraph.Adj :=
  inferInstanceAs (DecidableRel NarrativeDynamics.FitnessABMPath4.pathAdj)

private theorem path_connected : pathGraph.Connected where
  preconnected := by
    intro i j
    fin_cases i <;> fin_cases j
    · exact ⟨.nil⟩
    · exact ⟨.cons (show pathGraph.Adj 0 1 from by decide) (.nil)⟩
    · exact ⟨.cons (show pathGraph.Adj 0 1 from by decide) (.cons (show pathGraph.Adj 1 2 from by decide) (.nil))⟩
    · exact ⟨.cons (show pathGraph.Adj 0 1 from by decide) (.cons (show pathGraph.Adj 1 2 from by decide) (.cons (show pathGraph.Adj 2 3 from by decide) (.nil)))⟩
    · exact ⟨.cons (show pathGraph.Adj 1 0 from by decide) (.nil)⟩
    · exact ⟨.nil⟩
    · exact ⟨.cons (show pathGraph.Adj 1 2 from by decide) (.nil)⟩
    · exact ⟨.cons (show pathGraph.Adj 1 2 from by decide) (.cons (show pathGraph.Adj 2 3 from by decide) (.nil))⟩
    · exact ⟨.cons (show pathGraph.Adj 2 1 from by decide) (.cons (show pathGraph.Adj 1 0 from by decide) (.nil))⟩
    · exact ⟨.cons (show pathGraph.Adj 2 1 from by decide) (.nil)⟩
    · exact ⟨.nil⟩
    · exact ⟨.cons (show pathGraph.Adj 2 3 from by decide) (.nil)⟩
    · exact ⟨.cons (show pathGraph.Adj 3 2 from by decide) (.cons (show pathGraph.Adj 2 1 from by decide) (.cons (show pathGraph.Adj 1 0 from by decide) (.nil)))⟩
    · exact ⟨.cons (show pathGraph.Adj 3 2 from by decide) (.cons (show pathGraph.Adj 2 1 from by decide) (.nil))⟩
    · exact ⟨.cons (show pathGraph.Adj 3 2 from by decide) (.nil)⟩
    · exact ⟨.nil⟩
  nonempty := inferInstance

private def pathNetwork : FitnessAttachment.State 4 :=
  { snapshot := ⟨pathGraph, inferInstanceAs (DecidableRel NarrativeDynamics.FitnessABMPath4.pathAdj), fun _ => 1⟩
    valid := ⟨by decide, path_connected, by intro i; norm_num⟩ }

def baselineBeliefs : History → NarrativeDynamics.FitnessABMPath4.Beliefs
  | .bbii | .bibi => ![3/4,11/16,5/8,1/4]
  | .iibb => ![3/4,3/4,9/16,0]
def baselineExposures : History → Fin 4 → Nat
  | .bbii | .bibi => ![3,5,3,1]
  | .iibb => ![3,4,2,0]
def baseState (h : History) : JointState 4 :=
  { network := pathNetwork
    population := NarrativeDynamics.FitnessABMPath4.population (baselineBeliefs h) (baselineExposures h)
    populationValid := by
      constructor
      · intro i; norm_num [NarrativeDynamics.FitnessABMPath4.population, AgentProfile.Valid]
      · intro i; cases h <;> fin_cases i <;>
          norm_num [NarrativeDynamics.FitnessABMPath4.population, baselineBeliefs, AgentState.Valid] }

def tailState (h : History) (k : Nat) : JointState 4 := advance^[k] (baseState h)
def tailBelief (h : History) (k : Nat) : NarrativeDynamics.FitnessABMPath4.Beliefs := NarrativeDynamics.FitnessABMPath4.project (tailState h k).population

private def network3 : FitnessAttachment.State 3 :=
  (grow (initial one) (targets 2 1) positiveM birthData).network
private def network4 : FitnessAttachment.State 4 :=
  applyBirth network3 (targets 3 2) positiveM one

private def stage_a : JointState 2 :=
  { network := (initial one).network
    population := ⟨fun _ => ⟨1/2,1/2⟩, ![⟨1,0⟩,⟨1/2,1⟩]⟩
    populationValid := by
      constructor
      · intro i; norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def stage_b : JointState 2 :=
  { network := (initial one).network
    population := ⟨fun _ => ⟨1/2,1/2⟩, ![⟨3/4,1⟩,⟨3/4,2⟩]⟩
    populationValid := by
      constructor
      · intro i; norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def stage_c : JointState 3 :=
  { network := network3
    population := ⟨fun _ => ⟨1/2,1/2⟩, ![⟨1,0⟩,⟨1/2,1⟩,⟨0,0⟩]⟩
    populationValid := by
      constructor
      · intro i; norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def stage_d : JointState 3 :=
  { network := network3
    population := ⟨fun _ => ⟨1/2,1/2⟩, ![⟨3/4,1⟩,⟨3/4,2⟩,⟨1/4,1⟩]⟩
    populationValid := by
      constructor
      · intro i; norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def stage_e : JointState 3 :=
  { network := network3
    population := ⟨fun _ => ⟨1/2,1/2⟩, ![⟨3/4,2⟩,⟨3/4,3⟩,⟨3/8,1⟩]⟩
    populationValid := by
      constructor
      · intro i; norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def stage_f : JointState 4 :=
  { network := network4
    population := ⟨fun _ => ⟨1/2,1/2⟩, ![⟨3/4,1⟩,⟨3/4,2⟩,⟨1/4,1⟩,⟨0,0⟩]⟩
    populationValid := by
      constructor
      · intro i; norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def stage_g : JointState 4 :=
  { network := network4
    population := ⟨fun _ => ⟨1/2,1/2⟩, ![⟨3/4,2⟩,⟨3/4,3⟩,⟨1/2,2⟩,⟨0,0⟩]⟩
    populationValid := by
      constructor
      · intro i; norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private theorem lastCases_eq_if {α : Sort u} {n : Nat}
    (last : α) (old : Fin n → α) (i : Fin (n + 1)) :
    Fin.lastCases last old i =
      if h : i.val < n then old ⟨i.val, h⟩ else last := by
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simp
  · simp [j.isLt]

private theorem network3_adj (i j : Fin 3) :
    network3.snapshot.graph.Adj i j ↔ i.val+1 = j.val ∨ j.val+1 = i.val := by
  fin_cases i <;> fin_cases j <;>
    norm_num [network3, grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      lastCases_eq_if, initial, seedNetwork, targets, Targets.selected, Fin.ext_iff]

private theorem network4_adj (i j : Fin 4) :
    network4.snapshot.graph.Adj i j ↔ i.val+1 = j.val ∨ j.val+1 = i.val := by
  fin_cases i <;> fin_cases j
  · change network3.snapshot.graph.Adj 0 0 ↔ _
    exact network3_adj 0 0
  · change network3.snapshot.graph.Adj 0 1 ↔ _
    exact network3_adj 0 1
  · change network3.snapshot.graph.Adj 0 2 ↔ _
    exact network3_adj 0 2
  · change (0 : Fin 3) ∈ (targets 3 2).selected ↔ _
    decide_cbv
  · change network3.snapshot.graph.Adj 1 0 ↔ _
    exact network3_adj 1 0
  · change network3.snapshot.graph.Adj 1 1 ↔ _
    exact network3_adj 1 1
  · change network3.snapshot.graph.Adj 1 2 ↔ _
    exact network3_adj 1 2
  · change (1 : Fin 3) ∈ (targets 3 2).selected ↔ _
    decide_cbv
  · change network3.snapshot.graph.Adj 2 0 ↔ _
    exact network3_adj 2 0
  · change network3.snapshot.graph.Adj 2 1 ↔ _
    exact network3_adj 2 1
  · change network3.snapshot.graph.Adj 2 2 ↔ _
    exact network3_adj 2 2
  · change (2 : Fin 3) ∈ (targets 3 2).selected ↔ _
    decide_cbv
  · change (0 : Fin 3) ∈ (targets 3 2).selected ↔ _
    decide_cbv
  · change (1 : Fin 3) ∈ (targets 3 2).selected ↔ _
    decide_cbv
  · change (2 : Fin 3) ∈ (targets 3 2).selected ↔ _
    decide_cbv
  · change False ↔ _
    decide_cbv

private theorem network4_eq : network4 = pathNetwork := by
  apply network_eq
  · ext i j
    exact network4_adj i j
  · funext i
    fin_cases i <;> decide_cbv

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

-- Every transition is checked coordinatewise on the actual state. Graph and
-- profile equalities are separate from finite agent-state evaluation.
private theorem incoming_a :
    let s := initial one
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅,{0}] : Fin 2 → Finset (Fin 2)) := by
  have hg : ∀ i j, (initial one).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := by
    intro i j
    change (i ≠ j) ↔ _
    fin_cases i <;> fin_cases j <;> decide_cbv
  have hb : ∀ j, broadcasting ((initial one).population.profiles j)
      ((initial one).population.agents j) = decide ((1/2 : Rat) ≤ ![1,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_a : advance (initial one) = stage_a := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_a]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_b :
    let s := stage_a
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0}] : Fin 2 → Finset (Fin 2)) := by
  have hg : ∀ i j, (stage_a).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := by
    intro i j
    change (i ≠ j) ↔ _
    fin_cases i <;> fin_cases j <;> decide_cbv
  have hb : ∀ j, broadcasting ((stage_a).population.profiles j)
      ((stage_a).population.agents j) = decide ((1/2 : Rat) ≤ ![1,1/2] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_b : advance (stage_a) = stage_b := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_b]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_c :
    let s := grow (initial one) (targets 2 1) positiveM birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅,{0},∅] : Fin 3 → Finset (Fin 3)) := by
  have hg : ∀ i j, (grow (initial one) (targets 2 1) positiveM birthData).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network3_adj i j
  have hb : ∀ j, broadcasting ((grow (initial one) (targets 2 1) positiveM birthData).population.profiles j)
      ((grow (initial one) (targets 2 1) positiveM birthData).population.agents j) = decide ((1/2 : Rat) ≤ ![1,0,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_c : advance (grow (initial one) (targets 2 1) positiveM birthData) = stage_c := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_c]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_d :
    let s := stage_c
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0},{1}] : Fin 3 → Finset (Fin 3)) := by
  have hg : ∀ i j, (stage_c).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network3_adj i j
  have hb : ∀ j, broadcasting ((stage_c).population.profiles j)
      ((stage_c).population.agents j) = decide ((1/2 : Rat) ≤ ![1,1/2,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_d : advance (stage_c) = stage_d := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_d]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_e :
    let s := grow stage_b (targets 2 1) positiveM birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0},{1}] : Fin 3 → Finset (Fin 3)) := by
  have hg : ∀ i j, (grow stage_b (targets 2 1) positiveM birthData).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network3_adj i j
  have hb : ∀ j, broadcasting ((grow stage_b (targets 2 1) positiveM birthData).population.profiles j)
      ((grow stage_b (targets 2 1) positiveM birthData).population.agents j) = decide ((1/2 : Rat) ≤ ![3/4,3/4,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_e : advance (grow stage_b (targets 2 1) positiveM birthData) = stage_e := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_e]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_f :
    let s := grow stage_c (targets 3 2) positiveM birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0},{1},∅] : Fin 4 → Finset (Fin 4)) := by
  have hg : ∀ i j, (grow stage_c (targets 3 2) positiveM birthData).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network4_adj i j
  have hb : ∀ j, broadcasting ((grow stage_c (targets 3 2) positiveM birthData).population.profiles j)
      ((grow stage_c (targets 3 2) positiveM birthData).population.agents j) = decide ((1/2 : Rat) ≤ ![1,1/2,0,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_f : advance (grow stage_c (targets 3 2) positiveM birthData) = stage_f := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_f]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_g :
    let s := stage_f
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0},{1},∅] : Fin 4 → Finset (Fin 4)) := by
  have hg : ∀ i j, (stage_f).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network4_adj i j
  have hb : ∀ j, broadcasting ((stage_f).population.profiles j)
      ((stage_f).population.agents j) = decide ((1/2 : Rat) ≤ ![3/4,3/4,1/4,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_g : advance (stage_f) = stage_g := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_g]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_g_birth :
    let s := grow stage_d (targets 3 2) positiveM birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0},{1},∅] : Fin 4 → Finset (Fin 4)) := by
  have hg : ∀ i j, (grow stage_d (targets 3 2) positiveM birthData).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network4_adj i j
  have hb : ∀ j, broadcasting ((grow stage_d (targets 3 2) positiveM birthData).population.profiles j)
      ((grow stage_d (targets 3 2) positiveM birthData).population.agents j) = decide ((1/2 : Rat) ≤ ![3/4,3/4,1/4,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_g_birth : advance (grow stage_d (targets 3 2) positiveM birthData) = stage_g := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_g_birth]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_early :
    let s := stage_g
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0,2},{1},{2}] : Fin 4 → Finset (Fin 4)) := by
  have hg : ∀ i j, (stage_g).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network4_adj i j
  have hb : ∀ j, broadcasting ((stage_g).population.profiles j)
      ((stage_g).population.agents j) = decide ((1/2 : Rat) ≤ ![3/4,3/4,1/2,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_early : advance (stage_g) = baseState .bbii := by
  apply joint_eq
  · exact network4_eq
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_early]
    funext i; fin_cases i <;> decide_cbv

private theorem incoming_late :
    let s := grow stage_e (targets 3 2) positiveM birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1},{0},{1},∅] : Fin 4 → Finset (Fin 4)) := by
  have hg : ∀ i j, (grow stage_e (targets 3 2) positiveM birthData).network.snapshot.graph.Adj i j ↔
      i.val+1 = j.val ∨ j.val+1 = i.val := fun i j => network4_adj i j
  have hb : ∀ j, broadcasting ((grow stage_e (targets 3 2) positiveM birthData).population.profiles j)
      ((grow stage_e (targets 3 2) positiveM birthData).population.agents j) = decide ((1/2 : Rat) ≤ ![3/4,3/4,3/8,0] j) := by
    intro j
    fin_cases j <;> rfl
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hg, hb]
  fin_cases i <;> fin_cases j <;> decide_cbv

private theorem step_late : advance (grow stage_e (targets 3 2) positiveM birthData) = baseState .iibb := by
  apply joint_eq
  · exact network4_eq
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming_late]
    funext i; fin_cases i <;> decide_cbv

private theorem mass_first (s : JointState 2)
    (hn : s.network = (initial one).network) :
    orderedMass s.network (targets 2 1) = 1/2 := by
  rw [hn]
  decide_cbv
private theorem mass_second (s : JointState 3) (hn : s.network = network3) :
    orderedMass s.network (targets 3 2) = 1/4 := by
  rw [hn]
  decide_cbv

private theorem run_idle {n : Nat} (m t b r k : Nat) (s : JointState n) :
    runInputs m t b ⟨n,r,s⟩ (List.replicate k none) =
      .ok ⟨⟨n,r+k,advance^[k] s⟩,1⟩ := by
  induction k generalizing s t r with
  | zero => rfl
  | succ k ih =>
      simp only [List.replicate_succ, runInputs, ih, Function.iterate_succ_apply]
      congr 3; omega

theorem raw_tail_bridge (h : History) (k : Nat) :
    rawTail h k = .ok ⟨⟨4,4+k,tailState h k⟩,1/8⟩ := by
  have hm0 := mass_first (initial one) rfl
  have hmb := mass_first stage_b rfl
  have hmc := mass_second stage_c rfl
  have hmd := mass_second stage_d rfl
  have hme := mass_second stage_e rfl
  cases h <;>
    simp only [rawTail, ticks, List.cons_append, List.nil_append,
      show rawSeed = ⟨2, #[1, one.val], #[(0,1)]⟩ from rfl,
      replay_start one 1 (by decide), runInputs, checked_first, checked_second,
      step_a, step_b, step_c, step_d, step_e, step_f, step_g,
      step_g_birth, step_early, step_late, hm0, hmb, hmc, hmd, hme,
      run_idle, tailState]
  all_goals norm_num <;> rfl

theorem replay_baseline (h : History) :
    rawTail h 0 = .ok ⟨⟨4,4,baseState h⟩,1/8⟩ := raw_tail_bridge h 0

theorem baseState_adj (h : History) (i j : Fin 4) :
    (baseState h).network.snapshot.graph.Adj i j ↔ NarrativeDynamics.FitnessABMPath4.pathAdj i j := by
  fin_cases i <;> fin_cases j <;> rfl

theorem baseState_profiles (h : History) :
    (baseState h).population.profiles = fun _ => ⟨1/2,1/2⟩ := rfl

theorem baseState_exposures (h : History) :
    (fun i => ((baseState h).population.agents i).exposures) =
      baselineExposures h := rfl

theorem tail_network (h : History) (k : Nat) :
    (tailState h k).network = (baseState h).network := by
  induction k with
  | zero => rfl
  | succ k ih => simpa [tailState, Function.iterate_succ_apply'] using ih

theorem tail_profiles (h : History) (k : Nat) :
    (tailState h k).population.profiles = fun _ => ⟨1/2,1/2⟩ := by
  induction k with
  | zero => rfl
  | succ k ih => simpa [tailState, Function.iterate_succ_apply'] using ih

private theorem tail_population (h : History) (k : Nat) :
    (tailState h k).population = NarrativeDynamics.FitnessABMPath4.population (tailBelief h k)
      (fun i => ((tailState h k).population.agents i).exposures) := by
  have hp := tail_profiles h k
  cases he : (tailState h k).population with
  | mk profiles agents =>
    simp only [he] at hp
    cases hp
    simp only [tailBelief, NarrativeDynamics.FitnessABMPath4.project, he,
      NarrativeDynamics.FitnessABMPath4.population]

theorem tailBelief_step (h : History) (k : Nat) :
    tailBelief h (k+1) = NarrativeDynamics.FitnessABMPath4.beliefStep (tailBelief h k) := by
  simp only [tailBelief, tailState, Function.iterate_succ_apply']
  change NarrativeDynamics.FitnessABMPath4.project (advance (tailState h k)).population = _
  change NarrativeDynamics.FitnessABMPath4.project
    (@propagate 4 (tailState h k).network.snapshot.graph.Adj
      (tailState h k).network.snapshot.adjDec (tailState h k).population) = _
  rw [tail_network, tail_population]
  exact NarrativeDynamics.FitnessABMPath4.propagate_independent_exposures _ _

def activationOffset : History → Nat
  | .bbii | .bibi => 2
  | .iibb => 3

theorem tail_from_activation (h : History) (n : Nat) :
    tailBelief h (activationOffset h+n) =
      NarrativeDynamics.FitnessABMPath4.trajectory (tailBelief h (activationOffset h)) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [Nat.add_succ, tailBelief_step, ih]
      simp only [NarrativeDynamics.FitnessABMPath4.trajectory, Function.iterate_succ_apply']

theorem equal_control (k : Nat) : tailBelief .bbii k = tailBelief .bibi k := by
  have hb : baseState .bbii = baseState .bibi := rfl
  exact congrArg (fun s => NarrativeDynamics.FitnessABMPath4.project (advance^[k] s).population) hb

private theorem tail_zero (h : History) : tailBelief h 0 = baselineBeliefs h := rfl
private theorem early_one : tailBelief .bbii 1 = ![23/32,11/16,21/32,7/16] := by
  rw [show 1 = 0+1 from rfl, tailBelief_step, tail_zero]
  decide_cbv

theorem activation_bbii :
    tailBelief .bbii 2 = ![45/64,44/64,43/64,35/64] ∧
      NarrativeDynamics.FitnessABMPath4.allBroadcast (tailBelief .bbii 2) := by
  have hv : tailBelief .bbii 2 = ![45/64,44/64,43/64,35/64] := by
    rw [show 2 = 1+1 from rfl, tailBelief_step, early_one]
    decide_cbv
  refine ⟨hv, ?_⟩
  rw [hv]
  intro i
  fin_cases i <;> decide_cbv

theorem activation_bibi :
    tailBelief .bibi 2 = ![45/64,44/64,43/64,35/64] ∧
      NarrativeDynamics.FitnessABMPath4.allBroadcast (tailBelief .bibi 2) := by
  rw [← equal_control]
  exact activation_bbii

private theorem late_one : tailBelief .iibb 1 = ![3/4,45/64,21/32,9/32] := by
  rw [show 1 = 0+1 from rfl, tailBelief_step, tail_zero]
  decide_cbv
private theorem late_two : tailBelief .iibb 2 = ![93/128,45/64,87/128,15/32] := by
  rw [show 2 = 1+1 from rfl, tailBelief_step, late_one]
  decide_cbv

theorem activation_iibb :
    tailBelief .iibb 3 = ![183/256,180/256,177/256,147/256] ∧
      NarrativeDynamics.FitnessABMPath4.allBroadcast (tailBelief .iibb 3) := by
  have hv : tailBelief .iibb 3 = ![183/256,180/256,177/256,147/256] := by
    rw [show 3 = 2+1 from rfl, tailBelief_step, late_two]
    decide_cbv
  refine ⟨hv, ?_⟩
  rw [hv]
  intro i
  fin_cases i <;> decide_cbv

theorem baseline_not_allBroadcast (h : History) : ¬ NarrativeDynamics.FitnessABMPath4.allBroadcast (tailBelief h 0) := by
  rw [tail_zero]
  cases h <;> intro hx <;> have hx3 := (hx 3).1 <;> exact (by decide_cbv : ¬ ((1/2 : Rat) ≤ baselineBeliefs _ 3)) hx3

theorem predecessor_bbii : tailBelief .bbii 1 3 = 7/16 := by rw [early_one]; rfl
theorem predecessor_bibi : tailBelief .bibi 1 3 = 7/16 := by rw [← equal_control]; exact predecessor_bbii
theorem predecessor_iibb : tailBelief .iibb 2 3 = 15/32 := by rw [late_two]; rfl

theorem preactivation_not_allBroadcast (h : History) (k : Nat)
    (hk : k < activationOffset h) : ¬ NarrativeDynamics.FitnessABMPath4.allBroadcast (tailBelief h k) := by
  cases h with
  | bbii =>
      have : k = 0 ∨ k = 1 := by simp only [activationOffset] at hk; omega
      rcases this with rfl | rfl
      · exact baseline_not_allBroadcast _
      · rw [early_one]; intro hx; have hx3 := (hx 3).1; exact (by decide_cbv : ¬ ((1/2 : Rat) ≤ _)) hx3
  | bibi =>
      rw [← equal_control]
      have : k = 0 ∨ k = 1 := by simp only [activationOffset] at hk; omega
      rcases this with rfl | rfl
      · exact baseline_not_allBroadcast _
      · rw [early_one]; intro hx; have hx3 := (hx 3).1; exact (by decide_cbv : ¬ ((1/2 : Rat) ≤ _)) hx3
  | iibb =>
      have : k = 0 ∨ k = 1 ∨ k = 2 := by simp only [activationOffset] at hk; omega
      rcases this with rfl | rfl | rfl
      · exact baseline_not_allBroadcast _
      · rw [late_one]; intro hx; have hx3 := (hx 3).1; exact (by decide_cbv : ¬ ((1/2 : Rat) ≤ _)) hx3
      · rw [late_two]; intro hx; have hx3 := (hx 3).1; exact (by decide_cbv : ¬ ((1/2 : Rat) ≤ _)) hx3

example (h : History) (k : Nat) :
    rawTail h k = .ok ⟨⟨4,4+k,tailState h k⟩,1/8⟩ := raw_tail_bridge h k
example : tailBelief .bbii 2 = ![45/64,44/64,43/64,35/64] := activation_bbii.1

end FiniteReplayFixtures

#print axioms replay_baseline
#print axioms raw_tail_bridge
#print axioms baseState_adj
#print axioms baseState_profiles
#print axioms baseState_exposures
#print axioms tail_network
#print axioms tail_profiles
#print axioms tailBelief_step
#print axioms activation_bbii
#print axioms activation_bibi
#print axioms activation_iibb
#print axioms baseline_not_allBroadcast
#print axioms preactivation_not_allBroadcast
#print axioms predecessor_bbii
#print axioms predecessor_bibi
#print axioms predecessor_iibb
#print axioms tail_from_activation
#print axioms equal_control

end NarrativeDynamics.Tests.FitnessABMPath4

open Filter Topology

example (x : Beliefs) (hx : allBroadcast x) (i : Fin 4) :
    Tendsto (fun n : Nat => (trajectory x n i : Real))
      atTop (nhds (mean x : Real)) := trajectory_tendsto x hx i

namespace NarrativeDynamics.Tests.FitnessABMPath4

-- Each tail index k denotes global tick 4+k, as certified by raw_tail_bridge.
-- Removing the finite activation offset therefore retains the common clock.
private theorem tail_tendsto_from_activation (h : History)
    (hx : allBroadcast (tailBelief h (activationOffset h))) (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief h k i : Real))
      atTop (nhds (mean (tailBelief h (activationOffset h)) : Real)) := by
  apply (tendsto_add_atTop_iff_nat (activationOffset h)).mp
  have hlimit := trajectory_tendsto (tailBelief h (activationOffset h)) hx i
  convert hlimit using 1
  funext n
  rw [Nat.add_comm n (activationOffset h), tail_from_activation]

theorem bbii_tendsto (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief .bbii k i : Real))
      atTop (nhds (127/192 : Real)) := by
  have hlimit := tail_tendsto_from_activation .bbii activation_bbii.2 i
  have hm : mean (tailBelief .bbii 2) = 127/192 := by
    rw [activation_bbii.1]
    dsimp [mean]
    norm_num
  norm_num [activationOffset, hm] at hlimit
  exact hlimit

theorem bibi_tendsto (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief .bibi k i : Real))
      atTop (nhds (127/192 : Real)) := by
  have hlimit := tail_tendsto_from_activation .bibi activation_bibi.2 i
  have hm : mean (tailBelief .bibi 2) = 127/192 := by
    rw [activation_bibi.1]
    dsimp [mean]
    norm_num
  norm_num [activationOffset, hm] at hlimit
  exact hlimit

theorem iibb_tendsto (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief .iibb k i : Real))
      atTop (nhds (87/128 : Real)) := by
  have hlimit := tail_tendsto_from_activation .iibb activation_iibb.2 i
  have hm : mean (tailBelief .iibb 3) = 87/128 := by
    rw [activation_iibb.1]
    dsimp [mean]
    norm_num
  norm_num [activationOffset, hm] at hlimit
  exact hlimit

theorem separation_limit (i : Fin 4) :
    Tendsto
      (fun k : Nat => (tailBelief .iibb k i : Real) - (tailBelief .bbii k i : Real))
      atTop (nhds (7/384 : Real)) ∧ (0 : Real) < 7/384 := by
  constructor
  · convert (iibb_tendsto i).sub (bbii_tendsto i) using 1
    norm_num
  · norm_num

-- Independently transport BBII's limit through the all-k equality control.
example (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief .bibi k i : Real))
      atTop (nhds (127/192 : Real)) := by
  simpa only [← equal_control] using bbii_tendsto i

example (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief .bbii k i : Real))
      atTop (nhds (127/192 : Real)) := bbii_tendsto i

example (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief .bibi k i : Real))
      atTop (nhds (127/192 : Real)) := bibi_tendsto i

example (i : Fin 4) :
    Tendsto (fun k : Nat => (tailBelief .iibb k i : Real))
      atTop (nhds (87/128 : Real)) := iibb_tendsto i

example (i : Fin 4) :
    Tendsto
      (fun k : Nat => (tailBelief .iibb k i : Real) - (tailBelief .bbii k i : Real))
      atTop (nhds (7/384 : Real)) ∧ (0 : Real) < 7/384 := separation_limit i

end NarrativeDynamics.Tests.FitnessABMPath4

#print axioms NarrativeDynamics.FitnessABMPath4.trajectory_tendsto
#print axioms NarrativeDynamics.Tests.FitnessABMPath4.bbii_tendsto
#print axioms NarrativeDynamics.Tests.FitnessABMPath4.bibi_tendsto
#print axioms NarrativeDynamics.Tests.FitnessABMPath4.iibb_tendsto
#print axioms NarrativeDynamics.Tests.FitnessABMPath4.separation_limit
