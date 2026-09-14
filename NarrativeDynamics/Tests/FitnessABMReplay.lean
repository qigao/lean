import NarrativeDynamics.Conformance.FitnessABMVectors

namespace NarrativeDynamics.Tests.FitnessABMReplay

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open NarrativeDynamics.FitnessABM
open NarrativeDynamics.Conformance.BBPropagation
open scoped BigOperators

def seedRaw : FitnessAttachment.RawSeed := ⟨2, #[1, 1], #[(0, 1)]⟩
def agentsRaw : Array RawAgent := #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩]
def attach0 : RawBirthInput := ⟨⟨1, #[0]⟩, 1, 1/2, 0⟩
def attach1 : RawBirthInput := ⟨⟨1, #[1]⟩, 1, 1/2, 0⟩

def summary (r : Except JointError Result) :
    Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat) :=
  r.map fun out =>
    (out.final.nodeCount, out.final.roundIndex,
     FitnessAttachment.actualEdgeCount out.final.state.network.snapshot,
     List.ofFn (fun i => (out.final.state.population.agents i).belief),
     List.ofFn (fun i => (out.final.state.population.agents i).exposures),
     out.probability)

private theorem summary_ok (out : Result) (nodes rounds edges : Nat)
    (beliefs : List Rat) (exposures : List Nat) (mass : Rat) :
    summary (.ok out) = .ok (nodes, rounds, edges, beliefs, exposures, mass) ↔
      out.final.nodeCount = nodes ∧ out.final.roundIndex = rounds ∧
      actualEdgeCount out.final.state.network.snapshot = edges ∧
      List.ofFn (fun i => (out.final.state.population.agents i).belief) = beliefs ∧
      List.ofFn (fun i => (out.final.state.population.agents i).exposures) = exposures ∧
      out.probability = mass := by
  change (Except.ok (_, _, _, _, _, _) : Except JointError _) =
    Except.ok (_, _, _, _, _, _) ↔ _
  simp only [Except.ok.injEq, Prod.mk.injEq]

-- Decidability is needed only for the finite observable result, never graph functions.
private instance : DecidableEq
    (Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat)) :=
  fun x y => match x, y with
  | .error x, .error y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.error.inj he))
  | .ok x, .ok y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.ok.inj he))
  | .error _, .ok _ => .isFalse (by intro h; cases h)
  | .ok _, .error _ => .isFalse (by intro h; cases h)

section FiniteReplayFixtures
-- Match the established finite BB fixture budget; keep default heartbeats.
set_option maxRecDepth 4096

private def one : PosFitness := ⟨1, by norm_num⟩
private def three : PosFitness := ⟨3, by norm_num⟩
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
    population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨0, 0⟩]⟩
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

private theorem parsed_seedRaw : parseSeed seedRaw = .ok (seedNetwork one) :=
  parsed_seed one

private theorem parsed_agents :
    parseAgents 2 agentsRaw = .ok
      ⟨(initial one).population, (initial one).populationValid⟩ := by
  have valid : ∀ a ∈ agentsRaw.toList, a.Valid := by
    intro a ha
    simp [agentsRaw] at ha
    rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]
  obtain ⟨p, hp⟩ := parseAgents_complete 2 agentsRaw rfl valid
  have values := (parseAgents_sound 2 agentsRaw p hp).2.2
  have he : p.val = (initial one).population := by
    rcases p with ⟨⟨profiles, agents⟩, hv⟩
    have hprofiles : profiles = (initial one).population.profiles := by
      funext i
      have h := (values i (by have := i.isLt; simpa [agentsRaw] using this)).1
      fin_cases i <;> simpa [agentsRaw, initial] using h
    have hagents : agents = (initial one).population.agents := by
      funext i
      have h := (values i (by have := i.isLt; simpa [agentsRaw] using this)).2
      fin_cases i <;> simpa [agentsRaw, initial] using h
    cases hprofiles
    cases hagents
    rfl
  have hp' : p = ⟨(initial one).population, (initial one).populationValid⟩ :=
    Subtype.ext he
  exact hp.trans (congrArg Except.ok hp')

private theorem replay_start (eta : PosFitness) (m : Nat) (hm : 0 < m ∧ m ≤ 2)
    (ticks : List RawTick) :
    FitnessABM.replay ⟨2, #[1, eta.val], #[(0, 1)]⟩ m agentsRaw ticks =
      runInputs m 0 0 ⟨2, 0, initial eta⟩ ticks := by
  simp only [FitnessABM.replay, parsed_seed, if_pos hm, parsed_agents]
  rfl

private def targets0 : Targets 2 1 := ⟨![0], by decide⟩
private def targets1 : Targets 2 1 := ⟨![1], by decide⟩
private def targets01 : Targets 2 2 := ⟨![0, 1], by decide⟩
private def targets10 : Targets 2 2 := ⟨![1, 0], by decide⟩
private def birthData : BirthData :=
  ⟨one, ⟨⟨1, 1/2⟩, 0, by norm_num [AgentProfile.Valid], by norm_num⟩⟩

private theorem parsed_newborn : parseAgent ⟨1, 1/2, 0, 0⟩ =
    .ok ⟨(⟨1, 1/2⟩, ⟨0, 0⟩), by norm_num [AgentProfile.Valid, AgentState.Valid]⟩ := by
  norm_num [parseAgent]

private theorem checked0 (eta : PosFitness) :
    checkedBirth (initial eta) 1 attach0 = .ok
      (grow (initial eta) targets0 positiveM birthData,
        orderedMass (initial eta).network targets0) := by
  apply (checkedBirth_spec _ _ _).mpr
  refine ⟨⟨targets0, one, positiveM⟩, _, ?_, parsed_newborn, rfl⟩
  have hm : 0 < (1 : Nat) ∧ 1 ≤ 2 := by decide
  have hf : 0 < attach0.birth.fitness := by norm_num [attach0]
  have hs : attach0.birth.targets.size = 1 := rfl
  have hb : targetsBounded 2 attach0.birth.targets := by decide_cbv
  have hd : targetsDistinct attach0.birth.targets := by decide_cbv
  have he : (⟨hb, hd⟩ : CheckedTargets 2 attach0.birth.targets).embedding =
      targets0 := by
    apply Function.Embedding.ext
    intro i
    fin_cases i
    rfl
  simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]
  rw [he]
  rfl

private theorem checked1 (eta : PosFitness) :
    checkedBirth (initial eta) 1 attach1 = .ok
      (grow (initial eta) targets1 positiveM birthData,
        orderedMass (initial eta).network targets1) := by
  apply (checkedBirth_spec _ _ _).mpr
  refine ⟨⟨targets1, one, positiveM⟩, _, ?_, parsed_newborn, rfl⟩
  have hm : 0 < (1 : Nat) ∧ 1 ≤ 2 := by decide
  have hf : 0 < attach1.birth.fitness := by norm_num [attach1]
  have hs : attach1.birth.targets.size = 1 := rfl
  have hb : targetsBounded 2 attach1.birth.targets := by decide_cbv
  have hd : targetsDistinct attach1.birth.targets := by decide_cbv
  have he : (⟨hb, hd⟩ : CheckedTargets 2 attach1.birth.targets).embedding =
      targets1 := by
    apply Function.Embedding.ext
    intro i
    fin_cases i
    rfl
  simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]
  rw [he]
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

private theorem lastCases_eq_if {α : Sort u} {n : Nat}
    (last : α) (old : Fin n → α) (i : Fin (n + 1)) :
    Fin.lastCases last old i =
      if h : i.val < n then old ⟨i.val, h⟩ else last := by
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simp
  · simp [j.isLt]

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

private def round0 : JointState 3 :=
  { network := (grow (initial one) targets0 positiveM birthData).network
    population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,0⟩, ⟨1,1⟩, ⟨1,1⟩]⟩
    populationValid := by
      constructor
      · intro i; fin_cases i <;> norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def round1 : JointState 3 :=
  { network := (grow (initial one) targets1 positiveM birthData).network
    population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,0⟩, ⟨1,1⟩, ⟨0,0⟩]⟩
    populationValid := by
      constructor
      · intro i; fin_cases i <;> norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private def round2 : JointState 3 :=
  { network := round1.network
    population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,1⟩, ⟨1,2⟩, ⟨1,1⟩]⟩
    populationValid := by
      constructor
      · intro i; fin_cases i <;> norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private theorem top2_adj (i j : Fin 2) :
    (⊤ : SimpleGraph (Fin 2)).Adj i j ↔ i ≠ j := Iff.rfl

private theorem incoming0 :
    let s := grow (initial one) targets0 positiveM birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
  have selected0 : targets0.selected = {0} := by
    ext i
    simp only [Targets.selected, Finset.mem_image, Finset.mem_univ, true_and,
      Finset.mem_singleton]
    constructor
    · rintro ⟨j, hj⟩
      fin_cases j
      exact hj.symm
    · intro hi
      subst i
      exact ⟨0, rfl⟩
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      lastCases_eq_if, initial, seedNetwork, extendPopulation,
      birthData, one, selected0, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem incoming1 :
    let s := grow (initial one) targets1 positiveM birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, ∅] : Fin 3 → Finset (Fin 3)) := by
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    norm_num [grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      lastCases_eq_if, Fin.ext_iff, initial, seedNetwork, extendPopulation,
      birthData, one, targets1, Targets.selected, broadcasting]

private theorem incoming2 :
    letI := round1.network.snapshot.adjDec
    incoming round1.network.snapshot.graph.Adj round1.population =
      (![{1}, {0}, {1}] : Fin 3 → Finset (Fin 3)) := by
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    norm_num [round1, grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      lastCases_eq_if, Fin.ext_iff, initial, seedNetwork, extendPopulation,
      birthData, one, targets1, Targets.selected, broadcasting]

private theorem round0_step : advance (grow (initial one) targets0 positiveM birthData) = round0 := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming0]
    funext i; fin_cases i <;> decide_cbv

private theorem round1_step : advance (grow (initial one) targets1 positiveM birthData) = round1 := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming1]
    funext i; fin_cases i <;> decide_cbv

private theorem round2_step : advance round1 = round2 := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incoming2]
    funext i; fin_cases i <;> decide_cbv

private theorem mass0 : orderedMass (initial one).network targets0 = 1/2 := by decide_cbv
private theorem mass1 : orderedMass (initial one).network targets1 = 1/2 := by decide_cbv

private theorem replay0 : FitnessABM.replay seedRaw 1 agentsRaw [some attach0] =
    .ok ⟨⟨3, 1, round0⟩, 1/2⟩ := by
  rw [show seedRaw = ⟨2, #[1, one.val], #[(0, 1)]⟩ from rfl,
    replay_start one 1 (by decide), runInputs, checked0]
  simp only [runInputs, round0_step, mass0, mul_one]

private theorem replay1 : FitnessABM.replay seedRaw 1 agentsRaw [some attach1] =
    .ok ⟨⟨3, 1, round1⟩, 1/2⟩ := by
  rw [show seedRaw = ⟨2, #[1, one.val], #[(0, 1)]⟩ from rfl,
    replay_start one 1 (by decide), runInputs, checked1]
  simp only [runInputs, round1_step, mass1, mul_one]

private theorem replay2 : FitnessABM.replay seedRaw 1 agentsRaw [some attach1, none] =
    .ok ⟨⟨3, 2, round2⟩, 1/2⟩ := by
  rw [show seedRaw = ⟨2, #[1, one.val], #[(0, 1)]⟩ from rfl,
    replay_start one 1 (by decide), runInputs, checked1]
  simp only [runInputs, round1_step, round2_step, mass1, mul_one]

private theorem seed_edges_one : actualEdgeCount (initial one).network.snapshot = 1 := by
  decide_cbv

private theorem round0_edges : actualEdgeCount round0.network.snapshot = 2 := by
  change actualEdgeCount
    (applyBirth (initial one).network targets0 positiveM birthData.fitness).snapshot = 2
  rw [birth_edges, seed_edges_one]

private theorem round1_edges : actualEdgeCount round1.network.snapshot = 2 := by
  change actualEdgeCount
    (applyBirth (initial one).network targets1 positiveM birthData.fitness).snapshot = 2
  rw [birth_edges, seed_edges_one]

private theorem round2_edges : actualEdgeCount round2.network.snapshot = 2 := round1_edges

-- Exact original RED observations, with symbolic rounds before finite reduction.
example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some attach0]) =
    .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/2) := by
  rw [replay0]
  apply (summary_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, round0_edges, ?_, ?_, rfl⟩
  · decide_cbv
  · decide_cbv

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some attach1]) =
    .ok (3, 1, 2, [1, 1, 0], [0, 1, 0], 1/2) := by
  rw [replay1]
  apply (summary_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, round1_edges, ?_, ?_, rfl⟩
  · decide_cbv
  · decide_cbv

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some attach1, none]) =
    .ok (3, 2, 2, [1, 1, 1], [1, 2, 1], 1/2) := by
  rw [replay2]
  apply (summary_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, round2_edges, ?_, ?_, rfl⟩
  · decide_cbv
  · decide_cbv

example : summary (FitnessABM.replay seedRaw 1 agentsRaw []) =
    .ok (2, 0, 1, [1, 0], [0, 0], 1) := by
  rw [show seedRaw = ⟨2, #[1, one.val], #[(0, 1)]⟩ from rfl,
    replay_start one 1 (by decide)]
  decide_cbv

-- Errors remain observable without asking for decidable equality of graph functions.
example : summary (FitnessABM.replay ⟨0, #[], #[]⟩ 1 #[⟨-1, 2, 2, 0⟩] []) =
    .error (.seedNetwork .invalidNodeCount) := by decide_cbv

example : summary (FitnessABM.replay seedRaw 0 agentsRaw []) = .error .initialM := by
  unfold FitnessABM.replay
  rw [parsed_seedRaw]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 3 agentsRaw []) = .error .initialM := by
  unfold FitnessABM.replay
  rw [parsed_seedRaw]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 1 #[⟨1, 1/2, 1, 0⟩] []) =
    .error (.seedAgentCount 2 1) := by
  unfold FitnessABM.replay
  rw [parsed_seedRaw]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 1 #[⟨-1, 2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩] []) =
    .error (.seedAgent 0 .receptivity) := by
  unfold FitnessABM.replay
  rw [parsed_seedRaw]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 1 #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 2, 0⟩] []) =
    .error (.seedAgent 1 .belief) := by
  unfold FitnessABM.replay
  rw [parsed_seedRaw]
  decide_cbv

private theorem start_one (m : Nat) (hm : 0 < m ∧ m ≤ 2) (ticks : List RawTick) :
    FitnessABM.replay seedRaw m agentsRaw ticks = runInputs m 0 0 ⟨2, 0, initial one⟩ ticks :=
  replay_start one m hm ticks

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some ⟨⟨0, #[0]⟩, 1, 1/2, 2⟩]) =
    .error (.tickNetwork 0 0 .nonpositiveFitness) := by
  rw [start_one 1 (by decide)]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some ⟨⟨1, #[0]⟩, 1, 2, 2⟩]) =
    .error (.tickAgent 0 0 .threshold) := by
  rw [start_one 1 (by decide)]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some ⟨⟨1, #[0, 0]⟩, 1, 1/2, 0⟩]) =
    .error (.tickNetwork 0 0 .targetCountMismatch) := by
  rw [start_one 1 (by decide)]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 2 agentsRaw [some ⟨⟨1, #[0, 0]⟩, 1, 1/2, 0⟩]) =
    .error (.tickNetwork 0 0 .duplicateTarget) := by
  rw [start_one 2 (by decide)]
  decide_cbv

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some ⟨⟨1, #[2]⟩, 1, 1/2, 0⟩]) =
    .error (.tickNetwork 0 0 .targetOutOfRange) := by
  rw [start_one 1 (by decide)]
  decide_cbv

private def bad : RawBirthInput := ⟨⟨1, #[3]⟩, 1, 1/2, 0⟩

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [none, some attach1, none, some bad]) =
    .error (.tickNetwork 3 1 .targetOutOfRange) := by
  rw [start_one 1 (by decide)]
  decide_cbv

-- A prefix failure wins over a conflicting later network error.
private def badAgent : RawBirthInput := ⟨⟨1, #[0]⟩, 1, 2, 0⟩
private def badFitness : RawBirthInput := ⟨⟨0, #[0]⟩, 1, 1/2, 0⟩

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some badAgent, some badFitness]) =
    .error (.tickAgent 0 0 .threshold) := by
  rw [start_one 1 (by decide)]
  have hp : runInputs 1 0 0 ⟨2, 0, initial one⟩ [some badAgent] =
      .error (.tickAgent 0 0 .threshold) := by rfl
  rw [show [some badAgent, some badFitness] = [some badAgent] ++ [some badFitness] from rfl,
    runInputs_append_error 1 0 0 _ _ _ _ hp]
  rfl

example : summary (FitnessABM.replay seedRaw 1 agentsRaw [some badFitness, some badAgent]) =
    .error (.tickNetwork 0 0 .nonpositiveFitness) := by
  rw [start_one 1 (by decide)]
  decide_cbv

-- Idle-only successful replay projects to exactly the empty BB replay.
example (out : Result) (h : FitnessABM.replay seedRaw 1 agentsRaw [none, none] = .ok out) :
    ∃ bb, FitnessAttachment.replay seedRaw 1 [] = .ok bb ∧
      bb.final = ⟨out.final.nodeCount, out.final.state.network⟩ ∧
      ∃ hn : out.final.nodeCount = bb.final.nodeCount,
        (∀ i j, out.final.state.network.snapshot.graph.Adj i j ↔
          bb.final.state.snapshot.graph.Adj (Fin.cast hn i) (Fin.cast hn j)) ∧
        (∀ i, out.final.state.network.snapshot.fitness i =
          bb.final.state.snapshot.fitness (Fin.cast hn i)) ∧
        out.probability = bb.probability :=
  replay_projection seedRaw 1 agentsRaw [none, none] out h

example : ∃ out, FitnessABM.replay seedRaw 1 agentsRaw [none, none] = .ok out := by
  rw [start_one 1 (by decide)]
  exact ⟨_, rfl⟩

private def weightedSeed : RawSeed := ⟨2, #[1, 3], #[(0, 1)]⟩
private def attach01 : RawBirthInput := ⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩
private def attach10 : RawBirthInput := ⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩
private theorem two_pos : 0 < (2 : Nat) := by decide

private theorem checked01 :
    checkedBirth (initial three) 2 attach01 = .ok
      (grow (initial three) targets01 two_pos birthData,
        orderedMass (initial three).network targets01) := by
  apply (checkedBirth_spec _ _ _).mpr
  refine ⟨⟨targets01, one, two_pos⟩, _, ?_, parsed_newborn, rfl⟩
  have hm : 0 < (2 : Nat) ∧ 2 ≤ 2 := by decide
  have hf : 0 < attach01.birth.fitness := by norm_num [attach01]
  have hs : attach01.birth.targets.size = 2 := rfl
  have hb : targetsBounded 2 attach01.birth.targets := by decide_cbv
  have hd : targetsDistinct attach01.birth.targets := by decide_cbv
  have he : (⟨hb, hd⟩ : CheckedTargets 2 attach01.birth.targets).embedding =
      targets01 := by
    apply Function.Embedding.ext
    intro i
    fin_cases i <;> rfl
  simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]
  rw [he]
  rfl

private theorem checked10 :
    checkedBirth (initial three) 2 attach10 = .ok
      (grow (initial three) targets10 two_pos birthData,
        orderedMass (initial three).network targets10) := by
  apply (checkedBirth_spec _ _ _).mpr
  refine ⟨⟨targets10, one, two_pos⟩, _, ?_, parsed_newborn, rfl⟩
  have hm : 0 < (2 : Nat) ∧ 2 ≤ 2 := by decide
  have hf : 0 < attach10.birth.fitness := by norm_num [attach10]
  have hs : attach10.birth.targets.size = 2 := rfl
  have hb : targetsBounded 2 attach10.birth.targets := by decide_cbv
  have hd : targetsDistinct attach10.birth.targets := by decide_cbv
  have he : (⟨hb, hd⟩ : CheckedTargets 2 attach10.birth.targets).embedding =
      targets10 := by
    apply Function.Embedding.ext
    intro i
    fin_cases i <;> rfl
  simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]
  rw [he]
  rfl

private def roundBoth : JointState 3 :=
  { network := (grow (initial three) targets01 two_pos birthData).network
    population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1,0⟩, ⟨1,1⟩, ⟨1,1⟩]⟩
    populationValid := by
      constructor
      · intro i; fin_cases i <;> norm_num [AgentProfile.Valid]
      · intro i; fin_cases i <;> norm_num [AgentState.Valid] }

private theorem incomingBoth :
    let s := grow (initial three) targets01 two_pos birthData
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    norm_num [grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      lastCases_eq_if, Fin.ext_iff, initial, seedNetwork, extendPopulation,
      birthData, one, targets01, Targets.selected, broadcasting]

private theorem roundBoth_step :
    advance (grow (initial three) targets01 two_pos birthData) = roundBoth := by
  apply joint_eq
  · rfl
  · funext i; fin_cases i <;> decide_cbv
  · rw [advance_agents _ _ incomingBoth]
    funext i; fin_cases i <;> decide_cbv

private theorem order_same : grow (initial three) targets10 two_pos birthData =
    grow (initial three) targets01 two_pos birthData := by
  have hs : targets10.selected = targets01.selected := by decide_cbv
  unfold grow
  rw [birth_order_irrelevant _ targets10 targets01 two_pos _ hs]

private theorem mass01 : orderedMass (initial three).network targets01 = 1/4 := by decide_cbv
private theorem mass10 : orderedMass (initial three).network targets10 = 3/4 := by decide_cbv

private theorem replay01 : FitnessABM.replay weightedSeed 2 agentsRaw [some attach01] =
    .ok ⟨⟨3, 1, roundBoth⟩, 1/4⟩ := by
  rw [show weightedSeed = ⟨2, #[1, three.val], #[(0, 1)]⟩ from rfl,
    replay_start three 2 (by decide), runInputs, checked01]
  simp only [runInputs, roundBoth_step, mass01, mul_one]

private theorem replay10 : FitnessABM.replay weightedSeed 2 agentsRaw [some attach10] =
    .ok ⟨⟨3, 1, roundBoth⟩, 3/4⟩ := by
  rw [show weightedSeed = ⟨2, #[1, three.val], #[(0, 1)]⟩ from rfl,
    replay_start three 2 (by decide), runInputs, checked10]
  simp only [runInputs, order_same, roundBoth_step, mass10, mul_one]

private theorem seed_edges_three : actualEdgeCount (initial three).network.snapshot = 1 := by
  decide_cbv

private theorem roundBoth_edges : actualEdgeCount roundBoth.network.snapshot = 3 := by
  change actualEdgeCount
    (applyBirth (initial three).network targets01 two_pos birthData.fitness).snapshot = 3
  rw [birth_edges, seed_edges_three]

-- Both original orders are replayed, with identical complete grown states.
example : summary (FitnessABM.replay weightedSeed 2 agentsRaw [some attach01]) =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 1/4) := by
  rw [replay01]
  apply (summary_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, roundBoth_edges, ?_, ?_, rfl⟩
  · decide_cbv
  · decide_cbv

example : summary (FitnessABM.replay weightedSeed 2 agentsRaw [some attach10]) =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 3/4) := by
  rw [replay10]
  apply (summary_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, roundBoth_edges, ?_, ?_, rfl⟩
  · decide_cbv
  · decide_cbv

example : ∃ final,
    FitnessABM.replay weightedSeed 2 agentsRaw [some attach01] = .ok ⟨final, 1/4⟩ ∧
    FitnessABM.replay weightedSeed 2 agentsRaw [some attach10] = .ok ⟨final, 3/4⟩ :=
  ⟨_, replay01, replay10⟩

private def orderedPairs {n : Nat}
    (keepPair : Fin n → Fin n → Bool) : List (Nat × Nat) :=
  let ids : Array (Fin n) := Array.ofFn id
  (ids.flatMap fun source =>
    ids.filterMap fun target =>
      if keepPair source target then some (source.val, target.val) else none).toList

private def propagationObservation {n : Nat} (g : MeshGraph (Fin n))
    [DecidableRel g] (population : Population n) :
    List Rat × List Nat × List Bool × List (Nat × Nat) :=
  let next := propagate g population
  let delivered := transmissions g population
  (List.ofFn (fun i => (next.agents i).belief),
   List.ofFn (fun i => (next.agents i).exposures),
   List.ofFn (fun i => broadcasting (next.profiles i) (next.agents i)),
   orderedPairs fun source target => decide ((source, target) ∈ delivered))

private def stateObservation {n : Nat} (prior : JointState n) :
    List Rat × List Nat × List Bool × List (Nat × Nat) :=
  let next := advance prior
  letI := prior.network.snapshot.adjDec
  let delivered := transmissions prior.network.snapshot.graph.Adj prior.population
  (List.ofFn (fun i => (next.population.agents i).belief),
   List.ofFn (fun i => (next.population.agents i).exposures),
   List.ofFn (fun i => broadcasting (next.population.profiles i)
     (next.population.agents i)),
   orderedPairs fun source target => decide ((source, target) ∈ delivered))

private theorem stateObservation_eq_propagationObservation {n : Nat}
    (prior : JointState n) :
    stateObservation prior =
      @propagationObservation n prior.network.snapshot.graph.Adj
        prior.network.snapshot.adjDec prior.population := by
  rfl

private def propagationSummary (seed : RawSeed) (agents : Array RawAgent) :
    Except JointError (List Rat × List Nat × List Bool × List (Nat × Nat)) :=
  match FitnessABM.replay seed 1 agents [] with
  | .error error => .error error
  | .ok result =>
    let prior := result.final.state
    .ok (stateObservation prior)

private theorem propagationSummary_of_parses (seed : RawSeed) (agents : Array RawAgent)
    (network : FitnessAttachment.State seed.nodeCount)
    (population : {p : Population seed.nodeCount // p.Valid})
    (hseed : parseSeed seed = .ok network)
    (hm : 0 < (1 : Nat) ∧ 1 ≤ seed.nodeCount)
    (hpopulation : parseAgents seed.nodeCount agents = .ok population) :
    propagationSummary seed agents =
      .ok (@propagationObservation seed.nodeCount network.snapshot.graph.Adj
        network.snapshot.adjDec population.val) := by
  simpa only [propagationSummary, FitnessABM.replay, hseed, if_pos hm,
    hpopulation, runInputs] using
      congrArg Except.ok (stateObservation_eq_propagationObservation
        (prior := ⟨network, population.val, population.property⟩))

private theorem propagationObservation_congr {n : Nat}
    {g h : MeshGraph (Fin n)} [dg : DecidableRel g] [dh : DecidableRel h]
    {p q : Population n} (hg : g = h) (hp : p = q) :
    @propagationObservation n g dg p = @propagationObservation n h dh q := by
  cases hg
  cases hp
  have hi : dg = dh := Subsingleton.elim _ _
  cases hi
  rfl

private theorem population_eq {n : Nat} (p q : Population n)
    (hprofiles : p.profiles = q.profiles) (hagents : p.agents = q.agents) : p = q := by
  cases p
  cases q
  simp_all

-- Parser soundness identifies the concrete population and adjacency before any
-- finite propagation evaluation; connectivity is never unfolded by CBV.
private theorem propagationSummary_flat (seed : RawSeed) (agents : Array RawAgent)
    (hseed : seed.Valid) (hsize : agents.size = seed.nodeCount)
    (hvalid : ∀ a ∈ agents.toList, a.Valid)
    (population : Population seed.nodeCount)
    (hvalues : ∀ (i : Fin seed.nodeCount) (hi : i.val < agents.size),
      population.profiles i = ⟨agents[i.val].receptivity, agents[i.val].threshold⟩ ∧
      population.agents i = ⟨agents[i.val].belief, agents[i.val].exposures⟩) :
    propagationSummary seed agents =
      .ok (@propagationObservation seed.nodeCount (seedGraph seed).Adj
        (seedAdjDec seed) population) := by
  obtain ⟨network, hn⟩ := parseSeed_complete seed hseed
  obtain ⟨parsed, hp⟩ := parseAgents_complete seed.nodeCount agents hsize hvalid
  have hm : 0 < (1 : Nat) ∧ 1 ≤ seed.nodeCount := by
    have := hseed.nodes
    omega
  rw [propagationSummary_of_parses seed agents network parsed hn hm hp]
  apply congrArg Except.ok
  apply propagationObservation_congr (dg := network.snapshot.adjDec)
    (dh := seedAdjDec seed)
  · exact congrArg SimpleGraph.Adj (parseSeed_graph seed network hn)
  · apply population_eq
    · funext i
      have hi : i.val < agents.size := by rw [hsize]; exact i.isLt
      exact ((parseAgents_sound _ _ _ hp).2.2 i hi).1.trans
        ((hvalues i hi).1.symm)
    · funext i
      have hi : i.val < agents.size := by rw [hsize]; exact i.isLt
      exact ((parseAgents_sound _ _ _ hp).2.2 i hi).2.trans
        ((hvalues i hi).2.symm)

private theorem reachable_edge {n : Nat} (g : SimpleGraph (Fin n))
    (i j : Fin n) (hedge : g.Adj i j) : g.Reachable i j :=
  ⟨.cons hedge .nil⟩

private theorem attachSource_connected :
    (seedGraph attachSource.seed).Connected := by
  let g : SimpleGraph (Fin 3) := seedGraph attachSource.seed
  letI : DecidableRel g.Adj := seedAdjDec attachSource.seed
  change g.Connected
  refine { preconnected := ?_, nonempty := inferInstance }
  intro i j
  fin_cases i <;> fin_cases j <;>
    solve
    | exact ⟨.nil⟩
    | apply reachable_edge g
      norm_num [g, seedGraph, attachSource, canonicalEdge, Fin.ext_iff]
    | exact ⟨.cons
        (show g.Adj 1 0 by norm_num [g, seedGraph, attachSource, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 0 2 by norm_num [g, seedGraph, attachSource, canonicalEdge, Fin.ext_iff]) .nil)⟩
    | exact ⟨.cons
        (show g.Adj 2 0 by norm_num [g, seedGraph, attachSource, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 0 1 by norm_num [g, seedGraph, attachSource, canonicalEdge, Fin.ext_iff]) .nil)⟩

private theorem attachRelay_connected :
    (seedGraph attachRelay.seed).Connected := by
  let g : SimpleGraph (Fin 3) := seedGraph attachRelay.seed
  letI : DecidableRel g.Adj := seedAdjDec attachRelay.seed
  change g.Connected
  refine { preconnected := ?_, nonempty := inferInstance }
  intro i j
  fin_cases i <;> fin_cases j <;>
    solve
    | exact ⟨.nil⟩
    | apply reachable_edge g
      norm_num [g, seedGraph, attachRelay, canonicalEdge, Fin.ext_iff]
    | exact ⟨.cons
        (show g.Adj 0 1 by norm_num [g, seedGraph, attachRelay, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 1 2 by norm_num [g, seedGraph, attachRelay, canonicalEdge, Fin.ext_iff]) .nil)⟩
    | exact ⟨.cons
        (show g.Adj 2 1 by norm_num [g, seedGraph, attachRelay, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 1 0 by norm_num [g, seedGraph, attachRelay, canonicalEdge, Fin.ext_iff]) .nil)⟩

private theorem secondBirth_connected :
    (seedGraph secondBirth.seed).Connected := by
  let g : SimpleGraph (Fin 4) := seedGraph secondBirth.seed
  letI : DecidableRel g.Adj := seedAdjDec secondBirth.seed
  change g.Connected
  refine { preconnected := ?_, nonempty := inferInstance }
  intro i j
  fin_cases i <;> fin_cases j <;>
    solve
    | exact ⟨.nil⟩
    | apply reachable_edge g
      norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff]
    | exact ⟨.cons
        (show g.Adj 0 1 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 1 2 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff]) .nil)⟩
    | exact ⟨.cons
        (show g.Adj 2 1 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 1 0 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff]) .nil)⟩
    | exact ⟨.cons
        (show g.Adj 1 2 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 2 3 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff]) .nil)⟩
    | exact ⟨.cons
        (show g.Adj 3 2 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 2 1 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff]) .nil)⟩
    | exact ⟨.cons
        (show g.Adj 0 1 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 1 2 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
          (.cons (show g.Adj 2 3 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff]) .nil))⟩
    | exact ⟨.cons
        (show g.Adj 3 2 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
        (.cons (show g.Adj 2 1 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff])
          (.cons (show g.Adj 1 0 by norm_num [g, seedGraph, secondBirth, canonicalEdge, Fin.ext_iff]) .nil))⟩

private theorem attachSource_seed_valid : attachSource.seed.Valid :=
  { nodes := by decide
    size := rfl
    fitness := by intro i; fin_cases i <;> norm_num [attachSource, positiveSeedFitness]
    edges := by intro i; fin_cases i <;> decide_cbv
    distinct := by decide_cbv
    connected := attachSource_connected }

private theorem attachRelay_seed_valid : attachRelay.seed.Valid :=
  { nodes := by decide
    size := rfl
    fitness := by intro i; fin_cases i <;> norm_num [attachRelay, positiveSeedFitness]
    edges := by intro i; fin_cases i <;> decide_cbv
    distinct := by decide_cbv
    connected := attachRelay_connected }

private theorem relayNextRound_seed_valid : relayNextRound.seed.Valid := by
  simpa [relayNextRound, attachRelay] using attachRelay_seed_valid

private theorem secondBirth_seed_valid : secondBirth.seed.Valid :=
  { nodes := by decide
    size := rfl
    fitness := by intro i; fin_cases i <;> norm_num [secondBirth, positiveSeedFitness]
    edges := by intro i; fin_cases i <;> decide_cbv
    distinct := by decide_cbv
    connected := secondBirth_connected }

private def attachInitialPopulation : Population 3 :=
  ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]⟩

private def relayPopulation : Population 3 :=
  ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨1, 1⟩, ⟨0, 0⟩]⟩

private def secondBirthPopulation : Population 4 :=
  ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨1, 1⟩, ⟨0, 0⟩, ⟨0, 0⟩]⟩

private def halfReceptivePopulation : Population 2 :=
  ⟨![⟨1, 1/2⟩, ⟨1/2, 1/2⟩], ![⟨1, 0⟩, ⟨0, 0⟩]⟩

private def zeroReceptivePopulation : Population 2 :=
  ⟨![⟨1, 1/2⟩, ⟨0, 1/2⟩], ![⟨1, 0⟩, ⟨0, 0⟩]⟩

private def silentPopulation : Population 2 :=
  ⟨fun _ => ⟨1, 1/2⟩, ![⟨0, 0⟩, ⟨0, 0⟩]⟩

private def zeroThresholdPopulation : Population 2 :=
  ⟨![⟨1, 0⟩, ⟨1, 1/2⟩], ![⟨0, 0⟩, ⟨0, 0⟩]⟩

-- Compute each finite set and updated state once, then assemble observations.
private theorem propagationObservation_of_next {n : Nat} (g : MeshGraph (Fin n))
    [DecidableRel g] (population : Population n) (next : Fin n → AgentState)
    (delivered : Finset (Fin n × Fin n))
    (hnext : nextAgent g population = next)
    (hdelivered : transmissions g population = delivered) :
    propagationObservation g population =
      (List.ofFn (fun i => (next i).belief),
       List.ofFn (fun i => (next i).exposures),
       List.ofFn (fun i => broadcasting (population.profiles i) (next i)),
       orderedPairs fun source target => decide ((source, target) ∈ delivered)) := by
  simp only [propagationObservation, propagate, hnext, hdelivered]

private theorem attachSource_observation :
    @propagationObservation 3 (seedGraph attachSource.seed).Adj
      (seedAdjDec attachSource.seed) attachInitialPopulation =
      ([1, 1, 1], [0, 1, 1], [true, true, true], [(0, 1), (0, 2)]) := by
  let g : MeshGraph (Fin 3) := (seedGraph attachSource.seed).Adj
  letI : DecidableRel g := seedAdjDec attachSource.seed
  have hincoming : incoming g attachInitialPopulation =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
    funext i
    ext j
    fin_cases i <;> fin_cases j <;>
      norm_num [incoming, g, seedGraph, attachSource, canonicalEdge,
        attachInitialPopulation, broadcasting, Fin.ext_iff]
  have hnext : nextAgent g attachInitialPopulation =
      (![⟨1, 0⟩, ⟨1, 1⟩, ⟨1, 1⟩] : Fin 3 → AgentState) := by
    funext i
    simp only [nextAgent, hincoming]
    fin_cases i <;> norm_num [attachInitialPopulation]
  have hdelivered : transmissions g attachInitialPopulation =
      ({(0, 1), (0, 2)} : Finset (Fin 3 × Fin 3)) := by
    ext pair
    rcases pair with ⟨j, i⟩
    fin_cases j <;> fin_cases i <;>
      norm_num [transmissions, g, seedGraph, attachSource, canonicalEdge,
        attachInitialPopulation, broadcasting, Fin.ext_iff]
  change propagationObservation g attachInitialPopulation = _
  rw [propagationObservation_of_next g attachInitialPopulation _ _ hnext hdelivered]
  decide_cbv

private theorem attachRelay_observation :
    @propagationObservation 3 (seedGraph attachRelay.seed).Adj
      (seedAdjDec attachRelay.seed) attachInitialPopulation =
      ([1, 1, 0], [0, 1, 0], [true, true, false], [(0, 1)]) := by
  let g : MeshGraph (Fin 3) := (seedGraph attachRelay.seed).Adj
  letI : DecidableRel g := seedAdjDec attachRelay.seed
  have hincoming : incoming g attachInitialPopulation =
      (![∅, {0}, ∅] : Fin 3 → Finset (Fin 3)) := by
    funext i
    ext j
    fin_cases i <;> fin_cases j <;>
      norm_num [incoming, g, seedGraph, attachRelay, canonicalEdge,
        attachInitialPopulation, broadcasting, Fin.ext_iff]
  have hnext : nextAgent g attachInitialPopulation =
      (![⟨1, 0⟩, ⟨1, 1⟩, ⟨0, 0⟩] : Fin 3 → AgentState) := by
    funext i
    simp only [nextAgent, hincoming]
    fin_cases i <;> norm_num [attachInitialPopulation]
  have hdelivered : transmissions g attachInitialPopulation =
      ({(0, 1)} : Finset (Fin 3 × Fin 3)) := by
    ext pair
    rcases pair with ⟨j, i⟩
    fin_cases j <;> fin_cases i <;>
      norm_num [transmissions, g, seedGraph, attachRelay, canonicalEdge,
        attachInitialPopulation, broadcasting, Fin.ext_iff]
  change propagationObservation g attachInitialPopulation = _
  rw [propagationObservation_of_next g attachInitialPopulation _ _ hnext hdelivered]
  decide_cbv

private theorem relayNextRound_observation :
    @propagationObservation 3 (seedGraph relayNextRound.seed).Adj
      (seedAdjDec relayNextRound.seed) relayPopulation =
      ([1, 1, 1], [1, 2, 1], [true, true, true], [(0, 1), (1, 0), (1, 2)]) := by
  let g : MeshGraph (Fin 3) := (seedGraph relayNextRound.seed).Adj
  letI : DecidableRel g := seedAdjDec relayNextRound.seed
  have hincoming : incoming g relayPopulation =
      (![{1}, {0}, {1}] : Fin 3 → Finset (Fin 3)) := by
    funext i
    ext j
    fin_cases i <;> fin_cases j <;>
      norm_num [incoming, g, seedGraph, relayNextRound, canonicalEdge,
        relayPopulation, broadcasting, Fin.ext_iff]
  have hnext : nextAgent g relayPopulation =
      (![⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 1⟩] : Fin 3 → AgentState) := by
    funext i
    simp only [nextAgent, hincoming]
    fin_cases i <;> norm_num [relayPopulation]
  have hdelivered : transmissions g relayPopulation =
      ({(0, 1), (1, 0), (1, 2)} : Finset (Fin 3 × Fin 3)) := by
    ext pair
    rcases pair with ⟨j, i⟩
    fin_cases j <;> fin_cases i <;>
      norm_num [transmissions, g, seedGraph, relayNextRound, canonicalEdge,
        relayPopulation, broadcasting, Fin.ext_iff]
  change propagationObservation g relayPopulation = _
  rw [propagationObservation_of_next g relayPopulation _ _ hnext hdelivered]
  decide_cbv

private theorem secondBirth_incoming :
    @incoming 4 (seedGraph secondBirth.seed).Adj (seedAdjDec secondBirth.seed)
      secondBirthPopulation =
      (![{1}, {0}, {1}, ∅] : Fin 4 → Finset (Fin 4)) := by
  let g : MeshGraph (Fin 4) := (seedGraph secondBirth.seed).Adj
  letI : DecidableRel g := seedAdjDec secondBirth.seed
  change incoming g secondBirthPopulation = _
  funext i
  ext j
  fin_cases i <;> fin_cases j <;>
    norm_num [incoming, g, seedGraph, secondBirth, canonicalEdge,
      secondBirthPopulation, broadcasting, Fin.ext_iff]

private theorem secondBirth_next :
    @nextAgent 4 (seedGraph secondBirth.seed).Adj (seedAdjDec secondBirth.seed)
      secondBirthPopulation =
      (![⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 1⟩, ⟨0, 0⟩] : Fin 4 → AgentState) := by
  funext i
  simp only [nextAgent, secondBirth_incoming]
  fin_cases i <;> norm_num [secondBirthPopulation]

private theorem secondBirth_delivered :
    @transmissions 4 (seedGraph secondBirth.seed).Adj (seedAdjDec secondBirth.seed)
      secondBirthPopulation =
      ({(0, 1), (1, 0), (1, 2)} : Finset (Fin 4 × Fin 4)) := by
  let g : MeshGraph (Fin 4) := (seedGraph secondBirth.seed).Adj
  letI : DecidableRel g := seedAdjDec secondBirth.seed
  change transmissions g secondBirthPopulation = _
  ext pair
  rcases pair with ⟨j, i⟩
  have hmem : (j, i) ∈ transmissions g secondBirthPopulation ↔
      j ∈ incoming g secondBirthPopulation i := by
    simp only [transmissions, incoming, Finset.mem_filter, Finset.mem_product,
      Finset.mem_univ, true_and]
  rw [hmem, secondBirth_incoming]
  fin_cases j <;> fin_cases i <;> norm_num [Fin.ext_iff]

private theorem secondBirth_observation :
    @propagationObservation 4 (seedGraph secondBirth.seed).Adj
      (seedAdjDec secondBirth.seed) secondBirthPopulation =
      ([1, 1, 1, 0], [1, 2, 1, 0], [true, true, true, false], [(0, 1), (1, 0), (1, 2)]) := by
  let g : MeshGraph (Fin 4) := (seedGraph secondBirth.seed).Adj
  letI : DecidableRel g := seedAdjDec secondBirth.seed
  change propagationObservation g secondBirthPopulation = _
  rw [propagationObservation_of_next g secondBirthPopulation _ _
    secondBirth_next secondBirth_delivered]
  decide_cbv

private theorem silent_observation :
    @propagationObservation 2 (seedGraph silent.seed).Adj
      (seedAdjDec silent.seed) silentPopulation =
      ([0, 0], [0, 0], [false, false], []) := by
  let g : MeshGraph (Fin 2) := (seedGraph silent.seed).Adj
  letI : DecidableRel g := seedAdjDec silent.seed
  have hincoming : incoming g silentPopulation =
      (![∅, ∅] : Fin 2 → Finset (Fin 2)) := by
    funext i
    ext j
    fin_cases i <;> fin_cases j <;>
      norm_num [incoming, g, seedGraph, silent, canonicalEdge,
        silentPopulation, broadcasting, Fin.ext_iff]
  have hnext : nextAgent g silentPopulation =
      (![⟨0, 0⟩, ⟨0, 0⟩] : Fin 2 → AgentState) := by
    funext i
    simp only [nextAgent, hincoming]
    fin_cases i <;> norm_num [silentPopulation]
  have hdelivered : transmissions g silentPopulation =
      (∅ : Finset (Fin 2 × Fin 2)) := by
    ext pair
    rcases pair with ⟨j, i⟩
    fin_cases j <;> fin_cases i <;>
      norm_num [transmissions, g, seedGraph, silent, canonicalEdge,
        silentPopulation, broadcasting, Fin.ext_iff]
  change propagationObservation g silentPopulation = _
  rw [propagationObservation_of_next g silentPopulation _ _ hnext hdelivered]
  decide_cbv

-- These literal raw snapshots and complete observations fix the acceptance
-- semantics independently of corpus generation.
example : attachSource =
    ⟨"attach-source", ⟨3, #[1, 1, 1], #[(0, 1), (0, 2)]⟩,
      #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary attachSource.seed attachSource.agents =
    .ok ([1, 1, 1], [0, 1, 1], [true, true, true], [(0, 1), (0, 2)]) := by
  have hvalid : ∀ a ∈ attachSource.agents.toList, a.Valid := by
    intro a ha
    simp [attachSource] at ha
    rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]
  rw [propagationSummary_flat attachSource.seed attachSource.agents
    attachSource_seed_valid rfl hvalid attachInitialPopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  exact attachSource_observation

example : attachRelay =
    ⟨"attach-relay", ⟨3, #[1, 1, 1], #[(0, 1), (1, 2)]⟩,
      #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary attachRelay.seed attachRelay.agents =
    .ok ([1, 1, 0], [0, 1, 0], [true, true, false], [(0, 1)]) := by
  have hvalid : ∀ a ∈ attachRelay.agents.toList, a.Valid := by
    intro a ha
    simp [attachRelay] at ha
    rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]
  rw [propagationSummary_flat attachRelay.seed attachRelay.agents
    attachRelay_seed_valid rfl hvalid attachInitialPopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  exact attachRelay_observation

example : relayNextRound =
    ⟨"relay-next-round", ⟨3, #[1, 1, 1], #[(0, 1), (1, 2)]⟩,
      #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 1, 1⟩, ⟨1, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary relayNextRound.seed relayNextRound.agents =
    .ok ([1, 1, 1], [1, 2, 1], [true, true, true],
      [(0, 1), (1, 0), (1, 2)]) := by
  have hvalid : ∀ a ∈ relayNextRound.agents.toList, a.Valid := by
    intro a ha
    simp [relayNextRound] at ha
    rcases ha with rfl | rfl | rfl <;> norm_num [RawAgent.Valid]
  rw [propagationSummary_flat relayNextRound.seed relayNextRound.agents
    relayNextRound_seed_valid rfl hvalid relayPopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  exact relayNextRound_observation

example : secondBirth =
    ⟨"second-birth", ⟨4, #[1, 1, 1, 1], #[(0, 1), (1, 2), (2, 3)]⟩,
      #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 1, 1⟩, ⟨1, 1/2, 0, 0⟩,
        ⟨1, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary secondBirth.seed secondBirth.agents =
    .ok ([1, 1, 1, 0], [1, 2, 1, 0], [true, true, true, false],
      [(0, 1), (1, 0), (1, 2)]) := by
  have hvalid : ∀ a ∈ secondBirth.agents.toList, a.Valid := by
    intro a ha
    simp [secondBirth] at ha
    rcases ha with rfl | rfl | rfl <;> norm_num [RawAgent.Valid]
  rw [propagationSummary_flat secondBirth.seed secondBirth.agents
    secondBirth_seed_valid rfl hvalid secondBirthPopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  exact secondBirth_observation

example : halfReceptive =
    ⟨"half-receptive", ⟨2, #[1, 1], #[(0, 1)]⟩,
      #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary halfReceptive.seed halfReceptive.agents =
    .ok ([1, 1/2], [0, 1], [true, true], [(0, 1)]) := by
  have hvalid : ∀ a ∈ halfReceptive.agents.toList, a.Valid := by
    intro a ha
    simp [halfReceptive] at ha
    rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]
  rw [propagationSummary_flat halfReceptive.seed halfReceptive.agents
    (parseSeed_sound seedRaw (seedNetwork one) parsed_seedRaw) rfl hvalid halfReceptivePopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  decide_cbv

example : zeroReceptive =
    ⟨"zero-receptive", ⟨2, #[1, 1], #[(0, 1)]⟩,
      #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary zeroReceptive.seed zeroReceptive.agents =
    .ok ([1, 0], [0, 1], [true, false], [(0, 1)]) := by
  have hvalid : ∀ a ∈ zeroReceptive.agents.toList, a.Valid := by
    intro a ha
    simp [zeroReceptive] at ha
    rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]
  rw [propagationSummary_flat zeroReceptive.seed zeroReceptive.agents
    (parseSeed_sound seedRaw (seedNetwork one) parsed_seedRaw) rfl hvalid zeroReceptivePopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  decide_cbv

example : silent =
    ⟨"silent", ⟨2, #[1, 1], #[(0, 1)]⟩,
      #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary silent.seed silent.agents =
    .ok ([0, 0], [0, 0], [false, false], []) := by
  have hvalid : ∀ a ∈ silent.agents.toList, a.Valid := by
    intro a ha
    simp [silent] at ha
    rcases ha with rfl
    norm_num [RawAgent.Valid]
  rw [propagationSummary_flat silent.seed silent.agents
    (parseSeed_sound seedRaw (seedNetwork one) parsed_seedRaw) rfl hvalid silentPopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  exact silent_observation

example : zeroThreshold =
    ⟨"zero-threshold", ⟨2, #[1, 1], #[(0, 1)]⟩,
      #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩ := rfl
example : propagationSummary zeroThreshold.seed zeroThreshold.agents =
    .ok ([0, 0], [0, 1], [true, false], [(0, 1)]) := by
  have hvalid : ∀ a ∈ zeroThreshold.agents.toList, a.Valid := by
    intro a ha
    simp [zeroThreshold] at ha
    rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]
  rw [propagationSummary_flat zeroThreshold.seed zeroThreshold.agents
    (parseSeed_sound seedRaw (seedNetwork one) parsed_seedRaw) rfl hvalid zeroThresholdPopulation (by
      intro i hi
      fin_cases i <;> constructor <;> rfl)]
  apply congrArg Except.ok
  decide_cbv

end FiniteReplayFixtures

#print axioms NarrativeDynamics.FitnessABM.parseAgent_sound
#print axioms NarrativeDynamics.FitnessABM.parseAgent_complete
#print axioms NarrativeDynamics.FitnessABM.parseAgents_sound
#print axioms NarrativeDynamics.FitnessABM.parseAgents_complete
#print axioms NarrativeDynamics.FitnessABM.checkedBirth_spec
#print axioms NarrativeDynamics.FitnessABM.replay_success_iff_valid
#print axioms NarrativeDynamics.FitnessABM.replay_projection
#print axioms NarrativeDynamics.FitnessABM.replay_counts
#print axioms NarrativeDynamics.FitnessABM.replay_probability_pos
#print axioms NarrativeDynamics.FitnessABM.runInputs_append_error

end NarrativeDynamics.Tests.FitnessABMReplay
