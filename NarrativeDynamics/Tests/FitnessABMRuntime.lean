import NarrativeDynamics.Conformance.FitnessABMRuntimeVectors

namespace NarrativeDynamics.Tests.FitnessABMRuntime

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open NarrativeDynamics.FitnessABM
open NarrativeDynamics.Conformance.BBRuntime
open scoped BigOperators

def summary (input : RuntimeCaseInput) :
    Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat) :=
  (FitnessABM.replay input.seed input.m input.agents input.ticks).map fun out =>
    (out.final.nodeCount, out.final.roundIndex,
      actualEdgeCount out.final.state.network.snapshot,
      List.ofFn (fun i => (out.final.state.population.agents i).belief),
      List.ofFn (fun i => (out.final.state.population.agents i).exposures),
      out.probability)

private theorem summary_result_ok (out : Result) (nodes rounds edges : Nat)
    (beliefs : List Rat) (exposures : List Nat) (mass : Rat) :
    (Except.ok out : Except JointError Result).map (fun result =>
      (result.final.nodeCount, result.final.roundIndex,
        actualEdgeCount result.final.state.network.snapshot,
        List.ofFn (fun i => (result.final.state.population.agents i).belief),
        List.ofFn (fun i => (result.final.state.population.agents i).exposures),
        result.probability)) = .ok (nodes, rounds, edges, beliefs, exposures, mass) ↔
      out.final.nodeCount = nodes ∧ out.final.roundIndex = rounds ∧
      actualEdgeCount out.final.state.network.snapshot = edges ∧
      List.ofFn (fun i => (out.final.state.population.agents i).belief) = beliefs ∧
      List.ofFn (fun i => (out.final.state.population.agents i).exposures) = exposures ∧
      out.probability = mass := by
  change (Except.ok (_, _, _, _, _, _) : Except JointError _) =
    Except.ok (_, _, _, _, _, _) ↔ _
  simp only [Except.ok.injEq, Prod.mk.injEq]

private def exceptDecidableEq {α : Type} [DecidableEq α] : DecidableEq (Except JointError α) :=
  fun x y => match x, y with
  | .error x, .error y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.error.inj he))
  | .ok x, .ok y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.ok.inj he))
  | .error _, .ok _ => .isFalse (by intro h; cases h)
  | .ok _, .error _ => .isFalse (by intro h; cases h)

-- Register only the finite observable result types used by these literals.
private instance : DecidableEq
    (Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat)) :=
  @exceptDecidableEq _ (fun _ _ => inferInstance)
private instance : DecidableEq
    (Except JointError (List Rat × List Nat × Rat × List (Nat × Nat × Rat))) :=
  @exceptDecidableEq _ (fun _ _ => inferInstance)
private instance : DecidableEq (Except JointError (List Nat)) :=
  @exceptDecidableEq _ (fun _ _ => inferInstance)
private instance : DecidableEq (Except JointError (List (Nat × Nat × Rat))) :=
  @exceptDecidableEq _ (fun _ _ => inferInstance)

section FiniteLiterals
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

private def roster (raw : Array RawAgent) (hs : raw.size = 2)
    (hv : ∀ a ∈ raw.toList, a.Valid) : {p : Population 2 // p.Valid} :=
  let record (i : Fin 2) := raw[i.val]'(by have := i.isLt; omega)
  have valid (i : Fin 2) : (record i).Valid :=
    hv _ (Array.getElem_mem_toList (by have := i.isLt; omega))
  ⟨⟨fun i => ⟨(record i).receptivity, (record i).threshold⟩,
    fun i => ⟨(record i).belief, (record i).exposures⟩⟩,
    ⟨fun i => ⟨(valid i).1.1, (valid i).1.2,
      (valid i).2.1.1, (valid i).2.1.2⟩, fun i => (valid i).2.2⟩⟩

private theorem parsed_roster (raw : Array RawAgent) (hs : raw.size = 2)
    (hv : ∀ a ∈ raw.toList, a.Valid) :
    parseAgents 2 raw = .ok (roster raw hs hv) := by
  obtain ⟨p, hp⟩ := parseAgents_complete 2 raw hs hv
  have values := (parseAgents_sound 2 raw p hp).2.2
  have he : p.val = (roster raw hs hv).val := by
    rcases p with ⟨⟨profiles, agents⟩, valid⟩
    have hprofiles : profiles = (roster raw hs hv).val.profiles := by
      funext i
      exact (values i (by have := i.isLt; omega)).1
    have hagents : agents = (roster raw hs hv).val.agents := by
      funext i
      exact (values i (by have := i.isLt; omega)).2
    cases hprofiles
    cases hagents
    rfl
  exact hp.trans (congrArg Except.ok (Subtype.ext he))

private def initial (eta : PosFitness) (raw : Array RawAgent)
    (hs : raw.size = 2) (hv : ∀ a ∈ raw.toList, a.Valid) : JointState 2 :=
  ⟨seedNetwork eta, (roster raw hs hv).val, (roster raw hs hv).property⟩

private theorem replay_start (eta : PosFitness) (raw : Array RawAgent)
    (hs : raw.size = 2) (hv : ∀ a ∈ raw.toList, a.Valid)
    (m : Nat) (hm : 0 < m ∧ m ≤ 2) (ticks : List RawTick) :
    FitnessABM.replay ⟨2, #[1, eta.val], #[(0, 1)]⟩ m raw ticks =
      runInputs m 0 0 ⟨2, 0, initial eta raw hs hv⟩ ticks := by
  simp only [FitnessABM.replay, parsed_seed, if_pos hm, parsed_roster raw hs hv]
  rfl

private theorem parsed_unit : parseSeed unitSeed = .ok (seedNetwork one) :=
  parsed_seed one

private theorem replay_input (input : RuntimeCaseInput) (eta : PosFitness)
    (hn : input.seed = ⟨2, #[1, eta.val], #[(0, 1)]⟩)
    (hs : input.agents.size = 2) (hv : ∀ a ∈ input.agents.toList, a.Valid)
    (hm : 0 < input.m ∧ input.m ≤ 2) :
    FitnessABM.replay input.seed input.m input.agents input.ticks =
      runInputs input.m 0 0 ⟨2, 0, initial eta input.agents hs hv⟩ input.ticks := by
  rw [hn]
  exact replay_start eta input.agents hs hv input.m hm input.ticks

private theorem prefix_start (input : RuntimeCaseInput) (eta : PosFitness)
    (hn : input.seed = ⟨2, #[1, eta.val], #[(0, 1)]⟩)
    (hs : input.agents.size = 2) (hv : ∀ a ∈ input.agents.toList, a.Valid)
    (hm : 0 < input.m ∧ input.m ≤ 2) (count : Nat) :
    replayPrefix input count =
      runInputs input.m 0 0 ⟨2, 0, initial eta input.agents hs hv⟩ (input.ticks.take count) := by
  unfold replayPrefix
  rw [hn]
  exact replay_start eta input.agents hs hv input.m hm _

private def singletonTarget {n : Nat} (j : Fin n) : Targets n 1 :=
  ⟨fun _ => j, fun a b _ => Subsingleton.elim a b⟩

private def newbornData (b : Rat) (hb : 0 ≤ b ∧ b ≤ 1) : BirthData :=
  ⟨one, ⟨⟨1, 1/2⟩, b, by norm_num [AgentProfile.Valid], hb⟩⟩

private theorem checked_single {n : Nat} (s : JointState n) (j : Fin n)
    (b : Rat) (hb : 0 ≤ b ∧ b ≤ 1) :
    checkedBirth s 1 ⟨⟨1, #[j.val]⟩, 1, 1/2, b⟩ = .ok
      (grow s (singletonTarget j) positiveM (newbornData b hb),
        orderedMass s.network (singletonTarget j)) := by
  apply (checkedBirth_spec _ _ _).mpr
  have hm : 0 < (1 : Nat) ∧ 1 ≤ n := ⟨positiveM, by have := j.isLt; omega⟩
  have bound : targetsBounded n #[j.val] := by
    intro i
    fin_cases i
    exact j.isLt
  have distinct : targetsDistinct #[j.val] := by
    intro a b _
    fin_cases a
    fin_cases b
    rfl
  have he : (⟨bound, distinct⟩ : CheckedTargets n #[j.val]).embedding =
      singletonTarget j := by
    apply Function.Embedding.ext
    intro i
    fin_cases i
    rfl
  have hp : parseAgent ⟨1, 1/2, b, 0⟩ =
      .ok ⟨(⟨1, 1/2⟩, ⟨b, 0⟩), by
        constructor
        · norm_num [AgentProfile.Valid]
        · exact hb⟩ := by
    simp only [parseAgent,
      dif_pos (show (0 : Rat) ≤ 1 ∧ (1 : Rat) ≤ 1 from by norm_num),
      dif_pos (show (0 : Rat) ≤ 1/2 ∧ (1/2 : Rat) ≤ 1 from by norm_num), dif_pos hb]
  refine ⟨⟨singletonTarget j, one, positiveM⟩, _, ?_, hp, rfl⟩
  simp only [validateBirth, dif_pos hm, dif_pos (show (0 : Rat) < 1 from by norm_num),
    dif_pos (show (#[j.val] : Array Nat).size = 1 from rfl),
    checkTargets, dif_pos bound, dif_pos distinct]
  rw [he]
  rfl

private theorem checked0 (s : JointState 2) : checkedBirth s 1 (birth 0) = .ok
    (grow s (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num)),
      orderedMass s.network (singletonTarget (0 : Fin 2))) :=
  checked_single s (0 : Fin 2) 0 (by norm_num)

private theorem checked1 (s : JointState 2) : checkedBirth s 1 (birth 1) = .ok
    (grow s (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num)),
      orderedMass s.network (singletonTarget (1 : Fin 2))) :=
  checked_single s (1 : Fin 2) 0 (by norm_num)

private theorem checked2 (s : JointState 3) : checkedBirth s 1 (birth 2) = .ok
    (grow s (singletonTarget (2 : Fin 3)) positiveM (newbornData 0 (by norm_num)),
      orderedMass s.network (singletonTarget (2 : Fin 3))) :=
  checked_single s (2 : Fin 3) 0 (by norm_num)

private theorem checked_broadcast (s : JointState 2) :
    checkedBirth s 1 ⟨⟨1, #[1]⟩, 1, 1/2, 1⟩ = .ok
      (grow s (singletonTarget (1 : Fin 2)) positiveM (newbornData 1 (by norm_num)),
        orderedMass s.network (singletonTarget (1 : Fin 2))) :=
  checked_single s (1 : Fin 2) 1 (by norm_num)

private def targets01 : Targets 2 2 := ⟨![0, 1], by decide⟩
private def targets10 : Targets 2 2 := ⟨![1, 0], by decide⟩
private theorem two_pos : 0 < (2 : Nat) := by decide

private theorem parsed_newborn : parseAgent ⟨1, 1/2, 0, 0⟩ =
    .ok ⟨(⟨1, 1/2⟩, ⟨0, 0⟩), by norm_num [AgentProfile.Valid, AgentState.Valid]⟩ := by
  norm_num [parseAgent]

private theorem checked01 (s : JointState 2) :
    checkedBirth s 2 (⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩ : RawBirthInput) = .ok
      (grow s targets01 two_pos (newbornData 0 (by norm_num)),
        orderedMass s.network targets01) := by
  apply (checkedBirth_spec _ _ _).mpr
  refine ⟨⟨targets01, one, two_pos⟩, _, ?_, parsed_newborn, rfl⟩
  have hm : 0 < (2 : Nat) ∧ 2 ≤ 2 := by decide
  have hf : 0 < (⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.fitness := by norm_num
  have hs : (⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets.size = 2 := rfl
  have hb : targetsBounded 2 (⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets := by decide_cbv
  have hd : targetsDistinct (⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets := by decide_cbv
  have he : (⟨hb, hd⟩ : CheckedTargets 2 (⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets).embedding =
      targets01 := by
    apply Function.Embedding.ext
    intro i
    fin_cases i <;> rfl
  simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]
  rw [he]
  rfl

private theorem checked10 (s : JointState 2) :
    checkedBirth s 2 (⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩ : RawBirthInput) = .ok
      (grow s targets10 two_pos (newbornData 0 (by norm_num)),
        orderedMass s.network targets10) := by
  apply (checkedBirth_spec _ _ _).mpr
  refine ⟨⟨targets10, one, two_pos⟩, _, ?_, parsed_newborn, rfl⟩
  have hm : 0 < (2 : Nat) ∧ 2 ≤ 2 := by decide
  have hf : 0 < (⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.fitness := by norm_num
  have hs : (⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets.size = 2 := rfl
  have hb : targetsBounded 2 (⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets := by decide_cbv
  have hd : targetsDistinct (⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets := by decide_cbv
  have he : (⟨hb, hd⟩ : CheckedTargets 2 (⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩ : RawBirthInput).birth.targets).embedding =
      targets10 := by
    apply Function.Embedding.ext
    intro i
    fin_cases i <;> rfl
  simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]
  rw [he]
  rfl

private theorem empty_agents_valid : ∀ a ∈ empty.agents.toList, a.Valid := by
  intro a ha
  simp [empty, agentsRaw] at ha
  rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]

private theorem zeroReceptive_agents_valid : ∀ a ∈ zeroReceptive.agents.toList, a.Valid := by
  intro a ha
  simp [zeroReceptive, attachSource, empty, agentsRaw] at ha
  rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]

private theorem silent_agents_valid : ∀ a ∈ silent.agents.toList, a.Valid := by
  intro a ha
  simp [silent, attachSource, empty, agentsRaw] at ha
  rcases ha with rfl
  norm_num [RawAgent.Valid]

private theorem zeroThreshold_agents_valid : ∀ a ∈ zeroThreshold.agents.toList, a.Valid := by
  intro a ha
  simp [zeroThreshold, attachSource, empty, agentsRaw] at ha
  rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]

private theorem retainedExposures_agents_valid : ∀ a ∈ retainedExposures.agents.toList, a.Valid := by
  intro a ha
  simp [retainedExposures, attachRelay, empty, agentsRaw] at ha
  rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]

private theorem halfReceptive_agents_valid : ∀ a ∈ halfReceptive.agents.toList, a.Valid := by
  intro a ha
  simp [halfReceptive, attachSource, empty, agentsRaw] at ha
  rcases ha with rfl | rfl <;> norm_num [RawAgent.Valid]

private theorem lastCases_eq_if {α : Sort u} {n : Nat}
    (last : α) (old : Fin n → α) (i : Fin (n + 1)) :
    Fin.lastCases last old i =
      if h : i.val < n then old ⟨i.val, h⟩ else last := by
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simp
  · simp [j.isLt]

private theorem grow_agents {n m : Nat} (s : JointState n) (T : Targets n m)
    (hm : 0 < m) (b : BirthData) :
    (grow s T hm b).population.agents =
      Fin.lastCases ⟨b.agent.initialBelief, 0⟩ s.population.agents := rfl

private theorem grow_profiles {n m : Nat} (s : JointState n) (T : Targets n m)
    (hm : 0 < m) (b : BirthData) :
    (grow s T hm b).population.profiles =
      Fin.lastCases b.agent.profile s.population.profiles := rfl

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

private theorem singleton_selected {n : Nat} (target : Fin n) :
    (singletonTarget target).selected = {target} := by
  ext i
  simp only [Targets.selected, Finset.mem_image, Finset.mem_univ, true_and,
    Finset.mem_singleton]
  constructor
  · rintro ⟨j, hj⟩
    exact hj.symm
  · intro hi
    subst i
    exact ⟨0, rfl⟩

private theorem initial_network (eta : PosFitness) (raw : Array RawAgent)
    (hs : raw.size = 2) (hv : ∀ a ∈ raw.toList, a.Valid) :
    (initial eta raw hs hv).network = seedNetwork eta := rfl

private theorem default_profiles (eta : PosFitness) :
    (initial eta agentsRaw rfl empty_agents_valid).population.profiles =
      fun _ => ⟨1, 1/2⟩ := by
  funext i
  fin_cases i <;> rfl

private theorem default_agents (eta : PosFitness) :
    (initial eta agentsRaw rfl empty_agents_valid).population.agents =
      ![⟨1, 0⟩, ⟨0, 0⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem top2_adj (i j : Fin 2) :
    (⊤ : SimpleGraph (Fin 2)).Adj i j ↔ i ≠ j := Iff.rfl

private theorem growth0_incoming (eta : PosFitness) :
    let s := grow (initial eta agentsRaw rfl empty_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      lastCases_eq_if, initial_network, default_profiles, default_agents,
      seedNetwork, extendPopulation, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem growth1_incoming (eta : PosFitness) :
    let s := grow (initial eta agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, ∅] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow, applyBirth, birthSnapshot, birthGraph, birthAdj,
      lastCases_eq_if, initial_network, default_profiles, default_agents,
      seedNetwork, extendPopulation, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem default_growth_incoming (eta : PosFitness) (target : Fin 2) :
    let s := grow (initial eta agentsRaw rfl empty_agents_valid)
      (singletonTarget target) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, if target = 0 then {0} else ∅] : Fin 3 → Finset (Fin 3)) := by
  fin_cases target <;>
    first | exact growth0_incoming eta | exact growth1_incoming eta

private theorem default_growth_agents (eta : PosFitness) (target : Fin 2) :
    (advance (grow (initial eta agentsRaw rfl empty_agents_valid)
      (singletonTarget target) positiveM (newbornData 0 (by norm_num)))).population.agents =
      ![⟨1, 0⟩, ⟨1, 1⟩, ⟨if target = 0 then 1 else 0, if target = 0 then 1 else 0⟩] := by
  rw [advance_agents _ _ (default_growth_incoming eta target)]
  simp only [grow, extendPopulation, default_profiles, default_agents]
  funext i
  fin_cases target <;> fin_cases i <;> decide_cbv

private theorem relay_idle_incoming (eta : PosFitness) :
    let s := advance (grow (initial eta agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num)))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1}, {0}, {1}] : Fin 3 → Finset (Fin 3)) := by
  have h0_2 : ( 0 : Nat) < 2 := by decide
  have h1_2 : ( 1 : Nat) < 2 := by decide
  have h2_2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_profiles, advance_projection,
      advance_profiles, default_growth_agents, default_profiles, initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0_2, h1_2, h2_2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem relay_idle_agents (eta : PosFitness) :
    (advance (advance (grow (initial eta agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM
      (newbornData 0 (by norm_num))))).population.agents =
      ![⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 1⟩] := by
  rw [advance_agents _ _ (relay_idle_incoming eta)]
  simp only [grow_profiles, advance_profiles, default_growth_agents,
    default_profiles]
  funext i
  fin_cases i <;> decide_cbv

private theorem successive_incoming :
    let s := grow (advance (grow (initial one agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num))))
      (singletonTarget (2 : Fin 3)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1}, {0}, {1}, ∅] : Fin 4 → Finset (Fin 4)) := by
  have h0_2 : ( 0 : Nat) < 2 := by decide
  have h1_2 : ( 1 : Nat) < 2 := by decide
  have h2_2 : ¬ (2 : Nat) < 2 := by decide
  have h0_3 : ( 0 : Nat) < 3 := by decide
  have h1_3 : ( 1 : Nat) < 3 := by decide
  have h2_3 : ( 2 : Nat) < 3 := by decide
  have h3_3 : ¬ (3 : Nat) < 3 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, advance_projection,
      advance_profiles, default_growth_agents, default_profiles, initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0_2, h1_2, h2_2, h0_3, h1_3, h2_3, h3_3, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem successive_agents :
    (advance (grow (advance (grow (initial one agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num))))
      (singletonTarget (2 : Fin 3)) positiveM
      (newbornData 0 (by norm_num)))).population.agents =
      ![⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 1⟩, ⟨0, 0⟩] := by
  rw [advance_agents _ _ successive_incoming]
  simp only [grow_agents, grow_profiles, advance_profiles, default_growth_agents,
    default_profiles]
  funext i
  fin_cases i <;> decide_cbv

private theorem successive_idle_incoming :
    let s := advance (grow (advance (grow (initial one agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num))))
      (singletonTarget (2 : Fin 3)) positiveM (newbornData 0 (by norm_num)))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![{1}, {0, 2}, {1}, {2}] : Fin 4 → Finset (Fin 4)) := by
  have h0_2 : ( 0 : Nat) < 2 := by decide
  have h1_2 : ( 1 : Nat) < 2 := by decide
  have h2_2 : ¬ (2 : Nat) < 2 := by decide
  have h0_3 : ( 0 : Nat) < 3 := by decide
  have h1_3 : ( 1 : Nat) < 3 := by decide
  have h2_3 : ( 2 : Nat) < 3 := by decide
  have h3_3 : ¬ (3 : Nat) < 3 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_profiles, advance_projection,
      advance_profiles, successive_agents, default_profiles, initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0_2, h1_2, h2_2, h0_3, h1_3, h2_3, h3_3, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem successive_idle_agents :
    (advance (advance (grow (advance (grow (initial one agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num))))
      (singletonTarget (2 : Fin 3)) positiveM
      (newbornData 0 (by norm_num))))).population.agents =
      ![⟨1, 2⟩, ⟨1, 4⟩, ⟨1, 2⟩, ⟨1, 1⟩] := by
  rw [advance_agents _ _ successive_idle_incoming]
  simp only [grow_profiles, advance_profiles, successive_agents,
    default_profiles]
  funext i
  fin_cases i <;> decide_cbv

private theorem zeroReceptive_initial_profiles :
    (initial one #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩] rfl zeroReceptive_agents_valid).population.profiles = ![⟨1, 1/2⟩, ⟨0, 1/2⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem zeroReceptive_initial_agents :
    (initial one #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩] rfl zeroReceptive_agents_valid).population.agents = ![⟨1, 0⟩, ⟨0, 0⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem zeroReceptive_incoming :
    let s := grow (initial one #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩] rfl zeroReceptive_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, zeroReceptive_initial_profiles, zeroReceptive_initial_agents,
      initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem zeroReceptive_step_agents :
    (advance (grow (initial one #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩] rfl zeroReceptive_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num)))).population.agents =
      ![⟨1, 0⟩, ⟨0, 1⟩, ⟨1, 1⟩] := by
  rw [advance_agents _ _ zeroReceptive_incoming]
  simp only [grow_agents, grow_profiles, zeroReceptive_initial_profiles, zeroReceptive_initial_agents]
  funext i
  fin_cases i <;> decide_cbv

private theorem silent_initial_profiles :
    (initial one #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl silent_agents_valid).population.profiles = ![⟨1, 1/2⟩, ⟨1, 1/2⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem silent_initial_agents :
    (initial one #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl silent_agents_valid).population.agents = ![⟨0, 0⟩, ⟨0, 0⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem silent_incoming :
    let s := grow (initial one #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl silent_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, ∅, ∅] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, silent_initial_profiles, silent_initial_agents,
      initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem silent_step_agents :
    (advance (grow (initial one #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl silent_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num)))).population.agents =
      ![⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩] := by
  rw [advance_agents _ _ silent_incoming]
  simp only [grow_agents, grow_profiles, silent_initial_profiles, silent_initial_agents]
  funext i
  fin_cases i <;> decide_cbv

private theorem zeroThreshold_initial_profiles :
    (initial one #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl zeroThreshold_agents_valid).population.profiles = ![⟨1, 0⟩, ⟨1, 1/2⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem zeroThreshold_initial_agents :
    (initial one #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl zeroThreshold_agents_valid).population.agents = ![⟨0, 0⟩, ⟨0, 0⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem zeroThreshold_incoming :
    let s := grow (initial one #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl zeroThreshold_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, zeroThreshold_initial_profiles, zeroThreshold_initial_agents,
      initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem zeroThreshold_step_agents :
    (advance (grow (initial one #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] rfl zeroThreshold_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num)))).population.agents =
      ![⟨0, 0⟩, ⟨0, 1⟩, ⟨0, 1⟩] := by
  rw [advance_agents _ _ zeroThreshold_incoming]
  simp only [grow_agents, grow_profiles, zeroThreshold_initial_profiles, zeroThreshold_initial_agents]
  funext i
  fin_cases i <;> decide_cbv

private theorem broadcastNewborn_incoming :
    let s := grow (initial one agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 1 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0, 2}, ∅] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, default_profiles, default_agents,
      initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem broadcastNewborn_step_agents :
    (advance (grow (initial one agentsRaw rfl empty_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 1 (by norm_num)))).population.agents =
      ![⟨1, 0⟩, ⟨1, 2⟩, ⟨1, 0⟩] := by
  rw [advance_agents _ _ broadcastNewborn_incoming]
  simp only [grow_agents, grow_profiles, default_profiles, default_agents]
  funext i
  fin_cases i <;> decide_cbv

private theorem retainedExposures_initial_profiles :
    (initial one #[⟨1, 1/2, 1, 7⟩, ⟨1, 1/2, 0, 3⟩] rfl retainedExposures_agents_valid).population.profiles = ![⟨1, 1/2⟩, ⟨1, 1/2⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem retainedExposures_initial_agents :
    (initial one #[⟨1, 1/2, 1, 7⟩, ⟨1, 1/2, 0, 3⟩] rfl retainedExposures_agents_valid).population.agents = ![⟨1, 7⟩, ⟨0, 3⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem retainedExposures_incoming :
    let s := grow (initial one #[⟨1, 1/2, 1, 7⟩, ⟨1, 1/2, 0, 3⟩] rfl retainedExposures_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, ∅] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, retainedExposures_initial_profiles, retainedExposures_initial_agents,
      initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem retainedExposures_step_agents :
    (advance (grow (initial one #[⟨1, 1/2, 1, 7⟩, ⟨1, 1/2, 0, 3⟩] rfl retainedExposures_agents_valid)
      (singletonTarget (1 : Fin 2)) positiveM (newbornData 0 (by norm_num)))).population.agents =
      ![⟨1, 7⟩, ⟨1, 4⟩, ⟨0, 0⟩] := by
  rw [advance_agents _ _ retainedExposures_incoming]
  simp only [grow_agents, grow_profiles, retainedExposures_initial_profiles, retainedExposures_initial_agents]
  funext i
  fin_cases i <;> decide_cbv

private theorem halfReceptive_initial_profiles :
    (initial one #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩] rfl halfReceptive_agents_valid).population.profiles = ![⟨1, 1/2⟩, ⟨1/2, 1/2⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem halfReceptive_initial_agents :
    (initial one #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩] rfl halfReceptive_agents_valid).population.agents = ![⟨1, 0⟩, ⟨0, 0⟩] := by
  funext i
  fin_cases i <;> rfl

private theorem halfReceptive_incoming :
    let s := grow (initial one #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩] rfl halfReceptive_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, halfReceptive_initial_profiles, halfReceptive_initial_agents,
      initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, singleton_selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem halfReceptive_step_agents :
    (advance (grow (initial one #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩] rfl halfReceptive_agents_valid)
      (singletonTarget (0 : Fin 2)) positiveM (newbornData 0 (by norm_num)))).population.agents =
      ![⟨1, 0⟩, ⟨1/2, 1⟩, ⟨1, 1⟩] := by
  rw [advance_agents _ _ halfReceptive_incoming]
  simp only [grow_agents, grow_profiles, halfReceptive_initial_profiles, halfReceptive_initial_agents]
  funext i
  fin_cases i <;> decide_cbv

private theorem targets01_selected : targets01.selected = {0, 1} := by decide_cbv
private theorem targets10_selected : targets10.selected = {0, 1} := by decide_cbv

private theorem both_growth_incoming (eta : PosFitness) (targets : Targets 2 2)
    (selected : targets.selected = {0, 1}) :
    let s := grow (initial eta agentsRaw rfl empty_agents_valid)
      targets (by decide) (newbornData 0 (by norm_num))
    letI := s.network.snapshot.adjDec
    incoming s.network.snapshot.graph.Adj s.population =
      (![∅, {0}, {0}] : Fin 3 → Finset (Fin 3)) := by
  have h0 : (0 : Nat) < 2 := by decide
  have h1 : (1 : Nat) < 2 := by decide
  have h2 : ¬ (2 : Nat) < 2 := by decide
  funext i
  ext j
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and]
  fin_cases i <;> fin_cases j <;>
    (simp only [grow_projection, grow_agents, grow_profiles, default_profiles, default_agents,
      initial_network]
     simp only [applyBirth, birthSnapshot, birthGraph, birthAdj, lastCases_eq_if,
      seedNetwork, newbornData, selected, broadcasting, top2_adj]
     simp only [h0, h1, h2, dif_pos, dif_neg (show ¬ False from fun h => h)]
     norm_num [Fin.ext_iff])

private theorem both_growth_agents (eta : PosFitness) (targets : Targets 2 2)
    (selected : targets.selected = {0, 1}) :
    (advance (grow (initial eta agentsRaw rfl empty_agents_valid)
      targets (by decide) (newbornData 0 (by norm_num)))).population.agents =
      ![⟨1, 0⟩, ⟨1, 1⟩, ⟨1, 1⟩] := by
  rw [advance_agents _ _ (both_growth_incoming eta targets selected)]
  simp only [grow_agents, grow_profiles, default_profiles, default_agents]
  funext i
  fin_cases i <;> decide_cbv

private theorem empty_literal : summary empty =
    .ok (2, 0, 1, [1, 0], [0, 0], 1) := by
  unfold summary
  rw [replay_input empty one rfl rfl empty_agents_valid (by decide)]
  dsimp only [empty]
  decide_cbv

private theorem idleTwo_literal : summary idleTwo =
    .ok (2, 2, 1, [1, 1], [1, 2], 1) := by
  unfold summary
  rw [replay_input idleTwo one rfl rfl empty_agents_valid (by decide)]
  dsimp only [idleTwo, empty]
  simp only [runInputs]
  decide_cbv

private theorem attachSource_literal : summary attachSource =
    .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/2) := by
  unfold summary
  rw [replay_input attachSource one rfl rfl empty_agents_valid (by decide)]
  dsimp only [attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [default_growth_agents]
    decide_cbv
  · rw [default_growth_agents]
    decide_cbv
  · decide_cbv

private theorem attachRelay_literal : summary attachRelay =
    .ok (3, 1, 2, [1, 1, 0], [0, 1, 0], 1/2) := by
  unfold summary
  rw [replay_input attachRelay one rfl rfl empty_agents_valid (by decide)]
  dsimp only [attachRelay, empty]
  rw [runInputs, checked1]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [default_growth_agents]
    decide_cbv
  · rw [default_growth_agents]
    decide_cbv
  · decide_cbv

private theorem relayIdle_literal : summary relayIdle =
    .ok (3, 2, 2, [1, 1, 1], [1, 2, 1], 1/2) := by
  unfold summary
  rw [replay_input relayIdle one rfl rfl empty_agents_valid (by decide)]
  dsimp only [relayIdle, empty]
  rw [runInputs, checked1]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [relay_idle_agents]
    decide_cbv
  · rw [relay_idle_agents]
    decide_cbv
  · decide_cbv

private theorem successiveBirths_literal : summary successiveBirths =
    .ok (4, 2, 3, [1, 1, 1, 0], [1, 2, 1, 0], 1/8) := by
  unfold summary
  rw [replay_input successiveBirths one rfl rfl empty_agents_valid (by decide)]
  dsimp only [successiveBirths, empty]
  simp only [runInputs, checked1, checked2]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [successive_agents]
    decide_cbv
  · rw [successive_agents]
    decide_cbv
  · decide_cbv

private theorem successiveIdle_literal : summary successiveIdle =
    .ok (4, 3, 3, [1, 1, 1, 1], [2, 4, 2, 1], 1/8) := by
  unfold summary
  rw [replay_input successiveIdle one rfl rfl empty_agents_valid (by decide)]
  dsimp only [successiveIdle, empty]
  simp only [runInputs, checked1, checked2]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [successive_idle_agents]
    decide_cbv
  · rw [successive_idle_agents]
    decide_cbv
  · decide_cbv

private theorem weightedSource_literal : summary weightedSource =
    .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/4) := by
  unfold summary
  rw [replay_input weightedSource three rfl rfl empty_agents_valid (by decide)]
  dsimp only [weightedSource, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [default_growth_agents]
    decide_cbv
  · rw [default_growth_agents]
    decide_cbv
  · decide_cbv

private theorem ordered01_literal : summary ordered01 =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 1/4) := by
  unfold summary
  rw [replay_input ordered01 three rfl rfl empty_agents_valid (by decide)]
  dsimp only [ordered01, empty]
  rw [runInputs, checked01]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [both_growth_agents three targets01 targets01_selected]
    decide_cbv
  · rw [both_growth_agents three targets01 targets01_selected]
    decide_cbv
  · decide_cbv

private theorem ordered10_literal : summary ordered10 =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 3/4) := by
  unfold summary
  rw [replay_input ordered10 three rfl rfl empty_agents_valid (by decide)]
  dsimp only [ordered10, empty]
  rw [runInputs, checked10]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [both_growth_agents three targets10 targets10_selected]
    decide_cbv
  · rw [both_growth_agents three targets10 targets10_selected]
    decide_cbv
  · decide_cbv

private theorem zeroReceptive_literal : summary zeroReceptive =
    .ok (3, 1, 2, [1, 0, 1], [0, 1, 1], 1/2) := by
  unfold summary
  rw [replay_input zeroReceptive one rfl rfl zeroReceptive_agents_valid (by decide)]
  dsimp only [zeroReceptive, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [zeroReceptive_step_agents]
    decide_cbv
  · rw [zeroReceptive_step_agents]
    decide_cbv
  · decide_cbv

private theorem silent_literal : summary silent =
    .ok (3, 1, 2, [0, 0, 0], [0, 0, 0], 1/2) := by
  unfold summary
  rw [replay_input silent one rfl rfl silent_agents_valid (by decide)]
  dsimp only [silent, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [silent_step_agents]
    decide_cbv
  · rw [silent_step_agents]
    decide_cbv
  · decide_cbv

private theorem zeroThreshold_literal : summary zeroThreshold =
    .ok (3, 1, 2, [0, 0, 0], [0, 1, 1], 1/2) := by
  unfold summary
  rw [replay_input zeroThreshold one rfl rfl zeroThreshold_agents_valid (by decide)]
  dsimp only [zeroThreshold, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [zeroThreshold_step_agents]
    decide_cbv
  · rw [zeroThreshold_step_agents]
    decide_cbv
  · decide_cbv

private theorem broadcastNewborn_literal : summary broadcastNewborn =
    .ok (3, 1, 2, [1, 1, 1], [0, 2, 0], 1/2) := by
  unfold summary
  rw [replay_input broadcastNewborn one rfl rfl empty_agents_valid (by decide)]
  dsimp only [broadcastNewborn, attachRelay, empty]
  rw [runInputs, checked_broadcast]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [broadcastNewborn_step_agents]
    decide_cbv
  · rw [broadcastNewborn_step_agents]
    decide_cbv
  · decide_cbv

private theorem retainedExposures_literal : summary retainedExposures =
    .ok (3, 1, 2, [1, 1, 0], [7, 4, 0], 1/2) := by
  unfold summary
  rw [replay_input retainedExposures one rfl rfl retainedExposures_agents_valid (by decide)]
  dsimp only [retainedExposures, attachRelay, empty]
  rw [runInputs, checked1]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [retainedExposures_step_agents]
    decide_cbv
  · rw [retainedExposures_step_agents]
    decide_cbv
  · decide_cbv

private theorem halfReceptive_literal : summary halfReceptive =
    .ok (3, 1, 2, [1, 1/2, 1], [0, 1, 1], 1/2) := by
  unfold summary
  rw [replay_input halfReceptive one rfl rfl halfReceptive_agents_valid (by decide)]
  dsimp only [halfReceptive, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  apply (summary_result_ok _ _ _ _ _ _ _).mpr
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [advance_projection, grow_projection, birth_edges]
    decide_cbv
  · rw [halfReceptive_step_agents]
    decide_cbv
  · rw [halfReceptive_step_agents]
    decide_cbv
  · decide_cbv

private theorem seedNodes_literal : summary seedNodes =
    .error (.seedNetwork .invalidNodeCount) := by
  decide_cbv

private theorem seedSize_literal : summary seedSize =
    .error (.seedNetwork .fitnessSizeMismatch) := by
  decide_cbv

private theorem seedFitness_literal : summary seedFitness =
    .error (.seedNetwork .nonpositiveFitness) := by
  decide_cbv

private theorem seedEdge_literal : summary seedEdge =
    .error (.seedNetwork .invalidEdge) := by
  decide_cbv

private theorem seedDuplicate_literal : summary seedDuplicate =
    .error (.seedNetwork .duplicateEdge) := by
  decide_cbv

private theorem seedDisconnected_literal : summary seedDisconnected =
    .error (.seedNetwork .disconnectedSeed) := by
  have hn : 2 ≤ seedDisconnected.seed.nodeCount := by decide
  have hs : seedDisconnected.seed.fitness.size = seedDisconnected.seed.nodeCount := rfl
  have hf : positiveSeedFitness seedDisconnected.seed := by decide
  have he : validSeedEdges seedDisconnected.seed := by decide
  have hd : (seedDisconnected.seed.edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ¬ ∀ b, b ∈ reached (seedSnapshot seedDisconnected.seed hs)
      (⟨0, by have := hn; omega⟩ : Fin seedDisconnected.seed.nodeCount)
      (seedDisconnected.seed.nodeCount - 1) := by
    change ¬ ∀ b : Fin 3, b ∈ reached
      (seedSnapshot (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed) rfl) (0 : Fin 3) 2
    decide
  have parsed : parseSeed seedDisconnected.seed = .error .disconnectedSeed := by
    simp only [parseSeed, dif_pos hn, dif_pos hs, dif_pos hf, dif_pos he,
      dif_pos hd, dif_neg hc]
  unfold summary FitnessABM.replay
  rw [parsed]
  rfl

private theorem initialMZero_literal : summary initialMZero =
    .error (.initialM) := by
  unfold summary
  dsimp only [initialMZero, empty, idleTwo]
  unfold FitnessABM.replay
  rw [parsed_unit]
  decide_cbv

private theorem initialMTooLarge_literal : summary initialMTooLarge =
    .error (.initialM) := by
  unfold summary
  dsimp only [initialMTooLarge, empty, idleTwo]
  unfold FitnessABM.replay
  rw [parsed_unit]
  decide_cbv

private theorem seedAgentCount_literal : summary seedAgentCount =
    .error (.seedAgentCount 2 1) := by
  unfold summary
  dsimp only [seedAgentCount, empty, idleTwo]
  unfold FitnessABM.replay
  rw [parsed_unit]
  decide_cbv

private theorem seedAgentR_literal : summary seedAgentR =
    .error (.seedAgent 0 .receptivity) := by
  unfold summary
  dsimp only [seedAgentR, empty, idleTwo]
  unfold FitnessABM.replay
  rw [parsed_unit]
  decide_cbv

private theorem seedAgentThreshold_literal : summary seedAgentThreshold =
    .error (.seedAgent 0 .threshold) := by
  unfold summary
  dsimp only [seedAgentThreshold, empty, idleTwo]
  unfold FitnessABM.replay
  rw [parsed_unit]
  decide_cbv

private theorem seedAgentBelief_literal : summary seedAgentBelief =
    .error (.seedAgent 1 .belief) := by
  unfold summary
  dsimp only [seedAgentBelief, empty, idleTwo]
  unfold FitnessABM.replay
  rw [parsed_unit]
  decide_cbv

private theorem birthFitness_literal : summary birthFitness =
    .error (.tickNetwork 0 0 .nonpositiveFitness) := by
  unfold summary
  rw [replay_input birthFitness one rfl rfl empty_agents_valid (by decide)]
  dsimp only [birthFitness, empty]
  decide_cbv

private theorem birthTargetCount_literal : summary birthTargetCount =
    .error (.tickNetwork 0 0 .targetCountMismatch) := by
  unfold summary
  rw [replay_input birthTargetCount one rfl rfl empty_agents_valid (by decide)]
  dsimp only [birthTargetCount, empty]
  decide_cbv

private theorem birthTargetRange_literal : summary birthTargetRange =
    .error (.tickNetwork 0 0 .targetOutOfRange) := by
  unfold summary
  rw [replay_input birthTargetRange one rfl rfl empty_agents_valid (by decide)]
  dsimp only [birthTargetRange, empty]
  decide_cbv

private theorem birthTargetDuplicate_literal : summary birthTargetDuplicate =
    .error (.tickNetwork 0 0 .duplicateTarget) := by
  unfold summary
  rw [replay_input birthTargetDuplicate one rfl rfl empty_agents_valid (by decide)]
  dsimp only [birthTargetDuplicate, birthTargetCount, empty]
  decide_cbv

private theorem birthAgentThreshold_literal : summary birthAgentThreshold =
    .error (.tickAgent 0 0 .threshold) := by
  unfold summary
  rw [replay_input birthAgentThreshold one rfl rfl empty_agents_valid (by decide)]
  dsimp only [birthAgentThreshold, empty]
  decide_cbv

private theorem lateBirth_literal : summary lateBirth =
    .error (.tickNetwork 3 1 .targetOutOfRange) := by
  unfold summary
  rw [replay_input lateBirth one rfl rfl empty_agents_valid (by decide)]
  dsimp only [lateBirth, empty]
  decide_cbv

private theorem firstFailure_literal : summary firstFailure =
    .error (.tickAgent 0 0 .threshold) := by
  unfold summary
  rw [replay_input firstFailure one rfl rfl empty_agents_valid (by decide)]
  dsimp only [firstFailure, empty]
  decide_cbv

private theorem lateFirstBirth_literal : summary lateFirstBirth =
    .error (.tickNetwork 2 0 .targetOutOfRange) := by
  unfold summary
  rw [replay_input lateFirstBirth one rfl rfl empty_agents_valid (by decide)]
  dsimp only [lateFirstBirth, empty]
  decide_cbv

private theorem transmission_mem_incoming {n : Nat} (g : MeshGraph (Fin n))
    [DecidableRel g] (p : Population n) (source target : Fin n) :
    (source, target) ∈ transmissions g p ↔ source ∈ incoming g p target := by
  simp only [transmission_iff, incoming, Finset.mem_filter, Finset.mem_univ, true_and]

private theorem successive_growth_literal :
    (observeTransition successiveBirths ⟨1, by decide⟩).map
      (fun t => (t.postGrowth.beliefs, t.postGrowth.exposures, t.tickMass, t.transmissions)) =
      .ok ([1, 1, 0, 0], [0, 1, 0, 0], 1/4, [(0, 1, 1), (1, 0, 1), (1, 2, 1)]) := by
  unfold observeTransition
  rw [prefix_start successiveBirths one rfl rfl empty_agents_valid (by decide)]
  dsimp only [successiveBirths, empty, List.take]
  rw [runInputs, checked1]
  simp only [runInputs, List.getElem_cons_succ, List.getElem_cons_zero]
  rw [checked2]
  simp only [Except.map, observeState, observeTransmissions,
    transmission_mem_incoming, successive_incoming]
  simp only [grow_agents, default_growth_agents, advance_projection,
    grow_projection, initial_network]
  decide_cbv

private theorem retained_growth_literal :
    (observeTransition retainedExposures ⟨0, by decide⟩).map
      (fun t => t.postGrowth.exposures) = .ok [7, 3, 0] := by
  unfold observeTransition
  rw [prefix_start retainedExposures one rfl rfl retainedExposures_agents_valid (by decide)]
  dsimp only [retainedExposures, attachRelay, empty, List.take]
  simp only [runInputs, List.getElem_cons_zero]
  rw [checked1]
  simp only [Except.map, observeState, grow_agents, retainedExposures_initial_agents]
  decide_cbv

private theorem newborn_transmissions_literal :
    (observeTransition broadcastNewborn ⟨0, by decide⟩).map
      (fun t => t.transmissions) = .ok [(0, 1, 1), (2, 1, 1)] := by
  unfold observeTransition
  rw [prefix_start broadcastNewborn one rfl rfl empty_agents_valid (by decide)]
  dsimp only [broadcastNewborn, attachRelay, empty, List.take]
  simp only [runInputs, List.getElem_cons_zero]
  rw [checked_broadcast]
  simp only [Except.map, observeTransmissions, transmission_mem_incoming,
    broadcastNewborn_incoming]
  simp only [grow_agents, default_agents]
  decide_cbv

theorem success_literals :
  summary empty = .ok (2, 0, 1, [1, 0], [0, 0], 1) ∧
  summary idleTwo = .ok (2, 2, 1, [1, 1], [1, 2], 1) ∧
  summary attachSource = .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/2) ∧
  summary attachRelay = .ok (3, 1, 2, [1, 1, 0], [0, 1, 0], 1/2) ∧
  summary relayIdle = .ok (3, 2, 2, [1, 1, 1], [1, 2, 1], 1/2) ∧
  summary successiveBirths = .ok (4, 2, 3, [1, 1, 1, 0], [1, 2, 1, 0], 1/8) ∧
  summary successiveIdle = .ok (4, 3, 3, [1, 1, 1, 1], [2, 4, 2, 1], 1/8) ∧
  summary weightedSource = .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/4) ∧
  summary ordered01 = .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 1/4) ∧
  summary ordered10 = .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 3/4) ∧
  summary zeroReceptive = .ok (3, 1, 2, [1, 0, 1], [0, 1, 1], 1/2) ∧
  summary silent = .ok (3, 1, 2, [0, 0, 0], [0, 0, 0], 1/2) ∧
  summary zeroThreshold = .ok (3, 1, 2, [0, 0, 0], [0, 1, 1], 1/2) ∧
  summary broadcastNewborn = .ok (3, 1, 2, [1, 1, 1], [0, 2, 0], 1/2) ∧
  summary retainedExposures = .ok (3, 1, 2, [1, 1, 0], [7, 4, 0], 1/2) ∧
  summary halfReceptive = .ok (3, 1, 2, [1, 1/2, 1], [0, 1, 1], 1/2) :=
  ⟨empty_literal, idleTwo_literal, attachSource_literal, attachRelay_literal, relayIdle_literal, successiveBirths_literal, successiveIdle_literal, weightedSource_literal, ordered01_literal, ordered10_literal, zeroReceptive_literal, silent_literal, zeroThreshold_literal, broadcastNewborn_literal, retainedExposures_literal, halfReceptive_literal⟩

theorem shared_error_literals :
  summary seedNodes = .error (.seedNetwork .invalidNodeCount) ∧
  summary seedSize = .error (.seedNetwork .fitnessSizeMismatch) ∧
  summary seedFitness = .error (.seedNetwork .nonpositiveFitness) ∧
  summary seedEdge = .error (.seedNetwork .invalidEdge) ∧
  summary seedDuplicate = .error (.seedNetwork .duplicateEdge) ∧
  summary seedDisconnected = .error (.seedNetwork .disconnectedSeed) ∧
  summary initialMZero = .error (.initialM) ∧
  summary initialMTooLarge = .error (.initialM) ∧
  summary seedAgentCount = .error (.seedAgentCount 2 1) ∧
  summary seedAgentR = .error (.seedAgent 0 .receptivity) ∧
  summary seedAgentThreshold = .error (.seedAgent 0 .threshold) ∧
  summary seedAgentBelief = .error (.seedAgent 1 .belief) ∧
  summary birthFitness = .error (.tickNetwork 0 0 .nonpositiveFitness) ∧
  summary birthTargetCount = .error (.tickNetwork 0 0 .targetCountMismatch) ∧
  summary birthTargetRange = .error (.tickNetwork 0 0 .targetOutOfRange) ∧
  summary birthTargetDuplicate = .error (.tickNetwork 0 0 .duplicateTarget) ∧
  summary birthAgentThreshold = .error (.tickAgent 0 0 .threshold) ∧
  summary lateBirth = .error (.tickNetwork 3 1 .targetOutOfRange) ∧
  summary firstFailure = .error (.tickAgent 0 0 .threshold) ∧
  summary lateFirstBirth = .error (.tickNetwork 2 0 .targetOutOfRange) :=
  ⟨seedNodes_literal, seedSize_literal, seedFitness_literal, seedEdge_literal, seedDuplicate_literal, seedDisconnected_literal, initialMZero_literal, initialMTooLarge_literal, seedAgentCount_literal, seedAgentR_literal, seedAgentThreshold_literal, seedAgentBelief_literal, birthFitness_literal, birthTargetCount_literal, birthTargetRange_literal, birthTargetDuplicate_literal, birthAgentThreshold_literal, lateBirth_literal, firstFailure_literal, lateFirstBirth_literal⟩

theorem transition_literals :
    (observeTransition successiveBirths ⟨1, by decide⟩).map
      (fun t => (t.postGrowth.beliefs, t.postGrowth.exposures, t.tickMass, t.transmissions)) =
      .ok ([1, 1, 0, 0], [0, 1, 0, 0], 1/4, [(0, 1, 1), (1, 0, 1), (1, 2, 1)]) ∧
    (observeTransition retainedExposures ⟨0, by decide⟩).map
      (fun t => t.postGrowth.exposures) = .ok [7, 3, 0] ∧
    (observeTransition broadcastNewborn ⟨0, by decide⟩).map
      (fun t => t.transmissions) = .ok [(0, 1, 1), (2, 1, 1)] :=
  ⟨successive_growth_literal, retained_growth_literal, newborn_transmissions_literal⟩

end FiniteLiterals

#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.success_literals
#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.shared_error_literals
#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.transition_literals

end NarrativeDynamics.Tests.FitnessABMRuntime
