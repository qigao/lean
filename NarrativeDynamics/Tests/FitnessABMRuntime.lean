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

private instance {α : Type} [DecidableEq α] : DecidableEq (Except JointError α) :=
  fun x y => match x, y with
  | .error x, .error y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.error.inj he))
  | .ok x, .ok y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.ok.inj he))
  | .error _, .ok _ => .isFalse (by intro h; cases h)
  | .ok _, .error _ => .isFalse (by intro h; cases h)

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
  decide_cbv

private theorem attachRelay_literal : summary attachRelay =
    .ok (3, 1, 2, [1, 1, 0], [0, 1, 0], 1/2) := by
  unfold summary
  rw [replay_input attachRelay one rfl rfl empty_agents_valid (by decide)]
  dsimp only [attachRelay, empty]
  rw [runInputs, checked1]
  simp only [runInputs]
  decide_cbv

private theorem relayIdle_literal : summary relayIdle =
    .ok (3, 2, 2, [1, 1, 1], [1, 2, 1], 1/2) := by
  unfold summary
  rw [replay_input relayIdle one rfl rfl empty_agents_valid (by decide)]
  dsimp only [relayIdle, empty]
  rw [runInputs, checked1]
  simp only [runInputs]
  decide_cbv

private theorem successiveBirths_literal : summary successiveBirths =
    .ok (4, 2, 3, [1, 1, 1, 0], [1, 2, 1, 0], 1/8) := by
  unfold summary
  rw [replay_input successiveBirths one rfl rfl empty_agents_valid (by decide)]
  dsimp only [successiveBirths, empty]
  rw [runInputs, checked1]
  rw [runInputs, checked2]
  simp only [runInputs]
  decide_cbv

private theorem successiveIdle_literal : summary successiveIdle =
    .ok (4, 3, 3, [1, 1, 1, 1], [2, 4, 2, 1], 1/8) := by
  unfold summary
  rw [replay_input successiveIdle one rfl rfl empty_agents_valid (by decide)]
  dsimp only [successiveIdle, empty]
  rw [runInputs, checked1]
  rw [runInputs, checked2]
  simp only [runInputs]
  decide_cbv

private theorem weightedSource_literal : summary weightedSource =
    .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/4) := by
  unfold summary
  rw [replay_input weightedSource three rfl rfl empty_agents_valid (by decide)]
  dsimp only [weightedSource, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  decide_cbv

private theorem ordered01_literal : summary ordered01 =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 1/4) := by
  unfold summary
  rw [replay_input ordered01 three rfl rfl empty_agents_valid (by decide)]
  dsimp only [ordered01, empty]
  rw [runInputs, checked01]
  simp only [runInputs]
  decide_cbv

private theorem ordered10_literal : summary ordered10 =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 3/4) := by
  unfold summary
  rw [replay_input ordered10 three rfl rfl empty_agents_valid (by decide)]
  dsimp only [ordered10, empty]
  rw [runInputs, checked10]
  simp only [runInputs]
  decide_cbv

private theorem zeroReceptive_literal : summary zeroReceptive =
    .ok (3, 1, 2, [1, 0, 1], [0, 1, 1], 1/2) := by
  unfold summary
  rw [replay_input zeroReceptive one rfl rfl zeroReceptive_agents_valid (by decide)]
  dsimp only [zeroReceptive, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  decide_cbv

private theorem silent_literal : summary silent =
    .ok (3, 1, 2, [0, 0, 0], [0, 0, 0], 1/2) := by
  unfold summary
  rw [replay_input silent one rfl rfl silent_agents_valid (by decide)]
  dsimp only [silent, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  decide_cbv

private theorem zeroThreshold_literal : summary zeroThreshold =
    .ok (3, 1, 2, [0, 0, 0], [0, 1, 1], 1/2) := by
  unfold summary
  rw [replay_input zeroThreshold one rfl rfl zeroThreshold_agents_valid (by decide)]
  dsimp only [zeroThreshold, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  decide_cbv

private theorem broadcastNewborn_literal : summary broadcastNewborn =
    .ok (3, 1, 2, [1, 1, 1], [0, 2, 0], 1/2) := by
  unfold summary
  rw [replay_input broadcastNewborn one rfl rfl empty_agents_valid (by decide)]
  dsimp only [broadcastNewborn, attachRelay, empty]
  rw [runInputs, checked_broadcast]
  simp only [runInputs]
  decide_cbv

private theorem retainedExposures_literal : summary retainedExposures =
    .ok (3, 1, 2, [1, 1, 0], [7, 4, 0], 1/2) := by
  unfold summary
  rw [replay_input retainedExposures one rfl rfl retainedExposures_agents_valid (by decide)]
  dsimp only [retainedExposures, attachRelay, empty]
  rw [runInputs, checked1]
  simp only [runInputs]
  decide_cbv

private theorem halfReceptive_literal : summary halfReceptive =
    .ok (3, 1, 2, [1, 1/2, 1], [0, 1, 1], 1/2) := by
  unfold summary
  rw [replay_input halfReceptive one rfl rfl halfReceptive_agents_valid (by decide)]
  dsimp only [halfReceptive, attachSource, empty]
  rw [runInputs, checked0]
  simp only [runInputs]
  decide_cbv

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
  decide_cbv

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

private theorem successive_growth_literal :
    (observeTransition successiveBirths ⟨1, by decide⟩).map
      (fun t => (t.postGrowth.beliefs, t.postGrowth.exposures, t.tickMass, t.transmissions)) =
      .ok ([1, 1, 0, 0], [0, 1, 0, 0], 1/4, [(0, 1, 1), (1, 0, 1), (1, 2, 1)]) := by
  unfold observeTransition
  rw [prefix_start successiveBirths one rfl rfl empty_agents_valid (by decide)]
  dsimp only [successiveBirths, empty, List.take]
  rw [runInputs, checked1]
  simp only [runInputs]
  dsimp only
  rw [checked2]
  decide_cbv

private theorem retained_growth_literal :
    (observeTransition retainedExposures ⟨0, by decide⟩).map
      (fun t => t.postGrowth.exposures) = .ok [7, 3, 0] := by
  unfold observeTransition
  rw [prefix_start retainedExposures one rfl rfl retainedExposures_agents_valid (by decide)]
  dsimp only [retainedExposures, attachRelay, empty, List.take]
  simp only [runInputs]
  dsimp only
  rw [checked1]
  decide_cbv

private theorem newborn_transmissions_literal :
    (observeTransition broadcastNewborn ⟨0, by decide⟩).map
      (fun t => t.transmissions) = .ok [(0, 1, 1), (2, 1, 1)] := by
  unfold observeTransition
  rw [prefix_start broadcastNewborn one rfl rfl empty_agents_valid (by decide)]
  dsimp only [broadcastNewborn, attachRelay, empty, List.take]
  simp only [runInputs]
  dsimp only
  rw [checked_broadcast]
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
