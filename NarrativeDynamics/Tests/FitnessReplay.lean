import NarrativeDynamics.Core.FitnessReplay

open NarrativeDynamics NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

namespace NarrativeDynamics.FitnessAttachment.ReplayFixtures

def rawTriangle : RawSeed := ⟨3, #[1, 2, 4], #[(0, 1), (0, 2), (1, 2)]⟩
def twoBirths : List RawBirth := [⟨3/2, #[2, 1]⟩, ⟨1/3, #[3, 2]⟩]
def unitTriangle : RawSeed := ⟨3, #[1, 1, 1], #[(0, 1), (0, 2), (1, 2)]⟩

def summaryOf : Except ReplayError ReplayResult →
    Except ReplayError (Nat × Nat × List Nat × List Rat × Rat)
  | .error e => .error e
  | .ok r => .ok (actualNodeCount r.final.state.snapshot,
      actualEdgeCount r.final.state.snapshot,
      List.ofFn (degree r.final.state.snapshot),
      List.ofFn r.final.state.snapshot.fitness, r.probability)

/-- Establish one actual checked step without unfolding a concrete graph. -/
private theorem checkedStep {n : Nat} (s : State n) (m : Nat) (raw : RawBirth)
    (hm : 0 < m ∧ m ≤ n) (hf : 0 < raw.fitness) (hs : raw.targets.size = m)
    (hb : targetsBounded n raw.targets) (hd : targetsDistinct raw.targets) :
    step s m raw = .ok
      (applyBirth s (hs ▸ (⟨hb, hd⟩ : CheckedTargets n raw.targets).embedding)
        hm.1 ⟨raw.fitness, hf⟩,
       orderedMass s (hs ▸ (⟨hb, hd⟩ : CheckedTargets n raw.targets).embedding)) := by
  simp only [step, validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]

/-- Rewrite just the leading birth, leaving its entire continuation opaque. -/
private theorem checkedBirthEquation {n m index : Nat} {s : State n}
    {raw : RawBirth} {rest : List RawBirth}
    (hm : 0 < m ∧ m ≤ n) (hf : 0 < raw.fitness)
    (hs : raw.targets.size = m) (hb : targetsBounded n raw.targets)
    (hd : targetsDistinct raw.targets) :
    runBirths m index ⟨n, s⟩ (raw :: rest) =
      let T : Targets n m :=
        hs ▸ (⟨hb, hd⟩ : CheckedTargets n raw.targets).embedding
      let next := applyBirth s T hm.1 ⟨raw.fitness, hf⟩
      match runBirths m (index + 1) ⟨n + 1, next⟩ rest with
      | .error e => .error e
      | .ok out => .ok ⟨out.final, orderedMass s T * out.probability⟩ := by
  rewrite [replay_step, checkedStep s m raw hm hf hs hb hd]
  rfl

#print axioms checkedStep
#print axioms checkedBirthEquation

/-- Finish only the empty continuation, without reopening a checked birth. -/
private theorem runBirthsNil (m index : Nat) (s : RunState) :
    runBirths m index s [] = .ok ⟨s, 1⟩ := rfl

/-- The complete original summary is equivalent to five separate field facts. -/
private theorem summaryOk (r : ReplayResult) (nodes edges : Nat)
    (degrees : List Nat) (fitness : List Rat) (probability : Rat) :
    summaryOf (.ok r) = .ok (nodes, edges, degrees, fitness, probability) ↔
      actualNodeCount r.final.state.snapshot = nodes ∧
      actualEdgeCount r.final.state.snapshot = edges ∧
      List.ofFn (degree r.final.state.snapshot) = degrees ∧
      List.ofFn r.final.state.snapshot.fitness = fitness ∧
      r.probability = probability := by
  simp only [summaryOf, Except.ok.injEq, Prod.mk.injEq]

/-- Reuse proved neighbor updates, rather than enumerate each grown graph again. -/
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

/-- Project the real birth's fitness without unfolding its graph or validity proof. -/
private theorem birthFitnessFn {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    (applyBirth s T hm eta).snapshot.fitness =
      Fin.lastCases eta.val s.snapshot.fitness := rfl

#print axioms runBirthsNil
#print axioms summaryOk
#print axioms birthDegreeFn
#print axioms birthFitnessFn

end NarrativeDynamics.FitnessAttachment.ReplayFixtures

open NarrativeDynamics.FitnessAttachment.ReplayFixtures

-- Prove the actual parser guards first; do not evaluate proof-carrying parsing
-- and the entire multi-birth graph in one equation-reduction stack.
macro "prepareReplaySeed " raw:term " atSize " size:term : tactic =>
  `(tactic|
    (trace "replay-fixture seed: guards start"
     have hn : 2 ≤ ($raw : RawSeed).nodeCount := by decide
     have hs : ($raw : RawSeed).fitness.size = ($raw : RawSeed).nodeCount := rfl
     have hf : positiveSeedFitness $raw := by decide
     have he : validSeedEdges $raw := by decide
     have hd : (($raw : RawSeed).edges.toList.map canonicalEdge).Nodup := by decide
     trace "replay-fixture seed: reachability guard start"
     have hc : ∀ b,
         b ∈ reached (seedSnapshot $raw hs)
           (⟨0, by have := hn; omega⟩ : Fin ($raw : RawSeed).nodeCount)
           (($raw : RawSeed).nodeCount - 1) := by
       change ∀ b : Fin $size,
         b ∈ reached (seedSnapshot $raw rfl) (0 : Fin $size) ($size - 1)
       decide
     trace "replay-fixture seed: simplification start"
     simp only [replay, parseSeed, dif_pos hn, dif_pos hs, dif_pos hf,
       dif_pos he, dif_pos hd, dif_pos hc]
     trace "replay-fixture seed: simplification done"))

-- Prove concrete guards, then rewrite exactly one checked-step equation.
-- Do not recursively simplify validators under unresolved continuation matches.
-- `rewrite` deliberately leaves the goal open for `decide_cbv`; `rw` also
-- attempts reflexivity, which can evaluate the entire remaining replay.
macro "prepareReplayBirth " n:term " withM " m:term
    " fitness " eta:term " targets " xs:term : tactic =>
  `(tactic|
    (trace "replay-fixture birth: guards start"
     have hm : 0 < $m ∧ $m ≤ $n := by decide
     have hf : 0 < ($eta : Rat) := by norm_num
     have hs : ($xs : Array Nat).size = $m := rfl
     have hb : targetsBounded $n $xs := by decide
     have hd : targetsDistinct $xs := by decide
     trace "replay-fixture birth: prefix simplification start"
     simp only [twoBirths, if_pos hm]
     trace "replay-fixture birth: checked rewrite start"
     rewrite [checkedBirthEquation (n := $n) (m := $m) (raw := (⟨$eta, $xs⟩ : RawBirth))
       hm hf hs hb hd]
     trace "replay-fixture birth: checked rewrite done"))

-- Bound field rewrites by the actual number of births. Recursive simp can
-- rematch reducible graph fields beneath Fin.lastCases and create congruence
-- metavariables; each birth equation must instead be used exactly once.
-- The parsed State carries unfolded validity evidence. These field rewrites
-- must recognize its original Valid predicate and carrier at default transparency.
macro "finishReplaySummary " n:term " withM " m:term " births " rounds:num : tactic => do
  let degrees ← if rounds.getNat == 1 then
    `(tactic| rewrite (transparency := .default) [birthDegreeFn (n := $n) (m := $m) (hm := by decide)])
  else
    `(tactic|
      (rewrite (transparency := .default) [birthDegreeFn (n := ($n + 1)) (m := $m) (hm := by decide)]
       rewrite (transparency := .default) [birthDegreeFn (n := $n) (m := $m) (hm := by decide)]))
  let fitnessTac ← if rounds.getNat == 1 then
    `(tactic| rewrite (transparency := .default) [birthFitnessFn (n := $n) (m := $m) (hm := by decide)])
  else
    `(tactic|
      (rewrite (transparency := .default) [birthFitnessFn (n := ($n + 1)) (m := $m) (hm := by decide)]
       rewrite (transparency := .default) [birthFitnessFn (n := $n) (m := $m) (hm := by decide)]))
  let probability ← if rounds.getNat == 1 then
    `(tactic| decide_cbv)
  else
    `(tactic|
      (rewrite (transparency := .default) [birthDegreeFn (n := $n) (m := $m) (hm := by decide)]
       rewrite (transparency := .default) [birthFitnessFn (n := $n) (m := $m) (hm := by decide)]
       decide_cbv))
  `(tactic|
    (trace "replay-fixture result: empty tail start"
     simp only [runBirthsNil]
     trace "replay-fixture result: field split start"
     apply (summaryOk _ _ _ _ _ _).mpr
     refine ⟨?_, ?_, ?_, ?_, ?_⟩
     · trace "replay-fixture result: nodes"
       simp only [actualNodeCount, Fintype.card_fin] <;> rfl
     · trace "replay-fixture result: edges"
       simp only [birth_edges]
       decide_cbv
     · trace "replay-fixture result: degrees rewrite"
       $degrees:tactic
       trace "replay-fixture result: degrees evaluation"
       decide_cbv
     · trace "replay-fixture result: fitness rewrite"
       $fitnessTac:tactic
       trace "replay-fixture result: fitness evaluation"
       decide_cbv
     · trace "replay-fixture result: probability"
       simp only [orderedMass, weights]
       $probability:tactic))

section ExactReplayFixtures
-- Report reduction hotspots without changing any proof or resource limit.
set_option diagnostics true
set_option maxRecDepth 4096

-- Actual raw API output, including updated adjacency and the conditional product.
example : summaryOf (replay rawTriangle 2 []) =
    .ok (3, 3, [2, 2, 2], [1, 2, 4], 1) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩]) =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/21) := by
  prepareReplaySeed rawTriangle atSize 3
  prepareReplayBirth rawTriangle.nodeCount withM 2
    fitness (3/2) targets #[2, 1]
  finishReplaySummary 3 withM 2 births 1
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[1, 2]⟩]) =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/35) := by
  prepareReplaySeed rawTriangle atSize 3
  prepareReplayBirth rawTriangle.nodeCount withM 2
    fitness (3/2) targets #[1, 2]
  finishReplaySummary 3 withM 2 births 1
example : summaryOf (replay rawTriangle 2 twoBirths) =
    .ok (5, 7, [2, 3, 4, 3, 2], [1, 2, 4, 3/2, 1/3], 24/805) := by
  prepareReplaySeed rawTriangle atSize 3
  prepareReplayBirth rawTriangle.nodeCount withM 2
    fitness (3/2) targets #[2, 1]
  prepareReplayBirth (rawTriangle.nodeCount + 1) withM 2
    fitness (1/3) targets #[3, 2]
  finishReplaySummary 3 withM 2 births 2

-- All fitness values, not only the seed, must receive the same scale.
example : summaryOf (replay ⟨3, #[2, 4, 8], rawTriangle.edges⟩ 2
    [⟨3, #[2, 1]⟩, ⟨2/3, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [2, 4, 8, 3, 2/3], 24/805) := by
  prepareReplaySeed (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed) atSize 3
  prepareReplayBirth (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount withM 2
    fitness (3) targets #[2, 1]
  prepareReplayBirth ((⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount + 1) withM 2
    fitness (2/3) targets #[3, 2]
  finishReplaySummary 3 withM 2 births 2
example : summaryOf (replay ⟨3, #[2, 4, 8], rawTriangle.edges⟩ 2 twoBirths) =
    .ok (5, 7, [2, 3, 4, 3, 2], [2, 4, 8, 3/2, 1/3], 24/1505) := by
  prepareReplaySeed (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed) atSize 3
  prepareReplayBirth (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount withM 2
    fitness (3/2) targets #[2, 1]
  prepareReplayBirth ((⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount + 1) withM 2
    fitness (1/3) targets #[3, 2]
  finishReplaySummary 3 withM 2 births 2

-- Constant fitness 3 and the unit-fitness BA specialization have equal laws.
example : summaryOf (replay ⟨3, #[3, 3, 3], rawTriangle.edges⟩ 2
    [⟨3, #[2, 1]⟩, ⟨3, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [3, 3, 3, 3, 3], 1/80) := by
  prepareReplaySeed (⟨3, #[3, 3, 3], rawTriangle.edges⟩ : RawSeed) atSize 3
  prepareReplayBirth (⟨3, #[3, 3, 3], rawTriangle.edges⟩ : RawSeed).nodeCount withM 2
    fitness (3) targets #[2, 1]
  prepareReplayBirth ((⟨3, #[3, 3, 3], rawTriangle.edges⟩ : RawSeed).nodeCount + 1) withM 2
    fitness (3) targets #[3, 2]
  finishReplaySummary 3 withM 2 births 2
example : summaryOf (replay unitTriangle 2 [⟨1, #[2, 1]⟩, ⟨1, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [1, 1, 1, 1, 1], 1/80) := by
  prepareReplaySeed unitTriangle atSize 3
  prepareReplayBirth unitTriangle.nodeCount withM 2
    fitness (1) targets #[2, 1]
  prepareReplayBirth (unitTriangle.nodeCount + 1) withM 2
    fitness (1) targets #[3, 2]
  finishReplaySummary 3 withM 2 births 2

-- Arbitrary valid seed and m equal to the INITIAL size across several births.
example : summaryOf (replay ⟨2, #[1, 1], #[(1, 0)]⟩ 1
    [⟨1, #[1]⟩, ⟨1, #[2]⟩]) =
    .ok (4, 3, [1, 2, 2, 1], [1, 1, 1, 1], 1/8) := by
  prepareReplaySeed (⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed) atSize 2
  prepareReplayBirth (⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed).nodeCount withM 1
    fitness (1) targets #[1]
  prepareReplayBirth ((⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed).nodeCount + 1) withM 1
    fitness (1) targets #[2]
  finishReplaySummary 2 withM 1 births 2
example : summaryOf (replay rawTriangle 3
    [⟨1, #[0, 1, 2]⟩, ⟨1, #[1, 2, 3]⟩]) =
    .ok (5, 9, [3, 4, 4, 4, 3], [1, 2, 4, 1, 1], 1/252) := by
  prepareReplaySeed rawTriangle atSize 3
  prepareReplayBirth rawTriangle.nodeCount withM 3
    fitness (1) targets #[0, 1, 2]
  prepareReplayBirth (rawTriangle.nodeCount + 1) withM 3
    fitness (1) targets #[1, 2, 3]
  finishReplaySummary 3 withM 3 births 2

-- Empty input is not a validation bypass; seed errors precede invalid initial m.
example : summaryOf (replay rawTriangle 0 []) = .error .initialM := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 4 []) = .error .initialM := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay ⟨0, #[], #[]⟩ 0 []) =
    .error (.seed .invalidNodeCount) := by decide_cbv
example : summaryOf (replay ⟨2, #[1], #[(0, 1)]⟩ 1 []) =
    .error (.seed .fitnessSizeMismatch) := by decide_cbv
example : summaryOf (replay ⟨2, #[0, 1], #[(0, 1)]⟩ 1 []) =
    .error (.seed .nonpositiveFitness) := by decide_cbv

-- First failure has a zero-based index and carries no partial result.
example : summaryOf (replay rawTriangle 2 [⟨1, #[2, 2]⟩]) =
    .error (.atBirth 0 .duplicateTarget) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨1, #[3, 3]⟩]) =
    .error (.atBirth 1 .duplicateTarget) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨1, #[3, 4]⟩]) =
    .error (.atBirth 1 .targetOutOfRange) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨0, #[3, 2]⟩]) =
    .error (.atBirth 1 .nonpositiveFitness) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨1, #[3]⟩]) =
    .error (.atBirth 1 .targetCountMismatch) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 2
    [⟨1, #[0, 0]⟩, ⟨0, #[99]⟩]) = .error (.atBirth 0 .duplicateTarget) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv

end ExactReplayFixtures

-- Generic laws bind the actual raw API result, not a separate test simulator.
example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (h : replay seed m bs = .ok out) :
    actualNodeCount out.final.state.snapshot = seed.nodeCount + bs.length :=
  replay_nodes seed m bs out h
example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (s : State seed.nodeCount) (hs : parseSeed seed = .ok s)
    (h : replay seed m bs = .ok out) :
    actualEdgeCount out.final.state.snapshot = actualEdgeCount s.snapshot + m * bs.length :=
  replay_edges seed m bs out s hs h
example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (h : replay seed m bs = .ok out) :
    List.ofFn out.final.state.snapshot.fitness = seed.fitness.toList ++ bs.map RawBirth.fitness :=
  replay_fitness_prefix seed m bs out h
example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (h : replay seed m bs = .ok out) : 0 < out.probability :=
  replay_probability_pos seed m bs out h
example {n : Nat} (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
    (schedule : List PosFitness) : continuationMass s m hm hb schedule = 1 :=
  continuationMass_one s m hm hb schedule

#print axioms replay_empty
#print axioms replay_step
#print axioms replay_valid
#print axioms replay_nodes
#print axioms replay_edges
#print axioms replay_fitness_prefix
#print axioms replay_probability_pos
#print axioms continuationMass_one

-- Whole-run laws use the same raw replay and preserve the exact supplied order.
example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (c : PosFitness) (h : replay seed m bs = .ok out) :
    replay (scaleSeed seed c) m (bs.map (fun raw => scaleBirth raw c)) =
      .ok (scaleResult out c) := replay_scale seed m bs out c h

example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (c : PosFitness) (h : replay seed m bs = .ok out) :
    ∃ next, replay (scaleSeed seed c) m (bs.map (fun raw => scaleBirth raw c)) = .ok next ∧
      next.final = scaleRunState out.final c := replay_scale_topology seed m bs out c h

example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (c : PosFitness) (h : replay seed m bs = .ok out) :
    ∃ next, replay (scaleSeed seed c) m (bs.map (fun raw => scaleBirth raw c)) = .ok next ∧
      next.probability = out.probability := replay_scale_probability seed m bs out c h

-- BA is the unit-fitness specialization, not a separately implemented generator.
example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (c : PosFitness) (h : replay (unitSeed seed) m (bs.map unitBirth) = .ok out) :
    ∃ next, replay (constantSeed seed c) m (bs.map (fun raw => constantBirth raw c)) = .ok next ∧
      next.final = scaleRunState out.final c := replay_ba_topology seed m bs out c h

example (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (c : PosFitness) (h : replay (unitSeed seed) m (bs.map unitBirth) = .ok out) :
    ∃ next, replay (constantSeed seed c) m (bs.map (fun raw => constantBirth raw c)) = .ok next ∧
      next.probability = out.probability := replay_ba_probability seed m bs out c h

#print axioms replay_scale
#print axioms replay_scale_topology
#print axioms replay_scale_probability
#print axioms replay_ba_topology
#print axioms replay_ba_probability
