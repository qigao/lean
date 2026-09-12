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

private theorem checkedStep {n : Nat} (s : State n) (m : Nat) (raw : RawBirth)
    (hm : 0 < m ∧ m ≤ n) (hf : 0 < raw.fitness) (hs : raw.targets.size = m)
    (hb : targetsBounded n raw.targets) (hd : targetsDistinct raw.targets) :
    step s m raw = .ok
      (applyBirth s (hs ▸ (⟨hb, hd⟩ : CheckedTargets n raw.targets).embedding)
        hm.1 ⟨raw.fitness, hf⟩,
       orderedMass s (hs ▸ (⟨hb, hd⟩ : CheckedTargets n raw.targets).embedding)) := by
  simp only [step, validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]

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

private theorem runBirthsNil (m index : Nat) (s : RunState) :
    runBirths m index s [] = .ok ⟨s, 1⟩ := rfl

private theorem summaryOk (r : ReplayResult) (nodes edges : Nat)
    (degrees : List Nat) (fitness : List Rat) (probability : Rat) :
    summaryOf (.ok r) = .ok (nodes, edges, degrees, fitness, probability) ↔
      actualNodeCount r.final.state.snapshot = nodes ∧
      actualEdgeCount r.final.state.snapshot = edges ∧
      List.ofFn (degree r.final.state.snapshot) = degrees ∧
      List.ofFn r.final.state.snapshot.fitness = fitness ∧
      r.probability = probability := by
  simp only [summaryOf, Except.ok.injEq, Prod.mk.injEq]

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

private theorem birthFitnessFn {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    (applyBirth s T hm eta).snapshot.fitness =
      Fin.lastCases eta.val s.snapshot.fitness := rfl

private theorem triangleSeedDegreeFn (fitness : Array Rat) (hs : fitness.size = 3) :
    degree (seedSnapshot ⟨3, fitness, rawTriangle.edges⟩ hs) =
      (![2, 2, 2] : Fin 3 → Nat) := by
  funext i
  fin_cases i <;> decide_cbv

private theorem edgeSeedDegreeFn (fitness : Array Rat) (hs : fitness.size = 2) :
    degree (seedSnapshot ⟨2, fitness, #[(1, 0)]⟩ hs) =
      (![1, 1] : Fin 2 → Nat) := by
  funext i
  fin_cases i <;> decide_cbv

/-- Expose the complete seed weight function once, so traceMass never unfolds adjacency. -/
private theorem triangleSeedWeightsFn (fitness : Array Rat) (hs : fitness.size = 3) :
    weights (seedSnapshot ⟨3, fitness, rawTriangle.edges⟩ hs) =
      fun i => (seedSnapshot ⟨3, fitness, rawTriangle.edges⟩ hs).fitness i *
        ((![2, 2, 2] : Fin 3 → Nat) i : Rat) := by
  funext i
  unfold weights
  rw [triangleSeedDegreeFn fitness hs]

private theorem edgeSeedWeightsFn (fitness : Array Rat) (hs : fitness.size = 2) :
    weights (seedSnapshot ⟨2, fitness, #[(1, 0)]⟩ hs) =
      fun i => (seedSnapshot ⟨2, fitness, #[(1, 0)]⟩ hs).fitness i *
        ((![1, 1] : Fin 2 → Nat) i : Rat) := by
  funext i
  unfold weights
  rw [edgeSeedDegreeFn fitness hs]

private theorem triangleSeedWeightsVector (fitness : Array Rat) (hs : fitness.size = 3)
    (out : Fin 3 → Rat)
    (h : (fun i => (seedSnapshot ⟨3, fitness, rawTriangle.edges⟩ hs).fitness i *
      ((![2, 2, 2] : Fin 3 → Nat) i : Rat)) = out) :
    weights (seedSnapshot ⟨3, fitness, rawTriangle.edges⟩ hs) = out := by
  rw [triangleSeedWeightsFn fitness hs]
  exact h

private theorem edgeSeedWeightsVector (fitness : Array Rat) (hs : fitness.size = 2)
    (out : Fin 2 → Rat)
    (h : (fun i => (seedSnapshot ⟨2, fitness, #[(1, 0)]⟩ hs).fitness i *
      ((![1, 1] : Fin 2 → Nat) i : Rat)) = out) :
    weights (seedSnapshot ⟨2, fitness, #[(1, 0)]⟩ hs) = out := by
  rw [edgeSeedWeightsFn fitness hs]
  exact h

/-- Expose successor weights through the already-proved fitness/degree update laws. -/
private theorem birthWeightsFn {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    weights (applyBirth s T hm eta).snapshot =
      fun v => (Fin.lastCases eta.val s.snapshot.fitness v) *
        ((Fin.lastCases m
          (fun u => degree s.snapshot u + if u ∈ T.selected then 1 else 0) v : Nat) : Rat) := by
  funext v
  unfold weights
  rw [birthFitnessFn s T hm eta, birthDegreeFn s T hm eta]

private theorem birthWeightsVector {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) (deg : Fin n → Nat)
    (selected : Finset (Fin n)) (out : Fin (n + 1) → Rat)
    (hdeg : degree s.snapshot = deg) (hsel : T.selected = selected)
    (hout : (fun v => (Fin.lastCases eta.val s.snapshot.fitness v) *
      ((Fin.lastCases m
        (fun u => deg u + if u ∈ selected then 1 else 0) v : Nat) : Rat)) = out) :
    weights (applyBirth s T hm eta).snapshot = out := by
  rw [birthWeightsFn s T hm eta, hdeg, hsel]
  exact hout

/-- A checked request preserves the authoritative raw order without carrying size casts
into the numerical trace. -/
private theorem checkedTargetsOrdered {n m : Nat} (xs : Array Nat)
    (hs : xs.size = m) (hb : targetsBounded n xs) (hd : targetsDistinct xs) :
    Targets.ordered (hs ▸ (⟨hb, hd⟩ : CheckedTargets n xs).embedding) =
      List.ofFn (fun i : Fin xs.size => ⟨xs[i.val], hb i⟩) := by
  subst m
  rfl

private theorem selected_eq_ordered_toFinset {n m : Nat} (T : Targets n m) :
    T.selected = T.ordered.toFinset := by
  ext j
  simp [Targets.selected, Targets.ordered]

/-- Keep concrete rational normalization behind an opaque kernel boundary. -/
private theorem triangleTrace21 :
    traceMass (![2, 4, 8] : Fin 3 → Rat) ∅ [2, 1] = 8/21 := by
  decide_cbv

private theorem triangleTrace12 :
    traceMass (![2, 4, 8] : Fin 3 → Rat) ∅ [1, 2] = 8/35 := by
  decide_cbv

/-- The raw triangle is connected independently of the parser's reachability check. -/
private theorem rawTriangleConnected : (seedGraph rawTriangle).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v
      exact ⟨.nil⟩
    · have huv : (seedGraph rawTriangle).Adj u v := by
        fin_cases u <;> fin_cases v <;>
          simp_all [seedGraph, rawTriangle, canonicalEdge]
      exact ⟨.cons huv .nil⟩
  nonempty := ⟨⟨0, by decide⟩⟩

#print axioms rawTriangleConnected

/-- Package the semantic seed contract before invoking parser completeness. -/
private theorem rawTriangleValid : rawTriangle.Valid := by
  exact ⟨by decide, rfl, by decide, by decide, by decide, rawTriangleConnected⟩

#print axioms rawTriangleValid

/-- Seal the concrete parser proof before replay composition uses its dependent state. -/
private theorem triangleSeedAccepted :
    ∃ s : State rawTriangle.nodeCount, parseSeed rawTriangle = .ok s :=
  parseSeed_complete rawTriangle rawTriangleValid

#print axioms triangleSeedAccepted

/-- Keep the accepted state abstract while exposing only its authoritative snapshot. -/
private theorem triangleSeedSnapshot (s : State rawTriangle.nodeCount)
    (h : parseSeed rawTriangle = .ok s) :
    s.snapshot = seedSnapshot rawTriangle rfl := by
  obtain ⟨hs, hsnapshot⟩ := parseSeed_snapshot rawTriangle s h
  simpa using hsnapshot

#print axioms triangleSeedSnapshot

/-- Seal the concrete validator proof before its dependent target embedding is consumed. -/
private theorem triangleBirth12Accepted (s : State rawTriangle.nodeCount) :
    ∃ v : ValidatedBirth rawTriangle.nodeCount 2,
      validateBirth s 2 (⟨3/2, #[1, 2]⟩ : RawBirth) = .ok v := by
  apply validateBirth_complete
  exact ⟨by decide, by norm_num, rfl, by decide, by decide⟩

#print axioms triangleBirth12Accepted

/-- Expose only the stable data preserved by the accepted `[1, 2]` request. -/
private theorem triangleBirth12Data (s : State 3)
    (v : ValidatedBirth 3 2)
    (h : validateBirth s 2 (⟨3/2, #[1, 2]⟩ : RawBirth) = .ok v) :
    v.targets.ordered = ([1, 2] : List (Fin 3)) ∧
      v.targets.selected = ({1, 2} : Finset (Fin 3)) ∧
      v.fitness.val = 3/2 := by
  obtain ⟨hfitness, hs, hvalues⟩ :=
    validateBirth_values s (⟨3/2, #[1, 2]⟩ : RawBirth) v h
  have htargets : (v.targets : Fin 2 → Fin 3) = ![1, 2] := by
    funext i
    fin_cases i
    · exact Fin.ext (hvalues (0 : Fin 2))
    · exact Fin.ext (hvalues (1 : Fin 2))
  have hordered : v.targets.ordered = ([1, 2] : List (Fin 3)) :=
    (congrArg (fun f : Fin 2 → Fin 3 => List.ofFn f) htargets).trans (by decide_cbv)
  refine ⟨hordered, ?_, hfitness⟩
  rewrite [selected_eq_ordered_toFinset, hordered]
  decide_cbv

#print axioms triangleBirth12Data

/-- Unfold raw replay once with abstract parser and validator witnesses. -/
private theorem replayOneBirth {seed : RawSeed} {m : Nat} {raw : RawBirth}
    (s : State seed.nodeCount) (v : ValidatedBirth seed.nodeCount m)
    (hseed : parseSeed seed = .ok s)
    (hm : 0 < m ∧ m ≤ seed.nodeCount)
    (hbirth : validateBirth s m raw = .ok v) :
    replay seed m [raw] =
      .ok ⟨⟨seed.nodeCount + 1,
        applyBirth s v.targets v.positive v.fitness⟩,
        orderedMass s v.targets⟩ := by
  simp only [replay, hseed, if_pos hm, runBirths, step, hbirth, mul_one]

#print axioms replayOneBirth

private theorem rawTriangleEdgeCount :
    actualEdgeCount (seedSnapshot rawTriangle rfl) = 3 := by
  decide_cbv

#print axioms rawTriangleEdgeCount

private theorem rawTriangleDegree :
    degree (seedSnapshot rawTriangle rfl) = (![2, 2, 2] : Fin 3 → Nat) := by
  simpa only [rawTriangle] using
    (triangleSeedDegreeFn (#[1, 2, 4] : Array Rat) rfl)

#print axioms rawTriangleDegree

private theorem rawTriangleWeights :
    weights (seedSnapshot rawTriangle rfl) = (![2, 4, 8] : Fin 3 → Rat) := by
  simpa only [rawTriangle] using
    (triangleSeedWeightsVector (fitness := (#[1, 2, 4] : Array Rat))
      (hs := rfl) (out := (![2, 4, 8] : Fin 3 → Rat))
      (h := by funext i; fin_cases i <;> decide_cbv))

#print axioms rawTriangleWeights

private theorem triangleBirth12Nodes (s : State rawTriangle.nodeCount)
    (v : ValidatedBirth rawTriangle.nodeCount 2) :
    actualNodeCount (applyBirth s v.targets v.positive v.fitness).snapshot = 4 := by
  exact Fintype.card_fin (rawTriangle.nodeCount + 1)

#print axioms triangleBirth12Nodes

private theorem triangleBirth12Edges (s : State rawTriangle.nodeCount)
    (v : ValidatedBirth rawTriangle.nodeCount 2)
    (hseed : parseSeed rawTriangle = .ok s) :
    actualEdgeCount (applyBirth s v.targets v.positive v.fitness).snapshot = 5 := by
  rewrite (transparency := .default)
    [birth_edges, triangleSeedSnapshot s hseed, rawTriangleEdgeCount]
  rfl

#print axioms triangleBirth12Edges

private theorem triangleBirth12Degrees (s : State rawTriangle.nodeCount)
    (v : ValidatedBirth rawTriangle.nodeCount 2)
    (hseed : parseSeed rawTriangle = .ok s)
    (hbirth : validateBirth s 2 (⟨3/2, #[1, 2]⟩ : RawBirth) = .ok v) :
    List.ofFn (degree (applyBirth s v.targets v.positive v.fitness).snapshot) =
      [2, 3, 3, 2] := by
  obtain ⟨_, hselected, _⟩ := triangleBirth12Data s v hbirth
  rewrite (transparency := .default)
    [birthDegreeFn, triangleSeedSnapshot s hseed, rawTriangleDegree, hselected]
  decide_cbv

#print axioms triangleBirth12Degrees

private theorem triangleBirth12Fitness (s : State rawTriangle.nodeCount)
    (v : ValidatedBirth rawTriangle.nodeCount 2)
    (hseed : parseSeed rawTriangle = .ok s)
    (hbirth : validateBirth s 2 (⟨3/2, #[1, 2]⟩ : RawBirth) = .ok v) :
    List.ofFn (applyBirth s v.targets v.positive v.fitness).snapshot.fitness =
      [1, 2, 4, 3/2] := by
  obtain ⟨_, _, hfitness⟩ := triangleBirth12Data s v hbirth
  rewrite (transparency := .default)
    [birthFitnessFn, triangleSeedSnapshot s hseed, hfitness]
  decide_cbv

#print axioms triangleBirth12Fitness

private theorem triangleBirth12Mass (s : State rawTriangle.nodeCount)
    (v : ValidatedBirth rawTriangle.nodeCount 2)
    (hseed : parseSeed rawTriangle = .ok s)
    (hbirth : validateBirth s 2 (⟨3/2, #[1, 2]⟩ : RawBirth) = .ok v) :
    orderedMass s v.targets = 8/35 := by
  obtain ⟨hordered, _, _⟩ := triangleBirth12Data s v hbirth
  unfold orderedMass
  rewrite (transparency := .default)
    [triangleSeedSnapshot s hseed, rawTriangleWeights, hordered]
  exact triangleTrace12

#print axioms triangleBirth12Mass

/-- Compose opaque field facts without rechecking concrete parser/validator terms. -/
private theorem triangleBirth12Summary (s : State rawTriangle.nodeCount)
    (v : ValidatedBirth rawTriangle.nodeCount 2)
    (hseed : parseSeed rawTriangle = .ok s)
    (hbirth : validateBirth s 2 (⟨3/2, #[1, 2]⟩ : RawBirth) = .ok v) :
    summaryOf (.ok ⟨⟨rawTriangle.nodeCount + 1,
      applyBirth s v.targets v.positive v.fitness⟩,
      orderedMass s v.targets⟩) =
      .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/35) := by
  apply (summaryOk _ _ _ _ _ _).mpr
  exact ⟨triangleBirth12Nodes s v, triangleBirth12Edges s v hseed,
    triangleBirth12Degrees s v hseed hbirth,
    triangleBirth12Fitness s v hseed hbirth,
    triangleBirth12Mass s v hseed hbirth⟩

#print axioms triangleBirth12Summary

/-- Audit the complete raw replay fixture, including its accepted witnesses. -/
private theorem triangleReplay12Accepted :
    summaryOf (replay rawTriangle 2 [⟨3/2, #[1, 2]⟩]) =
      .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/35) := by
  obtain ⟨s, hseed⟩ := triangleSeedAccepted
  obtain ⟨v, hbirth⟩ := triangleBirth12Accepted s
  exact (congrArg summaryOf
    (replayOneBirth (seed := rawTriangle) (m := 2)
      (raw := (⟨3/2, #[1, 2]⟩ : RawBirth)) s v hseed (by decide) hbirth)).trans
      (triangleBirth12Summary s v hseed hbirth)

#print axioms triangleReplay12Accepted

/-- Read the checked request through its public value contract, keeping the validator
result abstract. No concrete parser or validator proof enters trace evaluation. -/
private theorem validatedTargetsData {n : Nat} (s : State n) (raw : RawBirth)
    (v : ValidatedBirth n raw.targets.size)
    (h : validateBirth s raw.targets.size raw = .ok v)
    (hb : targetsBounded n raw.targets) :
    v.targets.ordered =
      List.ofFn (fun i : Fin raw.targets.size => (⟨raw.targets[i.val], hb i⟩ : Fin n)) ∧
      v.fitness.val = raw.fitness := by
  obtain ⟨hfitness, hs, hvalues⟩ := validateBirth_values s raw v h
  have htargets : (v.targets : Fin raw.targets.size → Fin n) =
      fun i => ⟨raw.targets[i.val], hb i⟩ := by
    funext i
    exact Fin.ext (hvalues i)
  exact ⟨congrArg (fun f : Fin raw.targets.size → Fin n => List.ofFn f) htargets,
    hfitness⟩

/-- Compose a birth using an abstract validated result, as in replayOneBirth. -/
private theorem validatedBirthEquation {n m index : Nat} (s : State n)
    (raw : RawBirth) (rest : List RawBirth) (v : ValidatedBirth n m)
    (h : validateBirth s m raw = .ok v) :
    runBirths m index ⟨n, s⟩ (raw :: rest) =
      match runBirths m (index + 1)
          ⟨n + 1, applyBirth s v.targets v.positive v.fitness⟩ rest with
      | .error e => .error e
      | .ok out => .ok ⟨out.final, orderedMass s v.targets * out.probability⟩ := by
  rewrite [replay_step, step, h]
  rfl

private theorem rawEdgeConnected :
    (seedGraph (⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed)).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v
      exact ⟨.nil⟩
    · have huv : (seedGraph (⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed)).Adj u v := by
        fin_cases u <;> fin_cases v <;> simp_all [seedGraph, canonicalEdge]
      exact ⟨.cons huv .nil⟩
  nonempty := ⟨⟨0, by decide⟩⟩

#print axioms validatedTargetsData
#print axioms validatedBirthEquation
#print axioms rawEdgeConnected

#print axioms runBirthsNil
#print axioms summaryOk
#print axioms birthDegreeFn
#print axioms birthFitnessFn
#print axioms triangleSeedDegreeFn
#print axioms edgeSeedDegreeFn
#print axioms triangleSeedWeightsFn
#print axioms edgeSeedWeightsFn
#print axioms triangleSeedWeightsVector
#print axioms edgeSeedWeightsVector
#print axioms birthWeightsFn
#print axioms birthWeightsVector
#print axioms checkedTargetsOrdered
#print axioms selected_eq_ordered_toFinset
#print axioms triangleTrace21
#print axioms triangleTrace12

end NarrativeDynamics.FitnessAttachment.ReplayFixtures

open NarrativeDynamics.FitnessAttachment.ReplayFixtures

/-- Match local field equations at default transparency: fixture dimensions may be
written as a raw seed's nodeCount or as a numeral. Every rewrite produces a checked
equality proof; fields absent from the current goal are simply left for other goals. -/
elab "rewriteReplayFacts" : tactic => Lean.Elab.Tactic.withMainContext do
  let context ← Lean.getLCtx
  for decl in context do
    let type ← Lean.instantiateMVars decl.type
    let some (_, lhs, _) ← Lean.Meta.matchEq? type | continue
    let head := (← Lean.instantiateMVars lhs).getAppFn'
    unless head.isConstOf ``State.snapshot || head.isConstOf ``Targets.selected ||
        head.isConstOf ``Targets.ordered || head.isConstOf ``Subtype.val do continue
    let saved ← Lean.Elab.Tactic.saveState
    try
      let goal ← Lean.Elab.Tactic.getMainGoal
      let result ← goal.rewrite (← goal.getType) decl.toExpr
        (config := { transparency := .default })
      let next ← goal.replaceTargetEq result.eNew result.eqProof
      Lean.Elab.Tactic.replaceMainGoal (next :: result.mvarIds)
      Lean.logInfo m!"replay-fixture field rewritten: {decl.userName.eraseMacroScopes}"
    catch _ => saved.restore

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
     simp (config := { failIfUnchanged := false }) only [twoBirths, if_pos hm]
     trace "replay-fixture birth: checked rewrite start"
     rewrite [checkedBirthEquation (n := $n) (m := $m) (raw := (⟨$eta, $xs⟩ : RawBirth))
       hm hf hs hb hd]
     trace "replay-fixture birth: checked rewrite done"))

macro "prepareAcceptedReplaySeed " raw:term " atSize " size:num " as " s:ident : tactic => do
  let connected ← if size.getNat == 3 then
    `(tactic| exact rawTriangleConnected)
  else
    `(tactic| exact rawEdgeConnected)
  `(tactic|
    (trace "replay-fixture seed: abstract witness start"
     have hvalid : ($raw : RawSeed).Valid := by
       exact ⟨by decide, rfl, by decide, by decide, by decide, by $connected:tactic⟩
     obtain ⟨$s, hseed⟩ := parseSeed_complete $raw hvalid
     obtain ⟨hs, hsnapshot⟩ := parseSeed_snapshot $raw $s hseed
     simp only [replay, hseed]
     trace "replay-fixture seed: abstract witness done"))

macro "prepareAcceptedReplayBirth " n:term " withM " m:term
    " fitness " eta:term " targets " xs:term " from " s:term " as " v:ident : tactic =>
  `(tactic|
    (trace "replay-fixture birth: abstract witness start"
     have hm : 0 < $m ∧ $m ≤ $n := by decide
     have hf : 0 < ($eta : Rat) := by norm_num
     have hs : ($xs : Array Nat).size = $m := rfl
     have hb : targetsBounded $n $xs := by decide
     have hd : targetsDistinct $xs := by decide
     obtain ⟨$v, hbirth⟩ := validateBirth_complete $s (⟨$eta, $xs⟩ : RawBirth)
       (show RawBirth.Valid (⟨$eta, $xs⟩ : RawBirth) $n $m from ⟨hm, hf, hs, hb, hd⟩)
     obtain ⟨hordered, hfitness⟩ :=
       validatedTargetsData $s (⟨$eta, $xs⟩ : RawBirth) $v hbirth hb
     have hselected := (selected_eq_ordered_toFinset ($v).targets).trans
       (congrArg List.toFinset hordered)
     simp (config := { failIfUnchanged := false }) only [twoBirths, if_pos hm]
     rewrite [validatedBirthEquation $s (⟨$eta, $xs⟩ : RawBirth) _ $v hbirth]
     trace "replay-fixture birth: abstract witness done"))

macro "proveReplayProbability1 " "atSize " size:num " seedWeights " seedW:term : tactic => do
  let rewriteSeed ← if size.getNat == 3 then
    `(tactic| rewrite (transparency := .default) [triangleSeedWeightsVector (out := $seedW)
      (h := by funext i; fin_cases i <;> decide_cbv)])
  else
    `(tactic| rewrite (transparency := .default) [edgeSeedWeightsVector (out := $seedW)
      (h := by funext i; fin_cases i <;> decide_cbv)])
  `(tactic|
    (trace "replay-fixture probability: seed weights rewrite"
     $rewriteSeed:tactic
     trace "replay-fixture probability: abstract targets evaluation"
     decide_cbv))

macro "proveReplayProbability2 " "atSize " size:num " seedWeights " seedW:term
    " selected " selected:term " nextWeights " nextW:term : tactic => do
  let rewriteBirth ← if size.getNat == 3 then
    `(tactic| rewrite (transparency := .default) [birthWeightsVector
      (deg := (![2, 2, 2] : Fin 3 → Nat))
      (selected := $selected) (out := $nextW)
      (hdeg := by rewriteReplayFacts; exact triangleSeedDegreeFn _ (by decide))
      (hsel := by
        rewriteReplayFacts
        decide_cbv)
      (hout := by funext v; rewriteReplayFacts; fin_cases v <;> decide_cbv)])
  else
    `(tactic| rewrite (transparency := .default) [birthWeightsVector
      (deg := (![1, 1] : Fin 2 → Nat))
      (selected := $selected) (out := $nextW)
      (hdeg := by rewriteReplayFacts; exact edgeSeedDegreeFn _ (by decide))
      (hsel := by
        rewriteReplayFacts
        decide_cbv)
      (hout := by funext v; rewriteReplayFacts; fin_cases v <;> decide_cbv)])
  let rewriteSeed ← if size.getNat == 3 then
    `(tactic| rewrite (transparency := .default) [triangleSeedWeightsVector (out := $seedW)
      (h := by funext i; fin_cases i <;> decide_cbv)])
  else
    `(tactic| rewrite (transparency := .default) [edgeSeedWeightsVector (out := $seedW)
      (h := by funext i; fin_cases i <;> decide_cbv)])
  `(tactic|
    (trace "replay-fixture probability: successor weights rewrite"
     $rewriteBirth:tactic
     trace "replay-fixture probability: seed weights rewrite"
     $rewriteSeed:tactic
     trace "replay-fixture probability: numeric evaluation"
     decide_cbv
     trace "replay-fixture probability: numeric evaluation done"))

macro "finishReplaySummary " n:term " withM " m:term " births " rounds:num
    " probability " probability:tactic : tactic => do
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
       rewriteReplayFacts
       decide_cbv
     · trace "replay-fixture result: degrees rewrite"
       $degrees:tactic
       rewriteReplayFacts
       first | rewrite (transparency := .default) [triangleSeedDegreeFn (hs := by decide)]
             | rewrite (transparency := .default) [edgeSeedDegreeFn (hs := by decide)]
       trace "replay-fixture result: degrees evaluation"
       decide_cbv
     · trace "replay-fixture result: fitness rewrite"
       $fitnessTac:tactic
       rewriteReplayFacts
       trace "replay-fixture result: fitness evaluation"
       decide_cbv
     · trace "replay-fixture result: probability"
       simp only [ReplayResult.probability, mul_one, orderedMass, checkedTargetsOrdered]
       rewriteReplayFacts
       simp only [rawTriangle, unitTriangle]
       $probability:tactic))

section ExactReplayFixtures
set_option maxRecDepth 4096

example : summaryOf (replay rawTriangle 2 []) =
    .ok (3, 3, [2, 2, 2], [1, 2, 4], 1) := by
  prepareReplaySeed rawTriangle atSize 3
  decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩]) =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/21) := by
  prepareAcceptedReplaySeed rawTriangle atSize 3 as s0
  prepareAcceptedReplayBirth rawTriangle.nodeCount withM 2
    fitness (3/2) targets #[2, 1] from s0 as v1
  finishReplaySummary 3 withM 2 births 1 probability
    (proveReplayProbability1 atSize 3 seedWeights (![2, 4, 8] : Fin 3 → Rat))
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[1, 2]⟩]) =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/35) := by
  exact triangleReplay12Accepted
example : summaryOf (replay rawTriangle 2 twoBirths) =
    .ok (5, 7, [2, 3, 4, 3, 2], [1, 2, 4, 3/2, 1/3], 24/805) := by
  prepareAcceptedReplaySeed rawTriangle atSize 3 as s0
  prepareAcceptedReplayBirth rawTriangle.nodeCount withM 2
    fitness (3/2) targets #[2, 1] from s0 as v1
  prepareAcceptedReplayBirth (rawTriangle.nodeCount + 1) withM 2
    fitness (1/3) targets #[3, 2] from (applyBirth s0 v1.targets v1.positive v1.fitness) as v2
  finishReplaySummary 3 withM 2 births 2 probability
    (proveReplayProbability2 atSize 3 seedWeights (![2, 4, 8] : Fin 3 → Rat)
      selected ({1, 2} : Finset (Fin 3))
      nextWeights (![2, 6, 12, 3] : Fin 4 → Rat))

example : summaryOf (replay ⟨3, #[2, 4, 8], rawTriangle.edges⟩ 2
    [⟨3, #[2, 1]⟩, ⟨2/3, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [2, 4, 8, 3, 2/3], 24/805) := by
  prepareAcceptedReplaySeed (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed) atSize 3 as s0
  prepareAcceptedReplayBirth (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount withM 2
    fitness (3) targets #[2, 1] from s0 as v1
  prepareAcceptedReplayBirth ((⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount + 1) withM 2
    fitness (2/3) targets #[3, 2] from (applyBirth s0 v1.targets v1.positive v1.fitness) as v2
  finishReplaySummary 3 withM 2 births 2 probability
    (proveReplayProbability2 atSize 3 seedWeights (![4, 8, 16] : Fin 3 → Rat)
      selected ({1, 2} : Finset (Fin 3))
      nextWeights (![4, 12, 24, 6] : Fin 4 → Rat))
example : summaryOf (replay ⟨3, #[2, 4, 8], rawTriangle.edges⟩ 2 twoBirths) =
    .ok (5, 7, [2, 3, 4, 3, 2], [2, 4, 8, 3/2, 1/3], 24/1505) := by
  prepareAcceptedReplaySeed (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed) atSize 3 as s0
  prepareAcceptedReplayBirth (⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount withM 2
    fitness (3/2) targets #[2, 1] from s0 as v1
  prepareAcceptedReplayBirth ((⟨3, #[2, 4, 8], rawTriangle.edges⟩ : RawSeed).nodeCount + 1) withM 2
    fitness (1/3) targets #[3, 2] from (applyBirth s0 v1.targets v1.positive v1.fitness) as v2
  finishReplaySummary 3 withM 2 births 2 probability
    (proveReplayProbability2 atSize 3 seedWeights (![4, 8, 16] : Fin 3 → Rat)
      selected ({1, 2} : Finset (Fin 3))
      nextWeights (![4, 12, 24, 3] : Fin 4 → Rat))

example : summaryOf (replay ⟨3, #[3, 3, 3], rawTriangle.edges⟩ 2
    [⟨3, #[2, 1]⟩, ⟨3, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [3, 3, 3, 3, 3], 1/80) := by
  prepareAcceptedReplaySeed (⟨3, #[3, 3, 3], rawTriangle.edges⟩ : RawSeed) atSize 3 as s0
  prepareAcceptedReplayBirth (⟨3, #[3, 3, 3], rawTriangle.edges⟩ : RawSeed).nodeCount withM 2
    fitness (3) targets #[2, 1] from s0 as v1
  prepareAcceptedReplayBirth ((⟨3, #[3, 3, 3], rawTriangle.edges⟩ : RawSeed).nodeCount + 1) withM 2
    fitness (3) targets #[3, 2] from (applyBirth s0 v1.targets v1.positive v1.fitness) as v2
  finishReplaySummary 3 withM 2 births 2 probability
    (proveReplayProbability2 atSize 3 seedWeights (![6, 6, 6] : Fin 3 → Rat)
      selected ({1, 2} : Finset (Fin 3))
      nextWeights (![6, 9, 9, 6] : Fin 4 → Rat))
example : summaryOf (replay unitTriangle 2 [⟨1, #[2, 1]⟩, ⟨1, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [1, 1, 1, 1, 1], 1/80) := by
  prepareAcceptedReplaySeed unitTriangle atSize 3 as s0
  prepareAcceptedReplayBirth unitTriangle.nodeCount withM 2
    fitness (1) targets #[2, 1] from s0 as v1
  prepareAcceptedReplayBirth (unitTriangle.nodeCount + 1) withM 2
    fitness (1) targets #[3, 2] from (applyBirth s0 v1.targets v1.positive v1.fitness) as v2
  finishReplaySummary 3 withM 2 births 2 probability
    (proveReplayProbability2 atSize 3 seedWeights (![2, 2, 2] : Fin 3 → Rat)
      selected ({1, 2} : Finset (Fin 3))
      nextWeights (![2, 3, 3, 2] : Fin 4 → Rat))

example : summaryOf (replay ⟨2, #[1, 1], #[(1, 0)]⟩ 1
    [⟨1, #[1]⟩, ⟨1, #[2]⟩]) =
    .ok (4, 3, [1, 2, 2, 1], [1, 1, 1, 1], 1/8) := by
  prepareAcceptedReplaySeed (⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed) atSize 2 as s0
  prepareAcceptedReplayBirth (⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed).nodeCount withM 1
    fitness (1) targets #[1] from s0 as v1
  prepareAcceptedReplayBirth ((⟨2, #[1, 1], #[(1, 0)]⟩ : RawSeed).nodeCount + 1) withM 1
    fitness (1) targets #[2] from (applyBirth s0 v1.targets v1.positive v1.fitness) as v2
  finishReplaySummary 2 withM 1 births 2 probability
    (proveReplayProbability2 atSize 2 seedWeights (![1, 1] : Fin 2 → Rat)
      selected ({1} : Finset (Fin 2))
      nextWeights (![1, 2, 1] : Fin 3 → Rat))
example : summaryOf (replay rawTriangle 3
    [⟨1, #[0, 1, 2]⟩, ⟨1, #[1, 2, 3]⟩]) =
    .ok (5, 9, [3, 4, 4, 4, 3], [1, 2, 4, 1, 1], 1/252) := by
  prepareAcceptedReplaySeed rawTriangle atSize 3 as s0
  prepareAcceptedReplayBirth rawTriangle.nodeCount withM 3
    fitness (1) targets #[0, 1, 2] from s0 as v1
  prepareAcceptedReplayBirth (rawTriangle.nodeCount + 1) withM 3
    fitness (1) targets #[1, 2, 3] from (applyBirth s0 v1.targets v1.positive v1.fitness) as v2
  finishReplaySummary 3 withM 3 births 2 probability
    (proveReplayProbability2 atSize 3 seedWeights (![2, 4, 8] : Fin 3 → Rat)
      selected ({0, 1, 2} : Finset (Fin 3))
      nextWeights (![3, 6, 12, 3] : Fin 4 → Rat))

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
