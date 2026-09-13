import NarrativeDynamics.Core.FitnessReplay
import NarrativeDynamics.Core.SmallWorldMetrics

/-!
# Finite BB scope: a positive-probability seven-hop path

The acceptance boundary is the real checked replay and its typed successors.
Distances use the existing mesh semantics, including a structural lower bound.
-/

open NarrativeDynamics
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

namespace NarrativeDynamics.FitnessAttachment.ScopeFixtures

set_option maxRecDepth 4096

def pathSeed : RawSeed := ⟨2, #[1, 1], #[(0, 1)]⟩

def pathBirths : List RawBirth :=
  [⟨1, #[1]⟩, ⟨1, #[2]⟩, ⟨1, #[3]⟩,
   ⟨1, #[4]⟩, ⟨1, #[5]⟩, ⟨1, #[6]⟩]

def scopeSummary : Except ReplayError ReplayResult →
    Except ReplayError (Nat × Nat × List Nat × List Rat × Rat)
  | .error e => .error e
  | .ok out => .ok (actualNodeCount out.final.state.snapshot,
      actualEdgeCount out.final.state.snapshot,
      List.ofFn (degree out.final.state.snapshot),
      List.ofFn out.final.state.snapshot.fitness, out.probability)

/-- The seed validity proof is semantic; parser completeness supplies acceptance. -/
private theorem pathSeedConnected : (seedGraph pathSeed).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v
      exact ⟨.nil⟩
    · have huv : (seedGraph pathSeed).Adj u v := by
        fin_cases u <;> fin_cases v <;>
          simp_all [seedGraph, pathSeed, canonicalEdge]
      exact ⟨.cons huv .nil⟩
  nonempty := ⟨⟨0, by decide⟩⟩

private theorem pathSeedValid : pathSeed.Valid :=
  ⟨by decide, rfl, by decide, by decide, by decide, pathSeedConnected⟩

def path2State : State 2 :=
  ⟨seedSnapshot pathSeed rfl, by
    exact ⟨by decide, pathSeedConnected, by decide⟩⟩

private theorem path2Parsed : parseSeed pathSeed = .ok path2State := by
  obtain ⟨s, hparse⟩ := parseSeed_complete pathSeed pathSeedValid
  obtain ⟨hs, hsnapshot⟩ := parseSeed_snapshot pathSeed s hparse
  have hstate : s = path2State := by
    cases s with
    | mk snapshot valid =>
      change snapshot = seedSnapshot pathSeed hs at hsnapshot
      cases hsnapshot
      rfl
  simpa only [hstate] using hparse

private def unitFitness : PosFitness := ⟨1, by norm_num⟩

/-- The single old target is the last existing vertex, retaining its numeric ID. -/
private def lastTarget (n : Nat) : Targets (n + 1) 1 :=
  ⟨fun _ => Fin.last n, fun _ _ _ => Subsingleton.elim _ _⟩

def path3State : State 3 :=
  applyBirth path2State (lastTarget 1) (by decide) unitFitness

def path4State : State 4 :=
  applyBirth path3State (lastTarget 2) (by decide) unitFitness

def path5State : State 5 :=
  applyBirth path4State (lastTarget 3) (by decide) unitFitness

def path6State : State 6 :=
  applyBirth path5State (lastTarget 4) (by decide) unitFitness

def path7State : State 7 :=
  applyBirth path6State (lastTarget 5) (by decide) unitFitness

def path8State : State 8 :=
  applyBirth path7State (lastTarget 6) (by decide) unitFitness

/-- Abstract accepted validator data identify the exact typed successor. -/
private theorem pathStep (n : Nat) (s : State (n + 1)) :
    step s 1 (⟨1, #[n]⟩ : RawBirth) =
      .ok (applyBirth s (lastTarget n) (by decide) unitFitness,
        orderedMass s (lastTarget n)) := by
  have hvalid : RawBirth.Valid (⟨1, #[n]⟩ : RawBirth) (n + 1) 1 := by
    refine ⟨⟨by decide, by omega⟩, by norm_num, rfl, ?_, ?_⟩
    · intro i
      fin_cases i
      simp
    · intro i j _
      exact Subsingleton.elim (α := Fin 1) i j
  obtain ⟨v, hv⟩ := validateBirth_complete s (⟨1, #[n]⟩ : RawBirth) hvalid
  obtain ⟨hf, hs, hvalues⟩ := validateBirth_values s (⟨1, #[n]⟩ : RawBirth) v hv
  have ht : v.targets = lastTarget n := by
    apply Function.Embedding.ext
    intro i
    apply Fin.ext
    fin_cases i
    simpa [lastTarget] using hvalues (0 : Fin 1)
  have heta : v.fitness = unitFitness := by
    apply Subtype.ext
    exact hf
  apply (step_spec s (⟨1, #[n]⟩ : RawBirth) _).mpr
  refine ⟨v, hv, ?_⟩
  rw [ht, heta]

/-- Use the degree update law before finite arithmetic, avoiding replay reduction. -/
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

private theorem path2Degrees :
    degree path2State.snapshot = (![ 1, 1 ] : Fin 2 → Nat) := by
  funext i
  fin_cases i <;> decide_cbv

private theorem path2Fitness :
    path2State.snapshot.fitness = (![ 1, 1 ] : Fin 2 → Rat) := by
  funext i
  fin_cases i <;> decide_cbv

private theorem path2Weights :
    weights path2State.snapshot = (![ 1, 1 ] : Fin 2 → Rat) := by
  funext i
  unfold weights
  rw [path2Fitness, path2Degrees]
  fin_cases i <;> decide_cbv

private theorem path2Mass :
    orderedMass path2State (lastTarget 1) = 1/2 := by
  unfold orderedMass
  rw [path2Weights]
  decide_cbv

private theorem path3Degrees :
    degree path3State.snapshot = (![ 1, 2, 1 ] : Fin 3 → Nat) := by
  rw [path3State, birthDegreeFn, path2Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem path3Fitness :
    path3State.snapshot.fitness = (![ 1, 1, 1 ] : Fin 3 → Rat) := by
  rw [path3State, birthFitnessFn, path2Fitness]
  funext i
  fin_cases i <;> decide_cbv

private theorem path3Weights :
    weights path3State.snapshot = (![ 1, 2, 1 ] : Fin 3 → Rat) := by
  funext i
  unfold weights
  rw [path3Fitness, path3Degrees]
  fin_cases i <;> decide_cbv

private theorem path3Mass :
    orderedMass path3State (lastTarget 2) = 1/4 := by
  unfold orderedMass
  rw [path3Weights]
  decide_cbv

private theorem path4Degrees :
    degree path4State.snapshot = (![ 1, 2, 2, 1 ] : Fin 4 → Nat) := by
  rw [path4State, birthDegreeFn, path3Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem path4Fitness :
    path4State.snapshot.fitness = (![ 1, 1, 1, 1 ] : Fin 4 → Rat) := by
  rw [path4State, birthFitnessFn, path3Fitness]
  funext i
  fin_cases i <;> decide_cbv

private theorem path4Weights :
    weights path4State.snapshot = (![ 1, 2, 2, 1 ] : Fin 4 → Rat) := by
  funext i
  unfold weights
  rw [path4Fitness, path4Degrees]
  fin_cases i <;> decide_cbv

private theorem path4Mass :
    orderedMass path4State (lastTarget 3) = 1/6 := by
  unfold orderedMass
  rw [path4Weights]
  decide_cbv

private theorem path5Degrees :
    degree path5State.snapshot = (![ 1, 2, 2, 2, 1 ] : Fin 5 → Nat) := by
  rw [path5State, birthDegreeFn, path4Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem path5Fitness :
    path5State.snapshot.fitness = (![ 1, 1, 1, 1, 1 ] : Fin 5 → Rat) := by
  rw [path5State, birthFitnessFn, path4Fitness]
  funext i
  fin_cases i <;> decide_cbv

private theorem path5Weights :
    weights path5State.snapshot = (![ 1, 2, 2, 2, 1 ] : Fin 5 → Rat) := by
  funext i
  unfold weights
  rw [path5Fitness, path5Degrees]
  fin_cases i <;> decide_cbv

private theorem path5Mass :
    orderedMass path5State (lastTarget 4) = 1/8 := by
  unfold orderedMass
  rw [path5Weights]
  decide_cbv

private theorem path6Degrees :
    degree path6State.snapshot = (![ 1, 2, 2, 2, 2, 1 ] : Fin 6 → Nat) := by
  rw [path6State, birthDegreeFn, path5Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem path6Fitness :
    path6State.snapshot.fitness = (![ 1, 1, 1, 1, 1, 1 ] : Fin 6 → Rat) := by
  rw [path6State, birthFitnessFn, path5Fitness]
  funext i
  fin_cases i <;> decide_cbv

private theorem path6Weights :
    weights path6State.snapshot = (![ 1, 2, 2, 2, 2, 1 ] : Fin 6 → Rat) := by
  funext i
  unfold weights
  rw [path6Fitness, path6Degrees]
  fin_cases i <;> decide_cbv

private theorem path6Mass :
    orderedMass path6State (lastTarget 5) = 1/10 := by
  unfold orderedMass
  rw [path6Weights]
  decide_cbv

private theorem path7Degrees :
    degree path7State.snapshot = (![ 1, 2, 2, 2, 2, 2, 1 ] : Fin 7 → Nat) := by
  rw [path7State, birthDegreeFn, path6Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem path7Fitness :
    path7State.snapshot.fitness = (![ 1, 1, 1, 1, 1, 1, 1 ] : Fin 7 → Rat) := by
  rw [path7State, birthFitnessFn, path6Fitness]
  funext i
  fin_cases i <;> decide_cbv

private theorem path7Weights :
    weights path7State.snapshot = (![ 1, 2, 2, 2, 2, 2, 1 ] : Fin 7 → Rat) := by
  funext i
  unfold weights
  rw [path7Fitness, path7Degrees]
  fin_cases i <;> decide_cbv

private theorem path7Mass :
    orderedMass path7State (lastTarget 6) = 1/12 := by
  unfold orderedMass
  rw [path7Weights]
  decide_cbv

private theorem path8Degrees :
    degree path8State.snapshot = (![ 1, 2, 2, 2, 2, 2, 2, 1 ] : Fin 8 → Nat) := by
  rw [path8State, birthDegreeFn, path7Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem path8Fitness :
    path8State.snapshot.fitness = (![ 1, 1, 1, 1, 1, 1, 1, 1 ] : Fin 8 → Rat) := by
  rw [path8State, birthFitnessFn, path7Fitness]
  funext i
  fin_cases i <;> decide_cbv

/-- The product is evaluated only after all six conditional masses are checked. -/
private theorem pathMassProduct :
    (1/2 : Rat) * ((1/4) * ((1/6) * ((1/8) * ((1/10) * ((1/12) * 1))))) =
      1/46080 := by
  decide_cbv

/-- Raw replay equals six actual typed births, including the exact trace mass. -/
theorem path8_replay : replay pathSeed 1 pathBirths =
    .ok ⟨⟨8, path8State⟩, 1/46080⟩ := by
  have h2 := pathStep 1 path2State
  change step path2State 1 (⟨1, #[1]⟩ : RawBirth) =
    .ok (path3State, orderedMass path2State (lastTarget 1)) at h2
  have h3 := pathStep 2 path3State
  change step path3State 1 (⟨1, #[2]⟩ : RawBirth) =
    .ok (path4State, orderedMass path3State (lastTarget 2)) at h3
  have h4 := pathStep 3 path4State
  change step path4State 1 (⟨1, #[3]⟩ : RawBirth) =
    .ok (path5State, orderedMass path4State (lastTarget 3)) at h4
  have h5 := pathStep 4 path5State
  change step path5State 1 (⟨1, #[4]⟩ : RawBirth) =
    .ok (path6State, orderedMass path5State (lastTarget 4)) at h5
  have h6 := pathStep 5 path6State
  change step path6State 1 (⟨1, #[5]⟩ : RawBirth) =
    .ok (path7State, orderedMass path6State (lastTarget 5)) at h6
  have h7 := pathStep 6 path7State
  change step path7State 1 (⟨1, #[6]⟩ : RawBirth) =
    .ok (path8State, orderedMass path7State (lastTarget 6)) at h7
  simp only [replay, path2Parsed]
  change runBirths 1 0 ⟨2, path2State⟩ pathBirths = _
  simp only [pathBirths, runBirths, h2, h3, h4, h5, h6, h7,
    path2Mass, path3Mass, path4Mass, path5Mass, path6Mass, path7Mass,
    pathMassProduct]

private theorem path8Edges : actualEdgeCount path8State.snapshot = 7 := by
  simp only [path8State, path7State, path6State, path5State, path4State,
    path3State, birth_edges]
  decide_cbv

theorem path8_summary : scopeSummary (replay pathSeed 1 pathBirths) =
    .ok (8, 7, [1, 2, 2, 2, 2, 2, 2, 1],
      [1, 1, 1, 1, 1, 1, 1, 1], 1/46080) := by
  rw [path8_replay]
  change Except.ok (actualNodeCount path8State.snapshot,
    actualEdgeCount path8State.snapshot,
    List.ofFn (n := 8) (degree path8State.snapshot),
    List.ofFn (n := 8) path8State.snapshot.fitness, (1/46080 : Rat)) = _
  rw [path8Edges, path8Degrees, path8Fitness]
  decide_cbv

theorem path8_positive_mass (out : ReplayResult)
    (h : replay pathSeed 1 pathBirths = .ok out) : 0 < out.probability :=
  replay_probability_pos pathSeed 1 pathBirths out h

/-- Expose only the four adjacency cases, using the actual birth laws. -/
private theorem birthAdjFn {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    (applyBirth s T hm eta).snapshot.graph.Adj =
      Fin.lastCases (Fin.lastCases False (fun v => v ∈ T.selected))
        (fun u => Fin.lastCases (u ∈ T.selected) (fun v => s.snapshot.graph.Adj u v)) := by
  funext u v
  apply propext
  refine Fin.lastCases ?_ (fun a => ?_) u
  · refine Fin.lastCases ?_ (fun b => ?_) v
    · simp only [Fin.lastCases_last]
      exact ⟨(applyBirth s T hm eta).snapshot.graph.loopless.irrefl _, False.elim⟩
    · simp only [Fin.lastCases_last, Fin.lastCases_castSucc]
      exact birth_new_adj_iff_rev s T hm eta b
  · refine Fin.lastCases ?_ (fun b => ?_) v
    · simp only [Fin.lastCases_last, Fin.lastCases_castSucc]
      exact birth_new_adj_iff s T hm eta a
    · simp only [Fin.lastCases_castSucc]
      exact birth_old_adj_iff s T hm eta a b

private theorem lastTarget_selected (n : Nat) :
    (lastTarget n).selected = {Fin.last n} := by
  ext u
  simp [Targets.selected, lastTarget]

/-- The four seed pairs are checked once, independently of later proof records. -/
private theorem path2AdjFn : path2State.snapshot.graph.Adj =
    fun u v : Fin 2 => u.val + 1 = v.val ∨ v.val + 1 = u.val := by
  funext u v
  apply propext
  fin_cases u <;> fin_cases v <;>
    simp [path2State, seedSnapshot, seedGraph, pathSeed, canonicalEdge]

/-- For a nondependent result, expose last-vertex cases as an ordinary bounds test.
This avoids evaluating the reverse-induction implementation of `Fin.lastCases`. -/
private theorem lastCasesValue {n : Nat} {α : Type*}
    (fresh : α) (old : Fin n → α) (i : Fin (n + 1)) :
    Fin.lastCases fresh old i =
      if h : i.val < n then old ⟨i.val, h⟩ else fresh := by
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simp only [Fin.lastCases_last, Fin.val_last, Nat.lt_irrefl, dif_neg]
  · simp only [Fin.lastCases_castSucc, Fin.val_castSucc, dif_pos j.isLt]

/-- Check all 64 ordered pairs after exposing the actual successor adjacency fields. -/
theorem path8_adj (u v : Fin 8) : path8State.snapshot.graph.Adj u v ↔
    u.val + 1 = v.val ∨ v.val + 1 = u.val := by
  rw [path8State, birthAdjFn, path7State, birthAdjFn,
    path6State, birthAdjFn, path5State, birthAdjFn,
    path4State, birthAdjFn, path3State, birthAdjFn, path2AdjFn]
  simp only [lastTarget_selected]
  fin_cases u <;> fin_cases v <;> norm_num [lastCasesValue, Fin.ext_iff]

theorem path8_walk_seven_exact :
    MeshWalk path8State.snapshot.graph.Adj 7 (0 : Fin 8) 7 := by
  exact .step ((path8_adj 0 1).2 (by omega))
    (.step ((path8_adj 1 2).2 (by omega))
    (.step ((path8_adj 2 3).2 (by omega))
    (.step ((path8_adj 3 4).2 (by omega))
    (.step ((path8_adj 4 5).2 (by omega))
    (.step ((path8_adj 5 6).2 (by omega))
    (.step ((path8_adj 6 7).2 (by omega)) (.refl 7)))))))

theorem path8_walk_seven :
    ReachWithin path8State.snapshot.graph.Adj 7 (0 : Fin 8) 7 :=
  ⟨7, le_rfl, path8_walk_seven_exact⟩

private theorem path8_walk_label_bound {length : Nat} {source target : Fin 8}
    (walk : MeshWalk path8State.snapshot.graph.Adj length source target) :
    target.val ≤ source.val + length := by
  induction walk with
  | refl => omega
  | @step n source next target edge rest ih =>
      have hstep : next.val ≤ source.val + 1 := by
        rw [path8_adj] at edge
        omega
      omega

theorem path8_no_six :
    ¬ ReachWithin path8State.snapshot.graph.Adj 6 (0 : Fin 8) 7 := by
  rintro ⟨length, hlength, walk⟩
  have hlabel := path8_walk_label_bound walk
  omega

theorem path8_bounded :
    ∃ limit, GlobalHopBound path8State.snapshot.graph.Adj limit :=
  ⟨7, by simpa using state_bounded path8State⟩

theorem path8_shortest_seven :
    shortestHopCount path8State.snapshot.graph.Adj path8_bounded
      (0 : Fin 8) 7 = 7 := by
  apply Nat.le_antisymm
  · exact shortestHopCount_minimal _ path8_bounded path8_walk_seven
  · by_contra h
    have hle6 : shortestHopCount path8State.snapshot.graph.Adj path8_bounded
        (0 : Fin 8) 7 ≤ 6 := by omega
    exact path8_no_six (reachWithin_mono
      (shortestHopCount_spec _ path8_bounded (0 : Fin 8) 7) hle6)

theorem path8_diameter_seven :
    meshDiameter path8State.snapshot.graph.Adj path8_bounded = 7 := by
  apply Nat.le_antisymm
  · exact meshDiameter_minimal _ path8_bounded
      (by simpa using state_bounded path8State)
  · have hend :=
      (meshDiameter_spec path8State.snapshot.graph.Adj path8_bounded)
        (0 : Fin 8) 7
    have hshort := shortestHopCount_minimal
      path8State.snapshot.graph.Adj path8_bounded hend
    rw [path8_shortest_seven] at hshort
    exact hshort

-- Catches changed replay state updates, fitness retention, and trace weighting.
example : scopeSummary (replay pathSeed 1 pathBirths) =
    .ok (8, 7, [1, 2, 2, 2, 2, 2, 2, 1],
      [1, 1, 1, 1, 1, 1, 1, 1], 1/46080) := by
  exact path8_summary

-- The expected state must be built from six real typed applyBirth successors.
example : replay pathSeed 1 pathBirths =
    .ok ⟨⟨8, path8State⟩, 1/46080⟩ := by
  exact path8_replay

example (out : ReplayResult) (h : replay pathSeed 1 pathBirths = .ok out) :
    0 < out.probability := by
  exact path8_positive_mass out h

-- Catches a shortcut or missing path edge in the actual successor graph.
example (u v : Fin 8) : path8State.snapshot.graph.Adj u v ↔
    u.val + 1 = v.val ∨ v.val + 1 = u.val := by
  exact path8_adj u v

example : ReachWithin path8State.snapshot.graph.Adj 7 (0 : Fin 8) 7 := by
  exact path8_walk_seven

-- A supplied seven-edge walk alone cannot establish shortestness.
example : ¬ReachWithin path8State.snapshot.graph.Adj 6 (0 : Fin 8) 7 := by
  exact path8_no_six

example : shortestHopCount path8State.snapshot.graph.Adj path8_bounded
    (0 : Fin 8) 7 = 7 := by
  exact path8_shortest_seven

example : meshDiameter path8State.snapshot.graph.Adj path8_bounded = 7 := by
  exact path8_diameter_seven

#print axioms path8_summary
#print axioms path8_replay
#print axioms path8_positive_mass
#print axioms path8_adj
#print axioms path8_walk_seven
#print axioms path8_no_six
#print axioms path8_shortest_seven
#print axioms path8_diameter_seven

end NarrativeDynamics.FitnessAttachment.ScopeFixtures
