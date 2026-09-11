import NarrativeDynamics.Core.FitnessValidation

open NarrativeDynamics NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

namespace NarrativeDynamics.FitnessAttachment.ValidationFixtures

private theorem triangle_connected : (⊤ : SimpleGraph (Fin 3)).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v; exact ⟨.nil⟩
    · exact ⟨.cons h .nil⟩
  nonempty := inferInstance

def triangle : State 3 where
  snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![1, 2, 4] }
  valid := ⟨by decide, triangle_connected, by decide⟩

def rawTriangle : RawSeed := ⟨3, #[1, 2, 4], #[(0, 1), (0, 2), (1, 2)]⟩
def reversedTriangle : RawSeed := ⟨3, #[1, 2, 4], #[(2, 1), (2, 0), (1, 0)]⟩

def errorOf {α : Type} : Except Error α → Option Error
  | .error e => some e
  | .ok _ => none

def seedSummary (raw : RawSeed) : Except Error (Nat × Nat × List Nat × List Rat) :=
  match parseSeed raw with
  | .error e => .error e
  | .ok s => .ok (actualNodeCount s.snapshot, actualEdgeCount s.snapshot,
      List.ofFn (degree s.snapshot), List.ofFn s.snapshot.fitness)

def stepSummary {n : Nat} (s : State n) (m : Nat) (raw : RawBirth) :
    Except Error (Nat × Nat × List Nat × List Rat × Rat) :=
  match step s m raw with
  | .error e => .error e
  | .ok (t, mass) => .ok (actualNodeCount t.snapshot, actualEdgeCount t.snapshot,
      List.ofFn (degree t.snapshot), List.ofFn t.snapshot.fitness, mass)

end NarrativeDynamics.FitnessAttachment.ValidationFixtures

open NarrativeDynamics.FitnessAttachment.ValidationFixtures

section ExactValidationFixtures
set_option maxRecDepth 4096

-- Ordered raw seed checks; no dimension default, edge deduplication, or repair.
example : errorOf (parseSeed ⟨0, #[], #[]⟩) = some .invalidNodeCount := by decide_cbv
example : errorOf (parseSeed ⟨1, #[1], #[]⟩) = some .invalidNodeCount := by decide_cbv
example : errorOf (parseSeed ⟨3, #[1, 1], #[]⟩) = some .fitnessSizeMismatch := by decide_cbv
example : errorOf (parseSeed ⟨2, #[1, 1, 1], #[(0, 1)]⟩) = some .fitnessSizeMismatch := by decide_cbv
example : errorOf (parseSeed ⟨2, #[0, 1], #[(0, 1)]⟩) = some .nonpositiveFitness := by decide_cbv
example : errorOf (parseSeed ⟨2, #[1, -1], #[(0, 1)]⟩) = some .nonpositiveFitness := by decide_cbv
example : errorOf (parseSeed ⟨2, #[1, 1], #[(0, 0)]⟩) = some .invalidEdge := by decide_cbv
example : errorOf (parseSeed ⟨2, #[1, 1], #[(0, 2)]⟩) = some .invalidEdge := by decide_cbv
example : errorOf (parseSeed ⟨2, #[1, 1], #[(0, 1), (0, 1)]⟩) = some .duplicateEdge := by decide_cbv
example : errorOf (parseSeed ⟨2, #[1, 1], #[(0, 1), (1, 0)]⟩) = some .duplicateEdge := by decide_cbv
example : errorOf (parseSeed ⟨2, #[1, 1], #[]⟩) = some .disconnectedSeed := by
  have hn : 2 ≤ (⟨2, #[1, 1], #[]⟩ : RawSeed).nodeCount := by decide
  have hs : (⟨2, #[1, 1], #[]⟩ : RawSeed).fitness.size = (⟨2, #[1, 1], #[]⟩ : RawSeed).nodeCount := rfl
  have hf : positiveSeedFitness (⟨2, #[1, 1], #[]⟩ : RawSeed) := by decide
  have he : validSeedEdges (⟨2, #[1, 1], #[]⟩ : RawSeed) := by decide
  have hd : ((⟨2, #[1, 1], #[]⟩ : RawSeed).edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ¬ ∀ b, b ∈ reached (seedSnapshot (⟨2, #[1, 1], #[]⟩ : RawSeed) hs)
      (⟨0, by have := hn; omega⟩ : Fin (⟨2, #[1, 1], #[]⟩ : RawSeed).nodeCount) ((⟨2, #[1, 1], #[]⟩ : RawSeed).nodeCount - 1) := by
    decide
  simp only [errorOf, parseSeed, dif_pos hn, dif_pos hs,
    dif_pos hf, dif_pos he, dif_pos hd, dif_neg hc] <;> rfl
example : errorOf (parseSeed ⟨3, #[1, 1, 1], #[(0, 1)]⟩) = some .disconnectedSeed := by
  have hn : 2 ≤ (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed).nodeCount := by decide
  have hs : (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed).fitness.size = (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed).nodeCount := rfl
  have hf : positiveSeedFitness (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed) := by decide
  have he : validSeedEdges (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed) := by decide
  have hd : ((⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed).edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ¬ ∀ b, b ∈ reached (seedSnapshot (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed) hs)
      (⟨0, by have := hn; omega⟩ : Fin (⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed).nodeCount) ((⟨3, #[1, 1, 1], #[(0, 1)]⟩ : RawSeed).nodeCount - 1) := by
    decide
  simp only [errorOf, parseSeed, dif_pos hn, dif_pos hs,
    dif_pos hf, dif_pos he, dif_pos hd, dif_neg hc] <;> rfl
example : errorOf (parseSeed ⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩) = some .disconnectedSeed := by
  have hn : 2 ≤ (⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed).nodeCount := by decide
  have hs : (⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed).fitness.size = (⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed).nodeCount := rfl
  have hf : positiveSeedFitness (⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed) := by decide
  have he : validSeedEdges (⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed) := by decide
  have hd : ((⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed).edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ¬ ∀ b, b ∈ reached (seedSnapshot (⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed) hs)
      (⟨0, by have := hn; omega⟩ : Fin (⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed).nodeCount) ((⟨4, #[1, 1, 1, 1], #[(0, 1), (2, 3)]⟩ : RawSeed).nodeCount - 1) := by
    decide
  simp only [errorOf, parseSeed, dif_pos hn, dif_pos hs,
    dif_pos hf, dif_pos he, dif_pos hd, dif_neg hc] <;> rfl
example : seedSummary rawTriangle = .ok (3, 3, [2, 2, 2], [1, 2, 4]) := by
  have hn : 2 ≤ rawTriangle.nodeCount := by decide
  have hs : rawTriangle.fitness.size = rawTriangle.nodeCount := rfl
  have hf : positiveSeedFitness rawTriangle := by decide
  have he : validSeedEdges rawTriangle := by decide
  have hd : (rawTriangle.edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ∀ b, b ∈ reached (seedSnapshot rawTriangle hs)
      (⟨0, by have := hn; omega⟩ : Fin rawTriangle.nodeCount) (rawTriangle.nodeCount - 1) := by
    decide
  simp only [seedSummary, parseSeed, dif_pos hn, dif_pos hs,
    dif_pos hf, dif_pos he, dif_pos hd, dif_pos hc] <;> decide_cbv
example : seedSummary reversedTriangle = .ok (3, 3, [2, 2, 2], [1, 2, 4]) := by
  have hn : 2 ≤ reversedTriangle.nodeCount := by decide
  have hs : reversedTriangle.fitness.size = reversedTriangle.nodeCount := rfl
  have hf : positiveSeedFitness reversedTriangle := by decide
  have he : validSeedEdges reversedTriangle := by decide
  have hd : (reversedTriangle.edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ∀ b, b ∈ reached (seedSnapshot reversedTriangle hs)
      (⟨0, by have := hn; omega⟩ : Fin reversedTriangle.nodeCount) (reversedTriangle.nodeCount - 1) := by
    decide
  simp only [seedSummary, parseSeed, dif_pos hn, dif_pos hs,
    dif_pos hf, dif_pos he, dif_pos hd, dif_pos hc] <;> decide_cbv
example : seedSummary ⟨2, #[1/3, 2], #[(1, 0)]⟩ =
    .ok (2, 1, [1, 1], [1/3, 2]) := by
  have hn : 2 ≤ (⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed).nodeCount := by decide
  have hs : (⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed).fitness.size = (⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed).nodeCount := rfl
  have hf : positiveSeedFitness (⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed) := by decide
  have he : validSeedEdges (⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed) := by decide
  have hd : ((⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed).edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ∀ b, b ∈ reached (seedSnapshot (⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed) hs)
      (⟨0, by have := hn; omega⟩ : Fin (⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed).nodeCount) ((⟨2, #[1/3, 2], #[(1, 0)]⟩ : RawSeed).nodeCount - 1) := by
    decide
  simp only [seedSummary, parseSeed, dif_pos hn, dif_pos hs,
    dif_pos hf, dif_pos he, dif_pos hd, dif_pos hc] <;> decide_cbv
example : seedSummary ⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ =
    .ok (4, 3, [1, 2, 2, 1], [1, 1, 1, 1]) := by
  have hn : 2 ≤ (⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed).nodeCount := by decide
  have hs : (⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed).fitness.size = (⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed).nodeCount := rfl
  have hf : positiveSeedFitness (⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed) := by decide
  have he : validSeedEdges (⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed) := by decide
  have hd : ((⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed).edges.toList.map canonicalEdge).Nodup := by decide
  have hc : ∀ b, b ∈ reached (seedSnapshot (⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed) hs)
      (⟨0, by have := hn; omega⟩ : Fin (⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed).nodeCount) ((⟨4, #[1, 1, 1, 1], #[(2, 3), (1, 2), (0, 1)]⟩ : RawSeed).nodeCount - 1) := by
    decide
  simp only [seedSummary, parseSeed, dif_pos hn, dif_pos hs,
    dif_pos hf, dif_pos he, dif_pos hd, dif_pos hc] <;> decide_cbv

-- Whole-request validation precedes construction; the last entry can invalidate all.
example : errorOf (validateBirth triangle 0 ⟨1, #[]⟩) = some .invalidM := by decide_cbv
example : errorOf (validateBirth triangle 4 ⟨1, #[0, 1, 2, 0]⟩) = some .invalidM := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨0, #[1, 2]⟩) = some .nonpositiveFitness := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨-1, #[1, 2]⟩) = some .nonpositiveFitness := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨1, #[1]⟩) = some .targetCountMismatch := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨1, #[0, 1, 2]⟩) = some .targetCountMismatch := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨1, #[1, 1]⟩) = some .duplicateTarget := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨1, #[1, 3]⟩) = some .targetOutOfRange := by decide_cbv
example : errorOf (validateBirth triangle 3 ⟨1, #[0, 1, 99]⟩) = some .targetOutOfRange := by decide_cbv
example : errorOf (validateBirth triangle 3 ⟨1, #[0, 1, 0]⟩) = some .duplicateTarget := by decide_cbv
example : errorOf (validateBirth triangle 3 ⟨1, #[2, 0, 1]⟩) = none := by decide_cbv

-- Error precedence is explicit, including a valid prefix followed by a bad entry.
example : errorOf (validateBirth triangle 0 ⟨0, #[9]⟩) = some .invalidM := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨0, #[9]⟩) = some .nonpositiveFitness := by decide_cbv
example : errorOf (validateBirth triangle 2 ⟨1, #[9]⟩) = some .targetCountMismatch := by decide_cbv
example : errorOf (validateBirth triangle 3 ⟨1, #[1, 1, 3]⟩) = some .targetOutOfRange := by decide_cbv

example : rowValues (rawAttachmentRow triangle #[]) = .ok [1/7, 2/7, 4/7] := by decide_cbv
example : rowValues (rawAttachmentRow triangle #[2]) = .ok [1/3, 2/3, 0] := by decide_cbv
example : rowValues (rawAttachmentRow triangle #[2, 0]) = .ok [0, 1, 0] := by decide_cbv
example : errorOf (rawAttachmentRow triangle #[2, 2]) = some .duplicateTarget := by decide_cbv
example : errorOf (rawAttachmentRow triangle #[2, 3]) = some .targetOutOfRange := by decide_cbv
example : errorOf (rawAttachmentRow triangle #[0, 1, 2]) = some .zeroMass := by decide_cbv
example : errorOf (rawAttachmentRow triangle #[0, 0, 1, 2]) = some .duplicateTarget := by decide_cbv
example : errorOf (rawAttachmentRow triangle #[0, 1, 2, 3]) = some .targetOutOfRange := by decide_cbv

example : stepSummary triangle 2 ⟨3/2, #[2, 1]⟩ =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/21) := by
  have hm : 0 < 2 ∧ 2 ≤ 3 := by decide
  have hf : 0 < (3/2 : Rat) := by norm_num
  have hs : (#[2, 1] : Array Nat).size = 2 := rfl
  have hb : targetsBounded 3 #[2, 1] := by decide
  have hd : targetsDistinct #[2, 1] := by decide
  simp only [stepSummary, step, validateBirth, dif_pos hm, dif_pos hf,
    dif_pos hs, checkTargets, dif_pos hb, dif_pos hd]
  decide_cbv
example : stepSummary triangle 2 ⟨3/2, #[1, 2]⟩ =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/35) := by
  have hm : 0 < 2 ∧ 2 ≤ 3 := by decide
  have hf : 0 < (3/2 : Rat) := by norm_num
  have hs : (#[1, 2] : Array Nat).size = 2 := rfl
  have hb : targetsBounded 3 #[1, 2] := by decide
  have hd : targetsDistinct #[1, 2] := by decide
  simp only [stepSummary, step, validateBirth, dif_pos hm, dif_pos hf,
    dif_pos hs, checkTargets, dif_pos hb, dif_pos hd]
  decide_cbv
example : stepSummary triangle 3 ⟨1, #[0, 1, 2]⟩ =
    .ok (4, 6, [3, 3, 3, 3], [1, 2, 4, 1], 1/21) := by
  have hm : 0 < 3 ∧ 3 ≤ 3 := by decide
  have hf : 0 < (1 : Rat) := by norm_num
  have hs : (#[0, 1, 2] : Array Nat).size = 3 := rfl
  have hb : targetsBounded 3 #[0, 1, 2] := by decide
  have hd : targetsDistinct #[0, 1, 2] := by decide
  simp only [stepSummary, step, validateBirth, dif_pos hm, dif_pos hf,
    dif_pos hs, checkTargets, dif_pos hb, dif_pos hd]
  decide_cbv
example : stepSummary triangle 1 ⟨1, #[0]⟩ =
    .ok (4, 4, [3, 2, 2, 1], [1, 2, 4, 1], 1/7) := by
  have hm : 0 < 1 ∧ 1 ≤ 3 := by decide
  have hf : 0 < (1 : Rat) := by norm_num
  have hs : (#[0] : Array Nat).size = 1 := rfl
  have hb : targetsBounded 3 #[0] := by decide
  have hd : targetsDistinct #[0] := by decide
  simp only [stepSummary, step, validateBirth, dif_pos hm, dif_pos hf,
    dif_pos hs, checkTargets, dif_pos hb, dif_pos hd]
  decide_cbv
example : stepSummary triangle 3 ⟨1, #[0, 1, 0]⟩ = .error .duplicateTarget := by decide_cbv
example : stepSummary triangle 3 ⟨1, #[0, 1, 3]⟩ = .error .targetOutOfRange := by decide_cbv
example : stepSummary triangle 0 ⟨1, #[]⟩ = .error .invalidM := by decide_cbv
example : List.ofFn triangle.snapshot.fitness = [1, 2, 4] := by decide_cbv

end ExactValidationFixtures

-- Generic acceptance and output contracts; no local computation budget applies.
example {n : Nat} (s : Snapshot n) (a b : Fin n) (k : Nat) :
    b ∈ reached s a k ↔ ReachWithin s.graph.Adj k a b := reached_iff s a b k
example {n : Nat} (s : State n) :
    GlobalHopBound s.snapshot.graph.Adj (n - 1) := state_bounded s
example (raw : RawSeed) (s : State raw.nodeCount) (h : parseSeed raw = .ok s) :
    raw.Valid := parseSeed_sound raw s h
example (raw : RawSeed) (h : raw.Valid) :
    ∃ s, parseSeed raw = .ok s := parseSeed_complete raw h
example (raw : RawSeed) (s : State raw.nodeCount) (h : parseSeed raw = .ok s) :
    s.snapshot.graph = seedGraph raw := parseSeed_graph raw s h
example (raw : RawSeed) (s : State raw.nodeCount) (h : parseSeed raw = .ok s) :
    ∃ hs : raw.fitness.size = raw.nodeCount, s.snapshot = seedSnapshot raw hs :=
  parseSeed_snapshot raw s h
example {n m : Nat} (s : State n) (raw : RawBirth) (v : ValidatedBirth n m)
    (h : validateBirth s m raw = .ok v) : raw.Valid n m := validateBirth_sound s raw v h
example {n m : Nat} (s : State n) (raw : RawBirth) (h : raw.Valid n m) :
    ∃ v, validateBirth s m raw = .ok v := validateBirth_complete s raw h
example {n m : Nat} (s : State n) (raw : RawBirth) (v : ValidatedBirth n m)
    (h : validateBirth s m raw = .ok v) :
    v.fitness.val = raw.fitness ∧
      ∃ hs : raw.targets.size = m, ∀ i : Fin raw.targets.size,
        (v.targets (Fin.cast hs i)).val = raw.targets[i.val] := validateBirth_values s raw v h
example {n m : Nat} (s : State n) (raw : RawBirth) (out : State (n + 1) × Rat) :
    step s m raw = .ok out ↔
      ∃ v : ValidatedBirth n m, validateBirth s m raw = .ok v ∧
        out = (applyBirth s v.targets v.positive v.fitness, orderedMass s v.targets) :=
  step_spec s raw out
example {n m : Nat} (s : State n) (raw : RawBirth) (e : Error)
    (h : validateBirth s m raw = .error e) : step s m raw = .error e := step_error s raw e h

theorem raw_seed_accepts_iff (raw : RawSeed) :
    (∃ s, parseSeed raw = .ok s) ↔ raw.Valid := by
  constructor
  · rintro ⟨s, hs⟩
    exact parseSeed_sound raw s hs
  · exact parseSeed_complete raw

theorem raw_birth_accepts_iff {n m : Nat} (s : State n) (raw : RawBirth) :
    (∃ v, validateBirth s m raw = .ok v) ↔ raw.Valid n m := by
  constructor
  · rintro ⟨v, hv⟩
    exact validateBirth_sound s raw v hv
  · exact validateBirth_complete s raw

theorem checked_step_has_positive_mass {n m : Nat} (s : State n) (raw : RawBirth)
    (out : State (n + 1) × Rat) (h : step s m raw = .ok out) : 0 < out.2 := by
  obtain ⟨v, _, rfl⟩ := (step_spec s raw out).mp h
  exact orderedMass_pos s v.targets

#print axioms reached_iff
#print axioms connected_bounded
#print axioms state_bounded
#print axioms parseSeed_sound
#print axioms parseSeed_complete
#print axioms parseSeed_graph
#print axioms parseSeed_snapshot
#print axioms validateBirth_sound
#print axioms validateBirth_complete
#print axioms validateBirth_values
#print axioms step_spec
#print axioms step_error
#print axioms raw_seed_accepts_iff
#print axioms raw_birth_accepts_iff
#print axioms checked_step_has_positive_mass

section RawReductionWitnesses
set_option maxRecDepth 4096
private def isolatedPair : RawSeed := ⟨2, #[1, 1], #[]⟩
example : (letI := seedAdjDec isolatedPair;
    decide ((seedGraph isolatedPair).Adj (0 : Fin 2) (1 : Fin 2))) = false := by decide_cbv
example : reached (seedSnapshot isolatedPair rfl) (0 : Fin 2) 0 = ({0} : Finset (Fin 2)) := by decide_cbv
example : reached (seedSnapshot isolatedPair rfl) (0 : Fin 2) 1 = ({0} : Finset (Fin 2)) := by decide_cbv
example : ¬ (∀ b : Fin 2, b ∈ reached (seedSnapshot isolatedPair rfl) (0 : Fin 2) 1) := by
  decide_cbv
example : errorOf (parseSeed isolatedPair) = some .disconnectedSeed := by rfl
example : (letI := seedAdjDec rawTriangle;
    decide ((seedGraph rawTriangle).Adj (0 : Fin 3) (1 : Fin 3))) = true := by decide_cbv
example : reached (seedSnapshot rawTriangle rfl) (0 : Fin 3) 1 = (Finset.univ : Finset (Fin 3)) := by decide_cbv
end RawReductionWitnesses
