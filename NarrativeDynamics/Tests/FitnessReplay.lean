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

end NarrativeDynamics.FitnessAttachment.ReplayFixtures

open NarrativeDynamics.FitnessAttachment.ReplayFixtures

section ExactReplayFixtures
set_option maxRecDepth 4096

-- Actual raw API output, including updated adjacency and the conditional product.
example : summaryOf (replay rawTriangle 2 []) =
    .ok (3, 3, [2, 2, 2], [1, 2, 4], 1) := by decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩]) =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/21) := by decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[1, 2]⟩]) =
    .ok (4, 5, [2, 3, 3, 2], [1, 2, 4, 3/2], 8/35) := by decide_cbv
example : summaryOf (replay rawTriangle 2 twoBirths) =
    .ok (5, 7, [2, 3, 4, 3, 2], [1, 2, 4, 3/2, 1/3], 24/805) := by decide_cbv

-- All fitness values, not only the seed, must receive the same scale.
example : summaryOf (replay ⟨3, #[2, 4, 8], rawTriangle.edges⟩ 2
    [⟨3, #[2, 1]⟩, ⟨2/3, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [2, 4, 8, 3, 2/3], 24/805) := by decide_cbv
example : summaryOf (replay ⟨3, #[2, 4, 8], rawTriangle.edges⟩ 2 twoBirths) =
    .ok (5, 7, [2, 3, 4, 3, 2], [2, 4, 8, 3/2, 1/3], 24/1505) := by decide_cbv

-- Constant fitness 3 and the unit-fitness BA specialization have equal laws.
example : summaryOf (replay ⟨3, #[3, 3, 3], rawTriangle.edges⟩ 2
    [⟨3, #[2, 1]⟩, ⟨3, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [3, 3, 3, 3, 3], 1/80) := by decide_cbv
example : summaryOf (replay unitTriangle 2 [⟨1, #[2, 1]⟩, ⟨1, #[3, 2]⟩]) =
    .ok (5, 7, [2, 3, 4, 3, 2], [1, 1, 1, 1, 1], 1/80) := by decide_cbv

-- Arbitrary valid seed and m equal to the INITIAL size across several births.
example : summaryOf (replay ⟨2, #[1, 1], #[(1, 0)]⟩ 1
    [⟨1, #[1]⟩, ⟨1, #[2]⟩]) =
    .ok (4, 3, [1, 2, 2, 1], [1, 1, 1, 1], 1/8) := by decide_cbv
example : summaryOf (replay rawTriangle 3
    [⟨1, #[0, 1, 2]⟩, ⟨1, #[1, 2, 3]⟩]) =
    .ok (5, 9, [3, 4, 4, 4, 3], [1, 2, 4, 1, 1], 1/252) := by decide_cbv

-- Empty input is not a validation bypass; seed errors precede invalid initial m.
example : summaryOf (replay rawTriangle 0 []) = .error .initialM := by decide_cbv
example : summaryOf (replay rawTriangle 4 []) = .error .initialM := by decide_cbv
example : summaryOf (replay ⟨0, #[], #[]⟩ 0 []) =
    .error (.seed .invalidNodeCount) := by decide_cbv
example : summaryOf (replay ⟨2, #[1], #[(0, 1)]⟩ 1 []) =
    .error (.seed .fitnessSizeMismatch) := by decide_cbv
example : summaryOf (replay ⟨2, #[0, 1], #[(0, 1)]⟩ 1 []) =
    .error (.seed .nonpositiveFitness) := by decide_cbv

-- First failure has a zero-based index and carries no partial result.
example : summaryOf (replay rawTriangle 2 [⟨1, #[2, 2]⟩]) =
    .error (.atBirth 0 .duplicateTarget) := by decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨1, #[3, 3]⟩]) =
    .error (.atBirth 1 .duplicateTarget) := by decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨1, #[3, 4]⟩]) =
    .error (.atBirth 1 .targetOutOfRange) := by decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨0, #[3, 2]⟩]) =
    .error (.atBirth 1 .nonpositiveFitness) := by decide_cbv
example : summaryOf (replay rawTriangle 2 [⟨3/2, #[2, 1]⟩, ⟨1, #[3]⟩]) =
    .error (.atBirth 1 .targetCountMismatch) := by decide_cbv
example : summaryOf (replay rawTriangle 2
    [⟨1, #[0, 0]⟩, ⟨0, #[99]⟩]) = .error (.atBirth 0 .duplicateTarget) := by decide_cbv

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
