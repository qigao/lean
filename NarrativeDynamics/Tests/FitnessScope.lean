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
  exact path8_probability_pos out h

-- Catches a shortcut or missing path edge in the actual successor graph.
example (u v : Fin 8) : path8State.snapshot.graph.Adj u v ↔
    u.val + 1 = v.val ∨ v.val + 1 = u.val := by
  exact path8_adj u v

example : ReachWithin path8State.snapshot.graph.Adj 7 (0 : Fin 8) 7 := by
  exact path8_walk

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
#print axioms path8_probability_pos
#print axioms path8_adj
#print axioms path8_walk
#print axioms path8_no_six
#print axioms path8_shortest_seven
#print axioms path8_diameter_seven

end NarrativeDynamics.FitnessAttachment.ScopeFixtures
