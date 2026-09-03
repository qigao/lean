import Mathlib.Combinatorics.SimpleGraph.CycleGraph
import NarrativeDynamics.Core.SmallWorldMetrics

namespace NarrativeDynamics

/-- No node has an edge to itself. -/
def LooplessMesh {Node : Type*} (g : MeshGraph Node) : Prop :=
  ∀ node, ¬g node node

/-- The deterministic initial lattice for a Watts--Strogatz construction. Nodes
are arranged on a finite ring and connect within the given modular radius. -/
def wattsStrogatzRing (nodeCount radius : Nat) : MeshGraph (Fin nodeCount) :=
  fun source target => source ≠ target ∧
    ((source - target).val ≤ radius ∨ (target - source).val ≤ radius)

instance wattsStrogatzRingDecidable (nodeCount radius : Nat) :
    DecidableRel (wattsStrogatzRing nodeCount radius) := by
  intro source target
  unfold wattsStrogatzRing
  infer_instance

/-- Modular-radius ring adjacency is symmetric. -/
theorem wattsStrogatzRing_symmetric (nodeCount radius : Nat) :
    SymmetricMesh (wattsStrogatzRing nodeCount radius) := by
  intro source target edge
  exact ⟨edge.1.symm, edge.2.elim Or.inr Or.inl⟩

/-- The explicit inequality in ring adjacency excludes self-loops. -/
theorem wattsStrogatzRing_loopless (nodeCount radius : Nat) :
    LooplessMesh (wattsStrogatzRing nodeCount radius) := by
  intro node edge
  exact edge.1 rfl

/-- Non-degenerate deterministic parameters for the initial regular lattice. -/
structure WattsStrogatzParameters where
  nodeCount : Nat
  radius : Nat
  radiusPositive : 0 < radius
  localNotComplete : 2 * radius < nodeCount

/-- The regular ring selected by a valid parameter package. -/
def WattsStrogatzParameters.ringGraph (parameters : WattsStrogatzParameters) :
    MeshGraph (Fin parameters.nodeCount) :=
  wattsStrogatzRing parameters.nodeCount parameters.radius

/-- A Mathlib simple-graph walk is an exact project mesh walk over the same
adjacency relation. -/
theorem MeshWalk.ofSimpleGraphWalk {Node : Type*} {g : SimpleGraph Node}
    {source target : Node} (walk : g.Walk source target) :
    MeshWalk g.Adj walk.length source target := by
  induction walk with
  | nil => exact MeshWalk.refl _
  | cons edge rest ih => exact MeshWalk.step edge ih

/-- The finite cycle reaches every ordered node pair in at most `nodeCount - 1`
hops. -/
theorem cycleGraph_globalHopBound (nodeCount : Nat) :
    GlobalHopBound (SimpleGraph.cycleGraph nodeCount).Adj (nodeCount - 1) := by
  intro source target
  obtain ⟨walk, isPath⟩ :=
    (SimpleGraph.cycleGraph_preconnected source target).exists_isPath
  refine ⟨walk.length, ?_, MeshWalk.ofSimpleGraphWalk walk⟩
  have lengthLt : walk.length < nodeCount := by
    simpa using isPath.length_lt
  omega

/-- Every radius-positive WS ring retains all edges of the basic cycle. -/
theorem cycleGraph_subgraph_wattsStrogatzRing (nodeCount radius : Nat)
    (positive : 1 ≤ radius) :
    MeshSubgraph (SimpleGraph.cycleGraph nodeCount).Adj
      (wattsStrogatzRing nodeCount radius) := by
  intro source target adjacent
  refine ⟨adjacent.ne, ?_⟩
  rcases SimpleGraph.cycleGraph_adj'.mp adjacent with forward | backward
  · exact Or.inl (by simpa [forward] using positive)
  · exact Or.inr (by simpa [backward] using positive)

/-- The cycle's finite global bound lifts through ring-lattice edge inclusion. -/
theorem wattsStrogatzRing_globalHopBound (nodeCount radius : Nat)
    (positive : 1 ≤ radius) :
    GlobalHopBound (wattsStrogatzRing nodeCount radius) (nodeCount - 1) :=
  globalHopBound_of_subgraph
    (cycleGraph_subgraph_wattsStrogatzRing nodeCount radius positive)
    (cycleGraph_globalHopBound nodeCount)

/-- Every valid parameter package selects a globally bounded initial ring. -/
theorem WattsStrogatzParameters.globalHopBound
    (parameters : WattsStrogatzParameters) :
    GlobalHopBound parameters.ringGraph (parameters.nodeCount - 1) :=
  wattsStrogatzRing_globalHopBound parameters.nodeCount parameters.radius
    parameters.radiusPositive

end NarrativeDynamics
