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

end NarrativeDynamics
