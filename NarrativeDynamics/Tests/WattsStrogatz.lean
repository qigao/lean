import NarrativeDynamics.Core.WattsStrogatz

open NarrativeDynamics

example (nodeCount radius : Nat) :
    SymmetricMesh (wattsStrogatzRing nodeCount radius) := by
  exact wattsStrogatzRing_symmetric nodeCount radius

example (nodeCount radius : Nat) :
    LooplessMesh (wattsStrogatzRing nodeCount radius) := by
  exact wattsStrogatzRing_loopless nodeCount radius

example : localClusteringCoefficient
    (wattsStrogatzRing 6 2) (0 : Fin 6) = (2 : ℚ) / 3 := by
  native_decide

example (nodeCount radius : Nat) (positive : 1 ≤ radius) :
    GlobalHopBound (wattsStrogatzRing nodeCount radius) (nodeCount - 1) := by
  exact wattsStrogatzRing_globalHopBound nodeCount radius positive

example (parameters : WattsStrogatzParameters) :
    GlobalHopBound parameters.ringGraph (parameters.nodeCount - 1) := by
  exact parameters.globalHopBound

example {Node : Type*} (g : MeshGraph Node) (left right : Node) :
    MeshSubgraph g (addUndirectedShortcut g left right) := by
  exact base_subgraph_addUndirectedShortcut g left right

example {Node : Type*} {g : MeshGraph Node} (symmetric : SymmetricMesh g)
    (left right : Node) :
    SymmetricMesh (addUndirectedShortcut g left right) := by
  exact addUndirectedShortcut_symmetric symmetric left right

example {Node : Type*} {g : MeshGraph Node} (loopless : LooplessMesh g)
    {left right : Node} (distinct : left ≠ right) :
    LooplessMesh (addUndirectedShortcut g left right) := by
  exact addUndirectedShortcut_loopless loopless distinct

example {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) (source target : Node) :
    shortestHopCount h hBounded source target ≤
      shortestHopCount g gBounded source target := by
  exact shortestHopCount_mono_edges included gBounded hBounded source target

example {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) :
    meshDiameter h hBounded ≤ meshDiameter g gBounded := by
  exact meshDiameter_mono_edges included gBounded hBounded

example {Node : Type*} [Fintype Node] [DecidableEq Node]
    {g h : MeshGraph Node} (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) :
    averageShortestPathLength h hBounded ≤
      averageShortestPathLength g gBounded := by
  exact averageShortestPathLength_mono_edges included gBounded hBounded
