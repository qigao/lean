import NarrativeDynamics.Core.SmallWorld

open NarrativeDynamics

example {Node : Type*} (g h : MeshGraph Node)
    (included : MeshSubgraph g h) {limit : Nat} {source target : Node}
    (reachable : ReachWithin g limit source target) :
    ReachWithin h limit source target := by
  exact reachWithin_of_subgraph included reachable
