import NarrativeDynamics.Core.SmallWorldMetrics

open NarrativeDynamics

example {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node)
    (clustered : PerfectLocalClustering g)
    (pairsExist : (orderedNeighborPairs g center).Nonempty) :
    localClusteringCoefficient g center = 1 := by
  exact localClusteringCoefficient_eq_one_of_perfect g center clustered pairsExist
