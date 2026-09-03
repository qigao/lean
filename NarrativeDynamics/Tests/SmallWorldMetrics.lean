import NarrativeDynamics.Core.SmallWorldMetrics

open NarrativeDynamics

example {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node)
    (clustered : PerfectLocalClustering g)
    (pairsExist : (orderedNeighborPairs g center).Nonempty) :
    localClusteringCoefficient g center = 1 := by
  exact localClusteringCoefficient_eq_one_of_perfect g center clustered pairsExist

example {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) (source target : Node) :
    ReachWithin g (shortestHopCount g bounded source target) source target := by
  exact shortestHopCount_spec g bounded source target

example {Node : Type*} {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit) :
    meshDiameter g ⟨limit, certificate.shortPaths⟩ ≤ limit := by
  exact certificate.meshDiameter_le
