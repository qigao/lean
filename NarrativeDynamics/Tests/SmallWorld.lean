import NarrativeDynamics.Core.SmallWorld

open NarrativeDynamics

example {Node : Type*} (g h : MeshGraph Node)
    (included : MeshSubgraph g h) {limit : Nat} {source target : Node}
    (reachable : ReachWithin g limit source target) :
    ReachWithin h limit source target := by
  exact reachWithin_of_subgraph included reachable

example {Agent Society : Type*} {agentGraph : MeshGraph Agent}
    {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {limit : Nat} {source target : Society}
    (reachable : ReachWithin societyGraph limit source target) :
    ReachWithin agentGraph limit (model.gateway source) (model.gateway target) := by
  exact model.reach_lifts reachable

example {Agent Society : Type*} {agentGraph : MeshGraph Agent}
    {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    (cover : OneHopGatewayCover model)
    (societyBound : GlobalHopBound societyGraph 4) :
    GlobalHopBound agentGraph 6 := by
  exact global_six_hop_bound model cover societyBound

example {Agent Society : Type*} {agentGraph : MeshGraph Agent}
    {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    (cover : OneHopGatewayCover model)
    (societyBound : GlobalHopBound societyGraph 4)
    (symmetric : SymmetricMesh agentGraph)
    (clustered : PerfectLocalClustering agentGraph) :
    SmallWorldCertificate agentGraph 6 := by
  exact smallWorld_of_gateway_cover model cover societyBound symmetric clustered

example {Node : Type*} {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit)
    {center left right : Node} (different : left ≠ right)
    (leftNeighbor : g center left) (rightNeighbor : g center right) :
    g left right := by
  exact certificate.closes_neighbor_wedge different leftNeighbor rightNeighbor

example {Node : Type*} {g h : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit)
    (included : MeshSubgraph g h) : GlobalHopBound h limit := by
  exact certificate.short_paths_survive_edge_addition included
