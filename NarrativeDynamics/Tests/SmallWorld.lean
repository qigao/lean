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
