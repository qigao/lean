import NarrativeDynamics.Core.SocialMesh

namespace NarrativeDynamics

/-- Every edge of the first mesh is retained by the second mesh. -/
def MeshSubgraph {Node : Type*} (g h : MeshGraph Node) : Prop :=
  ∀ {source target}, g source target → h source target

/-- An edge-preserving node map sends an exact walk to an exact walk without
changing its length. -/
theorem MeshWalk.mapNodes {Source Target : Type*}
    {sourceGraph : MeshGraph Source} {targetGraph : MeshGraph Target}
    (mapNode : Source → Target)
    (mapEdge : ∀ {source target}, sourceGraph source target →
      targetGraph (mapNode source) (mapNode target))
    {length : Nat} {source target : Source}
    (walk : MeshWalk sourceGraph length source target) :
    MeshWalk targetGraph length (mapNode source) (mapNode target) := by
  induction walk with
  | refl => exact MeshWalk.refl _
  | step edge rest ih => exact MeshWalk.step (mapEdge edge) ih

/-- Bounded reachability composes and its inclusive hop budgets add. -/
theorem reachWithin_append {Node : Type*} {g : MeshGraph Node}
    {leftLimit rightLimit : Nat} {source middle target : Node}
    (left : ReachWithin g leftLimit source middle)
    (right : ReachWithin g rightLimit middle target) :
    ReachWithin g (leftLimit + rightLimit) source target := by
  rcases left with ⟨leftLength, leftBound, leftWalk⟩
  rcases right with ⟨rightLength, rightBound, rightWalk⟩
  exact ⟨leftLength + rightLength, Nat.add_le_add leftBound rightBound,
    leftWalk.append rightWalk⟩

/-- Adding edges cannot destroy an existing bounded path. -/
theorem reachWithin_of_subgraph {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h) {limit : Nat} {source target : Node}
    (reachable : ReachWithin g limit source target) :
    ReachWithin h limit source target := by
  rcases reachable with ⟨length, bound, walk⟩
  exact ⟨length, bound, walk.mapNodes id (fun edge => included edge)⟩

/-- Every ordered node pair has a path within one common inclusive hop budget. -/
def GlobalHopBound {Node : Type*} (g : MeshGraph Node) (limit : Nat) : Prop :=
  ∀ source target, ReachWithin g limit source target

/-- A global hop bound survives edge addition. -/
theorem globalHopBound_of_subgraph {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h) {limit : Nat}
    (bounded : GlobalHopBound g limit) : GlobalHopBound h limit := by
  intro source target
  exact reachWithin_of_subgraph included (bounded source target)

/-- A quotient-level society edge is realized by an Agent-level edge between
the societies' designated gateways. Membership remains explicit evidence. -/
structure SocietyGatewayModel {Agent Society : Type*}
    (agentGraph : MeshGraph Agent) (societyGraph : MeshGraph Society) where
  member : Agent → Society → Prop
  gateway : Society → Agent
  gatewayMember : ∀ society, member (gateway society) society
  bridge : ∀ {source target}, societyGraph source target →
    agentGraph (gateway source) (gateway target)

/-- Every exact society walk has an exact Agent-level realization between the
corresponding gateways. -/
theorem SocietyGatewayModel.walk_lifts {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {length : Nat} {source target : Society}
    (walk : MeshWalk societyGraph length source target) :
    MeshWalk agentGraph length (model.gateway source) (model.gateway target) := by
  exact walk.mapNodes model.gateway (fun edge => model.bridge edge)

/-- Society-level bounded reachability lifts without increasing the gateway hop
budget. -/
theorem SocietyGatewayModel.reach_lifts {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {limit : Nat} {source target : Society}
    (reachable : ReachWithin societyGraph limit source target) :
    ReachWithin agentGraph limit (model.gateway source) (model.gateway target) := by
  rcases reachable with ⟨length, bound, walk⟩
  exact ⟨length, bound, model.walk_lifts walk⟩

/-- Entering a source gateway, traversing the society mesh, and leaving a target
gateway compose into one Agent-level route with an additive hop budget. -/
theorem reachWithin_via_societies {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {entryLimit societyLimit exitLimit : Nat}
    {source target : Agent} {sourceSociety targetSociety : Society}
    (entry : ReachWithin agentGraph entryLimit source
      (model.gateway sourceSociety))
    (between : ReachWithin societyGraph societyLimit sourceSociety targetSociety)
    (exit : ReachWithin agentGraph exitLimit
      (model.gateway targetSociety) target) :
    ReachWithin agentGraph (entryLimit + societyLimit + exitLimit) source target := by
  exact reachWithin_append (reachWithin_append entry (model.reach_lifts between)) exit

end NarrativeDynamics
