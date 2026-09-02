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

end NarrativeDynamics
