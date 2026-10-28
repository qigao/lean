import NarrativeDynamics.Core.MeshProjection

open NarrativeDynamics

example {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (r : Relation) {a b : Agent}
    (edge : projectedMesh source member bridge r a b) :
    source r a b := by
  exact projectedMesh_source source member bridge r edge

example {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (r : Relation) {n : Nat} {a b : Agent}
    (walk : MeshWalk (projectedMesh source member bridge r) n a b) :
    MeshWalk (source r) n a b := by
  exact projectedMesh_walk_lifts source member bridge r walk

example {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (r : Relation) (s : Society) (cover : ∀ a, member a s) (a b : Agent) :
    projectedMesh source member bridge r a b ↔ source r a b := by
  exact projectedMesh_universal_society source member bridge r s cover a b

example {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (before after : Relation → Society → Society → Agent → Agent → Prop)
    (r : Relation)
    (added : ∀ r s t a b, before r s t a b → after r s t a b)
    {n : Nat} {a b : Agent}
    (reach : ReachWithin (projectedMesh source member before r) n a b) :
    ReachWithin (projectedMesh source member after r) n a b := by
  exact projectedMesh_reach_mono_bridges source member before after r added reach

example {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (r : Relation) (s : Society) {n : Nat} {a b : Agent}
    (reach : membershipQualifiedReach source member bridge r s n a b) :
    ReachWithin (source r) n a b := by
  exact membershipQualifiedReach_source source member bridge r s reach

example {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (r : Relation) (s : Society) (a : Agent) (outside : ¬ member a s) :
    ¬ membershipQualifiedReach source member bridge r s 0 a a := by
  exact membershipQualifiedReach_outsider source member bridge r s a outside

example {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (r : Relation) {a b : Agent}
    (notInSource : ¬ source r a b) :
    ¬ projectedMesh source member bridge r a b := by
  exact projectedMesh_no_unsourced_edge source member bridge r notInSource
