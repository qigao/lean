import NarrativeDynamics.Core.SmallWorld

namespace NarrativeDynamics

/-!
V24 is a selected-relation, source-backed topology projection.
It does not assert that a Python snapshot or a transmission history was
verified by Lean. The caller supplies the chosen source relation, membership
and explicit directed bridge declarations as mathematical predicates.
-/

/-- A society edge is internal to a shared membership or declared between
two distinct societies with named endpoints in their corresponding societies.
The source edge is always required independently of these conditions. -/
def projectedMesh {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) : MeshGraph Agent :=
  fun a b =>
    source relation a b ∧
      ((∃ s, member a s ∧ member b s) ∨
       (∃ s t, s ≠ t ∧ member a s ∧ member b t ∧
         bridge relation s t a b))

/-- No society declaration creates an edge absent from the selected source. -/
theorem projectedMesh_source {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) {a b : Agent}
    (edge : projectedMesh source member bridge relation a b) :
    source relation a b :=
  edge.1

/-- Projected walks lift to the actual selected source graph. -/
theorem projectedMesh_walk_lifts {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) {length : Nat} {a b : Agent}
    (walk : MeshWalk (projectedMesh source member bridge relation) length a b) :
    MeshWalk (source relation) length a b := by
  exact walk.mapNodes id (fun edge =>
    projectedMesh_source source member bridge relation edge)

/-- A projected bounded path always exists in the source graph as a walk. -/
theorem projectedMesh_reach_lifts {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) {budget : Nat} {a b : Agent}
    (reach : ReachWithin (projectedMesh source member bridge relation) budget a b) :
    ReachWithin (source relation) budget a b := by
  rcases reach with ⟨length, bound, walk⟩
  exact ⟨length, bound,
    projectedMesh_walk_lifts source member bridge relation walk⟩

/-- One universal society returns exactly the chosen source relation. -/
theorem projectedMesh_universal_society {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) (all : Society)
    (cover : ∀ a, member a all) (a b : Agent) :
    projectedMesh source member bridge relation a b ↔ source relation a b := by
  constructor
  · exact projectedMesh_source source member bridge relation
  · intro edge
    exact ⟨edge, Or.inl ⟨all, cover a, cover b⟩⟩

/-- Adding source-backed bridge declarations cannot remove an old edge. -/
theorem projectedMesh_bridge_mono {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (before after : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation)
    (added : ∀ r s t a b, before r s t a b → after r s t a b) :
    MeshSubgraph (projectedMesh source member before relation)
      (projectedMesh source member after relation) := by
  intro a b edge
  refine ⟨edge.1, ?_⟩
  rcases edge.2 with internal | declared
  · exact Or.inl internal
  · rcases declared with ⟨s, t, distinct, src, dst, declaredEdge⟩
    exact Or.inr ⟨s, t, distinct, src, dst,
      added relation s t a b declaredEdge⟩

/-- Finite reachability witnesses are monotone under bridge addition. -/
theorem projectedMesh_reach_mono_bridges {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (before after : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation)
    (added : ∀ r s t a b, before r s t a b → after r s t a b)
    {budget : Nat} {a b : Agent}
    (reach : ReachWithin (projectedMesh source member before relation) budget a b) :
    ReachWithin (projectedMesh source member after relation) budget a b := by
  exact reachWithin_of_subgraph
    (projectedMesh_bridge_mono source member before after relation added) reach

/-- Only the selected source and bridge relation are read by one projection. -/
theorem projectedMesh_selected_only {Agent Society Relation : Type*}
    (first second : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (before after : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation)
    (sameSource : first relation = second relation)
    (sameBridges : before relation = after relation)
    (a b : Agent) :
    projectedMesh first member before relation a b ↔
    projectedMesh second member after relation a b := by
  simp only [projectedMesh, sameSource, sameBridges]

/-- Local reachability includes the endpoints' membership, including at
zero hops. Edge restriction by itself does not enforce this for refl. -/
def membershipQualifiedReach {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) (society : Society) (budget : Nat)
    (a b : Agent) : Prop :=
  member a society ∧ member b society ∧
    ReachWithin
      (restrictMesh (projectedMesh source member bridge relation)
        (fun node => member node society)) budget a b

/-- A local member-qualified witness is also a global source witness. -/
theorem membershipQualifiedReach_source {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) (society : Society)
    {budget : Nat} {a b : Agent}
    (reach : membershipQualifiedReach source member bridge relation society budget a b) :
    ReachWithin (source relation) budget a b := by
  rcases reach with ⟨_, _, length, bound, walk⟩
  exact projectedMesh_reach_lifts source member bridge relation
    ⟨length, bound, restricted_walk_lifts walk⟩

/-- In particular, an outsider cannot claim a local zero-hop reflexive path. -/
theorem membershipQualifiedReach_outsider {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) (society : Society) (a : Agent)
    (outsider : ¬ member a society) :
    ¬ membershipQualifiedReach source member bridge relation society 0 a a := by
  intro reach
  exact outsider reach.1

/-- A bridge declaration never overrides an absent source edge. -/
theorem projectedMesh_no_unsourced_edge {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) {a b : Agent} (missing : ¬ source relation a b) :
    ¬ projectedMesh source member bridge relation a b := by
  intro edge
  exact missing (projectedMesh_source source member bridge relation edge)


/-- A society-to-society hop is admitted only if its designated gateways have
an actual projected Agent edge. Self reachability uses MeshWalk.refl rather
than manufacturing a reflexive Agent edge. -/
def projectedGatewaySocietyGraph {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation) (gateway : Society → Agent) : MeshGraph Society :=
  fun s t => s ≠ t ∧
    projectedMesh source member bridge relation (gateway s) (gateway t)

/-- A concrete selected-relation society overlay satisfies the existing
1 + 4 + 1 theorem when its real gateway model, assignment cover and
four-hop society graph are explicitly witnessed. Nothing about an arbitrary
runtime snapshot establishes these premises automatically. -/
theorem projectedMesh_global_six_hop_bound {Agent Society Relation : Type*}
    (source : Relation → MeshGraph Agent)
    (member : Agent → Society → Prop)
    (bridge : Relation → Society → Society → Agent → Agent → Prop)
    (relation : Relation)
    (gateway : Society → Agent)
    (gatewayMember : ∀ s, member (gateway s) s)
    (societyOf : Agent → Society)
    (assignedMember : ∀ a, member a (societyOf a))
    (toGateway : ∀ a,
      ReachWithin (projectedMesh source member bridge relation) 1
        a (gateway (societyOf a)))
    (fromGateway : ∀ a,
      ReachWithin (projectedMesh source member bridge relation) 1
        (gateway (societyOf a)) a)
    (societyBound : GlobalHopBound
      (projectedGatewaySocietyGraph source member bridge relation gateway) 4) :
    GlobalHopBound (projectedMesh source member bridge relation) 6 := by
  let model : SocietyGatewayModel
      (projectedMesh source member bridge relation)
      (projectedGatewaySocietyGraph source member bridge relation gateway) :=
    { member := member
      gateway := gateway
      gatewayMember := gatewayMember
      bridge := by
        intro s t edge
        exact edge.2 }
  let cover : OneHopGatewayCover model :=
    { societyOf := societyOf
      assignedMember := assignedMember
      toGateway := toGateway
      fromGateway := fromGateway }
  exact global_six_hop_bound model cover societyBound

end NarrativeDynamics
