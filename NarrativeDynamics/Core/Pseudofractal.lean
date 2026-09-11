import NarrativeDynamics.Core.SmallWorld

namespace NarrativeDynamics.Pseudofractal

/-- One generation retains the old vertices and creates one vertex for each
old unordered edge. The carrier freezes the parent graph before expansion. -/
abbrev ExpansionVertex {V : Type*} (G : SimpleGraph V) := V ⊕ G.edgeSet

/-- Old edges survive. Each newborn is adjacent precisely to its parent edge's
endpoints; two newborns are never adjacent in this generation. -/
def expansionAdj {V : Type*} (G : SimpleGraph V) :
    ExpansionVertex G → ExpansionVertex G → Prop
  | .inl u, .inl v => G.Adj u v
  | .inl v, .inr e => v ∈ e.val
  | .inr e, .inl v => v ∈ e.val
  | .inr _, .inr _ => False

/-- A single simultaneous edge-expansion step, on its actual enlarged carrier. -/
def expandGraph {V : Type*} (G : SimpleGraph V) :
    SimpleGraph (ExpansionVertex G) where
  Adj := expansionAdj G
  symm := by
    constructor
    intro u v adjacent
    cases u with
    | inl u =>
        cases v with
        | inl v => exact G.adj_symm adjacent
        | inr e => exact adjacent
    | inr e =>
        cases v with
        | inl v => exact adjacent
        | inr f => exact adjacent
  loopless := by
    constructor
    intro v
    cases v with
    | inl v => exact G.irrefl
    | inr e => exact fun impossible => impossible

/-- Endpoint membership is computable from vertex equality; the old adjacency
is supplied separately. No classical decision procedure is introduced. -/
instance expansionAdjDecidable {V : Type*} [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] : DecidableRel (expansionAdj G) := by
  intro u v
  cases u <;> cases v <;> dsimp [expansionAdj] <;> infer_instance

instance expandGraphDecidable {V : Type*} [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] : DecidableRel (expandGraph G).Adj :=
  expansionAdjDecidable G

/-- Old vertices retain their identities through an injective carrier map. -/
def oldVertex {V : Type*} (G : SimpleGraph V) : V ↪ ExpansionVertex G where
  toFun := Sum.inl
  inj' := by
    intro u v same
    exact Sum.inl.inj same

/-- Every old exact-length walk survives expansion with the same length. -/
theorem old_walk_lifts {V : Type*} (G : SimpleGraph V) {u v : V} {n : Nat}
    (walk : MeshWalk G.Adj n u v) :
    MeshWalk (expandGraph G).Adj n (Sum.inl u) (Sum.inl v) := by
  exact walk.mapNodes Sum.inl (fun edge => edge)

/-- Expansion preserves an old bounded path without spending any extra hops. -/
theorem old_reachWithin_lifts {V : Type*} (G : SimpleGraph V)
    {u v : V} {limit : Nat} (reachable : ReachWithin G.Adj limit u v) :
    ReachWithin (expandGraph G).Adj limit (Sum.inl u) (Sum.inl v) := by
  rcases reachable with ⟨length, bound, walk⟩
  exact ⟨length, bound, old_walk_lifts G walk⟩

end NarrativeDynamics.Pseudofractal
