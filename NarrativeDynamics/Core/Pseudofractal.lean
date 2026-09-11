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

namespace Internal

open scoped BigOperators

/-- An old edge together with one of its endpoints. These are precisely the
new mixed edges, not a second orientation-dependent birth mechanism. -/
abbrev EdgeIncidence {V : Type*} (G : SimpleGraph V) :=
  Σ e : G.edgeSet, {v : V // v ∈ e.val}

/-- Encode the disjoint old-edge/incidence partition as actual expanded edges. -/
def expandEdgeMap {V : Type*} (G : SimpleGraph V) :
    G.edgeSet ⊕ EdgeIncidence G → (expandGraph G).edgeSet
  | .inl e => ⟨Sym2.map Sum.inl e.val, by
      rcases e with ⟨uv, he⟩
      induction uv using Sym2.ind with
      | _ u v => exact he⟩
  | .inr i => ⟨s(Sum.inl i.2.val, Sum.inr i.1), i.2.property⟩

theorem expandEdgeMap_injective {V : Type*} (G : SimpleGraph V) :
    Function.Injective (expandEdgeMap G) := by
  intro a b same
  have raw := congrArg Subtype.val same
  cases a with
  | inl e =>
      cases b with
      | inl f =>
          change Sym2.map (Sum.inl : V → ExpansionVertex G) e.val =
            Sym2.map Sum.inl f.val at raw
          have included : Function.Injective (Sum.inl : V → ExpansionVertex G) :=
            fun _ _ h => Sum.inl.inj h
          exact congrArg Sum.inl (Subtype.ext (Sym2.map.injective included raw))
      | inr i =>
          change Sym2.map (Sum.inl : V → ExpansionVertex G) e.val =
            s(Sum.inl i.2.val, Sum.inr i.1) at raw
          have member : Sum.inr i.1 ∈ Sym2.map (Sum.inl : V → ExpansionVertex G) e.val := by
            rw [raw]
            exact Sym2.mem_mk_right _ _
          rcases Sym2.mem_map.mp member with ⟨v, _, impossible⟩
          cases impossible
  | inr i =>
      cases b with
      | inl e =>
          change s(Sum.inl i.2.val, Sum.inr i.1) =
            Sym2.map (Sum.inl : V → ExpansionVertex G) e.val at raw
          have member : Sum.inr i.1 ∈ Sym2.map (Sum.inl : V → ExpansionVertex G) e.val := by
            rw [← raw]
            exact Sym2.mem_mk_right _ _
          rcases Sym2.mem_map.mp member with ⟨v, _, impossible⟩
          cases impossible
      | inr j =>
          rcases i with ⟨e, v, hv⟩
          rcases j with ⟨f, w, hw⟩
          change s((Sum.inl v : ExpansionVertex G), Sum.inr e) =
            s(Sum.inl w, Sum.inr f) at raw
          have parts : v = w ∧ e = f := by simpa [Sym2.eq_iff] using raw
          rcases parts with ⟨rfl, rfl⟩
          rfl

theorem expandEdgeMap_surjective {V : Type*} (G : SimpleGraph V) :
    Function.Surjective (expandEdgeMap G) := by
  rintro ⟨uv, adjacent⟩
  induction uv using Sym2.ind with
  | _ x y =>
      cases x with
      | inl u =>
          cases y with
          | inl v => exact ⟨Sum.inl ⟨s(u, v), adjacent⟩, rfl⟩
          | inr e => exact ⟨Sum.inr ⟨e, ⟨u, adjacent⟩⟩, rfl⟩
      | inr e =>
          cases y with
          | inl v =>
              refine ⟨Sum.inr ⟨e, ⟨v, adjacent⟩⟩, ?_⟩
              apply Subtype.ext
              exact Sym2.eq_swap
          | inr f => exact False.elim adjacent

/-- This inverse is for mathematical counting only. The finite family below
never uses it for enumeration, equality, or adjacency decisions. -/
noncomputable def expandEdgesEquiv {V : Type*} (G : SimpleGraph V) :
    (expandGraph G).edgeSet ≃ G.edgeSet ⊕ EdgeIncidence G :=
  (Equiv.ofBijective (expandEdgeMap G)
    ⟨expandEdgeMap_injective G, expandEdgeMap_surjective G⟩).symm

/-- A genuine simple-graph edge has two distinct endpoints. -/
theorem endpoint_card {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) (e : G.edgeSet) :
    Fintype.card {v : V // v ∈ e.val} = 2 := by
  rcases e with ⟨uv, he⟩
  induction uv using Sym2.ind with
  | _ u v =>
      rw [Fintype.card_of_subtype (p := fun x : V => x ∈ s(u, v))
        ({u, v} : Finset V) (by intro x; simp)]
      simp [G.ne_of_adj he]

/-- The actual expanded edge set consists of one old edge and two incidences
per old edge. No recurrence counter participates in this identity. -/
theorem expand_edgeCount {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] :
    Fintype.card (expandGraph G).edgeSet = 3 * Fintype.card G.edgeSet := by
  calc
    Fintype.card (expandGraph G).edgeSet = Fintype.card G.edgeSet +
        ∑ e : G.edgeSet, Fintype.card {v : V // v ∈ e.val} := by
      simpa only [Fintype.card_sum, Fintype.card_sigma] using
        Fintype.card_congr (expandEdgesEquiv G)
    _ = Fintype.card G.edgeSet + Fintype.card G.edgeSet * 2 := by
      simp [endpoint_card]
    _ = 3 * Fintype.card G.edgeSet := by omega

/-- Every expanded vertex has an old vertex at distance at most one in both
directions. This is existential path evidence, not an executable choice. -/
theorem endpoint_cover {V : Type*} (G : SimpleGraph V) (x : ExpansionVertex G) :
    ∃ v : V, ReachWithin (expandGraph G).Adj 1 x (Sum.inl v) ∧
      ReachWithin (expandGraph G).Adj 1 (Sum.inl v) x := by
  cases x with
  | inl v =>
      exact ⟨v, ⟨0, by omega, MeshWalk.refl _⟩,
        ⟨0, by omega, MeshWalk.refl _⟩⟩
  | inr e =>
      have endpoint : ∃ v : V, v ∈ e.val := by
        refine Sym2.inductionOn e.val ?_
        intro u v
        exact ⟨u, Sym2.mem_mk_left u v⟩
      rcases endpoint with ⟨v, hv⟩
      exact ⟨v, ⟨1, le_rfl, MeshWalk.single hv⟩,
        ⟨1, le_rfl, MeshWalk.single hv⟩⟩

theorem expand_globalHopBound {V : Type*} (G : SimpleGraph V) {limit : Nat}
    (bound : GlobalHopBound G.Adj limit) :
    GlobalHopBound (expandGraph G).Adj (limit + 2) := by
  intro source target
  rcases endpoint_cover G source with ⟨u, entry, _⟩
  rcases endpoint_cover G target with ⟨v, _, exit⟩
  have route := reachWithin_append
    (reachWithin_append entry (old_reachWithin_lifts G (bound u v))) exit
  simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using route

end Internal

/-- Bundle the carrier and graph so that each recursive generation is finite
and computable without a circular carrier/adjacency definition. -/
structure FiniteStage where
  Vertex : Type
  vertices : Fintype Vertex
  eqDec : DecidableEq Vertex
  graph : SimpleGraph Vertex
  adjDec : DecidableRel graph.Adj

/-- Generation zero is a triangle, not a single edge. -/
def triangleStage : FiniteStage where
  Vertex := Fin 3
  vertices := inferInstance
  eqDec := inferInstance
  graph := ⊤
  adjDec := inferInstance

/-- Install only the supplied computable instances before expanding. -/
def expandStage (stage : FiniteStage) : FiniteStage := by
  letI : Fintype stage.Vertex := stage.vertices
  letI : DecidableEq stage.Vertex := stage.eqDec
  letI : DecidableRel stage.graph.Adj := stage.adjDec
  exact {
    Vertex := ExpansionVertex stage.graph
    vertices := inferInstance
    eqDec := inferInstance
    graph := expandGraph stage.graph
    adjDec := inferInstance
  }

/-- The graph and carrier grow together by simultaneous frozen-edge expansion. -/
def family : Nat → FiniteStage
  | 0 => triangleStage
  | t + 1 => expandStage (family t)

abbrev Vertex (t : Nat) := (family t).Vertex

instance vertexFintype (t : Nat) : Fintype (Vertex t) := (family t).vertices
instance vertexDecidableEq (t : Nat) : DecidableEq (Vertex t) := (family t).eqDec

def graph (t : Nat) : SimpleGraph (Vertex t) := (family t).graph

instance graphDecidable (t : Nat) : DecidableRel (graph t).Adj := (family t).adjDec

/-- A persistent seed witness prevents vacuous connectedness on an empty carrier. -/
def Internal.seedVertex : (t : Nat) → Vertex t
  | 0 => (0 : Fin 3)
  | t + 1 => Sum.inl (Internal.seedVertex t)

instance vertexNonempty (t : Nat) : Nonempty (Vertex t) := ⟨Internal.seedVertex t⟩

/-- Actual vertex cardinality of the recursively constructed graph. -/
def nodeCount (t : Nat) : Nat := Fintype.card (Vertex t)

/-- Actual unordered-edge cardinality, not an independently stored recurrence. -/
def edgeCount (t : Nat) : Nat := Fintype.card (graph t).edgeSet

theorem nodeCount_zero : nodeCount 0 = 3 := by decide

theorem edgeCount_zero : edgeCount 0 = 3 := by decide

theorem nodeCount_succ (t : Nat) : nodeCount (t + 1) = nodeCount t + edgeCount t := by
  change Fintype.card (Vertex t ⊕ (graph t).edgeSet) =
    Fintype.card (Vertex t) + Fintype.card (graph t).edgeSet
  simp only [Fintype.card_sum]

theorem edgeCount_succ (t : Nat) : edgeCount (t + 1) = 3 * edgeCount t := by
  change Fintype.card (expandGraph (graph t)).edgeSet = 3 * Fintype.card (graph t).edgeSet
  exact Internal.expand_edgeCount (graph t)

theorem edgeCount_formula (t : Nat) : edgeCount t = 3 ^ (t + 1) := by
  induction t with
  | zero => simpa using edgeCount_zero
  | succ t ih =>
      rw [edgeCount_succ, ih, pow_succ (3 : Nat) (t + 1)]
      omega

/-- Division-free vertex count derived from the actual carrier recurrence. -/
theorem nodeCount_formula (t : Nat) : 2 * nodeCount t = 3 ^ (t + 1) + 3 := by
  induction t with
  | zero => simp [nodeCount_zero]
  | succ t ih =>
      rw [nodeCount_succ, mul_add, ih, edgeCount_formula, pow_succ (3 : Nat) (t + 1)]
      omega

/-- A deliberately non-tight upper bound, not an exact diameter formula. -/
theorem coarseBound (t : Nat) : GlobalHopBound (graph t).Adj (2 * t + 1) := by
  induction t with
  | zero =>
      intro u v
      by_cases same : u = v
      · subst v
        exact ⟨0, by omega, MeshWalk.refl _⟩
      · exact ⟨1, by omega, MeshWalk.single same⟩
  | succ t ih =>
      change GlobalHopBound (expandGraph (graph t)).Adj (2 * (t + 1) + 1)
      convert Internal.expand_globalHopBound (graph t) ih using 1
      omega

theorem bounded (t : Nat) : ∃ k, GlobalHopBound (graph t).Adj k :=
  ⟨2 * t + 1, coarseBound t⟩

/-- Convert the already-proved mesh walks into mathlib reachability; the seed
vertex supplies the separate nonempty-carrier obligation. -/
theorem graph_connected (t : Nat) : (graph t).Connected := by
  refine { preconnected := ?_, nonempty := inferInstance }
  intro u v
  rcases coarseBound t u v with ⟨_, _, walk⟩
  induction walk with
  | refl => exact SimpleGraph.Reachable.refl _
  | step edge rest ih => exact edge.reachable.trans ih

end NarrativeDynamics.Pseudofractal
