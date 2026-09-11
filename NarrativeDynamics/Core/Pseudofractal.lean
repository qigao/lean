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
  rcases coarseBound t u v with ⟨length, bound, walk⟩
  clear bound
  induction walk with
  | refl => exact SimpleGraph.Reachable.refl _
  | step edge rest ih => exact edge.reachable.trans ih


namespace Internal

/-- Canonical executable data. Edges are sorted, unique pairs with the smaller
endpoint first; the invariant is proved separately, not assumed by the type. -/
structure EncodedGraph where
  nodeCount : Nat
  edges : Array (Nat × Nat)
  deriving DecidableEq, Repr

abbrev edgeList (g : EncodedGraph) := g.edges.toList

def edgeLE (a b : Nat × Nat) : Prop :=
  a.1 < b.1 ∨ a.1 = b.1 ∧ a.2 ≤ b.2

def edgeLT (a b : Nat × Nat) : Prop :=
  a.1 < b.1 ∨ a.1 = b.1 ∧ a.2 < b.2

instance : DecidableRel edgeLE := fun a b =>
  inferInstanceAs (Decidable (a.1 < b.1 ∨ a.1 = b.1 ∧ a.2 ≤ b.2))
instance : DecidableRel edgeLT := fun a b =>
  inferInstanceAs (Decidable (a.1 < b.1 ∨ a.1 = b.1 ∧ a.2 < b.2))
instance : Std.Total edgeLE := ⟨by intros; unfold edgeLE; omega⟩
instance : IsTrans (Nat × Nat) edgeLE := ⟨by intros; unfold edgeLE at *; omega⟩

def encodingWellFormed (g : EncodedGraph) : Prop :=
  (∀ p ∈ edgeList g, p.1 < p.2 ∧ p.2 < g.nodeCount) ∧
    (edgeList g).Pairwise edgeLT

instance (g : EncodedGraph) : Decidable (encodingWellFormed g) :=
  inferInstanceAs (Decidable ((∀ p ∈ edgeList g, p.1 < p.2 ∧ p.2 < g.nodeCount) ∧
    (edgeList g).Pairwise edgeLT))

theorem encoding_nodup (g : EncodedGraph) (h : encodingWellFormed g) :
    (edgeList g).Nodup :=
  h.2.imp (by intro a b hab same; subst b; unfold edgeLT at hab; omega)

/-- Each ranked frozen parent edge contributes exactly its two endpoint links. -/
def newEdge (g : EncodedGraph) (x : Fin (edgeList g).length × Bool) : Nat × Nat :=
  (if x.2 then ((edgeList g).get x.1).2 else ((edgeList g).get x.1).1,
    g.nodeCount + x.1.val)

def newEdges (g : EncodedGraph) : List (Nat × Nat) :=
  ((List.finRange (edgeList g).length).product [false, true]).map (newEdge g)

def rawExpansionEdges (g : EncodedGraph) : List (Nat × Nat) :=
  edgeList g ++ newEdges g

def expandEncoded (g : EncodedGraph) : EncodedGraph :=
  ⟨g.nodeCount + (edgeList g).length,
    ((rawExpansionEdges g).mergeSort (fun a b => decide (edgeLE a b))).toArray⟩

/-- Executable recursion is independent of the proof-level recursive carrier. -/
def encoded : Nat → EncodedGraph
  | 0 => ⟨3, #[(0,1),(0,2),(1,2)]⟩
  | t + 1 => expandEncoded (encoded t)

theorem newEdge_injective (g : EncodedGraph) (hg : encodingWellFormed g) :
    Function.Injective (newEdge g) := by
  rintro ⟨i, b⟩ ⟨j, c⟩ same
  have ids : i = j := by
    apply Fin.ext
    have second := congrArg Prod.snd same
    dsimp [newEdge] at second
    omega
  subst j
  have endpoints := hg.1 _ (List.get_mem (edgeList g) i)
  have first := congrArg Prod.fst same
  cases b <;> cases c <;> simp_all [newEdge]

theorem rawExpansion_nodup (g : EncodedGraph) (hg : encodingWellFormed g) :
    (rawExpansionEdges g).Nodup := by
  apply List.Nodup.append (encoding_nodup g hg)
  · have indices := List.nodup_finRange (edgeList g).length
    have sides : ([false, true] : List Bool).Nodup := by decide
    exact (indices.product sides).map (newEdge_injective g hg)
  · apply List.disjoint_left.mpr
    intro p old fresh
    rcases List.mem_map.mp fresh with ⟨⟨i, side⟩, _, rfl⟩
    have bound := (hg.1 _ old).2
    change g.nodeCount + i.val < g.nodeCount at bound
    omega

theorem mem_expanded_edges (g : EncodedGraph) (a b : Nat) :
    (a,b) ∈ edgeList (expandEncoded g) ↔
      (a,b) ∈ edgeList g ∨ ∃ i : Fin (edgeList g).length,
        b = g.nodeCount + i.val ∧
          (a = ((edgeList g).get i).1 ∨ a = ((edgeList g).get i).2) := by
  have perm := List.mergeSort_perm (rawExpansionEdges g) (fun a b => decide (edgeLE a b))
  change (a,b) ∈ ((rawExpansionEdges g).mergeSort _).toArray.toList ↔ _
  rw [List.toList_toArray, perm.mem_iff]
  simp only [rawExpansionEdges, List.mem_append]
  apply or_congr Iff.rfl
  constructor
  · intro h
    rcases List.mem_map.mp h with ⟨⟨i, side⟩, _, same⟩
    refine ⟨i, ?_, ?_⟩
    · exact (congrArg Prod.snd same).symm
    · have first := (congrArg Prod.fst same).symm
      cases side
      · exact Or.inl first
      · exact Or.inr first
  · rintro ⟨i, rfl, h | h⟩
    · exact List.mem_map.mpr ⟨(i, false), by simp, by simp [newEdge, h]⟩
    · exact List.mem_map.mpr ⟨(i, true), by simp, by simp [newEdge, h]⟩

theorem expandEncoded_wellFormed (g : EncodedGraph) (hg : encodingWellFormed g) :
    encodingWellFormed (expandEncoded g) := by
  constructor
  · rintro ⟨a,b⟩ hab
    change a < b ∧ b < g.nodeCount + (edgeList g).length
    rcases (mem_expanded_edges g a b).mp hab with old | ⟨i, rfl, h | h⟩
    · have bounds := hg.1 _ old
      omega
    · have bounds := hg.1 _ (List.get_mem (edgeList g) i)
      have hi := i.isLt
      omega
    · have bounds := hg.1 _ (List.get_mem (edgeList g) i)
      have hi := i.isLt
      omega
  · change (((rawExpansionEdges g).mergeSort _).toArray.toList).Pairwise edgeLT
    rw [List.toList_toArray]
    have weak := List.pairwise_mergeSort' edgeLE (rawExpansionEdges g)
    have unique : ((rawExpansionEdges g).mergeSort
        (fun a b => decide (edgeLE a b))).Nodup := (rawExpansion_nodup g hg).mergeSort
    apply (weak.and unique).imp
    rintro ⟨a,b⟩ ⟨c,d⟩ ⟨le, ne⟩
    dsimp [edgeLE, edgeLT] at *
    simp only [ne_eq, Prod.mk.injEq] at ne
    omega

theorem encoded_wellFormed (t : Nat) : encodingWellFormed (encoded t) := by
  induction t with
  | zero => decide
  | succ t ih => exact expandEncoded_wellFormed _ ih

def canonicalPair (u v : Nat) : Nat × Nat := (min u v, max u v)

/-- Reading raw data cannot silently wrap an invalid endpoint: the carrier is
bounded Fin and adjacency tests the exact canonical pair. -/
def toGraph (g : EncodedGraph) : SimpleGraph (Fin g.nodeCount) where
  Adj u v := u ≠ v ∧ canonicalPair u.val v.val ∈ edgeList g
  symm := by
    constructor
    intro u v h
    exact ⟨Ne.symm h.1, by simpa [canonicalPair, min_comm, max_comm] using h.2⟩
  loopless := by
    constructor
    intro v h
    exact h.1 rfl

instance (g : EncodedGraph) : DecidableRel (toGraph g).Adj := fun u v =>
  inferInstanceAs (Decidable (u ≠ v ∧ canonicalPair u.val v.val ∈ edgeList g))

def numberedGraph (t : Nat) : SimpleGraph (Fin (encoded t).nodeCount) := toGraph (encoded t)

instance (t : Nat) : DecidableRel (numberedGraph t).Adj :=
  inferInstanceAs (DecidableRel (toGraph (encoded t)).Adj)

theorem toGraph_adj (g : EncodedGraph) (u v : Fin g.nodeCount) :
    (toGraph g).Adj u v ↔ u.val ≠ v.val ∧ canonicalPair u.val v.val ∈ edgeList g := by
  simp [toGraph, Fin.ext_iff]

/-- Sorting two endpoints descends through the unordered-pair quotient. -/
def edgePair {n : Nat} : Sym2 (Fin n) → Nat × Nat :=
  Sym2.lift ⟨fun u v => canonicalPair u.val v.val,
    by intro u v; simp [canonicalPair, min_comm, max_comm]⟩

theorem edgePair_mem (g : EncodedGraph) (e : (toGraph g).edgeSet) :
    edgePair e.val ∈ edgeList g := by
  rcases e with ⟨p, hp⟩
  induction p using Sym2.ind with
  | _ u v => exact hp.2

/-- Decode a validated canonical pair using proved bounds, never modular casts. -/
def pairEdge (g : EncodedGraph) (hg : encodingWellFormed g)
    (p : {p // p ∈ edgeList g}) : (toGraph g).edgeSet :=
  let valid := hg.1 p.val p.property
  let u : Fin g.nodeCount := ⟨p.val.1, lt_trans valid.1 valid.2⟩
  let v : Fin g.nodeCount := ⟨p.val.2, valid.2⟩
  ⟨s(u,v), by
    change u ≠ v ∧ canonicalPair u.val v.val ∈ edgeList g
    refine ⟨?_, ?_⟩
    · intro same
      have h := congrArg Fin.val same
      dsimp [u, v] at h
      omega
    · simpa [canonicalPair, u, v, min_eq_left (le_of_lt valid.1),
        max_eq_right (le_of_lt valid.1)] using p.property⟩

def canonicalEdgeEquiv (g : EncodedGraph) (hg : encodingWellFormed g) :
    (toGraph g).edgeSet ≃ {p // p ∈ edgeList g} where
  toFun e := ⟨edgePair e.val, edgePair_mem g e⟩
  invFun := pairEdge g hg
  left_inv := by
    rintro ⟨p, hp⟩
    induction p using Sym2.ind with
    | _ u v =>
        apply Subtype.ext
        by_cases h : u.val ≤ v.val
        · simp [pairEdge, edgePair, canonicalPair, min_eq_left h, max_eq_right h]
        · have h' : v.val ≤ u.val := by omega
          simp [pairEdge, edgePair, canonicalPair, min_eq_right h', max_eq_left h',
            Sym2.eq_swap]
  right_inv := by
    intro p
    apply Subtype.ext
    have h := (hg.1 p.val p.property).1
    simp [pairEdge, edgePair, canonicalPair, min_eq_left (le_of_lt h),
      max_eq_right (le_of_lt h)]

/-- Constructive rank/unrank for a duplicate-free list. The inverse is lookup,
not a classically selected bijection. -/
def memberIndexEquiv {α : Type*} [DecidableEq α] (xs : List α) (h : xs.Nodup) :
    {a // a ∈ xs} ≃ Fin xs.length where
  toFun a := ⟨xs.idxOf a.val, List.idxOf_lt_length_iff.mpr a.property⟩
  invFun i := ⟨xs.get i, List.get_mem xs i⟩
  left_inv a := by
    apply Subtype.ext
    exact List.getElem_idxOf (List.idxOf_lt_length_iff.mpr a.property)
  right_inv i := Fin.ext (List.get_idxOf h i)

def edgeRankEquiv (g : EncodedGraph) (hg : encodingWellFormed g) :
    (toGraph g).edgeSet ≃ Fin (edgeList g).length :=
  (canonicalEdgeEquiv g hg).trans (memberIndexEquiv (edgeList g) (encoding_nodup g hg))

theorem edgeRank_lookup (g : EncodedGraph) (hg : encodingWellFormed g)
    (e : (toGraph g).edgeSet) :
    (edgeList g).get (edgeRankEquiv g hg e) = edgePair e.val := by
  exact congrArg Subtype.val ((memberIndexEquiv (edgeList g)
    (encoding_nodup g hg)).symm_apply_apply (canonicalEdgeEquiv g hg e))

theorem edgePair_endpoint {n : Nat} (e : Sym2 (Fin n)) (x : Fin n) :
    x ∈ e ↔ x.val = (edgePair e).1 ∨ x.val = (edgePair e).2 := by
  induction e using Sym2.ind with
  | _ u v =>
      by_cases h : u.val ≤ v.val
      · simp [edgePair, canonicalPair, min_eq_left h, max_eq_right h, Fin.ext_iff]
      · have h' : v.val ≤ u.val := by omega
        simp [edgePair, canonicalPair, min_eq_right h', max_eq_left h', Fin.ext_iff,
          or_comm]

theorem rank_endpoint (g : EncodedGraph) (hg : encodingWellFormed g)
    (e : (toGraph g).edgeSet) (x : Fin g.nodeCount) :
    x ∈ e.val ↔ x.val = ((edgeList g).get (edgeRankEquiv g hg e)).1 ∨
      x.val = ((edgeList g).get (edgeRankEquiv g hg e)).2 := by
  rw [edgeRank_lookup]
  exact edgePair_endpoint e.val x

/-- Split at the old vertex count, with checked subtraction for newborn IDs. -/
def sumIndexEquiv (n m : Nat) : Fin n ⊕ Fin m ≃ Fin (n + m) where
  toFun
    | .inl i => ⟨i.val, by omega⟩
    | .inr j => ⟨n + j.val, by omega⟩
  invFun k := if h : k.val < n then .inl ⟨k.val, h⟩ else
    .inr ⟨k.val - n, by omega⟩
  left_inv := by
    intro x
    cases x with
    | inl i => simp [i.isLt]
    | inr j => simp
  right_inv := by
    intro k
    by_cases h : k.val < n
    · simp [h]
    · simp [h, Nat.add_sub_of_le (Nat.le_of_not_gt h)]

def stepNumbering (g : EncodedGraph) (hg : encodingWellFormed g) :
    ExpansionVertex (toGraph g) ≃ Fin (expandEncoded g).nodeCount :=
  (Equiv.sumCongr (Equiv.refl _) (edgeRankEquiv g hg)).trans
    (sumIndexEquiv g.nodeCount (edgeList g).length)

@[simp] theorem stepNumbering_old (g : EncodedGraph) (hg : encodingWellFormed g)
    (u : Fin g.nodeCount) : (stepNumbering g hg (.inl u)).val = u.val := rfl
@[simp] theorem stepNumbering_new (g : EncodedGraph) (hg : encodingWellFormed g)
    (e : (toGraph g).edgeSet) :
    (stepNumbering g hg (.inr e)).val = g.nodeCount + (edgeRankEquiv g hg e).val := rfl

theorem step_old_adj (g : EncodedGraph) (hg : encodingWellFormed g)
    (u v : Fin g.nodeCount) :
    (toGraph (expandEncoded g)).Adj (stepNumbering g hg (.inl u))
      (stepNumbering g hg (.inl v)) ↔ (toGraph g).Adj u v := by
  simp only [toGraph_adj, stepNumbering_old]
  apply and_congr Iff.rfl
  rw [canonicalPair, mem_expanded_edges]
  constructor
  · rintro (old | ⟨i, hi, _⟩)
    · exact old
    · have bounds : max u.val v.val < g.nodeCount := max_lt u.isLt v.isLt
      omega
  · exact Or.inl

theorem step_mixed_adj (g : EncodedGraph) (hg : encodingWellFormed g)
    (u : Fin g.nodeCount) (e : (toGraph g).edgeSet) :
    (toGraph (expandEncoded g)).Adj (stepNumbering g hg (.inl u))
      (stepNumbering g hg (.inr e)) ↔ u ∈ e.val := by
  simp only [toGraph_adj, stepNumbering_old, stepNumbering_new]
  have hu : u.val < g.nodeCount + (edgeRankEquiv g hg e).val := by omega
  rw [canonicalPair, min_eq_left (le_of_lt hu), max_eq_right (le_of_lt hu),
    mem_expanded_edges]
  constructor
  · rintro ⟨_, old | ⟨i, same, endpoints⟩⟩
    · have bad := (hg.1 _ old).2
      omega
    · have ids : i = edgeRankEquiv g hg e := by apply Fin.ext; omega
      subst i
      exact (rank_endpoint g hg e u).mpr endpoints
  · intro member
    exact ⟨ne_of_lt hu, Or.inr ⟨edgeRankEquiv g hg e, rfl,
      (rank_endpoint g hg e u).mp member⟩⟩

theorem step_new_not_adj (g : EncodedGraph) (hg : encodingWellFormed g)
    (e f : (toGraph g).edgeSet) :
    ¬ (toGraph (expandEncoded g)).Adj (stepNumbering g hg (.inr e))
      (stepNumbering g hg (.inr f)) := by
  rw [toGraph_adj]
  simp only [stepNumbering_new, canonicalPair]
  rintro ⟨_, h⟩
  rcases (mem_expanded_edges g _ _).mp h with old | ⟨i, _, first | first⟩
  · have bad := (hg.1 _ old).2
    omega
  · have bounds := hg.1 _ (List.get_mem (edgeList g) i)
    omega
  · have bounds := hg.1 _ (List.get_mem (edgeList g) i)
    omega

/-- One canonical executable expansion represents exactly the mathematical step. -/
def stepNumberingIso (g : EncodedGraph) (hg : encodingWellFormed g) :
    expandGraph (toGraph g) ≃g toGraph (expandEncoded g) where
  toEquiv := stepNumbering g hg
  map_rel_iff' := by
    intro x y
    cases x with
    | inl u =>
        cases y with
        | inl v => exact step_old_adj g hg u v
        | inr e => exact step_mixed_adj g hg u e
    | inr e =>
        cases y with
        | inl v =>
            rw [(toGraph (expandEncoded g)).adj_comm]
            exact step_mixed_adj g hg v e
        | inr f => exact iff_false_intro (step_new_not_adj g hg e f)

/-- Expansion respects graph isomorphisms, including unordered edge identities. -/
def expandIso {V W : Type*} {G : SimpleGraph V} {H : SimpleGraph W}
    (f : G ≃g H) : expandGraph G ≃g expandGraph H where
  toEquiv := Equiv.sumCongr f.toEquiv f.mapEdgeSet
  map_rel_iff' := by
    intro x y
    cases x with
    | inl u =>
        cases y with
        | inl v => exact f.map_adj_iff
        | inr e =>
            change f u ∈ Sym2.map f e.val ↔ u ∈ e.val
            simp [Sym2.mem_map, f.injective.eq_iff]
    | inr e =>
        cases y with
        | inl v =>
            change f v ∈ Sym2.map f e.val ↔ v ∈ e.val
            simp [Sym2.mem_map, f.injective.eq_iff]
        | inr e => exact Iff.rfl

def numberingIso : (t : Nat) → graph t ≃g numberedGraph t
  | 0 => { toEquiv := Equiv.refl (Fin 3), map_rel_iff' := by decide }
  | t + 1 => (expandIso (numberingIso t)).trans
      (stepNumberingIso (encoded t) (encoded_wellFormed t))

/-- Both forward numbering and its inverse are executable. -/
def numbering (t : Nat) : Vertex t ≃ Fin (encoded t).nodeCount :=
  (numberingIso t).toEquiv

theorem numbered_adj_iff (t : Nat) (u v : Vertex t) :
    (numberedGraph t).Adj (numbering t u) (numbering t v) ↔
      (graph t).Adj u v := (numberingIso t).map_adj_iff

theorem numberedBounded (t : Nat) : ∃ k, GlobalHopBound (numberedGraph t).Adj k := by
  refine ⟨2 * t + 1, ?_⟩
  intro u v
  rcases coarseBound t ((numbering t).symm u) ((numbering t).symm v) with ⟨n, hn, path⟩
  refine ⟨n, hn, ?_⟩
  have mapped := path.mapNodes (numbering t)
    (fun {a b} hab => (numbered_adj_iff t a b).mpr hab)
  simpa using mapped

end Internal

end NarrativeDynamics.Pseudofractal
