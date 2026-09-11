import NarrativeDynamics.Core.Pseudofractal

open NarrativeDynamics NarrativeDynamics.Pseudofractal

-- All four cases use the frozen old graph and its unordered edges.
example {V : Type*} (G : SimpleGraph V) (u v : V) :
    (expandGraph G).Adj (Sum.inl u) (Sum.inl v) ↔ G.Adj u v := Iff.rfl

example {V : Type*} (G : SimpleGraph V) (v : V) (e : G.edgeSet) :
    (expandGraph G).Adj (Sum.inl v) (Sum.inr e) ↔ v ∈ e.val := Iff.rfl

example {V : Type*} (G : SimpleGraph V) (e : G.edgeSet) (v : V) :
    (expandGraph G).Adj (Sum.inr e) (Sum.inl v) ↔ v ∈ e.val := Iff.rfl

example {V : Type*} (G : SimpleGraph V) (e f : G.edgeSet) :
    ¬ (expandGraph G).Adj (Sum.inr e) (Sum.inr f) := by
  simp [expandGraph, expansionAdj]

example {V : Type*} (G : SimpleGraph V) : SymmetricMesh (expandGraph G).Adj := by
  intro u v adjacent
  exact (expandGraph G).adj_symm adjacent

example {V : Type*} (G : SimpleGraph V) (v : ExpansionVertex G) :
    ¬ (expandGraph G).Adj v v := (expandGraph G).irrefl

-- No classical/noncomputable adjacency instance is allowed here.
example {V : Type*} [DecidableEq V] (G : SimpleGraph V) [DecidableRel G.Adj] :
    DecidableRel (expandGraph G).Adj := inferInstance

example {V : Type*} (G : SimpleGraph V) (v : V) :
    oldVertex G v = Sum.inl v := rfl

example {V : Type*} (G : SimpleGraph V) : Function.Injective (oldVertex G) :=
  (oldVertex G).injective

example {V : Type*} (G : SimpleGraph V) {u v : V} {n : Nat}
    (walk : MeshWalk G.Adj n u v) :
    MeshWalk (expandGraph G).Adj n (Sum.inl u) (Sum.inl v) :=
  old_walk_lifts G walk

example {V : Type*} (G : SimpleGraph V) {u v : V} {limit : Nat}
    (reachable : ReachWithin G.Adj limit u v) :
    ReachWithin (expandGraph G).Adj limit (Sum.inl u) (Sum.inl v) :=
  old_reachWithin_lifts G reachable

-- Reversing the parent edge does not create a second newborn.
example {V : Type*} (G : SimpleGraph V) {u v : V}
    (forward : s(u, v) ∈ G.edgeSet) (backward : s(v, u) ∈ G.edgeSet) :
    (Sum.inr ⟨s(u, v), forward⟩ : ExpansionVertex G) =
      Sum.inr ⟨s(v, u), backward⟩ := by
  exact congrArg Sum.inr (Subtype.ext Sym2.eq_swap)

-- Actual finite carriers/edges, not separately stored recurrence counters.
private abbrev triangle : SimpleGraph (Fin 3) := ⊤

example : Fintype.card (ExpansionVertex triangle) = 6 := by decide
example : Fintype.card (expandGraph triangle).edgeSet = 9 := by decide

private def edge01 : triangle.edgeSet := ⟨s(0, 1), by decide⟩

example : (expandGraph triangle).Adj (Sum.inl 0) (Sum.inr edge01) := by decide
example : (expandGraph triangle).Adj (Sum.inr edge01) (Sum.inl 1) := by decide
example : ¬ (expandGraph triangle).Adj (Sum.inl 2) (Sum.inr edge01) := by decide
example : ¬ (expandGraph triangle).Adj (Sum.inr edge01) (Sum.inr edge01) := by decide

-- Empty edge sets and isolated vertices must not invent newborns or paths.
example : Fintype.card (ExpansionVertex (⊥ : SimpleGraph (Fin 3))) = 3 := by
  simp [ExpansionVertex, Fintype.card_sum, SimpleGraph.edgeSet_bot]
example : ¬ (expandGraph (⊥ : SimpleGraph (Fin 3))).Adj
    (Sum.inl 0) (Sum.inl 1) := by decide
example : Fintype.card (ExpansionVertex (⊥ : SimpleGraph (Fin 0))) = 0 := by
  simp [ExpansionVertex, SimpleGraph.edgeSet_bot]

example {V : Type*} (G : SimpleGraph V) (u : V) :
    MeshWalk (expandGraph G).Adj 0 (oldVertex G u) (oldVertex G u) :=
  old_walk_lifts G (MeshWalk.refl u)

#print axioms old_walk_lifts
#print axioms old_reachWithin_lifts

-- Task 2: the recursive graph and carrier must travel together.
example : family 0 = triangleStage := rfl
example (t : Nat) : family (t + 1) = expandStage (family t) := rfl
example (t : Nat) : Fintype (Vertex t) := inferInstance
example (t : Nat) : DecidableEq (Vertex t) := inferInstance
example (t : Nat) : DecidableRel (graph t).Adj := inferInstance
example (t : Nat) : Nonempty (Vertex t) := inferInstance

-- These definitional contracts forbid independent recurrence counters.
example (t : Nat) : nodeCount t = Fintype.card (Vertex t) := rfl
example (t : Nat) : edgeCount t = Fintype.card (graph t).edgeSet := rfl

example : nodeCount 0 = 3 := nodeCount_zero
example : edgeCount 0 = 3 := edgeCount_zero
example (t : Nat) : nodeCount (t + 1) = nodeCount t + edgeCount t :=
  nodeCount_succ t
example (t : Nat) : edgeCount (t + 1) = 3 * edgeCount t := edgeCount_succ t
example (t : Nat) : edgeCount t = 3 ^ (t + 1) := edgeCount_formula t
example (t : Nat) : 2 * nodeCount t = 3 ^ (t + 1) + 3 := nodeCount_formula t

-- Small concrete checks evaluate the actual finite graph, not the formulas.
example : nodeCount 1 = 6 := by decide
example : edgeCount 1 = 9 := by decide
example : nodeCount 2 = 15 := by decide
example : edgeCount 2 = 27 := by decide
example : nodeCount 3 = 42 := by decide
-- G3 unordered-edge enumeration needs more reduction depth than the default.
-- Keep direct computation, with a finite budget local to this one check.
set_option maxRecDepth 4096 in
example : edgeCount 3 = 81 := by decide

-- The bound is only an upper bound; it is not an exact diameter claim.
example (t : Nat) : GlobalHopBound (graph t).Adj (2 * t + 1) := coarseBound t
example (t : Nat) : ∃ k, GlobalHopBound (graph t).Adj k := bounded t
example (t : Nat) : (graph t).Connected := graph_connected t

#print axioms nodeCount_formula
#print axioms edgeCount_formula
#print axioms coarseBound
#print axioms graph_connected

-- Task 3: the executable constructor has fixed IDs and sorted unordered edges.
-- Kernel reduction unfolds the well-founded sort in these concrete checks;
-- this is not native evaluation and introduces no compiler-result axiom.
open NarrativeDynamics.Pseudofractal.Internal

example : (encoded 0).nodeCount = 3 := by decide
example : (encoded 0).edges = #[(0,1),(0,2),(1,2)] := by decide
example : (encoded 1).nodeCount = 6 := by decide
example : (encoded 1).edges =
    #[(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,5),(2,4),(2,5)] := by decide +kernel
example : (encoded 2).nodeCount = 15 := by decide +kernel
example : (encoded 2).edges.size = 27 := by decide +kernel
example : (encoded 3).nodeCount = 42 := by decide +kernel
example : (encoded 3).edges.size = 81 := by decide +kernel

example : encodingWellFormed (encoded 0) := by decide
example : encodingWellFormed (encoded 1) := by decide +kernel
example : encodingWellFormed (encoded 2) := by decide +kernel
example : encodingWellFormed (encoded 3) := by decide +kernel
example (t : Nat) : encodingWellFormed (encoded t) := encoded_wellFormed t

-- Reject loops, reversed/out-of-range edges, duplicates and unsorted data.
example : ¬ encodingWellFormed (⟨3, #[(0,0)]⟩ : EncodedGraph) := by decide
example : ¬ encodingWellFormed (⟨3, #[(1,0)]⟩ : EncodedGraph) := by decide
example : ¬ encodingWellFormed (⟨3, #[(0,3)]⟩ : EncodedGraph) := by decide
example : ¬ encodingWellFormed (⟨3, #[(0,1),(0,1)]⟩ : EncodedGraph) := by decide
example : ¬ encodingWellFormed (⟨3, #[(1,2),(0,1)]⟩ : EncodedGraph) := by decide

-- Both directions and both inverse equations are essential, not just counts.
example (t : Nat) : graph t ≃g numberedGraph t := numberingIso t
example (t : Nat) (u v : Vertex t) :
    (numberedGraph t).Adj (numbering t u) (numbering t v) ↔
      (graph t).Adj u v := numbered_adj_iff t u v
example (t : Nat) (u : Vertex t) : (numbering t).symm (numbering t u) = u :=
  (numbering t).symm_apply_apply u
example (t : Nat) (i : Fin (encoded t).nodeCount) :
    numbering t ((numbering t).symm i) = i := (numbering t).apply_symm_apply i
example (t : Nat) : DecidableRel (numberedGraph t).Adj := inferInstance
example (t : Nat) : ∃ k, GlobalHopBound (numberedGraph t).Adj k := numberedBounded t

#print axioms encoded_wellFormed
#print axioms numberingIso
#print axioms numbered_adj_iff
#print axioms numberedBounded

-- Task 3 acceptance: cardinalities are transferred through the actual isomorphism.
example (t : Nat) : nodeCount t = (encoded t).nodeCount := by
  simpa [nodeCount] using Fintype.card_congr (numbering t)
example (t : Nat) : edgeCount t = (encoded t).edges.size := by
  simpa [edgeCount] using Fintype.card_congr
    ((numberingIso t).mapEdgeSet.trans (edgeRankEquiv (encoded t) (encoded_wellFormed t)))

-- Every old numeric identity is preserved definitionally at the next stage.
example (t : Nat) (u : Vertex t) :
    (numbering (t + 1) (Sum.inl u)).val = (numbering t u).val := rfl

-- Instantiate the proved isomorphism, separately from executable checks.
example : graph 0 ≃g numberedGraph 0 := numberingIso 0
example : graph 1 ≃g numberedGraph 1 := numberingIso 1
example : graph 2 ≃g numberedGraph 2 := numberingIso 2
example : graph 3 ≃g numberedGraph 3 := numberingIso 3

-- Native execution here is only a regression test, never a theorem premise.
-- It checks every ordered pair, including nonedges and diagonal pairs.
private def verifyNumberedGeneration (t : Nat) : IO (Nat × Nat) := do
  let ids := List.finRange (encoded t).nodeCount
  let mut pairs := 0
  for i in ids do
    let u := (numbering t).symm i
    unless decide (numbering t u = i) do
      throw (IO.userError s!"numbering round-trip failed at generation {t}, ID {i.val}")
    unless decide ((numbering t).symm (numbering t u) = u) do
      throw (IO.userError s!"inverse round-trip failed at generation {t}, ID {i.val}")
    unless decide ((numbering (t + 1) (Sum.inl u)).val = i.val) do
      throw (IO.userError s!"old ID changed at generation {t}, ID {i.val}")
    for j in ids do
      let v := (numbering t).symm j
      unless decide ((numberedGraph t).Adj i j) == decide ((graph t).Adj u v) do
        throw (IO.userError s!"adjacency mismatch at generation {t}, IDs {i.val}/{j.val}")
      pairs := pairs + 1
  pure (ids.length, pairs)

#eval do
  let mut vertices := 0
  let mut pairs := 0
  for t in [0, 1, 2, 3] do
    let (n, p) ← verifyNumberedGeneration t
    vertices := vertices + n
    pairs := pairs + p
  IO.println s!"Canonical numbering executable checks: {vertices} vertices, {pairs} ordered pairs."
