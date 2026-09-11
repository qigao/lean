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
example : Fintype.card (ExpansionVertex (⊥ : SimpleGraph (Fin 3))) = 3 := by decide
example : ¬ (expandGraph (⊥ : SimpleGraph (Fin 3))).Adj
    (Sum.inl 0) (Sum.inl 1) := by decide
example : Fintype.card (ExpansionVertex (⊥ : SimpleGraph (Fin 0))) = 0 := by decide

example {V : Type*} (G : SimpleGraph V) (u : V) :
    MeshWalk (expandGraph G).Adj 0 (oldVertex G u) (oldVertex G u) :=
  old_walk_lifts G (MeshWalk.refl u)

#print axioms old_walk_lifts
#print axioms old_reachWithin_lifts
