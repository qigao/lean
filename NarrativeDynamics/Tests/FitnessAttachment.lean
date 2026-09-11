import NarrativeDynamics.Core.FitnessAttachment

open NarrativeDynamics NarrativeDynamics.FitnessAttachment
open scoped BigOperators
open NarrativeDynamics.FitnessAttachment.Internal

-- Exact values test the real normalizer, not a test-only arithmetic model.
theorem normalization_fixture :
    normalizeValues [2, 4, 8] = .ok [1/7, 2/7, 4/7] := by decide_cbv

example : normalizeValues [0, 5, 0] = .ok [0, 1, 0] := by decide_cbv
example : normalizeValues [0, 0] = .error .zeroMass := by decide_cbv
example : normalizeValues [] = .error .zeroMass := by decide_cbv
example : normalizeValues [-1, 2] = .error .negativeWeight := by decide_cbv

-- Preserve order and exact rational arithmetic; no epsilon or rounding.
example : normalizeValues [4, 2, 8] = .ok [2/7, 1/7, 4/7] := by decide_cbv
example : normalizeValues [1/2, 1/3, 1/6] = .ok [1/2, 1/3, 1/6] := by decide_cbv
example : normalizeValues [1/1000000, 3/1000000] = .ok [1/4, 3/4] := by decide_cbv
example : normalizeValues [5] = .ok [1] := by decide_cbv
example : normalizeValues [0] = .error .zeroMass := by decide_cbv

-- Reject any negative entry before considering whether the total is positive.
example : normalizeValues [2, -1] = .error .negativeWeight := by decide_cbv
example : normalizeValues [-1, 1] = .error .negativeWeight := by decide_cbv
example : normalizeValues [-1, -2] = .error .negativeWeight := by decide_cbv
example : normalizeValues [0, -1] = .error .negativeWeight := by decide_cbv

-- Exclusion masks the selected indices without reordering surviving entries.
example : total (![2, 4, 8] : Fin 3 → Rat) = 14 := by decide_cbv
example : List.ofFn (remaining (![2, 4, 8] : Fin 3 → Rat) {2}) =
    [2, 4, 0] := by decide_cbv
example : rowValues (normalize (remaining (![2, 4, 8] : Fin 3 → Rat) {2})) =
    .ok [1/3, 2/3, 0] := by decide_cbv
example : rowValues (normalize (remaining (![2, 4, 8] : Fin 3 → Rat) Finset.univ)) =
    .error .zeroMass := by decide_cbv

-- Kernel contracts hold for every dimension and successful output.
example {n : Nat} (w : Fin n → Rat) (row : Row n)
    (success : normalize w = .ok row) : (∑ i, row.mass i) = 1 :=
  normalize_sum_one success

example {n : Nat} (w : Fin n → Rat) (row : Row n)
    (success : normalize w = .ok row) (i : Fin n) :
    row.mass i = w i / total w := normalize_mass_eq success i

example {n : Nat} (w : Fin n → Rat) (row : Row n)
    (success : normalize w = .ok row) (i : Fin n) :
    (0 < row.mass i ↔ 0 < w i) := normalize_support success i

example {n : Nat} (w : Fin n → Rat) (row : Row n)
    (success : normalize w = .ok row) (i : Fin n) :
    0 ≤ row.mass i := normalize_nonneg success i

example {n : Nat} (w : Fin n → Rat) (row : Row n)
    (success : normalize w = .ok row) :
    (∀ i, 0 ≤ w i) ∧ 0 < total w := normalize_conditions success

-- Completeness prevents an implementation that rejects every request.
example {n : Nat} (w : Fin n → Rat) :
    (∃ row, normalize w = .ok row) ↔ (∀ i, 0 ≤ w i) ∧ 0 < total w :=
  normalize_exists_iff w

example {n : Nat} (w : Fin n → Rat) (nonneg : ∀ i, 0 ≤ w i) :
    total w = 0 ↔ ∀ i, w i = 0 := total_eq_zero_iff w nonneg

example (xs ys : List Rat) (success : normalizeValues xs = .ok ys) :
    ys.length = xs.length := normalizeValues_length success

#print axioms normalize_mass_eq
#print axioms normalize_sum_one
#print axioms normalize_support
#print axioms normalize_nonneg
#print axioms normalize_conditions
#print axioms normalize_exists_iff
#print axioms total_eq_zero_iff
#print axioms normalizeValues_length
#print axioms normalization_fixture

namespace NarrativeDynamics.FitnessAttachment.Fixtures

private theorem triangle_connected : (⊤ : SimpleGraph (Fin 3)).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v
      exact ⟨.nil⟩
    · exact ⟨.cons h .nil⟩
  nonempty := inferInstance

def triangle : State 3 where
  snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![1, 2, 4] }
  valid := ⟨by decide, triangle_connected, by decide⟩

def constantTriangle : State 3 where
  snapshot := { graph := ⊤, adjDec := inferInstance, fitness := fun _ => 1 }
  valid := ⟨by decide, triangle_connected, by decide⟩

def improvedTriangle : State 3 where
  snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![2, 2, 4] }
  valid := ⟨by decide, triangle_connected, by decide⟩

def starGraph : SimpleGraph (Fin 4) where
  Adj u v := u ≠ v ∧ (u = 0 ∨ v = 0)
  symm := ⟨fun _ _ h => ⟨h.1.symm, h.2.symm⟩⟩
  loopless := ⟨fun _ h => h.1 rfl⟩

private theorem star_connected : starGraph.Connected where
  preconnected := by
    have toCenter : ∀ u, starGraph.Reachable u 0 := by
      intro u
      by_cases h : u = 0
      · subst u
        exact ⟨.nil⟩
      · exact ⟨.cons ⟨h, Or.inr rfl⟩ .nil⟩
    intro u v
    exact (toCenter u).trans (toCenter v).symm
  nonempty := inferInstance

def star : State 4 where
  snapshot := { graph := starGraph, adjDec := fun _ _ => inferInstance,
                fitness := fun _ => 1 }
  valid := ⟨by decide, star_connected, by decide⟩

def isolated : Snapshot 3 where
  graph := ⊥
  adjDec := inferInstance
  fitness := fun _ => 1

end NarrativeDynamics.FitnessAttachment.Fixtures

-- Actual adjacency, not a caller-supplied degree vector or a uniform shortcut.
example : List.ofFn (degree Fixtures.triangle.snapshot) = [2, 2, 2] := by decide_cbv
example : List.ofFn (weights Fixtures.triangle.snapshot) = [2, 4, 8] := by decide_cbv
example : List.ofFn (degree Fixtures.star.snapshot) = [3, 1, 1, 1] := by decide_cbv
example : List.ofFn (weights Fixtures.isolated) = [0, 0, 0] := by decide_cbv
example : rowValues (normalize (weights Fixtures.isolated)) = .error .zeroMass := by
  decide_cbv

-- Fixed birth-time fitness and each prefix's remaining candidates determine the row.
theorem bb_triangle_fixture :
    rowValues (.ok (attachmentRow Fixtures.triangle ∅ (by decide))) =
      .ok [1/7, 2/7, 4/7] := by decide_cbv
example : rowValues (.ok (attachmentRow Fixtures.triangle {2} (by decide))) =
    .ok [1/3, 2/3, 0] := by decide_cbv
example : rowValues (.ok (attachmentRow Fixtures.triangle {0, 2} (by decide))) =
    .ok [0, 1, 0] := by decide_cbv
example : rowValues (.ok (attachmentRow Fixtures.constantTriangle ∅ (by decide))) =
    .ok [1/3, 1/3, 1/3] := by decide_cbv
example : rowValues (.ok (baRow Fixtures.star ∅ (by decide))) =
    .ok [1/2, 1/6, 1/6, 1/6] := by decide_cbv
example : rowValues (.ok (baRow Fixtures.star {0} (by decide))) =
    .ok [0, 1/3, 1/3, 1/3] := by decide_cbv
example : rowValues (.ok (attachmentRow
    (scaleFitness Fixtures.triangle ⟨3/2, by norm_num⟩) {2} (by decide))) =
    .ok [1/3, 2/3, 0] := by decide_cbv

-- Own-fitness gain is strict with competition, but not with only one eligible ID.
example : (attachmentRow Fixtures.triangle ∅ (by decide)).mass 0 = 1/7 := by decide_cbv
example : (attachmentRow Fixtures.improvedTriangle ∅ (by decide)).mass 0 = 1/4 := by
  decide_cbv
example : (attachmentRow Fixtures.triangle {1, 2} (by decide)).mass 0 =
    (attachmentRow Fixtures.improvedTriangle {1, 2} (by decide)).mass 0 := by decide_cbv

-- This is a weighted-vector arithmetic fixture, NOT a two-vertex graph of degrees 100/20.
example : normalizeValues [(1/10)*100, 1*20] = .ok [1/3, 2/3] := by decide_cbv
example : normalizeValues [100, 20] = .ok [5/6, 1/6] := by decide_cbv

-- Generic contracts retain graph, eligibility, and competitor premises explicitly.
example {n : Nat} (s : State n) (i : Fin n) : 0 < degree s.snapshot i :=
  state_degree_pos s i
example {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n) :
    0 < total (remaining (weights s.snapshot) S) := remaining_total_pos s S hS
example {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n) (i : Fin n) :
    (attachmentRow s S hS).mass i =
      remaining (weights s.snapshot) S i / total (remaining (weights s.snapshot) S) :=
  attachment_mass s S hS i
example {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n) (i : Fin n) :
    (0 < (attachmentRow s S hS).mass i ↔ i ∉ S) := attachment_support s S hS i
example {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n) :
    (∑ i, (attachmentRow s S hS).mass i) = 1 := (attachmentRow s S hS).total_one
example {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n)
    (c : PosFitness) (constant : ∀ j, s.snapshot.fitness j = c.val) (i : Fin n) :
    (attachmentRow s S hS).mass i = (baRow s S hS).mass i :=
  attachment_ba s S hS c constant i
example {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n)
    (c : PosFitness) (i : Fin n) :
    (attachmentRow (scaleFitness s c) S hS).mass i = (attachmentRow s S hS).mass i :=
  attachment_scale s S hS c i
example {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n)
    (i j : Fin n) (hi : i ∉ S) (hj : j ∉ S) :
    (attachmentRow s S hS).mass i / (attachmentRow s S hS).mass j =
      weights s.snapshot i / weights s.snapshot j := attachment_ratio s S hS i j hi hj
example {n : Nat} (s t : State n) (S : Finset (Fin n)) (hS : S.card < n)
    (i : Fin n) (sameGraph : t.snapshot.graph = s.snapshot.graph)
    (others : ∀ j, j ≠ i → t.snapshot.fitness j = s.snapshot.fitness j)
    (hi : i ∉ S) (increase : s.snapshot.fitness i ≤ t.snapshot.fitness i) :
    (attachmentRow s S hS).mass i ≤ (attachmentRow t S hS).mass i :=
  attachment_mono s t S hS i sameGraph others hi increase
example {n : Nat} (s t : State n) (S : Finset (Fin n)) (hS : S.card < n)
    (i : Fin n) (sameGraph : t.snapshot.graph = s.snapshot.graph)
    (others : ∀ j, j ≠ i → t.snapshot.fitness j = s.snapshot.fitness j)
    (hi : i ∉ S) (increase : s.snapshot.fitness i < t.snapshot.fitness i)
    (competitor : ∃ j, j ∉ S ∧ j ≠ i) :
    (attachmentRow s S hS).mass i < (attachmentRow t S hS).mass i :=
  attachment_strict_mono s t S hS i sameGraph others hi increase competitor

#print axioms state_degree_pos
#print axioms remaining_total_pos
#print axioms attachment_mass
#print axioms attachment_support
#print axioms attachment_ba
#print axioms attachment_scale
#print axioms attachment_ratio
#print axioms attachment_mono
#print axioms attachment_strict_mono
#print axioms bb_triangle_fixture
