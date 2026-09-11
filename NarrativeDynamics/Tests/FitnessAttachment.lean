import NarrativeDynamics.Core.FitnessAttachment

open NarrativeDynamics NarrativeDynamics.FitnessAttachment
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
