import Mathlib

/-!
# Exact finite fitness-attachment normalization

This first layer only constructs checked rational probability rows. Graph growth,
fitness-derived weights and target-sequence laws belong to subsequent tasks.
-/

namespace NarrativeDynamics.FitnessAttachment.Internal

open scoped BigOperators

/-- Explicit failures shared by the finite attachment input boundaries. -/
inductive Error where
  | negativeWeight
  | zeroMass
  | invalidNodeCount
  | fitnessSizeMismatch
  | nonpositiveFitness
  | invalidEdge
  | duplicateEdge
  | disconnectedSeed
  | invalidM
  | targetCountMismatch
  | targetOutOfRange
  | duplicateTarget
  deriving DecidableEq, Repr

/-- A stored fitness must be strictly positive, unlike a masked row weight. -/
abbrev PosFitness := { eta : Rat // 0 < eta }

/-- A finite rational row with validity proved from its computed masses. -/
structure Row (n : Nat) where
  mass : Fin n → Rat
  nonneg : ∀ i, 0 ≤ mass i
  total_one : (∑ i, mass i) = 1

/-- Sum actual weights, rather than accepting a caller-supplied denominator. -/
def total {n : Nat} (w : Fin n → Rat) : Rat := Finset.univ.sum w

/-- Exclude chosen old indices without changing the other weights or their IDs. -/
def remaining {n : Nat} (w : Fin n → Rat)
    (selected : Finset (Fin n)) (i : Fin n) : Rat :=
  if i ∈ selected then 0 else w i

/-- Construct a row only after nonnegative weights and positive mass are proved. -/
def normalizePositive {n : Nat} (w : Fin n → Rat)
    (h : ∀ i, 0 ≤ w i) (hz : 0 < total w) : Row n where
  mass i := w i / total w
  nonneg i := div_nonneg (h i) (le_of_lt hz)
  total_one := by
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt hz)

/-- Reject negatives first; empty/all-zero inputs never receive a fallback row. -/
def normalize {n : Nat} (w : Fin n → Rat) : Except Error (Row n) :=
  if h : ∀ i, 0 ≤ w i then
    if hz : 0 < total w then .ok (normalizePositive w h hz)
    else .error .zeroMass
  else .error .negativeWeight

/-- Expose row data in increasing finite-index order, preserving errors. -/
def rowValues {n : Nat} (result : Except Error (Row n)) : Except Error (List Rat) :=
  match result with
  | .error err => .error err
  | .ok row => .ok (List.ofFn row.mass)

/-- The list adapter uses bounded indexing, never a default for a missing entry. -/
def normalizeValues (xs : List Rat) : Except Error (List Rat) :=
  rowValues (normalize xs.get)

/-- A successful normalization certifies its actual input conditions. -/
theorem normalize_conditions {n : Nat} {w : Fin n → Rat} {row : Row n}
    (success : normalize w = .ok row) :
    (∀ i, 0 ≤ w i) ∧ 0 < total w := by
  unfold normalize at success
  split at success
  · rename_i h
    split at success
    · rename_i hz
      exact ⟨h, hz⟩
    · cases success
  · cases success

/-- Success identifies each output mass with the input's actual ratio. -/
theorem normalize_mass_eq {n : Nat} {w : Fin n → Rat} {row : Row n}
    (success : normalize w = .ok row) (i : Fin n) :
    row.mass i = w i / total w := by
  unfold normalize at success
  split at success
  · split at success
    · injection success with rowEq
      rw [← rowEq]
      rfl
    · cases success
  · cases success

/-- Sum-one follows from the ratios and positive denominator, not an oracle. -/
theorem normalize_sum_one {n : Nat} {w : Fin n → Rat} {row : Row n}
    (success : normalize w = .ok row) : (∑ i, row.mass i) = 1 := by
  calc
    (∑ i, row.mass i) = ∑ i, w i / total w :=
      Finset.sum_congr rfl (fun i _ => normalize_mass_eq success i)
    _ = total w / total w := by rw [← Finset.sum_div]; rfl
    _ = 1 := div_self (ne_of_gt (normalize_conditions success).2)

/-- Positive probability is equivalent to positive input weight at that index. -/
theorem normalize_support {n : Nat} {w : Fin n → Rat} {row : Row n}
    (success : normalize w = .ok row) (i : Fin n) :
    (0 < row.mass i ↔ 0 < w i) := by
  rw [normalize_mass_eq success i]
  exact div_pos_iff_of_pos_right (normalize_conditions success).2

/-- Successful masses are nonnegative, including masked zero-weight entries. -/
theorem normalize_nonneg {n : Nat} {w : Fin n → Rat} {row : Row n}
    (success : normalize w = .ok row) (i : Fin n) : 0 ≤ row.mass i := by
  rw [normalize_mass_eq success i]
  exact div_nonneg ((normalize_conditions success).1 i)
    (le_of_lt (normalize_conditions success).2)

/-- All valid weight vectors are accepted, and no invalid vector is accepted. -/
theorem normalize_exists_iff {n : Nat} (w : Fin n → Rat) :
    (∃ row, normalize w = .ok row) ↔ (∀ i, 0 ≤ w i) ∧ 0 < total w := by
  constructor
  · rintro ⟨row, success⟩
    exact normalize_conditions success
  · rintro ⟨h, hz⟩
    exact ⟨normalizePositive w h hz, by simp [normalize, h, hz]⟩

/-- For nonnegative weights, zero total means every weight is zero. -/
theorem total_eq_zero_iff {n : Nat} (w : Fin n → Rat) (h : ∀ i, 0 ≤ w i) :
    total w = 0 ↔ ∀ i, w i = 0 := by
  constructor
  · intro hz i
    have hi : w i ≤ total w :=
      Finset.single_le_sum (fun j _ => h j) (Finset.mem_univ i)
    rw [hz] at hi
    exact le_antisymm hi (h i)
  · intro hw
    simp [total, hw]

/-- Converting a checked row to a list preserves the finite carrier size. -/
theorem rowValues_length {n : Nat} {result : Except Error (Row n)}
    {values : List Rat} (success : rowValues result = .ok values) :
    values.length = n := by
  cases result with
  | error err => simp [rowValues] at success
  | ok row =>
    simp only [rowValues, Except.ok.injEq] at success
    rw [← success]
    simp

/-- A successful list normalization preserves its input dimension. -/
theorem normalizeValues_length {xs ys : List Rat}
    (success : normalizeValues xs = .ok ys) : ys.length = xs.length :=
  rowValues_length (result := normalize xs.get) success

end NarrativeDynamics.FitnessAttachment.Internal
