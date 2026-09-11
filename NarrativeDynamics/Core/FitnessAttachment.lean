import Mathlib
import NarrativeDynamics.Core.SmallWorldMetrics

/-!
# Exact finite fitness attachment and target laws

Checked rational rows and adjacency-derived, fixed-fitness attachment laws.
Complete target laws use sequential selection without replacement.
Graph growth and multi-birth replay belong to subsequent tasks.
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

namespace NarrativeDynamics.FitnessAttachment

open Internal
open scoped BigOperators

/-- One immutable graph/fitness snapshot; degrees are never stored separately. -/
structure Snapshot (n : Nat) where
  graph : SimpleGraph (Fin n)
  adjDec : DecidableRel graph.Adj
  fitness : Fin n → Rat

/-- Valid seeds and their later snapshots have positive fitness and no isolates. -/
def Snapshot.Valid {n : Nat} (s : Snapshot n) : Prop :=
  2 ≤ n ∧ s.graph.Connected ∧ ∀ i, 0 < s.fitness i

/-- A snapshot whose growth-profile assumptions have been established. -/
structure State (n : Nat) where
  snapshot : Snapshot n
  valid : snapshot.Valid

/-- Count actual neighbors using the snapshot's executable adjacency decision. -/
def degree {n : Nat} (s : Snapshot n) (i : Fin n) : Nat :=
  letI := s.adjDec
  (NarrativeDynamics.neighborSet s.graph.Adj i).card

/-- The BB weight is fixed fitness times the degree of this graph. -/
def weights {n : Nat} (s : Snapshot n) (i : Fin n) : Rat :=
  s.fitness i * (degree s i : Rat)

private theorem degree_congr {n : Nat} (s t : Snapshot n)
    (sameGraph : t.graph = s.graph) (i : Fin n) : degree t i = degree s i := by
  unfold degree
  congr 1
  ext j
  simp only [NarrativeDynamics.neighborSet, Finset.mem_filter, Finset.mem_univ, true_and]
  rw [sameGraph]

/-- Connectedness on at least two vertices supplies a genuine incident edge. -/
theorem state_degree_pos {n : Nat} (s : State n) (i : Fin n) :
    0 < degree s.snapshot i := by
  have hn : 2 ≤ n := s.valid.1
  letI : Nontrivial (Fin n) :=
    ⟨⟨⟨0, by omega⟩, ⟨1, by omega⟩, by
      intro h
      have hv := congrArg Fin.val h
      norm_num at hv⟩⟩
  obtain ⟨j, hj⟩ := exists_ne i
  obtain ⟨k, hk⟩ := (s.valid.2.1 i j).nonempty_neighborSet_left hj.symm
  letI := s.snapshot.adjDec
  change 0 < (Finset.univ.filter (s.snapshot.graph.Adj i)).card
  exact Finset.card_pos.mpr ⟨k, Finset.mem_filter.mpr ⟨Finset.mem_univ k, hk⟩⟩

private theorem weight_pos {n : Nat} (s : State n) (i : Fin n) :
    0 < weights s.snapshot i := by
  exact mul_pos (s.valid.2.2 i) (by exact_mod_cast state_degree_pos s i)

private theorem masked_nonneg {n : Nat} (s : State n)
    (S : Finset (Fin n)) (i : Fin n) : 0 ≤ remaining (weights s.snapshot) S i := by
  by_cases hi : i ∈ S
  · simp [remaining, hi]
  · simpa [remaining, hi] using (weight_pos s i).le

/-- Every proper selected set leaves at least one strictly positive weight. -/
theorem remaining_total_pos {n : Nat} (s : State n)
    (S : Finset (Fin n)) (hS : S.card < n) :
    0 < total (remaining (weights s.snapshot) S) := by
  have hex : ∃ i : Fin n, i ∉ S := by
    by_contra! allIn
    have hsub : Finset.univ ⊆ S := fun i _ => allIn i
    have hc := Finset.card_le_card hsub
    simp only [Finset.card_univ, Fintype.card_fin] at hc
    omega
  obtain ⟨i, hi⟩ := hex
  have hp : 0 < remaining (weights s.snapshot) S i := by
    simpa [remaining, hi] using weight_pos s i
  exact lt_of_lt_of_le hp
    (Finset.single_le_sum (fun j _ => masked_nonneg s S j) (Finset.mem_univ i))

/-- One conditional draw from remaining old vertices of a frozen snapshot. -/
def attachmentRow {n : Nat} (s : State n) (S : Finset (Fin n))
    (hS : S.card < n) : Row n :=
  normalizePositive (remaining (weights s.snapshot) S)
    (masked_nonneg s S) (remaining_total_pos s S hS)

/-- The conditional row uses the actual masked numerator and denominator. -/
theorem attachment_mass {n : Nat} (s : State n) (S : Finset (Fin n))
    (hS : S.card < n) (i : Fin n) :
    (attachmentRow s S hS).mass i =
      remaining (weights s.snapshot) S i / total (remaining (weights s.snapshot) S) := rfl

/-- Positive support is exactly the unselected old vertices. -/
theorem attachment_support {n : Nat} (s : State n) (S : Finset (Fin n))
    (hS : S.card < n) (i : Fin n) :
    (0 < (attachmentRow s S hS).mass i ↔ i ∉ S) := by
  rw [attachment_mass, div_pos_iff_of_pos_right (remaining_total_pos s S hS)]
  by_cases hi : i ∈ S
  · simp [remaining, hi]
  · simp [remaining, hi, weight_pos s i]

/-- BA is the same model with unit fitness, retaining the very same graph. -/
def asBA {n : Nat} (s : State n) : State n where
  snapshot := { s.snapshot with fitness := fun _ => 1 }
  valid := ⟨s.valid.1, s.valid.2.1, fun _ => by norm_num⟩

/-- Common positive rescaling changes fitness data, not the attachment law. -/
def scaleFitness {n : Nat} (s : State n) (c : PosFitness) : State n where
  snapshot := { s.snapshot with fitness := fun i => c.val * s.snapshot.fitness i }
  valid := ⟨s.valid.1, s.valid.2.1, fun i => mul_pos c.property (s.valid.2.2 i)⟩

/-- Degree-only BA uses the BB constructor, not a second normalization algorithm. -/
def baRow {n : Nat} (s : State n) (S : Finset (Fin n)) (hS : S.card < n) : Row n :=
  attachmentRow (asBA s) S hS

private theorem attachment_proportional {n : Nat} (s t : State n)
    (S : Finset (Fin n)) (hS : S.card < n) (c : PosFitness)
    (scaled : ∀ j, weights t.snapshot j = c.val * weights s.snapshot j) (i : Fin n) :
    (attachmentRow t S hS).mass i = (attachmentRow s S hS).mass i := by
  have hw : ∀ j, remaining (weights t.snapshot) S j =
      c.val * remaining (weights s.snapshot) S j := by
    intro j
    by_cases hj : j ∈ S
    · simp [remaining, hj]
    · simp only [remaining, if_neg hj, scaled j]
  have ht : total (remaining (weights t.snapshot) S) =
      c.val * total (remaining (weights s.snapshot) S) := by
    unfold total
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun j _ => hw j)
  rw [attachment_mass, attachment_mass, hw i, ht]
  exact mul_div_mul_left _ _ (ne_of_gt c.property)

/-- Every constant positive fitness profile reduces to BA on the same candidates. -/
theorem attachment_ba {n : Nat} (s : State n) (S : Finset (Fin n))
    (hS : S.card < n) (c : PosFitness)
    (constant : ∀ j, s.snapshot.fitness j = c.val) (i : Fin n) :
    (attachmentRow s S hS).mass i = (baRow s S hS).mass i := by
  apply attachment_proportional (asBA s) s S hS c
  intro j
  change s.snapshot.fitness j * (degree s.snapshot j : Rat) =
    c.val * (1 * (degree s.snapshot j : Rat))
  rw [constant j, one_mul]

/-- One common positive scale cancels from every conditional probability. -/
theorem attachment_scale {n : Nat} (s : State n) (S : Finset (Fin n))
    (hS : S.card < n) (c : PosFitness) (i : Fin n) :
    (attachmentRow (scaleFitness s c) S hS).mass i = (attachmentRow s S hS).mass i := by
  apply attachment_proportional s (scaleFitness s c) S hS c
  intro j
  change (c.val * s.snapshot.fitness j) * (degree s.snapshot j : Rat) =
    c.val * (s.snapshot.fitness j * (degree s.snapshot j : Rat))
  exact mul_assoc _ _ _

private theorem attachment_unselected {n : Nat} (s : State n)
    (S : Finset (Fin n)) (hS : S.card < n) (i : Fin n) (hi : i ∉ S) :
    (attachmentRow s S hS).mass i =
      weights s.snapshot i / total (remaining (weights s.snapshot) S) := by
  rw [attachment_mass]
  rw [show remaining (weights s.snapshot) S i = weights s.snapshot i from if_neg hi]

/-- Eligible vertices' probability ratio is their fitness-weighted degree ratio. -/
theorem attachment_ratio {n : Nat} (s : State n) (S : Finset (Fin n))
    (hS : S.card < n) (i j : Fin n) (hi : i ∉ S) (hj : j ∉ S) :
    (attachmentRow s S hS).mass i / (attachmentRow s S hS).mass j =
      weights s.snapshot i / weights s.snapshot j := by
  rw [attachment_unselected s S hS i hi, attachment_unselected s S hS j hj]
  exact div_div_div_cancel_right₀ (ne_of_gt (remaining_total_pos s S hS)) _ _

private theorem total_remaining_split {n : Nat} (w : Fin n → Rat)
    (S : Finset (Fin n)) (i : Fin n) (hi : i ∉ S) :
    total (remaining w S) = w i + total (remaining w (insert i S)) := by
  calc
    total (remaining w S) =
        ∑ j, ((if j = i then w i else 0) + remaining w (insert i S) j) := by
      apply Finset.sum_congr rfl
      intro j _
      by_cases hj : j = i
      · subst j
        simp [remaining, hi]
      · simp [remaining, hj]
    _ = w i + total (remaining w (insert i S)) := by
      rw [Finset.sum_add_distrib]
      simp [total]

private theorem competing_mass_congr {n : Nat} (s t : State n)
    (S : Finset (Fin n)) (i : Fin n) (sameGraph : t.snapshot.graph = s.snapshot.graph)
    (others : ∀ j, j ≠ i → t.snapshot.fitness j = s.snapshot.fitness j) :
    total (remaining (weights t.snapshot) (insert i S)) =
      total (remaining (weights s.snapshot) (insert i S)) := by
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : j ∈ insert i S
  · simp [remaining, hj]
  · have hji : j ≠ i := fun h => hj (Finset.mem_insert.mpr (Or.inl h))
    simp only [remaining, if_neg hj, weights, others j hji,
      degree_congr s.snapshot t.snapshot sameGraph j]

/-- Raising only one eligible vertex's fitness cannot decrease its probability. -/
theorem attachment_mono {n : Nat} (s t : State n) (S : Finset (Fin n))
    (hS : S.card < n) (i : Fin n) (sameGraph : t.snapshot.graph = s.snapshot.graph)
    (others : ∀ j, j ≠ i → t.snapshot.fitness j = s.snapshot.fitness j)
    (hi : i ∉ S) (increase : s.snapshot.fitness i ≤ t.snapshot.fitness i) :
    (attachmentRow s S hS).mass i ≤ (attachmentRow t S hS).mass i := by
  have hw : weights s.snapshot i ≤ weights t.snapshot i := by
    unfold weights
    rw [degree_congr s.snapshot t.snapshot sameGraph i]
    exact mul_le_mul_of_nonneg_right increase (Nat.cast_nonneg _)
  have hb : 0 ≤ total (remaining (weights s.snapshot) (insert i S)) :=
    Finset.sum_nonneg (fun j _ => masked_nonneg s (insert i S) j)
  rw [attachment_unselected s S hS i hi, attachment_unselected t S hS i hi]
  apply (div_le_div_iff₀ (remaining_total_pos s S hS) (remaining_total_pos t S hS)).2
  rw [total_remaining_split (weights s.snapshot) S i hi,
    total_remaining_split (weights t.snapshot) S i hi,
    competing_mass_congr s t S i sameGraph others]
  nlinarith [mul_nonneg (sub_nonneg.mpr hw) hb]

/-- Strict gain requires a real competitor; a sole eligible vertex stays at one. -/
theorem attachment_strict_mono {n : Nat} (s t : State n) (S : Finset (Fin n))
    (hS : S.card < n) (i : Fin n) (sameGraph : t.snapshot.graph = s.snapshot.graph)
    (others : ∀ j, j ≠ i → t.snapshot.fitness j = s.snapshot.fitness j)
    (hi : i ∉ S) (increase : s.snapshot.fitness i < t.snapshot.fitness i)
    (competitor : ∃ j, j ∉ S ∧ j ≠ i) :
    (attachmentRow s S hS).mass i < (attachmentRow t S hS).mass i := by
  have hw : weights s.snapshot i < weights t.snapshot i := by
    unfold weights
    rw [degree_congr s.snapshot t.snapshot sameGraph i]
    exact mul_lt_mul_of_pos_right increase (by exact_mod_cast state_degree_pos s i)
  obtain ⟨j, hj, hji⟩ := competitor
  have hj' : j ∉ insert i S := by simp [Finset.mem_insert, hj, hji]
  have hp : 0 < remaining (weights s.snapshot) (insert i S) j := by
    simpa [remaining, hj'] using weight_pos s j
  have hb : 0 < total (remaining (weights s.snapshot) (insert i S)) :=
    lt_of_lt_of_le hp (Finset.single_le_sum
      (fun k _ => masked_nonneg s (insert i S) k) (Finset.mem_univ j))
  rw [attachment_unselected s S hS i hi, attachment_unselected t S hS i hi]
  apply (div_lt_div_iff₀ (remaining_total_pos s S hS) (remaining_total_pos t S hS)).2
  rw [total_remaining_split (weights s.snapshot) S i hi,
    total_remaining_split (weights t.snapshot) S i hi,
    competing_mass_congr s t S i sameGraph others]
  nlinarith [mul_pos (sub_pos.mpr hw) hb]

end NarrativeDynamics.FitnessAttachment

namespace NarrativeDynamics.FitnessAttachment

open Internal
open scoped BigOperators

/-- Ordered, distinct old targets; the finite instance enumerates actual embeddings. -/
abbrev Targets (n m : Nat) := Fin m ↪ Fin n

/-- Enumerate finite functions and retain exactly the injective ones, without choice.
Mathlib's general embedding instance is noncomputable; these carriers have decidable equality. -/
instance Targets.instFintype (n m : Nat) : Fintype (Targets n m) :=
  Fintype.ofEquiv { f : Fin m → Fin n // Function.Injective f }
    (Equiv.subtypeInjectiveEquivEmbedding (Fin m) (Fin n))

/-- The order matters to the conditional probability, even for the same target set. -/
def Targets.ordered {n m : Nat} (T : Targets n m) : List (Fin n) := List.ofFn T

/-- Forget order only when forming the corresponding unordered event. -/
def Targets.selected {n m : Nat} (T : Targets n m) : Finset (Fin n) :=
  Finset.univ.image T

/-- Injectivity, not a separately assumed count, fixes the target-set cardinality. -/
theorem selected_card {n m : Nat} (T : Targets n m) : T.selected.card = m := by
  unfold Targets.selected
  rw [Finset.card_image_of_injective _ T.injective]
  simp

namespace Internal

/-- A finite mathematical event mass. Impossible draws have mass zero, not API success.
The empty continuation has mass one, including after the last candidate is used. -/
def traceMass {n : Nat} (w : Fin n → Rat) (S : Finset (Fin n)) : List (Fin n) → Rat
  | [] => 1
  | i :: xs =>
    if i ∈ S then 0
    else if total (remaining w S) ≤ 0 then 0
    else (w i / total (remaining w S)) * traceMass w (insert i S) xs

/-- Splitting the executable guards leaves the original event recurrence unchanged. -/
theorem traceMass_cons_eq {n : Nat} (w : Fin n → Rat) (S : Finset (Fin n))
    (i : Fin n) (xs : List (Fin n)) :
    traceMass w S (i :: xs) =
      if i ∈ S ∨ total (remaining w S) ≤ 0 then 0
      else (w i / total (remaining w S)) * traceMass w (insert i S) xs := by
  by_cases hi : i ∈ S <;> by_cases hz : total (remaining w S) ≤ 0 <;>
    simp [traceMass, hi, hz]

/-- Legal continuations avoid the already-selected prefix as well as each other. -/
abbrev Continuation (n m : Nat) (S : Finset (Fin n)) :=
  { T : Targets n m // ∀ j, T j ∉ S }

/-- Exact first-choice/tail partition: no ordering is omitted or counted twice. -/
def splitContinuation {n m : Nat} (S : Finset (Fin n)) :
    Continuation n (m + 1) S ≃
      (Σ i : { j : Fin n // j ∉ S }, Continuation n m (insert i.val S)) where
  toFun T := ⟨⟨T.val 0, T.property 0⟩, ⟨Fin.Embedding.tail T.val, by
    intro j hj
    rcases Finset.mem_insert.mp hj with heq | hin
    · exact Fin.succ_ne_zero j (T.val.injective heq)
    · exact T.property j.succ hin⟩⟩
  invFun z := ⟨Fin.Embedding.cons z.2.val (by
    rintro ⟨j, hj⟩
    exact z.2.property j (Finset.mem_insert.mpr (Or.inl hj))), by
    intro j
    refine Fin.cases ?_ ?_ j
    · exact z.1.property
    · intro k hk
      exact z.2.property k (Finset.mem_insert_of_mem hk)⟩
  left_inv T := by
    apply Subtype.ext
    ext j
    refine Fin.cases ?_ ?_ j
    · rfl
    · intro k
      rfl
  right_inv := by
    rintro ⟨i, T⟩
    rfl

private theorem remaining_pos_at {n : Nat} (w : Fin n → Rat)
    (hw : ∀ j, 0 < w j) (S : Finset (Fin n)) (i : Fin n) (hi : i ∉ S) :
    0 < total (remaining w S) := by
  have hn : ∀ j, 0 ≤ remaining w S j := by
    intro j
    by_cases hj : j ∈ S
    · simp [remaining, hj]
    · simpa [remaining, hj] using (hw j).le
  have hp : 0 < remaining w S i := by simpa [remaining, hi] using hw i
  exact lt_of_lt_of_le hp
    (Finset.single_le_sum (fun j _ => hn j) (Finset.mem_univ i))

private theorem remaining_pos_proper {n : Nat} (w : Fin n → Rat)
    (hw : ∀ j, 0 < w j) (S : Finset (Fin n)) (hS : S.card < n) :
    0 < total (remaining w S) := by
  have hex : ∃ i : Fin n, i ∉ S := by
    by_contra! allIn
    have hc := Finset.card_le_card (show Finset.univ ⊆ S from fun i _ => allIn i)
    simp only [Finset.card_univ, Fintype.card_fin] at hc
    omega
  obtain ⟨i, hi⟩ := hex
  exact remaining_pos_at w hw S i hi

/-- Restricting the sum to eligible IDs is exactly the zero-mask sum. -/
private theorem sum_eligible {n : Nat} (w : Fin n → Rat) (S : Finset (Fin n)) :
    (∑ i : { j : Fin n // j ∉ S }, w i.val) = total (remaining w S) := by
  have h := Fintype.sum_subtype_add_sum_subtype
    (fun i : Fin n => i ∉ S) (remaining w S)
  have good : (∑ i : { j : Fin n // j ∉ S }, remaining w S i.val) =
      ∑ i : { j : Fin n // j ∉ S }, w i.val := by
    apply Finset.sum_congr rfl
    intro i _
    exact if_neg i.property
  have bad : (∑ i : { j : Fin n // ¬ j ∉ S }, remaining w S i.val) = 0 := by
    apply Finset.sum_eq_zero
    intro i _
    exact if_pos (not_not.mp i.property)
  rw [good, bad, add_zero] at h
  exact h

/-- The partition also preserves the actual conditional-product computation. -/
theorem split_mass {n m : Nat} (w : Fin n → Rat) (hw : ∀ j, 0 < w j)
    (S : Finset (Fin n))
    (z : Σ i : { j : Fin n // j ∉ S }, Continuation n m (insert i.val S)) :
    traceMass w S ((splitContinuation S).symm z).val.ordered =
      (w z.1.val / total (remaining w S)) *
        traceMass w (insert z.1.val S) z.2.val.ordered := by
  have hz := remaining_pos_at w hw S z.1.val z.1.property
  change traceMass w S (List.ofFn (Fin.cons z.1.val z.2.val)) = _
  rw [List.ofFn_cons]
  simp only [traceMass, z.1.property, not_le.mpr hz, if_false, Targets.ordered]

/-- Every legal continuation has positive probability under positive weights. -/
theorem continuation_pos {n : Nat} (w : Fin n → Rat) (hw : ∀ j, 0 < w j)
    (m : Nat) (S : Finset (Fin n)) (T : Continuation n m S) :
    0 < traceMass w S T.val.ordered := by
  induction m generalizing S with
  | zero => simp [Targets.ordered, traceMass]
  | succ m ih =>
    obtain ⟨z, rfl⟩ := (splitContinuation (m := m) S).symm.surjective T
    rw [split_mass w hw S z]
    exact mul_pos
      (div_pos (hw z.1.val) (remaining_pos_at w hw S z.1.val z.1.property))
      (ih (insert z.1.val S) z.2)

/-- Complete normalization is proved by the head/tail bijection and prefix induction.
The dimension bound is essential; the empty continuation does not normalize a row. -/
theorem continuation_sum_one {n : Nat} (w : Fin n → Rat) (hw : ∀ j, 0 < w j)
    (m : Nat) (S : Finset (Fin n)) (bound : S.card + m ≤ n) :
    (∑ T : Continuation n m S, traceMass w S T.val.ordered) = 1 := by
  induction m generalizing S with
  | zero =>
    let z : Continuation n 0 S :=
      ⟨⟨Fin.elim0, by intro i; exact i.elim0⟩, by intro i; exact i.elim0⟩
    letI : Subsingleton (Continuation n 0 S) := ⟨by
      intro a b
      apply Subtype.ext
      ext i
      exact i.elim0⟩
    rw [Fintype.sum_subsingleton _ z]
    simp [Targets.ordered, traceMass]
  | succ m ih =>
    have hS : S.card < n := by omega
    let e := splitContinuation (n := n) (m := m) S
    rw [← e.symm.sum_comp (fun T : Continuation n (m + 1) S =>
      traceMass w S T.val.ordered)]
    rw [Fintype.sum_sigma]
    change (∑ i : { j : Fin n // j ∉ S },
      ∑ T : Continuation n m (insert i.val S),
        traceMass w S ((splitContinuation S).symm ⟨i, T⟩).val.ordered) = 1
    simp_rw [split_mass w hw S, ← Finset.mul_sum]
    calc
      (∑ i : { j : Fin n // j ∉ S },
          w i.val / total (remaining w S) *
            ∑ T : Continuation n m (insert i.val S),
              traceMass w (insert i.val S) T.val.ordered) =
          ∑ i : { j : Fin n // j ∉ S }, w i.val / total (remaining w S) := by
        apply Finset.sum_congr rfl
        intro i _
        have hb : (insert i.val S).card + m ≤ n := by
          rw [Finset.card_insert_of_notMem i.property]
          omega
        rw [ih (insert i.val S) hb, mul_one]
      _ = 1 := by
        rw [← Finset.sum_div, sum_eligible]
        exact div_self (ne_of_gt (remaining_pos_proper w hw S hS))

/-- At the empty prefix, every target embedding is a legal continuation. -/
def emptyContinuationEquiv {n m : Nat} : Targets n m ≃ Continuation n m ∅ where
  toFun T := ⟨T, by intro j; simp⟩
  invFun T := T.val
  left_inv _ := rfl
  right_inv _ := rfl

/-- Common positive scaling cancels at every conditional draw, not only the first. -/
theorem traceMass_scale {n : Nat} (w : Fin n → Rat) (c : PosFitness)
    (S : Finset (Fin n)) (xs : List (Fin n)) :
    traceMass (fun i => c.val * w i) S xs = traceMass w S xs := by
  induction xs generalizing S with
  | nil => rfl
  | cons i xs ih =>
    have ht : total (remaining (fun j => c.val * w j) S) =
        c.val * total (remaining w S) := by
      unfold total
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      by_cases hj : j ∈ S <;> simp [remaining, hj]
    by_cases hi : i ∈ S
    · simp [traceMass, hi]
    · by_cases hz : 0 < total (remaining w S)
      · have hz' := mul_pos c.property hz
        simp only [traceMass, ht, hi, not_le.mpr hz,
          not_le.mpr hz', if_false]
        rw [mul_div_mul_left _ _ (ne_of_gt c.property), ih]
      · have hz0 := le_of_not_gt hz
        have hz0' := mul_nonpos_of_nonneg_of_nonpos c.property.le hz0
        simp only [traceMass, ht, hi, hz0, hz0', if_false, if_true]

end Internal

/-- The probability of this exact ordered target tuple on the frozen old graph. -/
def orderedMass {n m : Nat} (s : State n) (T : Targets n m) : Rat :=
  traceMass (weights s.snapshot) ∅ T.ordered

/-- The mass of a set event sums all its orderings, including unequal ones. -/
def setMass {n : Nat} (s : State n) (m : Nat) (A : Finset (Fin n)) : Rat :=
  ∑ T : Targets n m, if T.selected = A then orderedMass s T else 0

theorem orderedMass_pos {n m : Nat} (s : State n) (T : Targets n m) :
    0 < orderedMass s T :=
  continuation_pos (weights s.snapshot) (weight_pos s) m ∅
    (emptyContinuationEquiv T)

theorem orderedMass_sum_one {n m : Nat} (s : State n) (hm : m ≤ n) :
    (∑ T : Targets n m, orderedMass s T) = 1 := by
  calc
    (∑ T : Targets n m, orderedMass s T) =
        ∑ U : Continuation n m ∅, traceMass (weights s.snapshot) ∅ U.val.ordered :=
      (emptyContinuationEquiv (n := n) (m := m)).sum_comp
        (fun U : Continuation n m ∅ => traceMass (weights s.snapshot) ∅ U.val.ordered)
    _ = 1 := continuation_sum_one (weights s.snapshot) (weight_pos s) m ∅ (by simpa using hm)

/-- Cardinality mismatches describe impossible events, never a renormalized law. -/
theorem setMass_wrong_card {n m : Nat} (s : State n) (A : Finset (Fin n))
    (hA : A.card ≠ m) : setMass s m A = 0 := by
  unfold setMass
  apply Finset.sum_eq_zero
  intro T _
  have hne : T.selected ≠ A := by
    intro he
    apply hA
    rw [← he]
    exact selected_card T
  exact if_neg hne

/-- Partitioning the complete ordered law by the selected set preserves mass one. -/
theorem setMass_sum_one {n m : Nat} (s : State n) (hm : m ≤ n) :
    (∑ A ∈ (Finset.univ : Finset (Fin n)).powersetCard m, setMass s m A) = 1 := by
  unfold setMass
  rw [Finset.sum_comm]
  calc
    (∑ T : Targets n m, ∑ A ∈ (Finset.univ : Finset (Fin n)).powersetCard m,
        if T.selected = A then orderedMass s T else 0) =
        ∑ T : Targets n m, orderedMass s T := by
      apply Finset.sum_congr rfl
      intro T _
      have hmem : T.selected ∈ (Finset.univ : Finset (Fin n)).powersetCard m := by
        simp [Finset.mem_powersetCard, selected_card]
      rw [Finset.sum_eq_single T.selected]
      · simp
      · intro B _ hB
        exact if_neg (Ne.symm hB)
      · intro hnot
        exact False.elim (hnot hmem)
    _ = 1 := orderedMass_sum_one s hm

/-- BA reduction holds for complete conditional products with constant fitness. -/
theorem orderedMass_ba {n m : Nat} (s : State n) (T : Targets n m)
    (c : PosFitness) (constant : ∀ i, s.snapshot.fitness i = c.val) :
    orderedMass s T = orderedMass (asBA s) T := by
  have hw : weights s.snapshot = fun i => c.val * weights (asBA s).snapshot i := by
    funext i
    change s.snapshot.fitness i * (degree s.snapshot i : Rat) =
      c.val * (1 * (degree s.snapshot i : Rat))
    rw [constant i, one_mul]
  unfold orderedMass
  rw [hw, traceMass_scale]

/-- A common scale preserves every complete finite target law. -/
theorem orderedMass_scale {n m : Nat} (s : State n) (T : Targets n m)
    (c : PosFitness) : orderedMass (scaleFitness s c) T = orderedMass s T := by
  have hw : weights (scaleFitness s c).snapshot = fun i => c.val * weights s.snapshot i := by
    funext i
    change (c.val * s.snapshot.fitness i) * (degree s.snapshot i : Rat) =
      c.val * (s.snapshot.fitness i * (degree s.snapshot i : Rat))
    exact mul_assoc _ _ _
  unfold orderedMass
  rw [hw, traceMass_scale]

end NarrativeDynamics.FitnessAttachment
