import NarrativeDynamics.Core.AnonymousTemporalCredit

open Finset

namespace NarrativeDynamics.MarginalizedTemporalCredit

structure DelayLaw where
  support : Finset Nat
  weight : Nat → ℝ

namespace DelayLaw

def Valid (law : DelayLaw) : Prop :=
  (∀ d, 0 ≤ law.weight d) ∧
  (∀ d, d ∉ law.support → law.weight d = 0) ∧
  (∑ d ∈ law.support, law.weight d) = 1

end DelayLaw

noncomputable def registeredLaw : DelayLaw :=
  { support := {1, 3, 5}
    weight := fun d => if d = 1 ∨ d = 3 ∨ d = 5 then (1 : ℝ) / 3 else 0 }

noncomputable def immediateLaw : DelayLaw :=
  { support := {0}
    weight := fun d => if d = 0 then 1 else 0 }

def validSource (n t d : Nat) : Prop :=
  d ≤ t ∧ t - d < n

instance validSourceDecidable (n t d : Nat) : Decidable (validSource n t d) := by
  unfold validSource
  infer_instance

noncomputable def expectedAggregate
    (law : DelayLaw) (reward : Nat → ℝ) (n t : Nat) : ℝ :=
  ∑ d ∈ law.support,
    if validSource n t d then law.weight d * reward (t - d) else 0

noncomputable def candidateCredit
    (law : DelayLaw) (credit : Nat → ℝ) (n t j : Nat) : ℝ :=
  ∑ d ∈ law.support,
    if validSource n t d ∧ t - d = j then law.weight d * credit j else 0

theorem registered_law_valid : registeredLaw.Valid := by
  constructor
  · intro d
    by_cases h : d = 1 ∨ d = 3 ∨ d = 5
    · simp [registeredLaw, h]
    · simp [registeredLaw, h]
  · constructor
    · intro d hd
      have hne : ¬ (d = 1 ∨ d = 3 ∨ d = 5) := by
        intro h
        apply hd
        simpa [registeredLaw] using h
      simp [registeredLaw, hne]
    · norm_num [registeredLaw]

theorem immediate_law_valid : immediateLaw.Valid := by
  constructor
  · intro d
    by_cases h : d = 0
    · simp [immediateLaw, h]
    · simp [immediateLaw, h]
  · constructor
    · intro d hd
      have hne : d ≠ 0 := by
        intro h
        subst d
        apply hd
        simp [immediateLaw]
      simp [immediateLaw, hne]
    · norm_num [immediateLaw]

theorem expected_aggregate_decomposition
    (law : DelayLaw) (reward : Nat → ℝ) (n t : Nat) :
    expectedAggregate law reward n t =
      ∑ d ∈ law.support,
        if d ≤ t ∧ t - d < n then law.weight d * reward (t - d) else 0 := by
  unfold expectedAggregate
  apply Finset.sum_congr rfl
  intro d hd
  by_cases h : d ≤ t ∧ t - d < n
  · simp [validSource, h]
  · simp [validSource, h]

theorem candidate_support_bounded
    (law : DelayLaw) (credit : Nat → ℝ) (n t j : Nat)
    (h : ∀ d ∈ law.support, ¬ (d ≤ t ∧ t - d = j ∧ j < n)) :
    candidateCredit law credit n t j = 0 := by
  unfold candidateCredit
  apply Finset.sum_eq_zero
  intro d hd
  have hnot : ¬ (validSource n t d ∧ t - d = j) := by
    intro hs
    rcases hs with ⟨⟨hdt, hlt⟩, heq⟩
    apply h d hd
    refine ⟨hdt, heq, ?_⟩
    simpa [heq] using hlt
  simp [hnot]

end NarrativeDynamics.MarginalizedTemporalCredit
