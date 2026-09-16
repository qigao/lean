import NarrativeDynamics.Core.MarginalizedTemporalCredit

open Finset

namespace NarrativeDynamics.MarginalizedTemporalCredit

example : registeredLaw.support = {1, 3, 5} := by
  norm_num [registeredLaw]

example : registeredLaw.weight 1 = (1 : ℝ) / 3 := by
  norm_num [registeredLaw]

example : registeredLaw.weight 3 = (1 : ℝ) / 3 := by
  norm_num [registeredLaw]

example : registeredLaw.weight 5 = (1 : ℝ) / 3 := by
  norm_num [registeredLaw]

example : registeredLaw.Valid := by
  exact registered_law_valid

example : immediateLaw.Valid := by
  exact immediate_law_valid

example (r : Nat → ℝ) (n t : Nat) :
    expectedAggregate registeredLaw r n t =
      ∑ d ∈ registeredLaw.support,
        if d ≤ t ∧ t - d < n then registeredLaw.weight d * r (t - d) else 0 := by
  exact expected_aggregate_decomposition registeredLaw r n t

example (law : DelayLaw) (r : Nat → ℝ) (n t j : Nat)
    (h : ∀ d ∈ law.support, ¬ (d ≤ t ∧ t - d = j ∧ j < n)) :
    candidateCredit law r n t j = 0 := by
  exact candidate_support_bounded law r n t j h

example : expectedAggregate registeredLaw (fun i => (i : ℝ)) 4 0 = 0 := by
  norm_num [expectedAggregate, registeredLaw, validSource]

example : expectedAggregate registeredLaw (fun i => (i : ℝ)) 4 8 = 1 := by
  norm_num [expectedAggregate, registeredLaw, validSource]

end NarrativeDynamics.MarginalizedTemporalCredit
