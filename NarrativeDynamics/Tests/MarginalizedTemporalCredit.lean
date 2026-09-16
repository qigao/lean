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

example (w z : Vector n) :
    marginalizedPrediction w z = dot w z := by
  exact current_weight_observation w z

example (w feature : Vector n) (denominator alpha reward : ℝ) :
    marginalizedUpdate w alpha reward feature
        (AnonymousTemporalCredit.normalizedCredit feature denominator) =
      AnonymousTemporalCredit.normalizedPhase3AUpdate
        w alpha reward (dot w feature)
        (AnonymousTemporalCredit.normalizedCredit feature denominator) := by
  exact immediate_reduction_to_phase3a w feature denominator alpha reward

example (law : DelayLaw) (x : Nat → Vector n) (count t : Nat)
    (h : ∀ d ∈ law.support, ¬ validSource count t d) :
    marginalizedFeature law x count t = zeroVector := by
  exact invalid_candidates_zero law x count t h

example (law : DelayLaw) (w : Vector n) (x c : Nat → Vector n)
    (alpha feedback : ℝ) (count t : Nat) (hDrain : count ≤ t) :
    marginalizedUpdate w alpha feedback
        (marginalizedFeature law x count t)
        (marginalizedFeature law c count t) =
      fun i =>
        w i + alpha *
          (feedback - dot w (marginalizedFeature law x count t)) *
          (marginalizedFeature law c count t) i := by
  exact drain_uses_same_equation law w x c alpha feedback count t hDrain

end NarrativeDynamics.MarginalizedTemporalCredit

#print axioms NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.current_weight_observation
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.drain_uses_same_equation
