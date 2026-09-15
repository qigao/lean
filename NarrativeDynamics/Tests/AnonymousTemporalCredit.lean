import NarrativeDynamics.Core.AnonymousTemporalCredit

open NarrativeDynamics.AnonymousTemporalCredit

namespace NarrativeDynamics.AnonymousTemporalCredit.Tests

private def multiplicityA : DeliveryHistory :=
  [[{ source := 0, reward := 1 }, { source := 1, reward := -1 }], []]

private def multiplicityB : DeliveryHistory :=
  [[], []]

private def reorderedA : DeliveryHistory :=
  [[{ source := 0, reward := 1 }, { source := 1, reward := 2 }]]

private def reorderedB : DeliveryHistory :=
  [[{ source := 99, reward := 2 }, { source := 42, reward := 1 }]]

example (obs : List Nat) (left right : DeliveryHistory)
    (h : aggregateStream left = aggregateStream right) :
    learnerView obs left = learnerView obs right := by
  exact aggregate_view_source_noninterference obs left right h

example (history : DeliveryHistory) :
    (aggregateStream history).sum =
      (history.flatten.map Delivery.reward).sum := by
  exact aggregate_conservation history

example : aggregateStream multiplicityA = aggregateStream multiplicityB := by
  norm_num [aggregateStream, aggregateBucket, multiplicityA, multiplicityB]

example : learnerView [10, 20] multiplicityA = learnerView [10, 20] multiplicityB := by
  apply aggregate_view_source_noninterference
  norm_num [aggregateStream, aggregateBucket, multiplicityA, multiplicityB]

example : learnerView [7] reorderedA = learnerView [7] reorderedB := by
  apply aggregate_view_source_noninterference
  norm_num [aggregateStream, aggregateBucket, reorderedA, reorderedB]

example (obs : List Nat) (history : DeliveryHistory) (f : Nat → Nat) :
    learnerView obs (relabelSources f history) = learnerView obs history := by
  exact learner_view_source_relabel_invariant obs history f

example (gamma lambda : ℝ) (d : Nat) :
    NarrativeDynamics.TemporalCredit.causalTraceCoeff gamma lambda d =
      (gamma * lambda) ^ d := by
  exact eligibility_historical_coefficient gamma lambda d

example (credit : Vector n) (prediction : ℝ) :
    advanceTrace 0 { eligibility := zeroVector, prediction := 0 }
        credit prediction =
      { eligibility := credit, prediction := prediction } := by
  exact zero_rho_trace_is_current_credit credit prediction

example
    (weights feature : Vector n)
    (denominator alpha reward prediction : ℝ) :
    let credit := normalizedCredit feature denominator
    let state := advanceTrace 0
      { eligibility := zeroVector, prediction := 0 }
      credit prediction
    anonymousEligibilityUpdate weights alpha reward state =
      normalizedPhase3AUpdate weights alpha reward prediction credit := by
  exact immediate_reduction_to_phase3a weights feature denominator alpha reward prediction

example :
    NarrativeDynamics.TemporalCredit.causalTraceCoeff ((9 : ℝ) / 10) ((8 : ℝ) / 10) 0 = 1 := by
  rw [eligibility_historical_coefficient]
  norm_num

example :
    NarrativeDynamics.TemporalCredit.causalTraceCoeff ((9 : ℝ) / 10) ((8 : ℝ) / 10) 1 =
      (18 : ℝ) / 25 := by
  rw [eligibility_historical_coefficient]
  norm_num

example :
    NarrativeDynamics.TemporalCredit.causalTraceCoeff ((9 : ℝ) / 10) ((8 : ℝ) / 10) 3 =
      (5832 : ℝ) / 15625 := by
  rw [eligibility_historical_coefficient]
  norm_num

example :
    NarrativeDynamics.TemporalCredit.causalTraceCoeff ((9 : ℝ) / 10) ((8 : ℝ) / 10) 5 =
      (1889568 : ℝ) / 9765625 := by
  rw [eligibility_historical_coefficient]
  norm_num

end NarrativeDynamics.AnonymousTemporalCredit.Tests
