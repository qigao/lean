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

end NarrativeDynamics.AnonymousTemporalCredit.Tests
