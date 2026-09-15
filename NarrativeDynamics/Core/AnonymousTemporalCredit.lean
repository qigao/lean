import NarrativeDynamics.Core.TemporalCredit

namespace NarrativeDynamics.AnonymousTemporalCredit

structure Delivery where
  source : Nat
  reward : ℝ
  deriving DecidableEq

abbrev DeliveryBucket := List Delivery
abbrev DeliveryHistory := List DeliveryBucket

def aggregateBucket (bucket : DeliveryBucket) : ℝ :=
  (bucket.map Delivery.reward).sum

def aggregateStream (history : DeliveryHistory) : List ℝ :=
  history.map aggregateBucket

structure LearnerView (Obs : Type) where
  observations : List Obs
  feedback : List ℝ
  deriving DecidableEq

def learnerView (observations : List Obs) (history : DeliveryHistory) : LearnerView Obs :=
  { observations := observations, feedback := aggregateStream history }

def relabelDelivery (f : Nat → Nat) (delivery : Delivery) : Delivery :=
  { delivery with source := f delivery.source }

def relabelSources (f : Nat → Nat) (history : DeliveryHistory) : DeliveryHistory :=
  history.map (List.map (relabelDelivery f))

theorem aggregate_view_source_noninterference
    (observations : List Obs)
    (left right : DeliveryHistory)
    (h : aggregateStream left = aggregateStream right) :
    learnerView observations left = learnerView observations right := by
  simp [learnerView, h]

theorem aggregate_conservation (history : DeliveryHistory) :
    (aggregateStream history).sum =
      (history.flatten.map Delivery.reward).sum := by
  induction history with
  | nil => rfl
  | cons bucket rest ih =>
      simp [aggregateStream, aggregateBucket, ih]

theorem aggregate_stream_relabel_sources
    (f : Nat → Nat) (history : DeliveryHistory) :
    aggregateStream (relabelSources f history) = aggregateStream history := by
  induction history with
  | nil => rfl
  | cons bucket rest ih =>
      simp [relabelSources, aggregateStream, aggregateBucket, relabelDelivery, ih]

theorem learner_view_source_relabel_invariant
    (observations : List Obs) (history : DeliveryHistory) (f : Nat → Nat) :
    learnerView observations (relabelSources f history) = learnerView observations history := by
  exact aggregate_view_source_noninterference observations _ _
    (aggregate_stream_relabel_sources f history)

end NarrativeDynamics.AnonymousTemporalCredit
