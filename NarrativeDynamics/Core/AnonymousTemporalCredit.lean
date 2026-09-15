import NarrativeDynamics.Core.TemporalCredit

namespace NarrativeDynamics.AnonymousTemporalCredit

structure Delivery where
  source : Nat
  reward : ℝ

abbrev DeliveryBucket := List Delivery
abbrev DeliveryHistory := List DeliveryBucket

def aggregateBucket (bucket : DeliveryBucket) : ℝ :=
  (bucket.map Delivery.reward).sum

def aggregateStream (history : DeliveryHistory) : List ℝ :=
  history.map aggregateBucket

structure LearnerView (Obs : Type) where
  observations : List Obs
  feedback : List ℝ

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
      change aggregateBucket bucket + (aggregateStream rest).sum =
        (List.map Delivery.reward (bucket ++ rest.flatten)).sum
      rw [List.map_append, List.sum_append, aggregateBucket, ih]

theorem aggregateBucket_relabel
    (f : Nat → Nat) (bucket : DeliveryBucket) :
    aggregateBucket (bucket.map (relabelDelivery f)) = aggregateBucket bucket := by
  induction bucket with
  | nil => rfl
  | cons delivery rest ih =>
      change delivery.reward + aggregateBucket (rest.map (relabelDelivery f)) =
        delivery.reward + aggregateBucket rest
      rw [ih]

theorem aggregate_stream_relabel_sources
    (f : Nat → Nat) (history : DeliveryHistory) :
    aggregateStream (relabelSources f history) = aggregateStream history := by
  induction history with
  | nil => rfl
  | cons bucket rest ih =>
      change aggregateBucket (bucket.map (relabelDelivery f)) ::
          aggregateStream (relabelSources f rest) =
        aggregateBucket bucket :: aggregateStream rest
      rw [aggregateBucket_relabel, ih]

theorem learner_view_source_relabel_invariant
    (observations : List Obs) (history : DeliveryHistory) (f : Nat → Nat) :
    learnerView observations (relabelSources f history) = learnerView observations history := by
  exact aggregate_view_source_noninterference observations _ _
    (aggregate_stream_relabel_sources f history)

abbrev Vector (n : Nat) := Fin n → ℝ

structure TraceState (n : Nat) where
  eligibility : Vector n
  prediction : ℝ

def zeroVector : Vector n := fun _ => 0

noncomputable def normalizedCredit (feature : Vector n) (denominator : ℝ) : Vector n :=
  fun i => feature i / denominator

def advanceTrace
    (rho : ℝ) (state : TraceState n)
    (currentCredit : Vector n) (currentPrediction : ℝ) : TraceState n :=
  {
    eligibility := fun i => rho * state.eligibility i + currentCredit i
    prediction := rho * state.prediction + currentPrediction
  }

def normalizedPhase3AUpdate
    (weights : Vector n) (alpha reward prediction : ℝ)
    (credit : Vector n) : Vector n :=
  fun i => weights i + alpha * (reward - prediction) * credit i

def anonymousEligibilityUpdate
    (weights : Vector n) (alpha feedback : ℝ)
    (state : TraceState n) : Vector n :=
  fun i => weights i + alpha * (feedback - state.prediction) * state.eligibility i

theorem eligibility_historical_coefficient
    (gamma lambda : ℝ) (d : Nat) :
    NarrativeDynamics.TemporalCredit.causalTraceCoeff gamma lambda d =
      (gamma * lambda) ^ d := by
  exact NarrativeDynamics.TemporalCredit.causal_trace_coeff_closed_form gamma lambda d

theorem zero_rho_trace_is_current_credit
    (credit : Vector n) (prediction : ℝ) :
    advanceTrace 0 { eligibility := zeroVector, prediction := 0 }
        credit prediction =
      { eligibility := credit, prediction := prediction } := by
  simp [advanceTrace, zeroVector]

theorem immediate_reduction_to_phase3a
    (weights feature : Vector n)
    (denominator alpha reward prediction : ℝ) :
    let credit := normalizedCredit feature denominator
    let state := advanceTrace 0
      { eligibility := zeroVector, prediction := 0 }
      credit prediction
    anonymousEligibilityUpdate weights alpha reward state =
      normalizedPhase3AUpdate weights alpha reward prediction credit := by
  funext i
  simp [advanceTrace, zeroVector, anonymousEligibilityUpdate, normalizedPhase3AUpdate]

end NarrativeDynamics.AnonymousTemporalCredit
