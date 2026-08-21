import Browser.Interaction.Recovery

namespace Browser.Interaction

/-- Adapter-facing observations. Protocol/version-specific decoding happens
    outside the proof core; anything not recognized must become `unrecognized`. -/
inductive RawObservation where
  | operationSucceeded
  | conditionPending
  | navigationStarted
  | networkTransient
  | protocolTransient
  | humanInput
  | elementDetached
  | executionContextDestroyed
  | frameDetached
  | sessionDetached
  | pageClosed
  | pageCrashed
  | contextDestroyed
  | browserExited
  | deadlineReached
  | unrecognized
  deriving Repr, DecidableEq, BEq

structure NormalizedFeedback where
  feedback : FeedbackClass
  fault : Option FaultClass := none
  deriving Repr, DecidableEq, BEq

/-- Total raw-observation normalization. The result contains the coarse feedback
    class plus an optional fault that carries the minimum recovery scope. -/
def normalizeObservation : RawObservation → NormalizedFeedback
  | .operationSucceeded => { feedback := .success }
  | .conditionPending => { feedback := .transient }
  | .navigationStarted => { feedback := .transient }
  | .networkTransient => { feedback := .transient }
  | .protocolTransient => { feedback := .transient }
  | .humanInput => { feedback := .conflict }
  | .elementDetached => { feedback := .stale, fault := some .elementStale }
  | .executionContextDestroyed => { feedback := .stale, fault := some .runtimeLost }
  | .frameDetached => { feedback := .stale, fault := some .runtimeLost }
  | .sessionDetached => { feedback := .unavailable, fault := some .sessionLost }
  | .pageClosed => { feedback := .terminal, fault := some .pageLost }
  | .pageCrashed => { feedback := .terminal, fault := some .pageLost }
  | .contextDestroyed => { feedback := .terminal, fault := some .contextLost }
  | .browserExited => { feedback := .terminal, fault := some .browserLost }
  | .deadlineReached => { feedback := .timeout, fault := some .timeoutFault }
  | .unrecognized => { feedback := .unknown, fault := some .unknownFault }

/-- Allowed semantic combinations. This detects a normalizer that classifies an
    observation with a fault incompatible with its coarse feedback category. -/
def normalizationCoherent (normalized : NormalizedFeedback) : Bool :=
  match normalized.feedback, normalized.fault with
  | .success, none => true
  | .transient, none => true
  | .conflict, none => true
  | .stale, some .elementStale => true
  | .stale, some .runtimeLost => true
  | .unavailable, some .sessionLost => true
  | .terminal, some .pageLost => true
  | .terminal, some .contextLost => true
  | .terminal, some .browserLost => true
  | .timeout, some .timeoutFault => true
  | .unknown, some .unknownFault => true
  | _, _ => false

theorem normalization_is_coherent (raw : RawObservation) :
    normalizationCoherent (normalizeObservation raw) = true := by
  cases raw <;> rfl

/-- Fault scope takes precedence over the coarse feedback policy. This is what
    lets two stale-looking observations select different minimum repairs. -/
def decideNormalized (normalized : NormalizedFeedback) : Decision :=
  match normalized.fault with
  | none => decideFeedback normalized.feedback
  | some fault =>
      let action := minimumRecovery fault
      if action = .fail then .failSafe else .recover action

theorem no_fault_uses_feedback_decision (feedback : FeedbackClass) :
    decideNormalized { feedback := feedback, fault := none } = decideFeedback feedback := by
  rfl

theorem recoverable_fault_uses_minimum
    (feedback : FeedbackClass) (fault : FaultClass)
    (h : minimumRecovery fault ≠ .fail) :
    decideNormalized { feedback := feedback, fault := some fault } =
      .recover (minimumRecovery fault) := by
  simp [decideNormalized, h]

theorem failing_fault_fails_safe
    (feedback : FeedbackClass) (fault : FaultClass)
    (h : minimumRecovery fault = .fail) :
    decideNormalized { feedback := feedback, fault := some fault } = .failSafe := by
  simp [decideNormalized, h]

theorem unrecognized_fails_safe :
    decideNormalized (normalizeObservation .unrecognized) = .failSafe := by
  rfl

theorem deadline_fails_safe :
    decideNormalized (normalizeObservation .deadlineReached) = .failSafe := by
  rfl

end Browser.Interaction
