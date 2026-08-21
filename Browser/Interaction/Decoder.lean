import Browser.Interaction.Normalizer

namespace Browser.Interaction

/-- Scope resolved by the adapter from target/session ownership information.
    CDP method name alone is intentionally not enough for page-level faults. -/
inductive AdapterScope where
  | browser
  | context
  | page
  | frame
  | worker
  | serviceWorker
  | other
  | unknown
  deriving Repr, DecidableEq, BEq

/-- Semantic failure disposition resolved by the protocol/network adapter.
    Unknown or policy failures are kept distinct from retryable failures. -/
inductive FailureDisposition where
  | transient
  | cancelled
  | blocked
  | cors
  | sessionGone
  | unknown
  deriving Repr, DecidableEq, BEq

/-- Stable adapter-facing event vocabulary. Version-specific JSON parsing and
    Chromium-specific strings terminate at this boundary. -/
inductive AdapterEvent where
  | runtimeExecutionContextDestroyed
  | runtimeExecutionContextsCleared
  | pageFrameDetached
  | targetSessionDetached
  | targetCrashed (scope : AdapterScope)
  | targetDestroyed (scope : AdapterScope)
  | inspectorDetached
  | networkLoadingFailed (disposition : FailureDisposition)
  | protocolCommandFailed (disposition : FailureDisposition)
  | domElementDetached
  | humanInputObserved
  | deadlineReached
  | browserProcessExited
  | contextDestroyed
  | operationSucceeded
  | conditionPending
  | navigationStarted
  | malformed
  | unknownSignal (name : String)
  deriving Repr, DecidableEq, BEq

/-- Total decoder into the already-proven raw observation vocabulary.
    Ambiguous/under-scoped input is never guessed: it becomes `unrecognized`. -/
def decodeEvent : AdapterEvent → RawObservation
  | .runtimeExecutionContextDestroyed => .executionContextDestroyed
  | .runtimeExecutionContextsCleared => .executionContextDestroyed
  | .pageFrameDetached => .frameDetached
  | .targetSessionDetached => .sessionDetached
  | .targetCrashed .page => .pageCrashed
  | .targetCrashed _ => .unrecognized
  | .targetDestroyed .page => .pageClosed
  | .targetDestroyed _ => .unrecognized
  | .inspectorDetached => .sessionDetached
  | .networkLoadingFailed .transient => .networkTransient
  | .networkLoadingFailed .cancelled => .conditionPending
  | .networkLoadingFailed _ => .unrecognized
  | .protocolCommandFailed .transient => .protocolTransient
  | .protocolCommandFailed .sessionGone => .sessionDetached
  | .protocolCommandFailed _ => .unrecognized
  | .domElementDetached => .elementDetached
  | .humanInputObserved => .humanInput
  | .deadlineReached => .deadlineReached
  | .browserProcessExited => .browserExited
  | .contextDestroyed => .contextDestroyed
  | .operationSucceeded => .operationSucceeded
  | .conditionPending => .conditionPending
  | .navigationStarted => .navigationStarted
  | .malformed => .unrecognized
  | .unknownSignal _ => .unrecognized

/-- Executable safety checker for the decoder's ambiguity boundaries. -/
def decoderOutputSafe (event : AdapterEvent) : Bool :=
  match event with
  | .targetCrashed scope =>
      match scope with
      | .page => decodeEvent event == .pageCrashed
      | _ => decodeEvent event == .unrecognized
  | .targetDestroyed scope =>
      match scope with
      | .page => decodeEvent event == .pageClosed
      | _ => decodeEvent event == .unrecognized
  | .networkLoadingFailed disposition =>
      match disposition with
      | .transient => decodeEvent event == .networkTransient
      | .cancelled => decodeEvent event == .conditionPending
      | _ => decodeEvent event == .unrecognized
  | .protocolCommandFailed disposition =>
      match disposition with
      | .transient => decodeEvent event == .protocolTransient
      | .sessionGone => decodeEvent event == .sessionDetached
      | _ => decodeEvent event == .unrecognized
  | .malformed | .unknownSignal _ => decodeEvent event == .unrecognized
  | _ => true

theorem decoder_output_safe (event : AdapterEvent) :
    decoderOutputSafe event = true := by
  cases event <;>
    try rfl
  case targetCrashed scope => cases scope <;> rfl
  case targetDestroyed scope => cases scope <;> rfl
  case networkLoadingFailed disposition => cases disposition <;> rfl
  case protocolCommandFailed disposition => cases disposition <;> rfl

theorem decoded_normalization_is_coherent (event : AdapterEvent) :
    normalizationCoherent (normalizeObservation (decodeEvent event)) = true := by
  exact normalization_is_coherent (decodeEvent event)

theorem unknown_signal_fails_safe (name : String) :
    decideNormalized (normalizeObservation (decodeEvent (.unknownSignal name))) = .failSafe := by
  rfl

theorem malformed_event_fails_safe :
    decideNormalized (normalizeObservation (decodeEvent .malformed)) = .failSafe := by
  rfl

end Browser.Interaction
