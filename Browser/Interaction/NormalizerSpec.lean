import Browser.Interaction.Normalizer

namespace Browser.Interaction

example : normalizeObservation .operationSucceeded = {
    feedback := .success
    fault := none
  } := by rfl

example : normalizeObservation .conditionPending = {
    feedback := .transient
    fault := none
  } := by rfl

example : normalizeObservation .navigationStarted = {
    feedback := .transient
    fault := none
  } := by rfl

example : normalizeObservation .networkTransient = {
    feedback := .transient
    fault := none
  } := by rfl

example : normalizeObservation .protocolTransient = {
    feedback := .transient
    fault := none
  } := by rfl

example : normalizeObservation .humanInput = {
    feedback := .conflict
    fault := none
  } := by rfl

example : normalizeObservation .elementDetached = {
    feedback := .stale
    fault := some .elementStale
  } := by rfl

example : normalizeObservation .executionContextDestroyed = {
    feedback := .stale
    fault := some .runtimeLost
  } := by rfl

example : normalizeObservation .frameDetached = {
    feedback := .stale
    fault := some .runtimeLost
  } := by rfl

example : normalizeObservation .sessionDetached = {
    feedback := .unavailable
    fault := some .sessionLost
  } := by rfl

example : normalizeObservation .pageClosed = {
    feedback := .terminal
    fault := some .pageLost
  } := by rfl

example : normalizeObservation .pageCrashed = {
    feedback := .terminal
    fault := some .pageLost
  } := by rfl

example : normalizeObservation .contextDestroyed = {
    feedback := .terminal
    fault := some .contextLost
  } := by rfl

example : normalizeObservation .browserExited = {
    feedback := .terminal
    fault := some .browserLost
  } := by rfl

example : normalizeObservation .deadlineReached = {
    feedback := .timeout
    fault := some .timeoutFault
  } := by rfl

example : normalizeObservation .unrecognized = {
    feedback := .unknown
    fault := some .unknownFault
  } := by rfl

example (raw : RawObservation) :
    normalizationCoherent (normalizeObservation raw) = true := by
  exact normalization_is_coherent raw

example : decideNormalized (normalizeObservation .elementDetached) =
    .recover .reResolve := by rfl

example : decideNormalized (normalizeObservation .executionContextDestroyed) =
    .recover .rebindRuntime := by rfl

example : decideNormalized (normalizeObservation .sessionDetached) =
    .recover .reattachSession := by rfl

example : decideNormalized (normalizeObservation .pageCrashed) =
    .recover .recreatePage := by rfl

example : decideNormalized (normalizeObservation .browserExited) =
    .recover .restartBrowser := by rfl

example : decideNormalized (normalizeObservation .deadlineReached) = .failSafe := by rfl
example : decideNormalized (normalizeObservation .unrecognized) = .failSafe := by rfl
example : decideNormalized (normalizeObservation .humanInput) = .suspend := by rfl
example : decideNormalized (normalizeObservation .networkTransient) = .wait := by rfl
example : decideNormalized (normalizeObservation .operationSucceeded) = .complete := by rfl

end Browser.Interaction
