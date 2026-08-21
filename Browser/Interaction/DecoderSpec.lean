import Browser.Interaction.Decoder

namespace Browser.Interaction

example : decodeEvent .runtimeExecutionContextDestroyed = .executionContextDestroyed := by rfl
example : decodeEvent .runtimeExecutionContextsCleared = .executionContextDestroyed := by rfl
example : decodeEvent .pageFrameDetached = .frameDetached := by rfl
example : decodeEvent .targetSessionDetached = .sessionDetached := by rfl
example : decodeEvent .inspectorDetached = .sessionDetached := by rfl

example : decodeEvent (.targetCrashed .page) = .pageCrashed := by rfl
example : decodeEvent (.targetDestroyed .page) = .pageClosed := by rfl
example : decodeEvent (.targetCrashed .worker) = .unrecognized := by rfl
example : decodeEvent (.targetDestroyed .worker) = .unrecognized := by rfl
example : decodeEvent (.targetDestroyed .unknown) = .unrecognized := by rfl

example : decodeEvent (.networkLoadingFailed .transient) = .networkTransient := by rfl
example : decodeEvent (.networkLoadingFailed .cancelled) = .conditionPending := by rfl
example : decodeEvent (.networkLoadingFailed .blocked) = .unrecognized := by rfl
example : decodeEvent (.networkLoadingFailed .cors) = .unrecognized := by rfl
example : decodeEvent (.networkLoadingFailed .unknown) = .unrecognized := by rfl

example : decodeEvent (.protocolCommandFailed .transient) = .protocolTransient := by rfl
example : decodeEvent (.protocolCommandFailed .sessionGone) = .sessionDetached := by rfl
example : decodeEvent (.protocolCommandFailed .unknown) = .unrecognized := by rfl

example : decodeEvent .domElementDetached = .elementDetached := by rfl
example : decodeEvent .humanInputObserved = .humanInput := by rfl
example : decodeEvent .deadlineReached = .deadlineReached := by rfl
example : decodeEvent .browserProcessExited = .browserExited := by rfl
example : decodeEvent .contextDestroyed = .contextDestroyed := by rfl
example : decodeEvent .operationSucceeded = .operationSucceeded := by rfl
example : decodeEvent .conditionPending = .conditionPending := by rfl
example : decodeEvent .navigationStarted = .navigationStarted := by rfl
example : decodeEvent .malformed = .unrecognized := by rfl
example : decodeEvent (.unknownSignal "Future.domainEvent") = .unrecognized := by rfl

example (event : AdapterEvent) : decoderOutputSafe event = true := by
  exact decoder_output_safe event

end Browser.Interaction
