import Browser.Async

namespace Browser

/-- A duplicate delivery is an exact runtime no-op. -/
theorem duplicate_delivery_noop
    (rt : AsyncRuntime) (envelope : AsyncEnvelope)
    (hseen : hasSeen rt envelope.id = true) :
    (deliver rt envelope).runtime = rt ∧
    (deliver rt envelope).emitted = [] ∧
    (deliver rt envelope).disposition = .duplicate := by
  simp [deliver, hseen]

/-- An unknown/dead current-epoch target cannot mutate the runtime. -/
theorem orphan_delivery_noop
    (rt : AsyncRuntime) (envelope : AsyncEnvelope)
    (hseen : hasSeen rt envelope.id = false)
    (hcause : causalValid rt envelope = true)
    (htarget : classifyTarget rt envelope.target = .orphan) :
    (deliver rt envelope).runtime = rt ∧
    (deliver rt envelope).emitted = [] ∧
    (deliver rt envelope).disposition = .orphan := by
  simp [deliver, hseen, hcause, htarget]

/-- A message for an old node epoch may be observed for causal diagnostics, but
    browser/graph/Page business state is unchanged and no child work is emitted. -/
theorem stale_delivery_preserves_model
    (rt : AsyncRuntime) (envelope : AsyncEnvelope)
    (hseen : hasSeen rt envelope.id = false)
    (hcause : causalValid rt envelope = true)
    (htarget : classifyTarget rt envelope.target = .stale) :
    (deliver rt envelope).runtime.model = rt.model ∧
    (deliver rt envelope).emitted = [] ∧
    (deliver rt envelope).disposition = .stale := by
  simp [deliver, hseen, hcause, htarget]

end Browser
