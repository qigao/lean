import Browser.Async

/-!
# AsyncRuntime integration/reference proofs

These theorems are retained as regression evidence for the original whole
`AsyncRuntime` model. They are not part of the supported public proof API.
New proof obligations should target `Browser.ProofAPI`, whose formal control
chain is `Interaction -> Feedback -> Recovery -> Aggregation -> ClosedLoop`.
-/

namespace Browser

/-- Integration/reference: re-delivering the exact same envelope is an exact
    runtime no-op. -/
theorem duplicate_delivery_noop
    (rt : AsyncRuntime) (envelope : AsyncEnvelope)
    (hdup : exactDuplicate rt envelope = true) :
    (deliver rt envelope).runtime = rt ∧
    (deliver rt envelope).emitted = [] ∧
    (deliver rt envelope).disposition = .duplicate := by
  simp [deliver, hdup]

/-- Integration/reference: reusing an observed MessageId for different content
    is rejected without mutating runtime state. -/
theorem message_id_collision_rejected
    (rt : AsyncRuntime) (envelope : AsyncEnvelope)
    (hdup : exactDuplicate rt envelope = false)
    (hcollision : messageIdCollision rt envelope = true) :
    (deliver rt envelope).runtime = rt ∧
    (deliver rt envelope).emitted = [] ∧
    (deliver rt envelope).disposition = .rejected := by
  simp [deliver, hdup, hcollision]

/-- Integration/reference: an unknown/dead current-epoch target cannot mutate
    the runtime. -/
theorem orphan_delivery_noop
    (rt : AsyncRuntime) (envelope : AsyncEnvelope)
    (hdup : exactDuplicate rt envelope = false)
    (hcollision : messageIdCollision rt envelope = false)
    (hcause : causalValid rt envelope = true)
    (htarget : classifyTarget rt envelope.target = .orphan) :
    (deliver rt envelope).runtime = rt ∧
    (deliver rt envelope).emitted = [] ∧
    (deliver rt envelope).disposition = .orphan := by
  simp [deliver, hdup, hcollision, hcause, htarget]

/-- Integration/reference: a message for an old node epoch may be observed for
    causal diagnostics, but browser/graph/Page business state is unchanged and
    no child work is emitted. -/
theorem stale_delivery_preserves_model
    (rt : AsyncRuntime) (envelope : AsyncEnvelope)
    (hdup : exactDuplicate rt envelope = false)
    (hcollision : messageIdCollision rt envelope = false)
    (hcause : causalValid rt envelope = true)
    (htarget : classifyTarget rt envelope.target = .stale) :
    (deliver rt envelope).runtime.model = rt.model ∧
    (deliver rt envelope).emitted = [] ∧
    (deliver rt envelope).disposition = .stale := by
  simp [deliver, hdup, hcollision, hcause, htarget]
  rfl

end Browser
