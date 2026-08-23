import Browser.AsyncConformance

/-!
# Async JSONL conformance integration/reference spec

Executable wire-format/replay examples for the historical whole-`AsyncRuntime`
model. Retained for integration and regression evidence and intentionally kept
outside `Browser.ProofAPI`.
-/

namespace Browser

private def tinyAsyncTrace : String :=
  "{\"kind\":\"bootstrap\",\"pages\":[{\"page\":1,\"context\":10}]}\n" ++
  "{\"kind\":\"message\",\"messageId\":1,\"correlationId\":100,\"depth\":0,\"source\":{\"kind\":\"browser\",\"epoch\":1},\"target\":{\"kind\":\"page\",\"id\":1,\"epoch\":1},\"payload\":\"humanInput\",\"page\":1}"

/-- The async wire decoder accepts actor-addressed message metadata. -/
example :
    (parseAsyncTraceLine
      "{\"kind\":\"message\",\"messageId\":1,\"correlationId\":100,\"depth\":0,\"source\":{\"kind\":\"browser\",\"epoch\":1},\"target\":{\"kind\":\"page\",\"id\":1,\"epoch\":1},\"payload\":\"humanInput\",\"page\":1}").isOk := by
  native_decide

/-- End-to-end JSONL parsing and async delivery share one verifier entry point. -/
example : (verifyAsyncJsonl tinyAsyncTrace).isOk := by
  native_decide

end Browser
