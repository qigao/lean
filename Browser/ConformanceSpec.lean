import Browser.Conformance

namespace Browser

private def bootstrap : TraceBootstrap := {
  kind := "bootstrap"
  pages := #[
    { page := 1, context := 10 },
    { page := 2, context := 20 }
  ]
}

private def initial : Model := modelFromBootstrap bootstrap

/-- A bootstrap line constructs the same page/context ownership model used by
    the runtime proofs. -/
example :
    initial.graph.contextOf 1 = 10 ∧
    initial.graph.contextOf 2 = 20 := by
  decide

/-- Driver JSON uses a declared ActionId for `actionStarted`; conformance must
    reject a trace when that id does not equal the model's next generation. -/
example :
    let record : DriverTraceRecord := {
      envelope := {
        id := 1
        event := .local { page := 1, kind := .automationStarted }
        cause := { correlation := 100, parent := none, depth := 0 }
      }
      declaredAction := some 99
    }
    verifyRecord { maxDepth := 4 } initial [] record = .error "action generation mismatch" := by
  rfl

/-- The JSONL decoder accepts a normalized action-start record carrying stable
    PageId/EventId/CorrelationId/ActionId metadata. -/
example :
    (parseTraceLine "{\"kind\":\"event\",\"eventId\":1,\"correlationId\":100,\"depth\":0,\"event\":\"actionStarted\",\"page\":1,\"action\":1}").isOk := by
  native_decide

end Browser
