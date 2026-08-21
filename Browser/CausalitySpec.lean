import Browser.Causality
import Browser.Policy
import Browser.Transition

namespace Browser

private def envelope : EventEnvelope := {
  id := 7
  event := .contextUnavailable 10
  cause := { correlation := 42, parent := none, depth := 3 }
}

/-- Policy children must preserve the correlation and advance the causal chain. -/
example :
    (childCause envelope).correlation = 42 ∧
    (childCause envelope).parent = some 7 ∧
    (childCause envelope).depth = 4 := by
  decide

/-- Bounded reaction must reject an event that is already deeper than budget. -/
example (m : Model) :
    reactEnvelope { maxDepth := 2 } m envelope = none := by
  rfl

end Browser
