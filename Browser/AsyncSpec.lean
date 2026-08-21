import Browser.AsyncProofs

namespace Browser

private def graph : RuntimeGraph := {
  contextOf := fun page => if page = 1 then 10 else 20
  pageOfFrame := fun frame => frame
  parentOfFrame := fun _ => none
}

private def model : Model := {
  browserAlive := true
  graph := graph
  knownPages := [1, 2]
  contextAvailable := fun _ => true
  page := fun _ => PageState.readyIdle
}

private def runtime : AsyncRuntime := asyncFromModel model

private def rootCause : Cause := {
  correlation := 100
  parent := none
  depth := 0
}

private def humanOnPage1 : AsyncEnvelope := {
  id := 1
  cause := rootCause
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .humanInput })
}

private def conflictingMessageId1 : AsyncEnvelope := {
  id := 1
  cause := rootCause
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 2, epoch := 1 }
  payload := .runtime (.local { page := 2, kind := .humanInput })
}

private def staleHumanOnPage1 : AsyncEnvelope := {
  id := 2
  cause := { correlation := 200, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 1, epoch := 2 }
  payload := .runtime (.local { page := 1, kind := .humanInput })
}

private def orphanHuman : AsyncEnvelope := {
  id := 3
  cause := { correlation := 300, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 99, epoch := 1 }
  payload := .runtime (.local { page := 99, kind := .humanInput })
}

/-- Bootstrap lifts known Page/Context actors at epoch 1. -/
example :
    (nodeSlot runtime (.page 1)).alive = true ∧
    (nodeSlot runtime (.page 1)).epoch = 1 ∧
    (nodeSlot runtime (.context 10)).alive = true := by
  native_decide

/-- Page-local delivery cannot mutate a sibling Page. -/
example :
    ((deliver runtime humanOnPage1).runtime.model.page 2) = model.page 2 := by
  native_decide

/-- Re-delivering the exact same message is classified as duplicate. -/
example :
    let first := deliver runtime humanOnPage1
    (deliver first.runtime humanOnPage1).disposition = .duplicate := by
  native_decide

/-- Reusing a MessageId for different content is corruption, not idempotent
    redelivery, and must be rejected. -/
example :
    let first := deliver runtime humanOnPage1
    (deliver first.runtime conflictingMessageId1).disposition = .rejected := by
  native_decide

/-- Old Page epoch is stale and cannot mutate the current Page state. -/
example :
    (deliver runtime staleHumanOnPage1).disposition = .stale ∧
    (deliver runtime staleHumanOnPage1).runtime.model.page 1 = model.page 1 := by
  native_decide

/-- A never-known target is orphan and leaves runtime state untouched. -/
example :
    (deliver runtime orphanHuman).disposition = .orphan ∧
    (deliver runtime orphanHuman).runtime.model.page 1 = model.page 1 := by
  native_decide

end Browser
