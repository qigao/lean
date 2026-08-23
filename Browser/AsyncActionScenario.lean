import Browser.AsyncProofs

/-!
# Async action integration/reference scenario

Executable ActionId freshness scenarios for the historical whole-`AsyncRuntime`
model. They are retained for regression evidence and remain outside the
supported `Browser.ProofAPI` theorem surface.
-/

namespace Browser

private def actionGraph : RuntimeGraph := {
  contextOf := fun _ => 10
  pageOfFrame := fun frame => frame
  parentOfFrame := fun _ => none
}

private def actionModel : Model := {
  browserAlive := true
  graph := actionGraph
  knownPages := [1]
  contextAvailable := fun _ => true
  page := fun _ => PageState.readyIdle
}

private def actionRuntime : AsyncRuntime := asyncFromModel actionModel

private def start1 : AsyncEnvelope := {
  id := 100
  cause := { correlation := 1000, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .automationStarted })
}

private def wait1 : AsyncEnvelope := {
  id := 101
  cause := { correlation := 1001, parent := none, depth := 0 }
  source := { actor := .page 1, epoch := 1 }
  target := { actor := .action 1 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .protocolWaitStarted 1 500 })
}

private def finish1 : AsyncEnvelope := {
  id := 102
  cause := { correlation := 1002, parent := none, depth := 0 }
  source := { actor := .page 1, epoch := 1 }
  target := { actor := .action 1 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .automationFinished 1 })
}

private def start2 : AsyncEnvelope := {
  id := 103
  cause := { correlation := 1003, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .automationStarted })
}

private def wait2 : AsyncEnvelope := {
  id := 104
  cause := { correlation := 1004, parent := none, depth := 0 }
  source := { actor := .page 1, epoch := 1 }
  target := { actor := .action 1 2, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .protocolWaitStarted 2 600 })
}

private def staleResponse1 : AsyncEnvelope := {
  id := 105
  cause := { correlation := 1005, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .action 1 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .protocolResponse 1 500 })
}

private def currentResponse2 : AsyncEnvelope := {
  id := 106
  cause := { correlation := 1006, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .action 1 2, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .protocolResponse 2 600 })
}

private def futureResponse3 : AsyncEnvelope := {
  id := 107
  cause := { correlation := 1007, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .action 1 3, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .protocolResponse 3 700 })
}

private def afterStart1 := (deliver actionRuntime start1).runtime
private def afterWait1 := (deliver afterStart1 wait1).runtime
private def afterFinish1 := (deliver afterWait1 finish1).runtime
private def afterStart2 := (deliver afterFinish1 start2).runtime
private def afterWait2 := (deliver afterStart2 wait2).runtime
private def staleResult := deliver afterWait2 staleResponse1
private def currentResult := deliver staleResult.runtime currentResponse2
private def futureResult := deliver afterWait2 futureResponse3

/-- Action-owned work is addressed to Action(PageId, ActionId). Old generations
    are stale before local action logic executes. -/
example :
    (afterWait2.model.page 1).currentAction = some 2 ∧
    staleResult.disposition = .stale ∧
    staleResult.runtime.model.page 1 = afterWait2.model.page 1 := by
  native_decide

/-- Current-generation action message is accepted and may consume its pending
    protocol state. -/
example :
    currentResult.disposition = .accepted ∧
    (currentResult.runtime.model.page 1).pendingProtocol = none := by
  native_decide

/-- A not-yet-created future ActionId is orphan rather than stale. -/
example : futureResult.disposition = .orphan := by
  native_decide

end Browser
