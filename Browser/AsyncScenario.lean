import Browser.AsyncProofs

/-!
# AsyncRuntime integration/reference scenario

Executable lifecycle/policy scenarios for the historical whole-`AsyncRuntime`
model. Retained for regression evidence; they are not a public proof API and
should not be extended into a parallel theorem family.
-/

namespace Browser

private def asyncGraph : RuntimeGraph := {
  contextOf := fun page => if page = 1 then 10 else 20
  pageOfFrame := fun frame => frame
  parentOfFrame := fun _ => none
}

private def asyncModel : Model := {
  browserAlive := true
  graph := asyncGraph
  knownPages := [1, 2]
  contextAvailable := fun _ => true
  page := fun _ => PageState.readyIdle
}

private def initialAsync : AsyncRuntime := asyncFromModel asyncModel

private def startPage1 : AsyncEnvelope := {
  id := 1
  cause := { correlation := 10, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .automationStarted })
}

private def context10Down : AsyncEnvelope := {
  id := 2
  cause := { correlation := 20, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .context 10, epoch := 1 }
  payload := .runtime (.contextUnavailable 10)
}

private def running : AsyncRuntime :=
  (deliver initialAsync startPage1).runtime

private def contextDownResult : DeliveryResult :=
  deliver running context10Down

private def afterPauseDelivery : DeliveryResult :=
  match contextDownResult.emitted with
  | command :: _ => deliver contextDownResult.runtime command
  | [] => contextDownResult

/-- Context delivery immediately constrains the parent but only emits child
    commands; it does not synchronously rewrite Page state. -/
example :
    contextDownResult.disposition = .accepted ∧
    contextDownResult.runtime.model.contextAvailable 10 = false ∧
    contextDownResult.runtime.model.page 1 = running.model.page 1 ∧
    contextDownResult.emitted.length = 1 := by
  native_decide

/-- The emitted Page command mutates Page state only when it is later delivered. -/
example :
    (running.model.page 1).action = .executing ∧
    (afterPauseDelivery.runtime.model.page 1).action = .suspended := by
  native_decide

private def destroyPage1 : AsyncEnvelope := {
  id := 20
  cause := { correlation := 200, parent := none, depth := 0 }
  source := { actor := .context 10, epoch := 1 }
  target := { actor := .page 1, epoch := 1 }
  payload := .graph (.pageDestroyed 1)
}

private def recreatePage1 : AsyncEnvelope := {
  id := 21
  cause := { correlation := 201, parent := none, depth := 0 }
  source := { actor := .context 10, epoch := 1 }
  target := { actor := .context 10, epoch := 1 }
  payload := .graph (.pageCreated 1 10 2)
}

private def lateOldPageReady : AsyncEnvelope := {
  id := 22
  cause := { correlation := 202, parent := none, depth := 0 }
  source := { actor := .browser, epoch := 1 }
  target := { actor := .page 1, epoch := 1 }
  payload := .runtime (.local { page := 1, kind := .pageReady })
}

private def destroyedPage : AsyncRuntime :=
  (deliver initialAsync destroyPage1).runtime

private def recreatedPage : AsyncRuntime :=
  (deliver destroyedPage recreatePage1).runtime

private def staleOldPageResult : DeliveryResult :=
  deliver recreatedPage lateOldPageReady

/-- Re-creation advances Page epoch; old-incarnation messages are stale and the
    new Page state remains unchanged. -/
example :
    (nodeSlot recreatedPage (.page 1)).epoch = 2 ∧
    (nodeSlot recreatedPage (.page 1)).alive = true ∧
    staleOldPageResult.disposition = .stale ∧
    staleOldPageResult.runtime.model.page 1 = recreatedPage.model.page 1 := by
  native_decide

private def createFrame100 : AsyncEnvelope := {
  id := 30
  cause := { correlation := 300, parent := none, depth := 0 }
  source := { actor := .page 1, epoch := 2 }
  target := { actor := .page 1, epoch := 2 }
  payload := .graph (.frameCreated 100 1 none 1)
}

private def destroyFrame100 : AsyncEnvelope := {
  id := 31
  cause := { correlation := 301, parent := none, depth := 0 }
  source := { actor := .page 1, epoch := 2 }
  target := { actor := .frame 100, epoch := 1 }
  payload := .graph (.frameDestroyed 100)
}

private def recreateFrame100 : AsyncEnvelope := {
  id := 32
  cause := { correlation := 302, parent := none, depth := 0 }
  source := { actor := .page 1, epoch := 2 }
  target := { actor := .page 1, epoch := 2 }
  payload := .graph (.frameCreated 100 1 none 2)
}

private def lateOldFrameDestroy : AsyncEnvelope := {
  id := 33
  cause := { correlation := 303, parent := none, depth := 0 }
  source := { actor := .page 1, epoch := 2 }
  target := { actor := .frame 100, epoch := 1 }
  payload := .graph (.frameDestroyed 100)
}

private def frameV1 : AsyncRuntime :=
  (deliver recreatedPage createFrame100).runtime

private def frameDead : AsyncRuntime :=
  (deliver frameV1 destroyFrame100).runtime

private def frameV2 : AsyncRuntime :=
  (deliver frameDead recreateFrame100).runtime

private def staleFrameResult : DeliveryResult :=
  deliver frameV2 lateOldFrameDestroy

/-- Frame topology follows the same epoch semantics as Page topology. -/
example :
    frameV2.model.graph.pageOfFrame 100 = 1 ∧
    (nodeSlot frameV2 (.frame 100)).epoch = 2 ∧
    (nodeSlot frameV2 (.frame 100)).alive = true ∧
    staleFrameResult.disposition = .stale ∧
    (nodeSlot staleFrameResult.runtime (.frame 100)).alive = true := by
  native_decide

end Browser
