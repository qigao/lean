import Browser

open Browser

private def graph : RuntimeGraph := {
  contextOf := fun p => if p = 1 then 10 else 20
  pageOfFrame := fun f => f
  parentOfFrame := fun _ => none
}

private def initial : Model := {
  browserAlive := true
  graph := graph
  knownPages := [1, 2]
  contextAvailable := fun _ => true
  page := fun _ => PageState.readyIdle
}

private def afterContextPause : Model :=
  let running := react initial (.local { page := 1, kind := .automationStarted })
  react running (.contextUnavailable 10)

private def contextIsolationOk : Bool :=
  ((afterContextPause.page 1).action == .suspended) &&
  ((afterContextPause.page 2).action == .idle) &&
  (!(afterContextPause.contextAvailable 10)) &&
  afterContextPause.contextAvailable 20

private def scenario : List RuntimeEvent := [
  .local { page := 1, kind := .automationStarted },
  .contextUnavailable 10,
  .contextAvailable 10,
  .local { page := 1, kind := .automationStarted },
  .local { page := 1, kind := .automationFinished },
  .local { page := 2, kind := .automationStarted },
  .browserDisconnected,
  .browserConnected,
  .local { page := 2, kind := .automationStarted },
  .local { page := 2, kind := .automationFinished }
]

def main : IO Unit := do
  let final := replay initial scenario
  if contextIsolationOk && traceSafe initial scenario && wellFormed? final then
    IO.println "browser-runtime formal model: scoped policy scenario verification passed"
  else
    throw <| IO.userError "browser-runtime formal model: scoped policy safety invariant violated"
