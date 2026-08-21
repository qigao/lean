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
  page := fun _ => PageState.readyIdle
}

private def scenario : List RuntimeEvent := [
  .local { page := 1, kind := .automationStarted },
  .local { page := 1, kind := .humanInput },
  .local { page := 1, kind := .executionContextDestroyed },
  .local { page := 1, kind := .executionContextReady },
  .local { page := 1, kind := .automationFinished },
  .local { page := 2, kind := .navigationStarted },
  .local { page := 2, kind := .executionContextReady },
  .local { page := 2, kind := .pageReady }
]

def main : IO Unit := do
  let final := replay initial scenario
  if wellFormed? final then
    IO.println "browser-runtime formal model: scenario verification passed"
  else
    throw <| IO.userError "browser-runtime formal model: safety invariant violated"
