import Browser.Interaction.ClosedLoop

open Browser.Interaction

private def recoverableTrace : List TaggedObservation := [
  { generation := 10, event := .targetCrashed .page },
  { generation := 9, event := .browserProcessExited },
  { generation := 10, event := .networkLoadingFailed .transient },
  { generation := 11, event := .contextDestroyed },
  { generation := 10, event := .targetSessionDetached }
]

private def pageAvailable : RecoveryAction → Bool
  | .recreatePage => true
  | _ => false

private def recovered := runClosedLoop 10 5 8 recoverableTrace pageAvailable

private def fatal :=
  runClosedLoop 20 5 8
    [{ generation := 20, event := .deadlineReached }]
    (fun _ => true)

private def idle :=
  runClosedLoop 30 5 8
    [{ generation := 30, event := .networkLoadingFailed .transient }]
    (fun _ => false)

private def exhausted :=
  runClosedLoop 40 1 1
    [{ generation := 40, event := .targetSessionDetached }]
    (fun _ => false)

private def scenarioVerified : Bool :=
  (recovered.runtimeSelected == .recreatePage) &&
  (recovered.selected == .recreatePage) &&
  (!recovered.final.failed) &&
  (recovered.final.generation == 11) &&
  (recovered.final.active == none) &&
  (fatal.runtimeSelected == .fail) &&
  (fatal.selected == .fail) &&
  fatal.final.failed &&
  (idle.runtimeSelected == .retry) &&
  (idle.selected == .retry) &&
  (!idle.final.failed) &&
  (idle.final.generation == 30) &&
  (idle.final.active == none) &&
  exhausted.final.failed

def main : IO UInt32 := do
  if scenarioVerified then
    IO.println "closed-loop recovery: minimal selection + fresh recovery + bounded failure verified"
    return 0
  else
    IO.eprintln "closed-loop recovery: violation"
    return 1
