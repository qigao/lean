import Browser.Interaction.ClosedLoop

namespace Browser.Interaction

private def mixedTrace : List TaggedObservation := [
  { generation := 10, event := .targetCrashed .page },
  { generation := 9, event := .browserProcessExited },
  { generation := 10, event := .networkLoadingFailed .transient },
  { generation := 11, event := .contextDestroyed },
  { generation := 10, event := .targetSessionDetached }
]

private def pageAvailable : RecoveryAction → Bool
  | .recreatePage => true
  | _ => false

private def recoveredLoop : ClosedLoopResult :=
  runClosedLoop 10 5 8 mixedTrace pageAvailable

example : recoveredLoop.selected = .recreatePage := by rfl
example : recoveredLoop.final.failed = false := by rfl
example : recoveredLoop.final.active = none := by rfl
example : recoveredLoop.final.generation = 11 := by rfl

example :
    recoveredLoop.selected = aggregateRecovery (effectiveFaults 10 mixedTrace) := by
  exact closed_loop_selects_aggregate 10 5 8 mixedTrace pageAvailable

example : supervisorResolved recoveredLoop.final = true := by
  exact closed_loop_is_resolved 10 5 8 mixedTrace pageAvailable

example :
    recoveredLoop.final.failed = false →
      recoveredLoop.final.generation = 10 + 1 := by
  intro h
  exact closed_loop_success_is_fresh 10 5 8 mixedTrace pageAvailable (by decide) h

private def fatalLoop : ClosedLoopResult :=
  runClosedLoop 20 5 8
    [{ generation := 20, event := .deadlineReached }]
    (fun _ => true)

example : fatalLoop.selected = .fail := by rfl
example : fatalLoop.final.failed = true := by rfl
example : supervisorResolved fatalLoop.final = true := by rfl

private def idleLoop : ClosedLoopResult :=
  runClosedLoop 30 5 8
    [{ generation := 30, event := .networkLoadingFailed .transient }]
    (fun _ => false)

example : idleLoop.selected = .retry := by rfl
example : idleLoop.final.failed = false := by rfl
example : idleLoop.final.generation = 30 := by rfl
example : idleLoop.final.active = none := by rfl

end Browser.Interaction
