import NarrativeDynamics.Core.IdleTail

open NarrativeDynamics

private def incrementModel : IdleTailModel Nat Nat :=
  { step := Nat.succ
    observe := id }

example : IdleTailModel.trajectory incrementModel 3 0 = 3 := by
  simpa using IdleTailModel.trajectory_zero incrementModel 3

example : IdleTailModel.trajectory incrementModel 3 2 = 5 := by
  decide

example (a b : Nat) :
    IdleTailModel.trajectory incrementModel 3 (a + b) =
      IdleTailModel.trajectory incrementModel
        (IdleTailModel.trajectory incrementModel 3 a) b := by
  exact IdleTailModel.trajectory_add incrementModel 3 a b

example (k : Nat) :
    IdleTailModel.observedTrajectory incrementModel 3 (k + 1) =
      incrementModel.observe
        (incrementModel.step
          (IdleTailModel.trajectory incrementModel 3 k)) := by
  exact IdleTailModel.observedTrajectory_succ incrementModel 3 k
