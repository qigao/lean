import NarrativeDynamics.Core.FitnessABMIdleTail

open NarrativeDynamics NarrativeDynamics.FitnessABM

example (s : RunState) :
    (idleRunStep s).nodeCount = s.nodeCount :=
  idleRunStep_nodeCount s

example (s : RunState) :
    (idleRunStep s).roundIndex = s.roundIndex + 1 :=
  idleRunStep_roundIndex s

example (s : RunState) (k : Nat) :
    runIdleTrajectory s k =
      ⟨s.nodeCount, s.roundIndex + k, (advance^[k]) s.state⟩ :=
  runIdleTrajectory_eq s k

example {n : Nat} (s : JointState n) (k : Nat) :
    IdleTailModel.trajectory (jointIdleTail n) s k = (advance^[k]) s := by
  rfl
