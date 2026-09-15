import NarrativeDynamics.Core.FitnessABMIdleTail
import NarrativeDynamics.Core.FitnessABMReplay

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

example (m tickIndex birthIndex k : Nat) (s : RunState) :
    runInputs m tickIndex birthIndex s (List.replicate k none) =
      .ok ⟨runIdleTrajectory s k, 1⟩ :=
  runInputs_replicate_idle m tickIndex birthIndex k s

example (m tickIndex birthIndex : Nat)
    (s : RunState) (ticks : List RawTick) (out : Result) (k : Nat)
    (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
    runInputs m tickIndex birthIndex s
        (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩ :=
  runInputs_append_idle m tickIndex birthIndex s ticks out k h

example (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : replay seed m agents ticks = .ok out) :
    replay seed m agents (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩ :=
  replay_append_idle seed m agents ticks out k h

#print axioms NarrativeDynamics.FitnessABM.runIdleTrajectory_eq
#print axioms NarrativeDynamics.FitnessABM.runInputs_replicate_idle
#print axioms NarrativeDynamics.FitnessABM.runInputs_append_idle
#print axioms NarrativeDynamics.FitnessABM.replay_append_idle
