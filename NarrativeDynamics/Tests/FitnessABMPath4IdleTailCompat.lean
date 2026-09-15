import NarrativeDynamics.Tests.FitnessABMPath4
import NarrativeDynamics.Core.FitnessABMIdleTail

namespace NarrativeDynamics.Tests.FitnessABMPath4

open NarrativeDynamics NarrativeDynamics.FitnessABM

example (h : History) (k : Nat) :
    tailState h k =
      IdleTailModel.trajectory (jointIdleTail 4) (baseState h) k := by
  rfl

example (h : History) (k : Nat) :
    tailBelief h k =
      IdleTailModel.observedTrajectory
        (observeJointWith
          (fun s : JointState 4 =>
            NarrativeDynamics.FitnessABMPath4.project s.population))
        (baseState h) k := by
  rfl

end NarrativeDynamics.Tests.FitnessABMPath4
