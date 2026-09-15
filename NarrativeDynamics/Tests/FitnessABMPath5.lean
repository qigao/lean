import NarrativeDynamics.Core.FitnessABMPath5

open NarrativeDynamics
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessABMPath5

example (x : Beliefs) (e : Fin 5 → Nat) :
    project (propagate path5Adj (population x e)) =
      project (propagate path5Adj (population x (fun _ => 0))) := by
  exact propagate_independent_exposures x e

example (x : Beliefs) (hx : allBroadcast x) :
    allBroadcast (beliefStep x) := by
  exact allBroadcast_step x hx

example (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := by
  exact mean_step x hx
