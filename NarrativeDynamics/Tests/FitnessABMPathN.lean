import NarrativeDynamics.Core.FitnessABMPathN

open NarrativeDynamics.FitnessABMPathN
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FiniteConsensus

example : pathAdj 6 (0 : Fin 6) 1 := by
  decide

example : degree 6 0 = 1 := by
  decide_cbv

example : degree 6 3 = 2 := by
  decide_cbv

example : pathKernel 6 0 0 = 1/2 := by
  decide_cbv

example : pathKernel 6 0 1 = 1/2 := by
  decide_cbv

example : pathKernel 6 3 2 = 1/4 := by
  decide_cbv

example (x : Beliefs 6) (e : Fin 6 → Nat) :
    project 6 (propagate (pathAdj 6) (population 6 x e)) = beliefStep 6 x := by
  exact propagate_independent_exposures 6 x e

example (x : Beliefs 6) (hx : allBroadcast 6 x) (k : Nat) :
    trajectory 6 x k = kernelTrajectory (pathKernel 6) x k := by
  exact trajectory_eq_kernelTrajectory 6 (by omega) x hx k

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) :
    mean n (beliefStep n x) = mean n x :=
  mean_step n hn x hx

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (k : Nat) :
    mean n (kernelTrajectory (pathKernel n) x k) = mean n x :=
  mean_kernel_iterate n hn x k
