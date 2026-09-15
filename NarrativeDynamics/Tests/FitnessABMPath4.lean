import NarrativeDynamics.Core.FitnessABMPath4

open NarrativeDynamics.NetworkPropagation NarrativeDynamics.FitnessABMPath4

example (x : Beliefs) (e : Fin 4 → Nat) (hx : allBroadcast x) :
    project (propagate pathAdj (population x e)) = linearStep x :=
  propagate_eq_linear x e hx

example (x : Beliefs) (e : Fin 4 → Nat) :
    project (propagate pathAdj (population x e)) = beliefStep x :=
  propagate_independent_exposures x e

example (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    allBroadcast (trajectory x n) :=
  allBroadcast_iterate x hx n

-- Node 3 is below the threshold: its belief must not enter node 2's mean.
example : beliefStep ![3/4, 11/16, 5/8, 1/4] =
    ![23/32, 11/16, 21/32, 7/16] := by
  decide_cbv

-- The symbolic contract catches losing any mode or applying the averaging
-- formula without maintaining the actual trajectory's broadcast region.
example (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    trajectory x n = closedForm x n := iterate_closedForm x hx n

example (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := mean_step x hx

-- The zero-eigenvalue mode contributes at n = 0, and the invariant is weighted.
example : closedForm ![45/64,44/64,43/64,35/64] 0 =
    ![45/64,44/64,43/64,35/64] := by decide_cbv

example : mean ![45/64,44/64,43/64,35/64] = 127/192 := by decide_cbv
