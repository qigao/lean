import NarrativeDynamics.Core.FitnessABMPath4

open NarrativeDynamics.NetworkPropagation NarrativeDynamics.FitnessABMPath4

example (x : Beliefs) (e : Fin 4 → Nat) (hx : allBroadcast x) :
    project (propagate pathAdj (population x e)) = linearStep x :=
  propagate_eq_linear x e hx

-- Node 3 is below the threshold: its belief must not enter node 2's mean.
example : beliefStep ![3/4, 11/16, 5/8, 1/4] =
    ![23/32, 11/16, 21/32, 7/16] := by
  decide_cbv
