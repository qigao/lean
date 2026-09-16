import NarrativeDynamics.Core.FitnessABMPathNParameters

open NarrativeDynamics
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessABMPathNParameters

example : ResponseParameters.Valid half := by
  norm_num [half, ResponseParameters.Valid]

example (n : Nat) (x : Beliefs n) :
    beliefStep half n x =
      project n
        (propagate (FitnessABMPathN.pathAdj n)
          (population half n x (fun _ => 0))) := by
  rfl

private def thresholdBeliefs : Beliefs 2 :=
  fun i => if i = 0 then (1 / 2 : Rat) else 0

private def thresholdPopulation :=
  population half 2 thresholdBeliefs (fun _ => 0)

example :
    broadcasting (thresholdPopulation.profiles 0)
      (thresholdPopulation.agents 0) = true := by
  decide_cbv

example (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
    population half n x e = FitnessABMPathN.population n x e := by
  exact population_half n x e

example (n : Nat) (x : Beliefs n) :
    beliefStep half n x = FitnessABMPathN.beliefStep n x := by
  exact beliefStep_half n x

example (n : Nat) (x : Beliefs n) :
    allBroadcast half n x ↔ FitnessABMPathN.allBroadcast n x := by
  exact allBroadcast_half n x

-- Task 3 RED: projected beliefs remain independent of stored exposures for
-- arbitrary response parameters, because baseline broadcasting and belief
-- arithmetic do not read the old exposure counters.
example (params : ResponseParameters) (n : Nat) (x : Beliefs n)
    (e : Fin n → Nat) :
    project n
        (propagate (FitnessABMPathN.pathAdj n) (population params n x e)) =
      beliefStep params n x := by
  exact propagate_independent_exposures params n x e

private def exposureA : Fin 3 → Nat := ![0, 0, 0]
private def exposureB : Fin 3 → Nat := ![2, 5, 9]

example (params : ResponseParameters) (x : Beliefs 3) :
    project 3
        (propagate (FitnessABMPathN.pathAdj 3) (population params 3 x exposureA)) =
      project 3
        (propagate (FitnessABMPathN.pathAdj 3) (population params 3 x exposureB)) := by
  rw [propagate_independent_exposures params 3 x exposureA,
    propagate_independent_exposures params 3 x exposureB]

-- The proof-only parameter kernel must specialize exactly to the existing
-- fixed half-response path kernel.
example (n : Nat) :
    pathKernel half n = FitnessABMPathN.pathKernel n := by
  exact pathKernel_half n
