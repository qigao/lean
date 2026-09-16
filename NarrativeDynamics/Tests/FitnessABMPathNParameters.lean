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

-- Task 2 RED: exact compatibility with the existing fixed `(1/2,1/2)` model.
example (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
    population half n x e = FitnessABMPathN.population n x e := by
  exact population_half n x e

example (n : Nat) (x : Beliefs n) :
    beliefStep half n x = FitnessABMPathN.beliefStep n x := by
  exact beliefStep_half n x

example (n : Nat) (x : Beliefs n) :
    allBroadcast half n x ↔ FitnessABMPathN.allBroadcast n x := by
  exact allBroadcast_half n x
