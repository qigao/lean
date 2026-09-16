import NarrativeDynamics.Core.FitnessABMPathNParameters

open NarrativeDynamics
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessABMPathNParameters
open scoped BigOperators

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

-- Task 3: projected beliefs remain independent of stored exposures for
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

example (n : Nat) :
    pathKernel half n = FitnessABMPathN.pathKernel n := by
  exact pathKernel_half n

-- Task 4 RED: valid parameter kernels are stochastic, the real executable
-- propagation agrees with the proof kernel inside the inclusive broadcast
-- region, and that region is preserved by one step and finite iteration.
private def threeQuarterThird : ResponseParameters :=
  ⟨3/4, 1/3⟩

private def thresholdVector : Beliefs 3 :=
  ![1/3, 1/2, 1]

example : ResponseParameters.Valid threeQuarterThird := by
  norm_num [threeQuarterThird, ResponseParameters.Valid]

example : allBroadcast threeQuarterThird 3 thresholdVector := by
  intro i
  fin_cases i <;> norm_num [threeQuarterThird, thresholdVector]

example (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n) :
    0 ≤ pathKernel params n i j := by
  exact pathKernel_nonneg params hvalid n hn i j

example (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    ∑ j, pathKernel params n i j = 1 := by
  exact pathKernel_rowsum params hvalid n hn i

example (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hall : allBroadcast params n x) :
    beliefStep params n x =
      FiniteConsensus.applyKernel (pathKernel params n) x := by
  exact propagate_eq_kernel params hvalid n hn x hall

example : allBroadcast threeQuarterThird 3
    (beliefStep threeQuarterThird 3 thresholdVector) := by
  exact allBroadcast_step threeQuarterThird
    (by norm_num [threeQuarterThird, ResponseParameters.Valid]) 3 (by decide)
    thresholdVector (by
      intro i
      fin_cases i <;> norm_num [threeQuarterThird, thresholdVector])

example (k : Nat) : allBroadcast threeQuarterThird 3
    ((beliefStep threeQuarterThird 3)^[k] thresholdVector) := by
  exact allBroadcast_iterate threeQuarterThird
    (by norm_num [threeQuarterThird, ResponseParameters.Valid]) 3 (by decide)
    thresholdVector (by
      intro i
      fin_cases i <;> norm_num [threeQuarterThird, thresholdVector]) k
