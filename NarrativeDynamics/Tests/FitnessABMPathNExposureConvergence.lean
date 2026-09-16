import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology

private def indexSchedule : ExposureParameters :=
  ⟨fun e => if e = 1 then 1/4 else if e = 2 then 1/2 else 1/3, 0⟩

private def path3zero : State 3 :=
  ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]

example : allBroadcast indexSchedule path3zero := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, indexSchedule, path3zero]

example : incoming indexSchedule path3zero 0 = 1 := by decide_cbv
example : incoming indexSchedule path3zero 1 = 2 := by decide_cbv

-- First transition must query alpha(1) at endpoints and alpha(2) at the middle.
example :
    exposureKernel indexSchedule 3 (fun _ => 0) 0 0 = 3/4 := by
  have hd : FitnessABMPathN.degree 3 (0 : Fin 3) = 1 := by decide_cbv
  norm_num [exposureKernel, indexSchedule, hd]

example :
    exposureKernel indexSchedule 3 (fun _ => 0) 1 1 = 1/2 := by
  have hd : FitnessABMPathN.degree 3 (1 : Fin 3) = 2 := by decide_cbv
  norm_num [exposureKernel, indexSchedule, hd]

example (k : Nat) (i : Fin 3) :
    (((step indexSchedule 3)^[k] path3zero) i).exposure =
      (path3zero i).exposure + k * FitnessABMPathN.degree 3 i := by
  exact exposure_iterate indexSchedule
    (by norm_num [indexSchedule, ExposureParameters.Valid]) 3 (by omega)
    path3zero (by
      intro j
      fin_cases j <;> norm_num [allBroadcast, indexSchedule, path3zero]) k i

example (k : Nat) :
    beliefs ((step indexSchedule 3)^[k] path3zero) =
      varyingTrajectory (kernelSchedule indexSchedule 3 path3zero)
        (beliefs path3zero) k := by
  exact beliefs_iterate_eq_varyingTrajectory indexSchedule
    (by norm_num [indexSchedule, ExposureParameters.Valid]) 3 (by omega)
    path3zero (by
      intro j
      fin_cases j <;> norm_num [allBroadcast, indexSchedule, path3zero]) k

#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.exposure_iterate
#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.beliefs_iterate_eq_varyingTrajectory
