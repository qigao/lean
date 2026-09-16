import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology
open scoped BigOperators

private def indexSchedule : ExposureParameters :=
  ⟨fun e => if e = 1 then 1/4 else if e = 2 then 1/2 else 1/3, 0⟩

private theorem indexSchedule_valid : indexSchedule.Valid := by
  constructor
  · intro e
    by_cases h1 : e = 1
    · norm_num [indexSchedule, h1]
    · by_cases h2 : e = 2
      · norm_num [indexSchedule, h1, h2]
      · norm_num [indexSchedule, h1, h2]
  · norm_num [indexSchedule]

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
  exact exposure_iterate indexSchedule indexSchedule_valid 3 (by omega)
    path3zero (by
      intro j
      fin_cases j <;> norm_num [allBroadcast, indexSchedule, path3zero]) k i

example (k : Nat) :
    beliefs ((step indexSchedule 3)^[k] path3zero) =
      varyingTrajectory (kernelSchedule indexSchedule 3 path3zero)
        (beliefs path3zero) k := by
  exact beliefs_iterate_eq_varyingTrajectory indexSchedule indexSchedule_valid
    3 (by omega) path3zero (by
      intro j
      fin_cases j <;> norm_num [allBroadcast, indexSchedule, path3zero]) k

private def path2Product (p : ExposureParameters) (e0 k : Nat) : Rat :=
  ∏ r ∈ Finset.range k,
    (1 - 2 * p.receptivityAt (e0 + r + 1))

example (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 - beliefs ((step p 2)^[k] s) 1) =
      ((s 0).belief - (s 1).belief) *
        path2Product p (s 0).exposure k := by
  exact path2_disagreement_product p hvalid s he hb k

example (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 + beliefs ((step p 2)^[k] s) 1) / 2 =
      ((s 0).belief + (s 1).belief) / 2 := by
  exact path2_mean_iterate p hvalid s he hb k

example (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hne : (s 0).belief ≠ (s 1).belief) :
    (Tendsto
      (fun k => |((path2Product p (s 0).exposure k : Rat) : Real)|)
      atTop (nhds 0)) ↔
    (∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop
        (nhds ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real))) := by
  exact path2_consensus_iff_product_tendsto_zero p hvalid s he hb hne

#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.exposure_iterate
#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.beliefs_iterate_eq_varyingTrajectory
