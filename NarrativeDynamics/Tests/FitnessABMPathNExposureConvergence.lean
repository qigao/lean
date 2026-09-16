import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology

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

example (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 - beliefs ((step p 2)^[k] s) 1) =
      ((s 0).belief - (s 1).belief) *
        path2MultiplierProduct p (s 0).exposure k := by
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
      (fun k => |((path2MultiplierProduct p (s 0).exposure k : Rat) : Real)|)
      atTop (nhds 0)) ↔
    (∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop
        (nhds ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real))) := by
  exact path2_consensus_iff_product_tendsto_zero p hvalid s he hb hne

-- Task 4 RED: exact schedule boundaries and the Path3 mean counterexample.
example : slowZeroSchedule.Valid := slowZeroSchedule_valid
example : nearOneSchedule.Valid := nearOneSchedule_valid
example : harmonicSchedule.Valid := harmonicSchedule_valid

example (k : Nat) :
    path2MultiplierProduct slowZeroSchedule 0 k =
      (k + 2 : Rat) / (2 * (k + 1 : Rat)) := by
  exact slowZero_product k

-- The merged same-step ordering queries exposure 1 first, so k=1 is 3/4.
example : path2MultiplierProduct slowZeroSchedule 0 1 = 3/4 := by
  simpa using slowZero_product 1

example (k : Nat) :
    path2MultiplierProduct nearOneSchedule 0 k =
      (-1 : Rat) ^ k * ((k + 2 : Rat) / (2 * (k + 1 : Rat))) := by
  exact nearOne_product k

example (k : Nat) :
    path2MultiplierProduct harmonicSchedule 0 k =
      1 / (k + 1 : Rat) := by
  exact harmonic_product k

example :
    Tendsto
      (fun k => (path2MultiplierProduct slowZeroSchedule 0 k : Real))
      atTop (nhds (1/2 : Real)) := by
  exact slowZero_product_tendsto_half

example :
    ¬ ∀ i : Fin 2,
      Tendsto
        (fun k =>
          (beliefs ((step slowZeroSchedule 2)^[k]
            (![⟨1, 0⟩, ⟨0, 0⟩] : State 2)) i : Real))
        atTop (nhds (1/2 : Real)) := by
  simpa using slowZero_not_consensus

example :
    ¬ ∃ c : Real, ∀ i : Fin 2,
      Tendsto
        (fun k =>
          (beliefs ((step nearOneSchedule 2)^[k]
            (![⟨1, 0⟩, ⟨0, 0⟩] : State 2)) i : Real))
        atTop (nhds c) := by
  simpa using nearOne_not_convergent

example :
    ∀ i : Fin 2,
      Tendsto
        (fun k =>
          (beliefs ((step harmonicSchedule 2)^[k]
            (![⟨1, 0⟩, ⟨0, 0⟩] : State 2)) i : Real))
        atTop (nhds (1/2 : Real)) := by
  simpa using harmonic_consensus

example :
    beliefs (step degreeSplitSchedule 3 degreeSplitState) =
      ![3/4, 1/4, 0] := by
  exact degreeSplit_step_beliefs

example : FitnessABMPathN.mean 3 (beliefs degreeSplitState) = 1/4 := by
  exact degreeSplit_mean_before

example :
    FitnessABMPathN.mean 3
      (beliefs (step degreeSplitSchedule 3 degreeSplitState)) = 5/16 := by
  exact degreeSplit_mean_after

example :
    FitnessABMPathN.mean 3
        (beliefs (step degreeSplitSchedule 3 degreeSplitState)) ≠
      FitnessABMPathN.mean 3 (beliefs degreeSplitState) := by
  exact exposure_degree_weighted_mean_not_invariant

#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.exposure_iterate
#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.beliefs_iterate_eq_varyingTrajectory
#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.path2_disagreement_product
#print axioms NarrativeDynamics.FitnessABMPathNExposureConvergence.path2_consensus_iff_product_tendsto_zero
