import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifier

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier
open Filter Topology

example : applyDecayTarget DecayTarget.zero (1/4 : Rat) = 1/4 := by
  norm_num [applyDecayTarget]

example : applyDecayTarget DecayTarget.one (1/4 : Rat) = 3/4 := by
  norm_num [applyDecayTarget]

example : mixingMass (1/4 : Rat) = 1/4 := by
  norm_num [mixingMass]

example : mixingMass (3/4 : Rat) = 1/4 := by
  norm_num [mixingMass]

example : |(1 : Rat) - 2 * (3/4 : Rat)| = 1 - 2 * mixingMass (3/4 : Rat) := by
  exact abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass
    (a := 3/4) (by norm_num) (by norm_num)

private def equalParams : ExposureParameters :=
  ⟨fun _ => 1/4, 0⟩

private def equalState : State 2 :=
  ![⟨1/2, 0⟩, ⟨1/2, 0⟩]

private theorem equalParams_valid : equalParams.Valid := by
  constructor
  · intro e
    norm_num [equalParams]
  · norm_num [equalParams]

private theorem equalState_allBroadcast : allBroadcast equalParams equalState := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, equalParams, equalState]

example :
    ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step equalParams 2)^[k] equalState) i : Real))
        atTop (nhds (1/2 : Real)) := by
  simpa [equalState] using
    (path2_equal_belief_consensus
      equalParams equalParams_valid equalState
      (by norm_num [equalState]) equalState_allBroadcast
      (by norm_num [equalState]))

example :
    Tendsto (fun _ : Nat => (2 : Real) * 0) atTop (nhds 0) ↔
      Tendsto (fun _ : Nat => (0 : Real)) atTop (nhds 0) := by
  exact tendsto_zero_const_mul_iff (f := fun _ : Nat => (0 : Real))
    (c := 2) (by norm_num)

example :
    Tendsto (fun _ : Nat => (3 : Real) * 2) atTop (nhds ((3 : Real) * 2)) := by
  have h : Tendsto (fun _ : Nat => (2 : Real)) atTop (nhds 2) :=
    tendsto_const_nhds
  exact tendsto_const_mul (c := 3) h

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests
