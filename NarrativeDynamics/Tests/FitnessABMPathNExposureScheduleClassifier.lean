import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifier

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier
open Filter Topology
open scoped BigOperators

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

example (f : Nat → Real) :
    (∏ r ∈ Finset.range (2 + 3), f r) =
      (∏ r ∈ Finset.range 2, f r) *
        (∏ r ∈ Finset.range 3, f (2 + r)) := by
  exact prod_range_add_split f 2 3

example (f : Nat → Real)
    (hzero : (∏ r ∈ Finset.range 2, f r) = 0) :
    (∏ r ∈ Finset.range (2 + 3), f r) = 0 := by
  exact prod_range_add_eq_zero_of_prefix_zero f 2 3 hzero

example (f : Nat → Real)
    (hprefix : (∏ r ∈ Finset.range 2, f r) ≠ 0) :
    Tendsto
        (fun k => ∏ r ∈ Finset.range (2 + k), f r)
        atTop (nhds 0) ↔
      Tendsto
        (fun k => ∏ r ∈ Finset.range k, f (2 + r))
        atTop (nhds 0) := by
  exact tendsto_zero_prod_range_add_iff f 2 hprefix

private def periodContracting : Fin 2 → Rat := ![1/4, 3/4]
private def periodBoundary : Fin 2 → Rat := ![0, 1]

private def periodContractingParams : ExposureParameters :=
  ⟨periodicReceptivity 2 (by decide) periodContracting, 0⟩

private def periodBoundaryParams : ExposureParameters :=
  ⟨periodicReceptivity 2 (by decide) periodBoundary, 0⟩

example :
    Tendsto
      (fun k => |(path2MultiplierProduct periodContractingParams 0 k : Real)|)
      atTop (nhds 0) := by
  exact periodic_abs_product_tendsto_zero_of_contracting_entry
    2 (by decide) periodContracting
    (by
      intro i
      fin_cases i <;> norm_num [periodContracting])
    (0 : Fin 2)
    (by norm_num [periodContracting])
    0

example :
    ¬ Tendsto
      (fun k => |(path2MultiplierProduct periodBoundaryParams 0 k : Real)|)
      atTop (nhds 0) := by
  exact periodic_abs_product_not_tendsto_zero_of_boundary_values
    2 (by decide) periodBoundary
    (by
      intro i
      fin_cases i <;> norm_num [periodBoundary])
    0

private def alternatingContractingParams : ExposureParameters :=
  ⟨alternatingReceptivity (1/4) (3/4), 0⟩

example : alternatingReceptivity (1/4) (3/4) 0 = 1/4 := by
  norm_num [alternatingReceptivity, periodicReceptivity]

example : alternatingReceptivity (1/4) (3/4) 1 = 3/4 := by
  norm_num [alternatingReceptivity, periodicReceptivity]

example :
    Tendsto
      (fun k => |(path2MultiplierProduct alternatingContractingParams 0 k : Real)|)
      atTop (nhds 0) := by
  simpa [alternatingContractingParams, alternatingReceptivity, periodContracting] using
    (periodic_abs_product_tendsto_zero_of_contracting_entry
      2 (by decide) periodContracting
      (by
        intro i
        fin_cases i <;> norm_num [periodContracting])
      (0 : Fin 2)
      (by norm_num [periodContracting])
      0)

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests