import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifierPolynomial

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

private def piecewiseOverrides : Finset Nat := {1, 2}

private def piecewiseOverrideValue (e : Nat) : Rat :=
  if e = 1 then 1/8 else if e = 2 then 3/8 else 0

private def piecewiseInteriorParams : ExposureParameters :=
  ⟨piecewiseConstantTailReceptivity
      piecewiseOverrides piecewiseOverrideValue (1/4), 0⟩

private def piecewiseBoundaryParams : ExposureParameters :=
  ⟨piecewiseConstantTailReceptivity
      piecewiseOverrides piecewiseOverrideValue 0, 0⟩

private def zeroPrefixOverrideValue (e : Nat) : Rat :=
  if e = 1 then 1/2 else 0

private def piecewiseZeroPrefixParams : ExposureParameters :=
  ⟨piecewiseConstantTailReceptivity {1} zeroPrefixOverrideValue 0, 0⟩

example :
    Tendsto
      (fun k => |(path2MultiplierProduct piecewiseInteriorParams 0 k : Real)|)
      atTop (nhds 0) := by
  exact piecewise_constant_tail_abs_product_tendsto_zero_of_interior_tail
    piecewiseInteriorParams 0 2 (1/4)
    (by norm_num)
    (by
      intro r hr
      simp [piecewiseInteriorParams, piecewiseConstantTailReceptivity,
        piecewiseOverrides]
      omega)

example :
    ¬ Tendsto
      (fun k => |(path2MultiplierProduct piecewiseBoundaryParams 0 k : Real)|)
      atTop (nhds 0) := by
  exact piecewise_constant_tail_abs_product_not_tendsto_zero_of_boundary_tail
    piecewiseBoundaryParams 0 2 0
    (Or.inl rfl)
    (by
      intro r hr
      simp [piecewiseBoundaryParams, piecewiseConstantTailReceptivity,
        piecewiseOverrides]
      omega)
    (by
      norm_num [piecewiseBoundaryParams, piecewiseConstantTailReceptivity,
        piecewiseOverrides, piecewiseOverrideValue, path2MultiplierProduct,
        Finset.prod_range_succ])

example :
    Tendsto
      (fun k => |(path2MultiplierProduct piecewiseZeroPrefixParams 0 k : Real)|)
      atTop (nhds 0) := by
  exact piecewise_constant_tail_abs_product_tendsto_zero_of_zero_prefix
    piecewiseZeroPrefixParams 0 1
    (by
      norm_num [piecewiseZeroPrefixParams, piecewiseConstantTailReceptivity,
        zeroPrefixOverrideValue, path2MultiplierProduct])

example (e : Nat) : 0 < polynomialDecay (1/4) 2 2 e := by
  exact polynomialDecay_pos (1/4) 2 2 e
    (by norm_num) (by norm_num) (by norm_num)

example (e : Nat) : polynomialDecay (1/4) 2 2 e ≤ 1 := by
  exact polynomialDecay_le_one (1/4) 2 2 e
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

example (e : Nat) (target : DecayTarget) :
    0 ≤ polynomialReceptivity (1/4) 2 2 target e ∧
      polynomialReceptivity (1/4) 2 2 target e ≤ 1 := by
  exact polynomialReceptivity_bounds (1/4) 2 2 target e
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

example (m : Nat → Real)
    (hm0 : ∀ k, 0 ≤ m k)
    (hmhalf : ∀ k, m k < 1/2)
    (hdiv : Tendsto (fun n => ∑ k ∈ Finset.range n, m k) atTop atTop) :
    Tendsto
      (fun n => ∏ k ∈ Finset.range n, (1 - 2 * m k))
      atTop (nhds 0) := by
  exact abs_product_tendsto_zero_of_mixing_sum_tendsto_atTop
    m hm0 hmhalf hdiv

example (m : Nat → Real)
    (hm0 : ∀ k, 0 ≤ m k)
    (hmhalf : ∀ k, m k < 1/2)
    (hsum : Summable m) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun n => ∏ k ∈ Finset.range n, (1 - 2 * m k))
        atTop (nhds L) := by
  exact abs_product_has_nonzero_limit_of_summable_mixing
    m hm0 hmhalf hsum

private def polyP1Params : ExposureParameters :=
  ⟨polynomialReceptivity (1/3) 1 2 DecayTarget.zero, 0⟩

private def polyP2Params : ExposureParameters :=
  ⟨polynomialReceptivity (1/4) 2 2 DecayTarget.zero, 0⟩

example :
    Tendsto
      (fun k => |(path2MultiplierProduct polyP1Params 0 k : Real)|)
      atTop (nhds 0) := by
  simpa [polyP1Params] using
    (polynomial_abs_product_tendsto_zero_of_p_eq_one
      (1/3) 2 0 DecayTarget.zero
      (by norm_num) (by norm_num) (by norm_num))

example
    (hnozero : ∀ r,
      polynomialReceptivity (1/4) 2 2 DecayTarget.zero (r + 1) ≠ 1/2) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun k => |(path2MultiplierProduct polyP2Params 0 k : Real)|)
        atTop (nhds L) := by
  simpa [polyP2Params] using
    (polynomial_abs_product_has_nonzero_limit_of_two_le_p
      (1/4) 2 2 0 DecayTarget.zero
      (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by simpa using hnozero))

private def harmonicGenericParams : ExposureParameters :=
  ⟨harmonicReceptivity (1/2) 1 DecayTarget.zero, 0⟩

example :
    Tendsto
      (fun k => |(path2MultiplierProduct harmonicGenericParams 0 k : Real)|)
      atTop (nhds 0) := by
  simpa [harmonicGenericParams] using
    (harmonic_abs_product_tendsto_zero
      (1/2) 1 0 DecayTarget.zero
      (by norm_num) (by norm_num) (by norm_num))


private def expZeroParams : ExposureParameters :=
  ⟨exponentialReceptivity (1/4) (1/2) 0 DecayTarget.zero, 0⟩

private def expOneParams : ExposureParameters :=
  ⟨exponentialReceptivity (1/4) (1/2) 0 DecayTarget.one, 0⟩

example
    (hnozero : ∀ r,
      exponentialReceptivity (1/4) (1/2) 0 DecayTarget.zero (r + 1) ≠ 1/2) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun k => |(path2MultiplierProduct expZeroParams 0 k : Real)|)
        atTop (nhds L) := by
  simpa [expZeroParams] using
    (exponential_abs_product_has_nonzero_limit
      (1/4) (1/2) 0 0 DecayTarget.zero
      (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by simpa using hnozero))

example
    (hnozero : ∀ r,
      exponentialReceptivity (1/4) (1/2) 0 DecayTarget.one (r + 1) ≠ 1/2) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun k => |(path2MultiplierProduct expOneParams 0 k : Real)|)
        atTop (nhds L) := by
  simpa [expOneParams] using
    (exponential_abs_product_has_nonzero_limit
      (1/4) (1/2) 0 0 DecayTarget.one
      (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by simpa using hnozero))

example
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    {L : Real}
    (hprod : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure k : Real))
      atTop (nhds L)) :
    ∃ c0 c1 : Real,
      Tendsto
          (fun k => (beliefs ((step p 2)^[k] s) 0 : Real))
          atTop (nhds c0) ∧
        Tendsto
          (fun k => (beliefs ((step p 2)^[k] s) 1 : Real))
          atTop (nhds c1) := by
  exact path2_nodewise_converges_of_signed_product_tendsto
    p hvalid s he hb hprod

example
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hne : (s 0).belief ≠ (s 1).belief)
    {L : Real} (hL : L ≠ 0)
    (heven : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure (2*k) : Real))
      atTop (nhds L))
    (hodd : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure (2*k+1) : Real))
      atTop (nhds (-L))) :
    ¬ ∃ c0 c1 : Real,
      Tendsto
          (fun k => (beliefs ((step p 2)^[k] s) 0 : Real))
          atTop (nhds c0) ∧
        Tendsto
          (fun k => (beliefs ((step p 2)^[k] s) 1 : Real))
          atTop (nhds c1) := by
  exact path2_oscillatory_nonconvergence_of_even_odd_product_limits
    p hvalid s he hb hne hL heven hodd

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests