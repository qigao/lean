import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifierExponential

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology

private theorem classifier_evenIndex_tendsto :
    Tendsto (fun k : Nat => 2 * k) atTop atTop := by
  refine Filter.tendsto_atTop.2 ?_
  intro b
  exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩

private theorem classifier_oddIndex_tendsto :
    Tendsto (fun k : Nat => 2 * k + 1) atTop atTop := by
  refine Filter.tendsto_atTop.2 ?_
  intro b
  exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩

theorem path2_nodewise_converges_of_signed_product_tendsto
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    {L : Real}
    (hprod : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure k : Real))
      atTop (nhds L)) :
    ∃ c0 c1 : Real,
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 0 : Real)) atTop (nhds c0) ∧
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 1 : Real)) atTop (nhds c1) := by
  let μ : Real := ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real)
  let d : Real := (((s 0).belief - (s 1).belief : Rat) : Real)
  let P : Nat → Real := fun k =>
    (path2MultiplierProduct p (s 0).exposure k : Real)
  have hP : Tendsto P atTop (nhds L) := by
    simpa [P] using hprod
  have hscaled0 : Tendsto (fun k => d * P k) atTop (nhds (d * L)) := by
    simpa using Filter.Tendsto.const_mul d hP
  have hscaled :
      Tendsto (fun k => d * P k / 2) atTop (nhds (d * L / 2)) := by
    have h := hscaled0.mul_const ((2 : Real)⁻¹)
    simpa [div_eq_mul_inv] using h
  have hzeroFormula : ∀ k,
      (beliefs ((step p 2)^[k] s) 0 : Real) = μ + d * P k / 2 := by
    intro k
    have hm := path2_mean_iterate p hvalid s he hb k
    have hd := path2_disagreement_product p hvalid s he hb k
    have hsum :
        beliefs ((step p 2)^[k] s) 0 +
            beliefs ((step p 2)^[k] s) 1 =
          (s 0).belief + (s 1).belief := by
      linarith [hm]
    have hRat :
        beliefs ((step p 2)^[k] s) 0 =
          ((s 0).belief + (s 1).belief) / 2 +
            (((s 0).belief - (s 1).belief) *
              path2MultiplierProduct p (s 0).exposure k) / 2) := by
      calc
        beliefs ((step p 2)^[k] s) 0 =
            ((beliefs ((step p 2)^[k] s) 0 +
                beliefs ((step p 2)^[k] s) 1) +
              (beliefs ((step p 2)^[k] s) 0 -
                beliefs ((step p 2)^[k] s) 1)) / 2 := by ring
        _ = ((s 0).belief + (s 1).belief) / 2 +
            (((s 0).belief - (s 1).belief) *
              path2MultiplierProduct p (s 0).exposure k) / 2) := by
          rw [hsum, hd]
          ring
    have hReal := congrArg (fun q : Rat => (q : Real)) hRat
    push_cast at hReal
    simpa [μ, d, P] using hReal
  have honeFormula : ∀ k,
      (beliefs ((step p 2)^[k] s) 1 : Real) = μ - d * P k / 2 := by
    intro k
    have hm := path2_mean_iterate p hvalid s he hb k
    have hd := path2_disagreement_product p hvalid s he hb k
    have hsum :
        beliefs ((step p 2)^[k] s) 0 +
            beliefs ((step p 2)^[k] s) 1 =
          (s 0).belief + (s 1).belief := by
      linarith [hm]
    have hRat :
        beliefs ((step p 2)^[k] s) 1 =
          ((s 0).belief + (s 1).belief) / 2 -
            (((s 0).belief - (s 1).belief) *
              path2MultiplierProduct p (s 0).exposure k) / 2) := by
      calc
        beliefs ((step p 2)^[k] s) 1 =
            ((beliefs ((step p 2)^[k] s) 0 +
                beliefs ((step p 2)^[k] s) 1) -
              (beliefs ((step p 2)^[k] s) 0 -
                beliefs ((step p 2)^[k] s) 1)) / 2 := by ring
        _ = ((s 0).belief + (s 1).belief) / 2 -
            (((s 0).belief - (s 1).belief) *
              path2MultiplierProduct p (s 0).exposure k) / 2) := by
          rw [hsum, hd]
          ring
    have hReal := congrArg (fun q : Rat => (q : Real)) hRat
    push_cast at hReal
    simpa [μ, d, P] using hReal
  refine ⟨μ + d * L / 2, μ - d * L / 2, ?_, ?_⟩
  · have hlim := Filter.Tendsto.const_add μ hscaled
    have hlim' :
        Tendsto (fun k => μ + d * P k / 2) atTop
          (nhds (μ + d * L / 2)) := by
      simpa using hlim
    exact hlim'.congr'
      (Filter.Eventually.of_forall fun k => (hzeroFormula k).symm)
  · have hlim := Filter.Tendsto.const_sub μ hscaled
    have hlim' :
        Tendsto (fun k => μ - d * P k / 2) atTop
          (nhds (μ - d * L / 2)) := by
      simpa using hlim
    exact hlim'.congr'
      (Filter.Eventually.of_forall fun k => (honeFormula k).symm)

theorem path2_not_consensus_of_signed_product_tendsto_nonzero
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hne : (s 0).belief ≠ (s 1).belief)
    {L : Real} (hL : L ≠ 0)
    (hprod : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure k : Real))
      atTop (nhds L)) :
    ¬ ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop
        (nhds ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real)) := by
  intro hcons
  have habs0 :=
    (path2_consensus_iff_product_tendsto_zero
      p hvalid s he hb hne).2 hcons
  have habsL :
      Tendsto
        (fun k => |(path2MultiplierProduct p (s 0).exposure k : Real)|)
        atTop (nhds |L|) := by
    simpa using hprod.abs
  have huniq := tendsto_nhds_unique habs0 habsL
  have hz : |L| = 0 := huniq
  exact hL (abs_eq_zero.mp hz)

theorem path2_oscillatory_nonconvergence_of_even_odd_product_limits
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
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 0 : Real)) atTop (nhds c0) ∧
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 1 : Real)) atTop (nhds c1) := by
  rintro ⟨c0, c1, h0, h1⟩
  let d : Real := (((s 0).belief - (s 1).belief : Rat) : Real)
  have hdRat : (s 0).belief - (s 1).belief ≠ 0 := sub_ne_zero.mpr hne
  have hd : d ≠ 0 := by
    dsimp [d]
    exact_mod_cast hdRat
  have hdiff :
      Tendsto
        (fun k =>
          (beliefs ((step p 2)^[k] s) 0 : Real) -
            (beliefs ((step p 2)^[k] s) 1 : Real))
        atTop (nhds (c0 - c1)) := by
    simpa using h0.sub h1
  have hdiffEven :=
    hdiff.comp classifier_evenIndex_tendsto
  have hdiffOdd :=
    hdiff.comp classifier_oddIndex_tendsto
  have hscaledEven :
      Tendsto
        (fun k => d *
          (path2MultiplierProduct p (s 0).exposure (2*k) : Real))
        atTop (nhds (d * L)) := by
    simpa using Filter.Tendsto.const_mul d heven
  have hscaledOdd :
      Tendsto
        (fun k => d *
          (path2MultiplierProduct p (s 0).exposure (2*k+1) : Real))
        atTop (nhds (d * (-L))) := by
    simpa using Filter.Tendsto.const_mul d hodd
  have hdeven :
      Tendsto
        (fun k =>
          (beliefs ((step p 2)^[2*k] s) 0 : Real) -
            (beliefs ((step p 2)^[2*k] s) 1 : Real))
        atTop (nhds (d * L)) := by
    refine hscaledEven.congr' (Filter.Eventually.of_forall ?_)
    intro k
    have hRat :=
      path2_disagreement_product p hvalid s he hb (2*k)
    dsimp [d]
    exact_mod_cast hRat.symm
  have hdodd :
      Tendsto
        (fun k =>
          (beliefs ((step p 2)^[2*k+1] s) 0 : Real) -
            (beliefs ((step p 2)^[2*k+1] s) 1 : Real))
        atTop (nhds (d * (-L))) := by
    refine hscaledOdd.congr' (Filter.Eventually.of_forall ?_)
    intro k
    have hRat :=
      path2_disagreement_product p hvalid s he hb (2*k+1)
    dsimp [d]
    exact_mod_cast hRat.symm
  have heqEven := tendsto_nhds_unique hdiffEven hdeven
  have heqOdd := tendsto_nhds_unique hdiffOdd hdodd
  have hzero : d * L = 0 := by
    linarith
  exact (mul_ne_zero hd hL) hzero

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier
