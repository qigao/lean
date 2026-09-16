import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

namespace NarrativeDynamics.FitnessABMPathNExposureConvergence

open NarrativeDynamics.FitnessABMPathNExposure
open Filter Topology
open scoped BigOperators

private theorem path2_degree_zero :
    FitnessABMPathN.degree 2 (0 : Fin 2) = 1 := by
  decide_cbv

private theorem path2_degree_one :
    FitnessABMPathN.degree 2 (1 : Fin 2) = 1 := by
  decide_cbv

private theorem path2_neighbors_zero :
    FitnessABMPathN.neighbors 2 (0 : Fin 2) = {(1 : Fin 2)} := by
  decide_cbv

private theorem path2_neighbors_one :
    FitnessABMPathN.neighbors 2 (1 : Fin 2) = {(0 : Fin 2)} := by
  decide_cbv

private theorem path2_step_belief_zero
    (p : ExposureParameters) (s : State 2)
    (hb : allBroadcast p s) :
    (step p 2 s 0).belief =
      (1 - p.receptivityAt ((s 0).exposure + 1)) * (s 0).belief +
        p.receptivityAt ((s 0).exposure + 1) * (s 1).belief := by
  have hi := incoming_eq_degree p 2 s hb (0 : Fin 2)
  have hbr := broadcasters_eq_neighbors p 2 s hb (0 : Fin 2)
  have hmean : broadcasterMean p s (0 : Fin 2) = (s 1).belief := by
    unfold broadcasterMean
    rw [hbr, path2_neighbors_zero, hi, path2_degree_zero]
    simp
  unfold step
  dsimp only
  rw [hi, path2_degree_zero]
  norm_num [hmean]

private theorem path2_step_belief_one
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    (step p 2 s 1).belief =
      (1 - p.receptivityAt ((s 0).exposure + 1)) * (s 1).belief +
        p.receptivityAt ((s 0).exposure + 1) * (s 0).belief := by
  have hi := incoming_eq_degree p 2 s hb (1 : Fin 2)
  have hbr := broadcasters_eq_neighbors p 2 s hb (1 : Fin 2)
  have hmean : broadcasterMean p s (1 : Fin 2) = (s 0).belief := by
    unfold broadcasterMean
    rw [hbr, path2_neighbors_one, hi, path2_degree_one]
    simp
  unfold step
  dsimp only
  rw [hi, path2_degree_one]
  norm_num [hmean, ← he]

theorem path2_equal_exposure_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (((step p 2)^[k] s) 0).exposure =
      (((step p 2)^[k] s) 1).exposure := by
  have h0 := exposure_iterate p hvalid 2 (by omega) s hb k (0 : Fin 2)
  have h1 := exposure_iterate p hvalid 2 (by omega) s hb k (1 : Fin 2)
  rw [path2_degree_zero] at h0
  rw [path2_degree_one] at h1
  omega

theorem path2_disagreement_step
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    (step p 2 s 0).belief - (step p 2 s 1).belief =
      (1 - 2 * p.receptivityAt ((s 0).exposure + 1)) *
        ((s 0).belief - (s 1).belief) := by
  rw [path2_step_belief_zero p s hb, path2_step_belief_one p s he hb]
  ring

def path2MultiplierProduct (p : ExposureParameters) (e0 k : Nat) : Rat :=
  ∏ r in Finset.range k,
    (1 - 2 * p.receptivityAt (e0 + r + 1))

theorem path2_disagreement_product
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 - beliefs ((step p 2)^[k] s) 1) =
      ((s 0).belief - (s 1).belief) *
        path2MultiplierProduct p (s 0).exposure k := by
  induction k with
  | zero =>
      simp [path2MultiplierProduct, beliefs]
  | succ k ih =>
      rw [Function.iterate_succ_apply']
      have hek := path2_equal_exposure_iterate p hvalid s he hb k
      have hbk := allBroadcast_iterate p hvalid 2 (by omega) s hb k
      rw [path2_disagreement_step p ((step p 2)^[k] s) hek hbk]
      rw [ih]
      have hex := exposure_iterate p hvalid 2 (by omega) s hb k (0 : Fin 2)
      rw [path2_degree_zero] at hex
      have hex' : (((step p 2)^[k] s) 0).exposure = (s 0).exposure + k := by
        simpa using hex
      rw [hex']
      simp [path2MultiplierProduct, Finset.prod_range_succ]
      ring

theorem path2_mean_step
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    ((step p 2 s 0).belief + (step p 2 s 1).belief) / 2 =
      ((s 0).belief + (s 1).belief) / 2 := by
  rw [path2_step_belief_zero p s hb, path2_step_belief_one p s he hb]
  ring

theorem path2_mean_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 + beliefs ((step p 2)^[k] s) 1) / 2 =
      ((s 0).belief + (s 1).belief) / 2 := by
  induction k with
  | zero => rfl
  | succ k ih =>
      rw [Function.iterate_succ_apply']
      have hek := path2_equal_exposure_iterate p hvalid s he hb k
      have hbk := allBroadcast_iterate p hvalid 2 (by omega) s hb k
      rw [path2_mean_step p ((step p 2)^[k] s) hek hbk]
      exact ih

private theorem path2_belief_zero_formula
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    beliefs ((step p 2)^[k] s) 0 =
      ((s 0).belief + (s 1).belief) / 2 +
        (((s 0).belief - (s 1).belief) *
          path2MultiplierProduct p (s 0).exposure k) / 2 := by
  have hm := path2_mean_iterate p hvalid s he hb k
  have hd := path2_disagreement_product p hvalid s he hb k
  linarith

private theorem path2_belief_one_formula
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    beliefs ((step p 2)^[k] s) 1 =
      ((s 0).belief + (s 1).belief) / 2 -
        (((s 0).belief - (s 1).belief) *
          path2MultiplierProduct p (s 0).exposure k) / 2 := by
  have hm := path2_mean_iterate p hvalid s he hb k
  have hd := path2_disagreement_product p hvalid s he hb k
  linarith

theorem path2_consensus_iff_product_tendsto_zero
    (p : ExposureParameters) (hvalid : p.Valid)
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
  let m : Real := ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real)
  let d : Real := (((s 0).belief - (s 1).belief : Rat) : Real)
  have hdRat : (s 0).belief - (s 1).belief ≠ 0 := sub_ne_zero.mpr hne
  have hd : d ≠ 0 := by
    dsimp [d]
    exact_mod_cast hdRat
  constructor
  · intro hprod
    have hP : Tendsto
        (fun k => ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real))
        atTop (nhds 0) := by
      rw [tendsto_zero_iff_norm_tendsto_zero]
      simpa [Real.norm_eq_abs] using hprod
    have hscaled0 : Tendsto
        (fun k => d * ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real))
        atTop (nhds 0) := by
      simpa [d] using Filter.Tendsto.const_mul d hP
    have hscaled : Tendsto
        (fun k =>
          d * ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real) / 2)
        atTop (nhds 0) := by
      simpa using hscaled0.div_const' (2 : Real)
    have hzeroFormula : ∀ k,
        (beliefs ((step p 2)^[k] s) 0 : Real) =
          m + d * ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real) / 2 := by
      intro k
      dsimp [m, d]
      exact_mod_cast path2_belief_zero_formula p hvalid s he hb k
    have honeFormula : ∀ k,
        (beliefs ((step p 2)^[k] s) 1 : Real) =
          m - d * ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real) / 2 := by
      intro k
      dsimp [m, d]
      exact_mod_cast path2_belief_one_formula p hvalid s he hb k
    have hz : Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) 0 : Real))
        atTop (nhds m) := by
      have hlim := Filter.Tendsto.const_add m hscaled
      simpa using hlim.congr'
        (Filter.Eventually.of_forall fun k => (hzeroFormula k).symm)
    have ho : Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) 1 : Real))
        atTop (nhds m) := by
      have hlim := Filter.Tendsto.const_sub m hscaled
      simpa using hlim.congr'
        (Filter.Eventually.of_forall fun k => (honeFormula k).symm)
    intro i
    fin_cases i
    · simpa [m] using hz
    · simpa [m] using ho
  · intro hcons
    have hdiff : Tendsto
        (fun k =>
          (beliefs ((step p 2)^[k] s) 0 : Real) -
            (beliefs ((step p 2)^[k] s) 1 : Real))
        atTop (nhds 0) := by
      have h := (hcons 0).sub (hcons 1)
      simpa using h
    have hscaled : Tendsto
        (fun k =>
          d * ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real))
        atTop (nhds 0) := by
      refine hdiff.congr' (Filter.Eventually.of_forall ?_)
      intro k
      dsimp [d]
      exact_mod_cast path2_disagreement_product p hvalid s he hb k
    have hdiv := hscaled.div_const' d
    have hP : Tendsto
        (fun k => ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real))
        atTop (nhds 0) := by
      have hdiv0 : Tendsto
          (fun k =>
            (d * ((path2MultiplierProduct p (s 0).exposure k : Rat) : Real)) / d)
          atTop (nhds 0) := by
        simpa [hd] using hdiv
      refine hdiv0.congr' (Filter.Eventually.of_forall ?_)
      intro k
      field_simp [hd]
    rw [tendsto_zero_iff_norm_tendsto_zero] at hP
    simpa [Real.norm_eq_abs] using hP

end NarrativeDynamics.FitnessABMPathNExposureConvergence
