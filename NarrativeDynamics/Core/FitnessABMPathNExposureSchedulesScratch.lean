import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

namespace NarrativeDynamics.FitnessABMPathNExposureConvergence

open NarrativeDynamics.FitnessABMPathNExposure
open Filter Topology
open scoped BigOperators

def slowZeroSchedule : ExposureParameters :=
  ⟨fun e => 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)), 0⟩

def nearOneSchedule : ExposureParameters :=
  ⟨fun e => 1 - 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)), 0⟩

def harmonicSchedule : ExposureParameters :=
  ⟨fun e => 1 / (2 * ((e + 1 : Nat) : Rat)), 0⟩

private theorem slowFraction_bounds (e : Nat) :
    0 ≤ (1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) : Rat) ∧
      (1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) : Rat) ≤ 1 := by
  let x : Rat := ((e + 1 : Nat) : Rat)
  have hx : (1 : Rat) ≤ x := by
    dsimp [x]
    exact_mod_cast Nat.succ_le_succ (Nat.zero_le e)
  have hx0 : (0 : Rat) < x := lt_of_lt_of_le (by norm_num) hx
  have hden : (0 : Rat) < 2 * x ^ 2 := by positivity
  constructor
  · exact div_nonneg (by norm_num) hden.le
  · rw [div_le_iff₀ hden]
    have hx2 : (1 : Rat) ≤ x ^ 2 := by
      nlinarith [sq_nonneg (x - 1)]
    nlinarith

private theorem harmonicFraction_bounds (e : Nat) :
    0 ≤ (1 / (2 * ((e + 1 : Nat) : Rat)) : Rat) ∧
      (1 / (2 * ((e + 1 : Nat) : Rat)) : Rat) ≤ 1 := by
  let x : Rat := ((e + 1 : Nat) : Rat)
  have hx : (1 : Rat) ≤ x := by
    dsimp [x]
    exact_mod_cast Nat.succ_le_succ (Nat.zero_le e)
  have hden : (0 : Rat) < 2 * x := by nlinarith
  constructor
  · exact div_nonneg (by norm_num) hden.le
  · rw [div_le_iff₀ hden]
    nlinarith

theorem slowZeroSchedule_valid : slowZeroSchedule.Valid := by
  constructor
  · intro e
    simpa [slowZeroSchedule] using slowFraction_bounds e
  · norm_num [slowZeroSchedule]

theorem nearOneSchedule_valid : nearOneSchedule.Valid := by
  constructor
  · intro e
    have h := slowFraction_bounds e
    change
      0 ≤ (1 : Rat) - 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) ∧
        (1 : Rat) - 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)) ≤ 1
    exact ⟨by linarith [h.2], by linarith [h.1]⟩
  · norm_num [nearOneSchedule]

theorem harmonicSchedule_valid : harmonicSchedule.Valid := by
  constructor
  · intro e
    simpa [harmonicSchedule] using harmonicFraction_bounds e
  · norm_num [harmonicSchedule]

private theorem slowZero_factor (r : Nat) :
    1 - 2 * slowZeroSchedule.receptivityAt (r + 1) =
      (((r + 1 : Nat) : Rat) * ((r + 3 : Nat) : Rat)) /
        (((r + 2 : Nat) : Rat) ^ 2) := by
  dsimp [slowZeroSchedule]
  have h : (((r + 2 : Nat) : Rat)) ≠ 0 := by positivity
  field_simp [h]
  push_cast
  ring

private theorem nearOne_factor (r : Nat) :
    1 - 2 * nearOneSchedule.receptivityAt (r + 1) =
      -(1 - 2 * slowZeroSchedule.receptivityAt (r + 1)) := by
  dsimp [nearOneSchedule, slowZeroSchedule]
  ring

private theorem harmonic_factor (r : Nat) :
    1 - 2 * harmonicSchedule.receptivityAt (r + 1) =
      ((r + 1 : Nat) : Rat) / ((r + 2 : Nat) : Rat) := by
  dsimp [harmonicSchedule]
  have h : (((r + 2 : Nat) : Rat)) ≠ 0 := by positivity
  field_simp [h]
  push_cast
  ring

theorem slowZero_product (k : Nat) :
    path2MultiplierProduct slowZeroSchedule 0 k =
      (k + 2 : Rat) / (2 * (k + 1 : Rat)) := by
  induction k with
  | zero => norm_num [path2MultiplierProduct]
  | succ k ih =>
      rw [show path2MultiplierProduct slowZeroSchedule 0 (Nat.succ k) =
          path2MultiplierProduct slowZeroSchedule 0 k *
            (1 - 2 * slowZeroSchedule.receptivityAt (k + 1)) by
        simp [path2MultiplierProduct, Finset.prod_range_succ]]
      rw [ih, slowZero_factor]
      push_cast
      have hk1 : ((k : Rat) + 1) ≠ 0 := by positivity
      have hk2 : ((k : Rat) + 2) ≠ 0 := by positivity
      field_simp [hk1, hk2]
      ring

theorem nearOne_product (k : Nat) :
    path2MultiplierProduct nearOneSchedule 0 k =
      (-1 : Rat) ^ k * ((k + 2 : Rat) / (2 * (k + 1 : Rat))) := by
  induction k with
  | zero => norm_num [path2MultiplierProduct]
  | succ k ih =>
      rw [show path2MultiplierProduct nearOneSchedule 0 (Nat.succ k) =
          path2MultiplierProduct nearOneSchedule 0 k *
            (1 - 2 * nearOneSchedule.receptivityAt (k + 1)) by
        simp [path2MultiplierProduct, Finset.prod_range_succ]]
      rw [ih, nearOne_factor, slowZero_factor, pow_succ]
      push_cast
      have hk1 : ((k : Rat) + 1) ≠ 0 := by positivity
      have hk2 : ((k : Rat) + 2) ≠ 0 := by positivity
      field_simp [hk1, hk2]
      ring

theorem harmonic_product (k : Nat) :
    path2MultiplierProduct harmonicSchedule 0 k =
      1 / (k + 1 : Rat) := by
  induction k with
  | zero => norm_num [path2MultiplierProduct]
  | succ k ih =>
      rw [show path2MultiplierProduct harmonicSchedule 0 (Nat.succ k) =
          path2MultiplierProduct harmonicSchedule 0 k *
            (1 - 2 * harmonicSchedule.receptivityAt (k + 1)) by
        simp [path2MultiplierProduct, Finset.prod_range_succ]]
      rw [ih, harmonic_factor]
      push_cast
      have hk1 : ((k : Rat) + 1) ≠ 0 := by positivity
      have hk2 : ((k : Rat) + 2) ≠ 0 := by positivity
      field_simp [hk1, hk2]
      ring

theorem slowZero_product_tendsto_half :
    Tendsto
      (fun k => (path2MultiplierProduct slowZeroSchedule 0 k : Real))
      atTop (nhds (1/2 : Real)) := by
  have hbase : Tendsto (fun k : Nat => (1 : Real) / ((k : Real) + 1))
      atTop (nhds 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  have hscaled : Tendsto
      (fun k : Nat => ((1 : Real) / ((k : Real) + 1)) * (1/2 : Real))
      atTop (nhds 0) := by
    simpa using hbase.mul_const (1/2 : Real)
  have hlim : Tendsto
      (fun k : Nat => (1/2 : Real) +
        ((1 : Real) / ((k : Real) + 1)) * (1/2 : Real))
      atTop (nhds (1/2 : Real)) := by
    have h := Filter.Tendsto.const_add (1/2 : Real) hscaled
    simpa using h
  refine hlim.congr' (Filter.Eventually.of_forall ?_)
  intro k
  change
    (1/2 : Real) + (1 / ((k : Real) + 1)) * (1/2 : Real) =
      (path2MultiplierProduct slowZeroSchedule 0 k : Real)
  rw [slowZero_product]
  push_cast
  have hk1 : (k : Real) + 1 ≠ 0 := by positivity
  field_simp [hk1]
  ring

private theorem harmonic_product_tendsto_zero :
    Tendsto
      (fun k => (path2MultiplierProduct harmonicSchedule 0 k : Real))
      atTop (nhds 0) := by
  have hbase : Tendsto (fun k : Nat => (1 : Real) / ((k : Real) + 1))
      atTop (nhds 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  refine hbase.congr' (Filter.Eventually.of_forall ?_)
  intro k
  change
    (1 : Real) / ((k : Real) + 1) =
      (path2MultiplierProduct harmonicSchedule 0 k : Real)
  rw [harmonic_product]
  push_cast

private theorem evenIndex_tendsto :
    Tendsto (fun k : Nat => 2 * k) atTop atTop := by
  refine Filter.tendsto_atTop.2 ?_
  intro b
  exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩

private theorem oddIndex_tendsto :
    Tendsto (fun k : Nat => 2 * k + 1) atTop atTop := by
  refine Filter.tendsto_atTop.2 ?_
  intro b
  exact Filter.eventually_atTop.2 ⟨b, fun a ha => by omega⟩

private theorem nearOne_even_product_tendsto_half :
    Tendsto
      (fun k => (path2MultiplierProduct nearOneSchedule 0 (2 * k) : Real))
      atTop (nhds (1/2 : Real)) := by
  have h := slowZero_product_tendsto_half.comp evenIndex_tendsto
  refine h.congr' (Filter.Eventually.of_forall ?_)
  intro k
  change
    (path2MultiplierProduct slowZeroSchedule 0 (2 * k) : Real) =
      (path2MultiplierProduct nearOneSchedule 0 (2 * k) : Real)
  rw [nearOne_product, slowZero_product]
  have hp : (-1 : Rat) ^ (2 * k) = 1 := by
    simp [pow_mul]
  rw [hp]
  norm_num

private theorem nearOne_odd_product_tendsto_neg_half :
    Tendsto
      (fun k => (path2MultiplierProduct nearOneSchedule 0 (2 * k + 1) : Real))
      atTop (nhds (-1/2 : Real)) := by
  have hslow := slowZero_product_tendsto_half.comp oddIndex_tendsto
  have hneg0 := hslow.neg
  have hneg : Tendsto
      (fun k => -(path2MultiplierProduct slowZeroSchedule 0 (2 * k + 1) : Real))
      atTop (nhds (-1/2 : Real)) := by
    simpa [Function.comp_def] using hneg0
  refine hneg.congr' (Filter.Eventually.of_forall ?_)
  intro k
  change
    -(path2MultiplierProduct slowZeroSchedule 0 (2 * k + 1) : Real) =
      (path2MultiplierProduct nearOneSchedule 0 (2 * k + 1) : Real)
  rw [nearOne_product, slowZero_product]
  have hp : (-1 : Rat) ^ (2 * k + 1) = -1 := by
    rw [pow_succ]
    simp [pow_mul]
  rw [hp]
  norm_num

private def split2 : State 2 := ![⟨1, 0⟩, ⟨0, 0⟩]

private theorem slowZero_split2_allBroadcast : allBroadcast slowZeroSchedule split2 := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, slowZeroSchedule, split2]

private theorem nearOne_split2_allBroadcast : allBroadcast nearOneSchedule split2 := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, nearOneSchedule, split2]

private theorem harmonic_split2_allBroadcast : allBroadcast harmonicSchedule split2 := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, harmonicSchedule, split2]

private theorem split2_equal_exposure :
    (split2 0).exposure = (split2 1).exposure := by
  norm_num [split2]

private theorem split2_beliefs_ne :
    (split2 0).belief ≠ (split2 1).belief := by
  norm_num [split2]

theorem slowZero_not_consensus :
    ¬ ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step slowZeroSchedule 2)^[k] split2) i : Real))
        atTop (nhds (1/2 : Real)) := by
  intro hcons
  have habs0 :=
    (path2_consensus_iff_product_tendsto_zero
      slowZeroSchedule slowZeroSchedule_valid split2 split2_equal_exposure
      slowZero_split2_allBroadcast split2_beliefs_ne).2 (by
        simpa [split2] using hcons)
  have habshalf : Tendsto
      (fun k => |(path2MultiplierProduct slowZeroSchedule 0 k : Real)|)
      atTop (nhds (1/2 : Real)) := by
    have h := slowZero_product_tendsto_half.abs
    simpa using h
  have huniq := tendsto_nhds_unique habs0 habshalf
  norm_num at huniq

theorem nearOne_not_convergent :
    ¬ ∃ c : Real, ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step nearOneSchedule 2)^[k] split2) i : Real))
        atTop (nhds c) := by
  rintro ⟨c, hcons⟩
  have hdiff : Tendsto
      (fun k =>
        (beliefs ((step nearOneSchedule 2)^[k] split2) 0 : Real) -
          (beliefs ((step nearOneSchedule 2)^[k] split2) 1 : Real))
      atTop (nhds 0) := by
    have h := (hcons 0).sub (hcons 1)
    simpa using h
  have hproduct0 : Tendsto
      (fun k => (path2MultiplierProduct nearOneSchedule 0 k : Real))
      atTop (nhds 0) := by
    refine hdiff.congr' (Filter.Eventually.of_forall ?_)
    intro k
    change
      (beliefs ((step nearOneSchedule 2)^[k] split2) 0 : Real) -
          (beliefs ((step nearOneSchedule 2)^[k] split2) 1 : Real) =
        (path2MultiplierProduct nearOneSchedule 0 k : Real)
    have h := path2_disagreement_product nearOneSchedule nearOneSchedule_valid
      split2 split2_equal_exposure nearOne_split2_allBroadcast k
    have h' :
        beliefs ((step nearOneSchedule 2)^[k] split2) 0 -
            beliefs ((step nearOneSchedule 2)^[k] split2) 1 =
          path2MultiplierProduct nearOneSchedule 0 k := by
      simpa [split2] using h
    have hReal := congrArg (fun q : Rat => (q : Real)) h'
    simpa using hReal
  have heven0 := hproduct0.comp evenIndex_tendsto
  have huniq := tendsto_nhds_unique heven0 nearOne_even_product_tendsto_half
  norm_num at huniq

theorem harmonic_consensus :
    ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step harmonicSchedule 2)^[k] split2) i : Real))
        atTop (nhds (1/2 : Real)) := by
  have habs0 : Tendsto
      (fun k => |(path2MultiplierProduct harmonicSchedule 0 k : Real)|)
      atTop (nhds 0) := by
    have h := harmonic_product_tendsto_zero.abs
    simpa using h
  have hcons :=
    (path2_consensus_iff_product_tendsto_zero
      harmonicSchedule harmonicSchedule_valid split2 split2_equal_exposure
      harmonic_split2_allBroadcast split2_beliefs_ne).1 habs0
  simpa [split2] using hcons

def degreeSplitSchedule : ExposureParameters :=
  ⟨fun e => if e = 1 then 1/4 else if e = 2 then 1/2 else 1/3, 0⟩

def degreeSplitState : State 3 :=
  ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]

theorem degreeSplit_step_beliefs :
    beliefs (step degreeSplitSchedule 3 degreeSplitState) =
      ![3/4, 1/4, 0] := by
  funext i
  fin_cases i <;> decide_cbv

theorem degreeSplit_mean_before :
    FitnessABMPathN.mean 3 (beliefs degreeSplitState) = 1/4 := by
  decide_cbv

theorem degreeSplit_mean_after :
    FitnessABMPathN.mean 3
      (beliefs (step degreeSplitSchedule 3 degreeSplitState)) = 5/16 := by
  rw [degreeSplit_step_beliefs]
  decide_cbv

theorem exposure_degree_weighted_mean_not_invariant :
    FitnessABMPathN.mean 3
        (beliefs (step degreeSplitSchedule 3 degreeSplitState)) ≠
      FitnessABMPathN.mean 3 (beliefs degreeSplitState) := by
  rw [degreeSplit_mean_after, degreeSplit_mean_before]
  norm_num

end NarrativeDynamics.FitnessABMPathNExposureConvergence
