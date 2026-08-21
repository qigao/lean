import Mathlib

namespace NarrativeDynamics

/-- Drive pressure: sensitivity multiplied by the positive gap between a
reference state and the agent's perceived current state. -/
def pressure (σ r x : ℝ) : ℝ :=
  σ * max (r - x) 0

/-- With nonnegative sensitivity, improving the perceived state cannot
increase pressure. Equivalently, pressure is antitone in `x`. -/
theorem pressure_antitone_state {σ r x₁ x₂ : ℝ}
    (hσ : 0 ≤ σ) (hx : x₁ ≤ x₂) :
    pressure σ r x₂ ≤ pressure σ r x₁ := by
  unfold pressure
  have hsub : r - x₂ ≤ r - x₁ := by
    linarith
  have hmax : max (r - x₂) 0 ≤ max (r - x₁) 0 := by
    exact max_le_max hsub (le_refl 0)
  exact mul_le_mul_of_nonneg_left hmax hσ

/-- A two-person minimal self-boundary model: own pressure plus the
weighted perceived pressure of one other person. -/
def effectivePressure (pSelf pOther boundary : ℝ) : ℝ :=
  pSelf + boundary * pOther

/-- If the perceived pressure of the other person is nonnegative, expanding
the self-boundary weight cannot reduce effective pressure. -/
theorem effectivePressure_monotone_boundary {p₁ p₂ w₁ w₂ : ℝ}
    (hp₂ : 0 ≤ p₂) (hw : w₁ ≤ w₂) :
    effectivePressure p₁ p₂ w₁ ≤ effectivePressure p₁ p₂ w₂ := by
  unfold effectivePressure
  nlinarith

/-- Multi-agent effective pressure: the weighted sum of perceived pressures
inside an agent's motivational self-boundary. -/
def effectivePressureN {ι : Type} [Fintype ι]
    (boundary pressures : ι → ℝ) : ℝ :=
  ∑ j, boundary j * pressures j

/-- For nonnegative perceived pressures, pointwise expansion of the
self-boundary cannot reduce total effective pressure. -/
theorem effectivePressureN_monotone_boundary {ι : Type} [Fintype ι]
    {p w₁ w₂ : ι → ℝ}
    (hp : ∀ j, 0 ≤ p j)
    (hw : ∀ j, w₁ j ≤ w₂ j) :
    effectivePressureN w₁ p ≤ effectivePressureN w₂ p := by
  unfold effectivePressureN
  refine Finset.sum_le_sum ?_
  intro j hj
  exact mul_le_mul_of_nonneg_right (hw j) (hp j)

/-- Minimal instrumental goal score: current pressure times the learned
instrumentality of the goal, minus cost and risk. -/
def goalScore (p instrumentality cost risk : ℝ) : ℝ :=
  p * instrumentality - cost - risk

/-- For nonnegative pressure, increasing instrumentality cannot lower the
goal score when cost and risk are held fixed. -/
theorem goalScore_monotone_instrumentality {p inst₁ inst₂ cost risk : ℝ}
    (hp : 0 ≤ p) (hinst : inst₁ ≤ inst₂) :
    goalScore p inst₁ cost risk ≤ goalScore p inst₂ cost risk := by
  unfold goalScore
  nlinarith

/-- Binary softmax probability for choosing `own` against one rival. -/
noncomputable def softmax2 (β own rival : ℝ) : ℝ :=
  Real.exp (β * own) /
    (Real.exp (β * own) + Real.exp (β * rival))

/-- With nonnegative inverse temperature, increasing a goal's own score while
holding its rival fixed cannot decrease its binary-softmax probability. -/
theorem softmax2_monotone_own_score {β score₁ score₂ rival : ℝ}
    (hβ : 0 ≤ β) (hs : score₁ ≤ score₂) :
    softmax2 β score₁ rival ≤ softmax2 β score₂ rival := by
  unfold softmax2
  have hscore : β * score₁ ≤ β * score₂ :=
    mul_le_mul_of_nonneg_left hs hβ
  have hexp : Real.exp (β * score₁) ≤ Real.exp (β * score₂) :=
    (Real.exp_le_exp).2 hscore
  have h₁ : 0 < Real.exp (β * score₁) := Real.exp_pos _
  have h₂ : 0 < Real.exp (β * score₂) := Real.exp_pos _
  have hr : 0 < Real.exp (β * rival) := Real.exp_pos _
  have hd₁ : 0 < Real.exp (β * score₁) + Real.exp (β * rival) :=
    add_pos h₁ hr
  have hd₂ : 0 < Real.exp (β * score₂) + Real.exp (β * rival) :=
    add_pos h₂ hr
  rw [div_le_div_iff₀ hd₁ hd₂]
  have hmul :
      Real.exp (β * score₁) * Real.exp (β * rival) ≤
        Real.exp (β * score₂) * Real.exp (β * rival) :=
    mul_le_mul_of_nonneg_right hexp (le_of_lt hr)
  calc
    Real.exp (β * score₁) *
        (Real.exp (β * score₂) + Real.exp (β * rival)) =
      Real.exp (β * score₁) * Real.exp (β * score₂) +
        Real.exp (β * score₁) * Real.exp (β * rival) := by ring
    _ ≤ Real.exp (β * score₁) * Real.exp (β * score₂) +
        Real.exp (β * score₂) * Real.exp (β * rival) := by
      exact add_le_add_right hmul _
    _ = Real.exp (β * score₂) *
        (Real.exp (β * score₁) + Real.exp (β * rival)) := by ring

end NarrativeDynamics
