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

/-- Minimal instrumental goal score: current pressure times the learned
instrumentality of the goal, minus cost and risk. -/
def goalScore (p instrumentality cost risk : ℝ) : ℝ :=
  p * instrumentality - cost - risk

/-- For nonnegative pressure, increasing positive instrumentality cannot
lower the goal score when cost and risk are held fixed. -/
theorem goalScore_monotone_instrumentality {p λ₁ λ₂ cost risk : ℝ}
    (hp : 0 ≤ p) (hλ : λ₁ ≤ λ₂) :
    goalScore p λ₁ cost risk ≤ goalScore p λ₂ cost risk := by
  unfold goalScore
  nlinarith

end NarrativeDynamics
