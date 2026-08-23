import NarrativeDynamics.Core.Drive

open NarrativeDynamics

example {σ r x₁ x₂ : ℝ} (hσ : 0 ≤ σ) (hx : x₁ ≤ x₂) :
    pressure σ r x₂ ≤ pressure σ r x₁ := by
  exact pressure_antitone_state hσ hx

example {p₁ p₂ w₁ w₂ : ℝ}
    (_hp₁ : 0 ≤ p₁) (hp₂ : 0 ≤ p₂) (hw : w₁ ≤ w₂) :
    effectivePressure p₁ p₂ w₁ ≤ effectivePressure p₁ p₂ w₂ := by
  exact effectivePressure_monotone_boundary hp₂ hw

example {p inst₁ inst₂ cost risk : ℝ}
    (hp : 0 ≤ p) (hinst : inst₁ ≤ inst₂) :
    goalScore p inst₁ cost risk ≤ goalScore p inst₂ cost risk := by
  exact goalScore_monotone_instrumentality hp hinst
