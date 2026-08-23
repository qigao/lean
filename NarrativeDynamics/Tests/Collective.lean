import NarrativeDynamics.Core.Drive

open NarrativeDynamics

example {ι : Type} [Fintype ι]
    {p w₁ w₂ : ι → ℝ}
    (hp : ∀ j, 0 ≤ p j)
    (hw : ∀ j, w₁ j ≤ w₂ j) :
    effectivePressureN w₁ p ≤ effectivePressureN w₂ p := by
  exact effectivePressureN_monotone_boundary hp hw

example {β score₁ score₂ rival : ℝ}
    (hβ : 0 ≤ β)
    (hs : score₁ ≤ score₂) :
    softmax2 β score₁ rival ≤ softmax2 β score₂ rival := by
  exact softmax2_monotone_own_score hβ hs
