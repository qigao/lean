import NarrativeDynamics.Core.Strategy

open NarrativeDynamics

example {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {g₁ g₂ : E} (hconflict : vectorStructuralConflict g₁ g₂) :
    ∃ δ : E,
      vectorPotentialChange g₁ δ < 0 ∧
      0 < vectorPotentialChange g₂ δ := by
  exact vectorStructuralConflict_has_tradeoff_direction hconflict
