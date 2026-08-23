import NarrativeDynamics.Core.Strategy

open NarrativeDynamics

example {g₁ g₂ : ℝ}
    (hconflict : structuralConflict g₁ g₂) :
    ∃ δ : ℝ,
      potentialChange g₁ δ < 0 ∧
      0 < potentialChange g₂ δ := by
  exact structuralConflict_has_tradeoff_direction hconflict
