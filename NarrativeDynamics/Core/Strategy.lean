import Mathlib

namespace NarrativeDynamics

/-- First-order change in an agent's motivational potential along a scalar
world-state direction. Negative means the direction locally improves the
agent's potential; positive means it worsens it. -/
def potentialChange (gradient direction : ℝ) : ℝ :=
  gradient * direction

/-- In the one-dimensional minimal model, two agents are structurally
conflicted when their motivational-potential gradients point in opposite
directions. -/
def structuralConflict (g₁ g₂ : ℝ) : Prop :=
  g₁ * g₂ < 0

/-- Opposed gradients imply the existence of a local world-state direction
that lowers the first agent's potential while raising the second's. -/
theorem structuralConflict_has_tradeoff_direction {g₁ g₂ : ℝ}
    (hconflict : structuralConflict g₁ g₂) :
    ∃ δ : ℝ,
      potentialChange g₁ δ < 0 ∧
      0 < potentialChange g₂ δ := by
  unfold structuralConflict at hconflict
  refine ⟨-g₁, ?_, ?_⟩
  · unfold potentialChange
    have hg₁ : g₁ ≠ 0 := by
      intro hz
      subst g₁
      norm_num at hconflict
    have hsq : 0 < g₁ * g₁ := mul_self_pos.mpr hg₁
    nlinarith
  · unfold potentialChange
    nlinarith

end NarrativeDynamics
