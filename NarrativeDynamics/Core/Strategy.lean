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

/-- First-order motivational-potential change in a real inner-product state
space. This is the directional derivative represented by the gradient inner
product with a candidate world-state direction. -/
def vectorPotentialChange {E : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (gradient direction : E) : ℝ :=
  inner ℝ gradient direction

/-- Vector structural conflict: two agents' local motivational gradients
have a negative inner product. -/
def vectorStructuralConflict {E : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (g₁ g₂ : E) : Prop :=
  inner ℝ g₁ g₂ < 0

/-- In any real inner-product state space, a negative gradient inner product
implies a local tradeoff direction. Choosing `-g₁` strictly improves agent 1
while strictly worsening agent 2. -/
theorem vectorStructuralConflict_has_tradeoff_direction {E : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {g₁ g₂ : E} (hconflict : vectorStructuralConflict g₁ g₂) :
    ∃ δ : E,
      vectorPotentialChange g₁ δ < 0 ∧
      0 < vectorPotentialChange g₂ δ := by
  unfold vectorStructuralConflict at hconflict
  refine ⟨-g₁, ?_, ?_⟩
  · have hg₁ : g₁ ≠ 0 := by
      intro hz
      subst g₁
      simp at hconflict
    have hself : 0 < inner ℝ g₁ g₁ := real_inner_self_pos.mpr hg₁
    have hneg : -(inner ℝ g₁ g₁) < 0 := neg_lt_zero.mpr hself
    simpa [vectorPotentialChange] using hneg
  · have hneg : 0 < -(inner ℝ g₁ g₂) := neg_pos.mpr hconflict
    have hcomm : inner ℝ g₂ g₁ = inner ℝ g₁ g₂ := real_inner_comm g₁ g₂
    simpa [vectorPotentialChange, hcomm] using hneg

end NarrativeDynamics
