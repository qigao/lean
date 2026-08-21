import NarrativeDynamics.Core.EpistemicGoal

namespace NarrativeDynamics

/-- Binary choice probability between two epistemically scored goals. -/
noncomputable def epistemicGoalChoiceProbability
    (β pressure belief : ℝ)
    (ownInstrumentalityH ownInstrumentalityNotH ownCost ownRisk : ℝ)
    (rivalInstrumentalityH rivalInstrumentalityNotH rivalCost rivalRisk : ℝ) : ℝ :=
  softmax2 β
    (epistemicGoalScore pressure belief
      ownInstrumentalityH ownInstrumentalityNotH ownCost ownRisk)
    (epistemicGoalScore pressure belief
      rivalInstrumentalityH rivalInstrumentalityNotH rivalCost rivalRisk)

/-- With positive inverse temperature, simultaneously raising the own goal's
score and lowering its rival's score strictly raises binary-softmax choice
probability. -/
theorem softmax2_strict_of_own_increases_rival_decreases
    {β own₁ own₂ rival₁ rival₂ : ℝ}
    (hβ : 0 < β)
    (hown : own₁ < own₂)
    (hrival : rival₂ < rival₁) :
    softmax2 β own₁ rival₁ < softmax2 β own₂ rival₂ := by
  unfold softmax2
  have hownβ : β * own₁ < β * own₂ :=
    mul_lt_mul_of_pos_left hown hβ
  have hrivalβ : β * rival₂ < β * rival₁ :=
    mul_lt_mul_of_pos_left hrival hβ
  have ha : Real.exp (β * own₁) < Real.exp (β * own₂) :=
    (Real.exp_lt_exp).2 hownβ
  have hb : Real.exp (β * rival₂) < Real.exp (β * rival₁) :=
    (Real.exp_lt_exp).2 hrivalβ
  have ha₁ : 0 < Real.exp (β * own₁) := Real.exp_pos _
  have ha₂ : 0 < Real.exp (β * own₂) := Real.exp_pos _
  have hb₁ : 0 < Real.exp (β * rival₁) := Real.exp_pos _
  have hb₂ : 0 < Real.exp (β * rival₂) := Real.exp_pos _
  have hd₁ :
      0 < Real.exp (β * own₁) + Real.exp (β * rival₁) :=
    add_pos ha₁ hb₁
  have hd₂ :
      0 < Real.exp (β * own₂) + Real.exp (β * rival₂) :=
    add_pos ha₂ hb₂
  rw [div_lt_div_iff₀ hd₁ hd₂]
  have hleft :
      Real.exp (β * own₁) * Real.exp (β * rival₂) <
        Real.exp (β * own₂) * Real.exp (β * rival₂) :=
    mul_lt_mul_of_pos_right ha hb₂
  have hright :
      Real.exp (β * own₂) * Real.exp (β * rival₂) <
        Real.exp (β * own₂) * Real.exp (β * rival₁) :=
    mul_lt_mul_of_pos_left hb ha₂
  calc
    Real.exp (β * own₁) *
        (Real.exp (β * own₂) + Real.exp (β * rival₂)) =
      Real.exp (β * own₁) * Real.exp (β * own₂) +
        Real.exp (β * own₁) * Real.exp (β * rival₂) := by ring
    _ < Real.exp (β * own₁) * Real.exp (β * own₂) +
        Real.exp (β * own₂) * Real.exp (β * rival₂) := by
      exact add_lt_add_right hleft _
    _ < Real.exp (β * own₁) * Real.exp (β * own₂) +
        Real.exp (β * own₂) * Real.exp (β * rival₁) := by
      exact add_lt_add_right hright _
    _ = Real.exp (β * own₂) *
        (Real.exp (β * own₁) + Real.exp (β * rival₁)) := by ring

/-- Positive evidence strictly increases choice probability for a goal whose
instrumentality is supported by `H` against a rival whose instrumentality is
opposed by `H`. This closes the local chain from evidence to action tendency
without introducing a direct belief-to-goal reward edge. -/
theorem positive_evidence_increases_supportive_goal_choice_probability
    {β pressure prior likelihoodH likelihoodNotH : ℝ}
    {aH aNotH aCost aRisk bH bNotH bCost bRisk : ℝ}
    (hβ : 0 < β)
    (hpressure : 0 < pressure)
    (hprior0 : 0 < prior)
    (hprior1 : prior < 1)
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH)
    (ha : aNotH < aH)
    (hb : bH < bNotH) :
    epistemicGoalChoiceProbability β pressure prior
        aH aNotH aCost aRisk bH bNotH bCost bRisk <
      epistemicGoalChoiceProbability β pressure
        (bayesPosterior prior likelihoodH likelihoodNotH)
        aH aNotH aCost aRisk bH bNotH bCost bRisk := by
  have hown :
      epistemicGoalScore pressure prior aH aNotH aCost aRisk <
        epistemicGoalScore pressure
          (bayesPosterior prior likelihoodH likelihoodNotH)
          aH aNotH aCost aRisk :=
    positive_evidence_raises_goal_score_of_supportive_instrumentality
      hpressure hprior0 hprior1 hnot hevidence ha
  have hrival :
      epistemicGoalScore pressure
          (bayesPosterior prior likelihoodH likelihoodNotH)
          bH bNotH bCost bRisk <
        epistemicGoalScore pressure prior bH bNotH bCost bRisk :=
    positive_evidence_lowers_goal_score_of_opposing_instrumentality
      hpressure hprior0 hprior1 hnot hevidence hb
  unfold epistemicGoalChoiceProbability
  exact softmax2_strict_of_own_increases_rival_decreases
    hβ hown hrival

end NarrativeDynamics
