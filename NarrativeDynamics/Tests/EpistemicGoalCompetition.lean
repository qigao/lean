import NarrativeDynamics.Core.EpistemicGoalCompetition

open NarrativeDynamics

example
    {β own₁ own₂ rival₁ rival₂ : ℝ}
    (hβ : 0 < β)
    (hown : own₁ < own₂)
    (hrival : rival₂ < rival₁) :
    softmax2 β own₁ rival₁ < softmax2 β own₂ rival₂ := by
  exact softmax2_strict_of_own_increases_rival_decreases
    hβ hown hrival

example
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
  exact positive_evidence_increases_supportive_goal_choice_probability
    hβ hpressure hprior0 hprior1 hnot hevidence ha hb
