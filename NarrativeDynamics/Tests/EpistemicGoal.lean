import NarrativeDynamics.Core.EpistemicGoal

open NarrativeDynamics

example
    {pressure prior likelihoodH likelihoodNotH instH instNotH cost risk : ℝ}
    (hpressure : 0 < pressure)
    (hprior0 : 0 < prior)
    (hprior1 : prior < 1)
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH)
    (hinst : instNotH < instH) :
    epistemicGoalScore pressure prior instH instNotH cost risk <
      epistemicGoalScore pressure
        (bayesPosterior prior likelihoodH likelihoodNotH)
        instH instNotH cost risk := by
  exact positive_evidence_raises_goal_score_of_supportive_instrumentality
    hpressure hprior0 hprior1 hnot hevidence hinst

example
    (pressure belief₁ belief₂ instrumentality cost risk : ℝ) :
    epistemicGoalScore pressure belief₁ instrumentality instrumentality cost risk =
      epistemicGoalScore pressure belief₂ instrumentality instrumentality cost risk := by
  exact hypothesis_independent_instrumentality_makes_goal_score_belief_invariant
    pressure belief₁ belief₂ instrumentality cost risk

example
    {pressure prior likelihoodH likelihoodNotH instH instNotH cost risk : ℝ}
    (hpressure : 0 < pressure)
    (hprior0 : 0 < prior)
    (hprior1 : prior < 1)
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH)
    (hinst : instH < instNotH) :
    epistemicGoalScore pressure
        (bayesPosterior prior likelihoodH likelihoodNotH)
        instH instNotH cost risk <
      epistemicGoalScore pressure prior instH instNotH cost risk := by
  exact positive_evidence_lowers_goal_score_of_opposing_instrumentality
    hpressure hprior0 hprior1 hnot hevidence hinst
