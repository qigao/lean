import NarrativeDynamics.Core.Drive
import NarrativeDynamics.Core.Belief

namespace NarrativeDynamics

/-- Expected goal instrumentality under a binary subjective world model. The
belief coordinate changes the mixture; the two instrumentality coordinates are
learned predictions of how useful the goal would be in each possible world. -/
def expectedInstrumentality
    (belief instrumentalityH instrumentalityNotH : ℝ) : ℝ :=
  belief * instrumentalityH + (1 - belief) * instrumentalityNotH

/-- Goal score after propagating epistemic uncertainty through the learned
hypothesis-conditioned instrumentality model. -/
def epistemicGoalScore
    (pressure belief instrumentalityH instrumentalityNotH cost risk : ℝ) : ℝ :=
  goalScore pressure
    (expectedInstrumentality belief instrumentalityH instrumentalityNotH)
    cost risk

/-- Shifting belief toward `H` strictly raises expected instrumentality exactly
in the supportive case where the goal is more useful under `H` than under its
negation. -/
theorem expectedInstrumentality_strict_mono_belief
    {belief₁ belief₂ instrumentalityH instrumentalityNotH : ℝ}
    (hbelief : belief₁ < belief₂)
    (hinst : instrumentalityNotH < instrumentalityH) :
    expectedInstrumentality belief₁ instrumentalityH instrumentalityNotH <
      expectedInstrumentality belief₂ instrumentalityH instrumentalityNotH := by
  have hprod :
      0 < (belief₂ - belief₁) * (instrumentalityH - instrumentalityNotH) :=
    mul_pos (sub_pos.mpr hbelief) (sub_pos.mpr hinst)
  have hid :
      expectedInstrumentality belief₂ instrumentalityH instrumentalityNotH -
          expectedInstrumentality belief₁ instrumentalityH instrumentalityNotH =
        (belief₂ - belief₁) * (instrumentalityH - instrumentalityNotH) := by
    unfold expectedInstrumentality
    ring
  nlinarith

/-- Shifting belief toward `H` strictly lowers expected instrumentality in the
opposing case where the goal is less useful under `H` than under its negation. -/
theorem expectedInstrumentality_strict_antitone_belief
    {belief₁ belief₂ instrumentalityH instrumentalityNotH : ℝ}
    (hbelief : belief₁ < belief₂)
    (hinst : instrumentalityH < instrumentalityNotH) :
    expectedInstrumentality belief₂ instrumentalityH instrumentalityNotH <
      expectedInstrumentality belief₁ instrumentalityH instrumentalityNotH := by
  have hprod :
      0 < (belief₂ - belief₁) * (instrumentalityNotH - instrumentalityH) :=
    mul_pos (sub_pos.mpr hbelief) (sub_pos.mpr hinst)
  have hid :
      expectedInstrumentality belief₁ instrumentalityH instrumentalityNotH -
          expectedInstrumentality belief₂ instrumentalityH instrumentalityNotH =
        (belief₂ - belief₁) * (instrumentalityNotH - instrumentalityH) := by
    unfold expectedInstrumentality
    ring
  nlinarith

/-- Positive evidence raises a goal score only when the agent's learned model
predicts that the goal is more instrumental if the supported hypothesis is
true. Evidence does not directly script or reward the goal. -/
theorem positive_evidence_raises_goal_score_of_supportive_instrumentality
    {pressure prior likelihoodH likelihoodNotH instrumentalityH instrumentalityNotH
      cost risk : ℝ}
    (hpressure : 0 < pressure)
    (hprior0 : 0 < prior)
    (hprior1 : prior < 1)
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH)
    (hinst : instrumentalityNotH < instrumentalityH) :
    epistemicGoalScore pressure prior instrumentalityH instrumentalityNotH cost risk <
      epistemicGoalScore pressure
        (bayesPosterior prior likelihoodH likelihoodNotH)
        instrumentalityH instrumentalityNotH cost risk := by
  have hposterior :
      prior < bayesPosterior prior likelihoodH likelihoodNotH :=
    bayesPosterior_gt_prior_of_positive_evidence
      hprior0 hprior1 hnot hevidence
  have hexpected := expectedInstrumentality_strict_mono_belief hposterior hinst
  have hweighted := mul_lt_mul_of_pos_left hexpected hpressure
  unfold epistemicGoalScore goalScore
  linarith

/-- If a goal has exactly the same learned instrumentality in both possible
worlds, changing belief cannot change its score. -/
theorem hypothesis_independent_instrumentality_makes_goal_score_belief_invariant
    (pressure belief₁ belief₂ instrumentality cost risk : ℝ) :
    epistemicGoalScore pressure belief₁ instrumentality instrumentality cost risk =
      epistemicGoalScore pressure belief₂ instrumentality instrumentality cost risk := by
  unfold epistemicGoalScore expectedInstrumentality goalScore
  ring

/-- The same positive evidence lowers a goal score when the supported
hypothesis predicts that the goal is less useful. This is the converse guard
against treating every belief update as goal activation. -/
theorem positive_evidence_lowers_goal_score_of_opposing_instrumentality
    {pressure prior likelihoodH likelihoodNotH instrumentalityH instrumentalityNotH
      cost risk : ℝ}
    (hpressure : 0 < pressure)
    (hprior0 : 0 < prior)
    (hprior1 : prior < 1)
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH)
    (hinst : instrumentalityH < instrumentalityNotH) :
    epistemicGoalScore pressure
        (bayesPosterior prior likelihoodH likelihoodNotH)
        instrumentalityH instrumentalityNotH cost risk <
      epistemicGoalScore pressure prior instrumentalityH instrumentalityNotH cost risk := by
  have hposterior :
      prior < bayesPosterior prior likelihoodH likelihoodNotH :=
    bayesPosterior_gt_prior_of_positive_evidence
      hprior0 hprior1 hnot hevidence
  have hexpected := expectedInstrumentality_strict_antitone_belief hposterior hinst
  have hweighted := mul_lt_mul_of_pos_left hexpected hpressure
  unfold epistemicGoalScore goalScore
  linarith

end NarrativeDynamics
