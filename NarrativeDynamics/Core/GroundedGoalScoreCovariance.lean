import NarrativeDynamics.Core.GroundedGoalCovariance

namespace NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

/-- A goal's score before the common evidence reweights the grounded
hypothesis space. Cost and risk are held fixed across the epistemic update. -/
noncomputable def groundedPriorEpistemicGoalScore
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (pressure cost risk : ℝ)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ) : ℝ :=
  goalScore pressure
    (groundedPriorExpectedInstrumentality g instrumentality)
    cost risk

/-- Evidence-induced goal-score change is positive drive pressure multiplied
by the evidence–instrumentality covariance, divided by evidence mass. The
fixed cost/risk terms cancel, so evidence has no direct reward edge. -/
theorem grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (pressure cost risk : ℝ)
    (hmass : 0 < evidenceMass g.competition) :
    groundedEpistemicGoalScore pressure cost risk g instrumentality -
        groundedPriorEpistemicGoalScore pressure cost risk g instrumentality =
      pressure * groundedLikelihoodInstrumentalityCovariance g instrumentality /
        evidenceMass g.competition := by
  have hchange :=
    grounded_expectedInstrumentality_change_eq_covariance_div_mass
      g instrumentality hmass
  calc
    groundedEpistemicGoalScore pressure cost risk g instrumentality -
        groundedPriorEpistemicGoalScore pressure cost risk g instrumentality =
      pressure *
        (groundedExpectedInstrumentality g instrumentality -
          groundedPriorExpectedInstrumentality g instrumentality) := by
            unfold groundedEpistemicGoalScore groundedPriorEpistemicGoalScore goalScore
            ring
    _ = pressure *
        (groundedLikelihoodInstrumentalityCovariance g instrumentality /
          evidenceMass g.competition) := by rw [hchange]
    _ = pressure * groundedLikelihoodInstrumentalityCovariance g instrumentality /
        evidenceMass g.competition := by ring

/-- Positive covariance raises a goal score whenever the currently active
drive pressure is strictly positive. -/
theorem positive_grounded_covariance_raises_goal_score
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (pressure cost risk : ℝ)
    (hpressure : 0 < pressure)
    (hmass : 0 < evidenceMass g.competition)
    (hcov : 0 < groundedLikelihoodInstrumentalityCovariance g instrumentality) :
    groundedPriorEpistemicGoalScore pressure cost risk g instrumentality <
      groundedEpistemicGoalScore pressure cost risk g instrumentality := by
  have hchange := grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentality pressure cost risk hmass
  have hpositive :
      0 < pressure * groundedLikelihoodInstrumentalityCovariance g instrumentality /
        evidenceMass g.competition :=
    div_pos (mul_pos hpressure hcov) hmass
  linarith

/-- Zero covariance leaves the goal score unchanged, irrespective of the
pressure level, because the observation does not reweight usefulness. -/
theorem zero_grounded_covariance_preserves_goal_score
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (pressure cost risk : ℝ)
    (hmass : 0 < evidenceMass g.competition)
    (hcov : groundedLikelihoodInstrumentalityCovariance g instrumentality = 0) :
    groundedEpistemicGoalScore pressure cost risk g instrumentality =
      groundedPriorEpistemicGoalScore pressure cost risk g instrumentality := by
  have hchange := grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentality pressure cost risk hmass
  rw [hcov] at hchange
  norm_num at hchange
  linarith

/-- Negative covariance lowers a goal score whenever drive pressure is
strictly positive. -/
theorem negative_grounded_covariance_lowers_goal_score
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (pressure cost risk : ℝ)
    (hpressure : 0 < pressure)
    (hmass : 0 < evidenceMass g.competition)
    (hcov : groundedLikelihoodInstrumentalityCovariance g instrumentality < 0) :
    groundedEpistemicGoalScore pressure cost risk g instrumentality <
      groundedPriorEpistemicGoalScore pressure cost risk g instrumentality := by
  have hchange := grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentality pressure cost risk hmass
  have hnegative :
      pressure * groundedLikelihoodInstrumentalityCovariance g instrumentality /
          evidenceMass g.competition < 0 :=
    div_neg_of_neg_of_pos (mul_neg_of_pos_of_neg hpressure hcov) hmass
  linarith

/-- Among two goals with equal prior scores, the one whose learned
instrumentality has greater covariance with the admitted evidence obtains the
strictly greater posterior score under positive pressure. Semantic goal labels
play no role in the comparison. -/
theorem greater_covariance_wins_equal_prior_goal_score
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentalityA instrumentalityB : ι → ℝ)
    (pressure costA riskA costB riskB : ℝ)
    (hpressure : 0 < pressure)
    (hmass : 0 < evidenceMass g.competition)
    (hprior :
      groundedPriorEpistemicGoalScore pressure costA riskA g instrumentalityA =
        groundedPriorEpistemicGoalScore pressure costB riskB g instrumentalityB)
    (hcov :
      groundedLikelihoodInstrumentalityCovariance g instrumentalityB <
        groundedLikelihoodInstrumentalityCovariance g instrumentalityA) :
    groundedEpistemicGoalScore pressure costB riskB g instrumentalityB <
      groundedEpistemicGoalScore pressure costA riskA g instrumentalityA := by
  have hchangeA := grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentalityA pressure costA riskA hmass
  have hchangeB := grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentalityB pressure costB riskB hmass
  have hshift :
      pressure * groundedLikelihoodInstrumentalityCovariance g instrumentalityB /
          evidenceMass g.competition <
        pressure * groundedLikelihoodInstrumentalityCovariance g instrumentalityA /
          evidenceMass g.competition := by
    apply (div_lt_div_iff_of_pos_right hmass).2
    exact mul_lt_mul_of_pos_left hcov hpressure
  linarith

end NarrativeDynamics
