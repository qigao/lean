import NarrativeDynamics.Core.GroundedGoalScoreCovariance

open NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

example
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
  exact grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentality pressure cost risk hmass

example
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
  exact positive_grounded_covariance_raises_goal_score
    g instrumentality pressure cost risk hpressure hmass hcov

example
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
  exact zero_grounded_covariance_preserves_goal_score
    g instrumentality pressure cost risk hmass hcov

example
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
  exact greater_covariance_wins_equal_prior_goal_score
    g instrumentalityA instrumentalityB pressure costA riskA costB riskB
    hpressure hmass hprior hcov
