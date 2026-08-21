import NarrativeDynamics.Core.GroundedGoalRankingReversal

open NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

/-- A goal may begin below a rival yet overtake it when its
likelihood–instrumentality covariance advantage exceeds the prior score gap. -/
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
      groundedPriorEpistemicGoalScore pressure costA riskA g instrumentalityA <
        groundedPriorEpistemicGoalScore pressure costB riskB g instrumentalityB)
    (hmargin :
      groundedPriorEpistemicGoalScore pressure costB riskB g instrumentalityB -
          groundedPriorEpistemicGoalScore pressure costA riskA g instrumentalityA <
        pressure *
          (groundedLikelihoodInstrumentalityCovariance g instrumentalityA -
            groundedLikelihoodInstrumentalityCovariance g instrumentalityB) /
          evidenceMass g.competition) :
    groundedEpistemicGoalScore pressure costB riskB g instrumentalityB <
      groundedEpistemicGoalScore pressure costA riskA g instrumentalityA := by
  exact covariance_advantage_reverses_unequal_prior_goal_ranking
    g instrumentalityA instrumentalityB pressure costA riskA costB riskB
    hpressure hmass hprior hmargin
