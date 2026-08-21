import NarrativeDynamics.Core.GroundedGoalScoreCovariance

namespace NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

/-- A goal can start below a rival and overtake it after one grounded
observation. The reversal occurs when the goal's evidence–instrumentality
covariance advantage, scaled by drive pressure and evidence mass, exceeds the
original score deficit. Semantic goal labels play no role. -/
theorem covariance_advantage_reverses_unequal_prior_goal_ranking
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentalityA instrumentalityB : ι → ℝ)
    (pressure costA riskA costB riskB : ℝ)
    (_hpressure : 0 < pressure)
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
  have hchangeA := grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentalityA pressure costA riskA hmass
  have hchangeB := grounded_goalScore_change_eq_pressure_mul_covariance_div_mass
    g instrumentalityB pressure costB riskB hmass
  have hgap :
      0 < groundedPriorEpistemicGoalScore pressure costB riskB g instrumentalityB -
        groundedPriorEpistemicGoalScore pressure costA riskA g instrumentalityA :=
    sub_pos.mpr hprior
  have hmargin' :
      groundedPriorEpistemicGoalScore pressure costB riskB g instrumentalityB -
          groundedPriorEpistemicGoalScore pressure costA riskA g instrumentalityA <
        pressure * groundedLikelihoodInstrumentalityCovariance g instrumentalityA /
            evidenceMass g.competition -
          pressure * groundedLikelihoodInstrumentalityCovariance g instrumentalityB /
            evidenceMass g.competition := by
    calc
      groundedPriorEpistemicGoalScore pressure costB riskB g instrumentalityB -
          groundedPriorEpistemicGoalScore pressure costA riskA g instrumentalityA <
        pressure *
          (groundedLikelihoodInstrumentalityCovariance g instrumentalityA -
            groundedLikelihoodInstrumentalityCovariance g instrumentalityB) /
          evidenceMass g.competition := hmargin
      _ = pressure * groundedLikelihoodInstrumentalityCovariance g instrumentalityA /
            evidenceMass g.competition -
          pressure * groundedLikelihoodInstrumentalityCovariance g instrumentalityB /
            evidenceMass g.competition := by ring
  linarith

end NarrativeDynamics
