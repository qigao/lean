import NarrativeDynamics.Core.GroundedGoalCovariance

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
    (hmass : 0 < evidenceMass g.competition) :
    groundedExpectedInstrumentality g instrumentality -
        groundedPriorExpectedInstrumentality g instrumentality =
      groundedLikelihoodInstrumentalityCovariance g instrumentality /
        evidenceMass g.competition := by
  exact grounded_expectedInstrumentality_change_eq_covariance_div_mass
    g instrumentality hmass

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (hmass : 0 < evidenceMass g.competition)
    (hcov : 0 < groundedLikelihoodInstrumentalityCovariance g instrumentality) :
    groundedPriorExpectedInstrumentality g instrumentality <
      groundedExpectedInstrumentality g instrumentality := by
  exact positive_grounded_covariance_raises_expected_instrumentality
    g instrumentality hmass hcov

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (hmass : 0 < evidenceMass g.competition)
    (hcov : groundedLikelihoodInstrumentalityCovariance g instrumentality = 0) :
    groundedExpectedInstrumentality g instrumentality =
      groundedPriorExpectedInstrumentality g instrumentality := by
  exact zero_grounded_covariance_preserves_expected_instrumentality
    g instrumentality hmass hcov

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (hmass : 0 < evidenceMass g.competition)
    (hcov : groundedLikelihoodInstrumentalityCovariance g instrumentality < 0) :
    groundedExpectedInstrumentality g instrumentality <
      groundedPriorExpectedInstrumentality g instrumentality := by
  exact negative_grounded_covariance_lowers_expected_instrumentality
    g instrumentality hmass hcov
