import NarrativeDynamics.Core.GroundedEpistemicGoal

namespace NarrativeDynamics

open scoped BigOperators

universe uH uP uA uE uO uL uI uC uN

/-- Prior expectation of a goal's learned hypothesis-conditioned
instrumentality. For its probabilistic interpretation, the finite priors are
understood to form a normalized distribution. -/
noncomputable def groundedPriorExpectedInstrumentality
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ) : ℝ :=
  ∑ i, g.prior i * instrumentality i

/-- Prior mixed moment between the common evidence likelihood and the learned
instrumentality of one goal. -/
noncomputable def groundedLikelihoodInstrumentalityMoment
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ) : ℝ :=
  ∑ i, hypothesisWeight g.competition i * instrumentality i

/-- Covariance numerator between evidence likelihood and learned goal
instrumentality under the prior hypothesis weights. With normalized priors it
is the ordinary prior covariance. -/
noncomputable def groundedLikelihoodInstrumentalityCovariance
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ) : ℝ :=
  groundedLikelihoodInstrumentalityMoment g instrumentality -
    evidenceMass g.competition *
      groundedPriorExpectedInstrumentality g instrumentality

/-- The posterior expectation is the likelihood–instrumentality mixed moment
divided by the evidence mass. -/
theorem grounded_expectedInstrumentality_eq_moment_div_mass
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ) :
    groundedExpectedInstrumentality g instrumentality =
      groundedLikelihoodInstrumentalityMoment g instrumentality /
        evidenceMass g.competition := by
  classical
  unfold groundedExpectedInstrumentality groundedLikelihoodInstrumentalityMoment
  calc
    (∑ i, posterior g.competition i * instrumentality i) =
        ∑ i, (hypothesisWeight g.competition i * instrumentality i) /
          evidenceMass g.competition := by
      apply Finset.sum_congr rfl
      intro i hi
      unfold posterior
      ring
    _ = (∑ i, hypothesisWeight g.competition i * instrumentality i) /
        evidenceMass g.competition := by
      rw [← Finset.sum_div]

/-- The evidence-induced change in expected goal instrumentality is exactly the
likelihood–instrumentality covariance divided by the positive evidence mass.
Thus evidence has no direct goal-value edge. -/
theorem grounded_expectedInstrumentality_change_eq_covariance_div_mass
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
  rw [grounded_expectedInstrumentality_eq_moment_div_mass]
  unfold groundedLikelihoodInstrumentalityCovariance
  have hne : evidenceMass g.competition ≠ 0 := ne_of_gt hmass
  field_simp [hne]
  <;> ring

/-- Positive evidence–instrumentality covariance strictly raises posterior
expected instrumentality above its prior expectation. -/
theorem positive_grounded_covariance_raises_expected_instrumentality
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
  have hchange := grounded_expectedInstrumentality_change_eq_covariance_div_mass
    g instrumentality hmass
  have hpositive :
      0 < groundedLikelihoodInstrumentalityCovariance g instrumentality /
        evidenceMass g.competition :=
    div_pos hcov hmass
  linarith

/-- Zero evidence–instrumentality covariance leaves expected instrumentality
unchanged by the observation. -/
theorem zero_grounded_covariance_preserves_expected_instrumentality
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
  have hchange := grounded_expectedInstrumentality_change_eq_covariance_div_mass
    g instrumentality hmass
  rw [hcov, zero_div] at hchange
  linarith

/-- Negative evidence–instrumentality covariance strictly lowers posterior
expected instrumentality below its prior expectation. -/
theorem negative_grounded_covariance_lowers_expected_instrumentality
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
  have hchange := grounded_expectedInstrumentality_change_eq_covariance_div_mass
    g instrumentality hmass
  have hnegative :
      groundedLikelihoodInstrumentalityCovariance g instrumentality /
          evidenceMass g.competition < 0 :=
    div_neg_of_neg_of_pos hcov hmass
  linarith

end NarrativeDynamics
