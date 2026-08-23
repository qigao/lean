import NarrativeDynamics.Core.GroundedHypothesisSpace

namespace NarrativeDynamics

open scoped BigOperators

universe uH uP uA uE uO uL uI uC uN

/-- Write one normalized finite posterior coordinate back into the corresponding
single-hypothesis epistemic state. Only confidence changes; private symbolic
knowledge, active proofs, and the hypothesis identity are preserved. -/
noncomputable def posteriorBeliefState
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) :
    EpistemicBeliefState ProofId Agent Event Object Location Institution Concept :=
  { g.state i with confidence := posterior g.competition i }

/-- Posterior normalization is a numerical update only: it cannot silently
replace the agent's private provenance graph, active proof set, or proposition. -/
theorem posterior_belief_preserves_symbolic_coordinates
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) :
    (posteriorBeliefState g i).kb = g.kb ∧
      (posteriorBeliefState g i).active = g.active ∧
      (posteriorBeliefState g i).hypothesis = g.hypothesis i := by
  simp [posteriorBeliefState, GroundedHypothesisSpace.state]

/-- The family of confidence coordinates written back from one grounded finite
competition remains a normalized distribution. -/
theorem posterior_belief_confidences_sum_one
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hmass : 0 < evidenceMass g.competition) :
    (∑ i, (posteriorBeliefState g i).confidence) = 1 := by
  simpa [posteriorBeliefState] using grounded_posterior_sum_one g hmass

/-- With equal positive priors, the grounded explanation assigning greater
likelihood to the common observation receives the greater written-back
confidence coordinate. -/
theorem grounded_higher_likelihood_yields_higher_confidence
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i j : ι)
    (hprior : g.prior i = g.prior j)
    (hpriorPos : 0 < g.prior i)
    (hlikelihood : (g.candidate j).likelihoodH < (g.candidate i).likelihoodH)
    (hmass : 0 < evidenceMass g.competition) :
    (posteriorBeliefState g j).confidence <
      (posteriorBeliefState g i).confidence := by
  have hp : g.competition.prior i = g.competition.prior j := by
    simpa [GroundedHypothesisSpace.competition] using hprior
  have hpp : 0 < g.competition.prior i := by
    simpa [GroundedHypothesisSpace.competition] using hpriorPos
  have hl : g.competition.likelihood j < g.competition.likelihood i := by
    simpa [GroundedHypothesisSpace.competition] using hlikelihood
  have h := higher_likelihood_wins_equal_prior
    g.competition i j hp hpp hl hmass
  simpa [posteriorBeliefState] using h

end NarrativeDynamics
