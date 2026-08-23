import NarrativeDynamics.Core.GroundedBeliefUpdate

open NarrativeDynamics
open scoped BigOperators

universe uH uP uA uE uO uL uI uC uN

example
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
  exact posterior_belief_preserves_symbolic_coordinates g i

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hmass : 0 < evidenceMass g.competition) :
    (∑ i, (posteriorBeliefState g i).confidence) = 1 := by
  exact posterior_belief_confidences_sum_one g hmass

example
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
  exact grounded_higher_likelihood_yields_higher_confidence
    g i j hprior hpriorPos hlikelihood hmass
