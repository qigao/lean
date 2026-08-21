import NarrativeDynamics.Core.InterpretationSelection

open NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

example
    {ι : Type uH} [Fintype ι] [DecidableEq ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hprior : ∀ k, 0 ≤ g.prior k)
    (hlikelihood : ∀ k, 0 ≤ (g.candidate k).likelihoodH)
    (hmass : 0 < evidenceMass g.competition)
    {i j : ι} (hneq : i ≠ j) :
    posterior g.competition i + posterior g.competition j ≤ 1 := by
  exact pair_grounded_posteriors_le_one
    g hprior hlikelihood hmass hneq

example
    {ι : Type uH} [Fintype ι] [DecidableEq ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ)
    (hthreshold : (1 : ℝ) / 2 ≤ threshold)
    (hprior : ∀ k, 0 ≤ g.prior k)
    (hlikelihood : ∀ k, 0 ≤ (g.candidate k).likelihoodH)
    (hmass : 0 < evidenceMass g.competition)
    {i j : ι}
    (hi : InterpretationCommittedAt g.active threshold g i)
    (hj : InterpretationCommittedAt g.active threshold g j) :
    i = j := by
  exact majority_interpretation_commitment_unique
    g threshold hthreshold hprior hlikelihood hmass hi hj
