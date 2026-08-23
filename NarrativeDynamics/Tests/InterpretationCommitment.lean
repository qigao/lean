import NarrativeDynamics.Core.InterpretationCommitment

open NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι)
    (habove : threshold < posterior g.competition i) :
    InterpretationCommittedAt g.active threshold g i := by
  exact grounded_posterior_above_threshold_commits_interpretation
    g threshold i habove

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι)
    (hcommit : InterpretationCommittedAt g.active threshold g i) :
    BeliefHeld threshold (posteriorBeliefState g i) := by
  exact committed_interpretation_implies_held_belief
    g threshold i hcommit

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι)
    (hbelow : posterior g.competition i ≤ threshold) :
    ¬ InterpretationCommittedAt g.active threshold g i := by
  exact posterior_below_threshold_blocks_commitment
    g threshold i hbelow

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι) :
    ¬ InterpretationCommittedAt
        (deactivateProof g.active g.commonEvidenceProof)
        threshold g i := by
  exact revoking_common_evidence_breaks_interpretation_commitment
    g threshold i
