import NarrativeDynamics.Core.GroundedHypothesisSpace

open NarrativeDynamics
open scoped BigOperators

universe uH uP uA uE uO uL uI uC uN

example
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node) (event : Event)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) :
    (g.candidate i).evidence = g.commonEvidence ∧
      (g.candidate i).evidenceProof = g.commonEvidenceProof := by
  exact grounded_candidate_shares_common_evidence g i

example
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node) (event : Event)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hw : WorldInvariant w) (i : ι) :
    infoReachable w.info (w.eventNode event) (w.agentNode g.kb.owner) := by
  exact grounded_candidate_requires_info_path g hw i

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node) (event : Event)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hmass : 0 < evidenceMass g.competition) :
    (∑ i, posterior g.competition i) = 1 := by
  exact grounded_posterior_sum_one g hmass

example
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node) (event : Event)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) :
    ¬ ProofValid g.kb.graph
        (deactivateProof g.active g.commonEvidenceProof)
        (g.candidate i).attributionProof := by
  exact grounded_common_evidence_revocation_invalidates_attribution g i
