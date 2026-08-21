import NarrativeDynamics.Core.BeliefSupport

open NarrativeDynamics

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop) (p : ProofId)
    (hvalid : ProofValid kb.graph active p) :
    BeliefSupported kb active (kb.graph.fact p) := by
  exact valid_proof_supports_belief kb active p hvalid

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop) (revoked : ProofId)
    (fact : HNode Agent Event Object Location Institution Concept)
    (hdepends : ∀ q, kb.graph.fact q = fact →
      q = revoked ∨ Relation.TransGen (provenanceSupport kb.graph) revoked q) :
    ¬ BeliefSupported kb (deactivateProof active revoked) fact := by
  exact revoking_all_support_retracts_belief kb active revoked fact hdepends

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked alternative : ProofId}
    (hvalid : ProofValid kb.graph active alternative)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport kb.graph) revoked alternative) :
    BeliefSupported kb (deactivateProof active revoked) (kb.graph.fact alternative) := by
  exact independent_support_preserves_belief kb active hvalid hneq hnotdep
