import NarrativeDynamics.Core.TruthMaintenance

open NarrativeDynamics

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked descendant : ProofId}
    (path : Relation.TransGen (provenanceSupport g) revoked descendant) :
    ¬ ProofValid g (deactivateProof active revoked) descendant := by
  exact deactivation_invalidates_descendant g active path

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked alternative : ProofId}
    (hvalid : ProofValid g active alternative)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport g) revoked alternative) :
    ProofValid g (deactivateProof active revoked) alternative := by
  exact alternative_proof_survives_deactivation g active hvalid hneq hnotdep

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked alternative : ProofId}
    (hvalid : ProofValid g active alternative)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport g) revoked alternative) :
    FactSupported g (deactivateProof active revoked) (g.fact alternative) := by
  exact alternative_proof_preserves_fact g active hvalid hneq hnotdep

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (revoked : ProofId)
    (fact : HNode Agent Event Object Location Institution Concept)
    (hdepends : ∀ q, g.fact q = fact →
      q = revoked ∨ Relation.TransGen (provenanceSupport g) revoked q) :
    ¬ FactSupported g (deactivateProof active revoked) fact := by
  exact no_remaining_independent_proof_retracts_fact g active revoked fact hdepends
