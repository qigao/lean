import NarrativeDynamics.Core.TruthMaintenance

namespace NarrativeDynamics

universe uP uA uE uO uL uI uC

/-- A private belief knowledge base owned by one agent. The proposition space
is the same typed heterogeneous fact language used by the world model, but the
provenance graph belongs to this agent's subjective model rather than to
objective reality. -/
structure AgentBeliefKB
    (ProofId : Type uP)
    (Agent : Type uA) (Event : Type uE) (Object : Type uO)
    (Location : Type uL) (Institution : Type uI) (Concept : Type uC) where
  owner : Agent
  graph : ProvenanceDAG ProofId Agent Event Object Location Institution Concept

/-- A proposition is believed exactly when the agent's private provenance
DAG still contains at least one valid active proof for it. -/
def BeliefSupported
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (fact : HNode Agent Event Object Location Institution Concept) : Prop :=
  FactSupported kb.graph active fact

/-- Any valid proof in an agent's private provenance graph supports the fact it
concludes as a current belief. -/
theorem valid_proof_supports_belief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop) (p : ProofId)
    (hvalid : ProofValid kb.graph active p) :
    BeliefSupported kb active (kb.graph.fact p) := by
  exact ⟨p, hvalid, rfl⟩

/-- If every proof supporting a belief is either the revoked proof itself or
transitively depends on it, revocation retracts that belief. -/
theorem revoking_all_support_retracts_belief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop) (revoked : ProofId)
    (fact : HNode Agent Event Object Location Institution Concept)
    (hdepends : ∀ q, kb.graph.fact q = fact →
      q = revoked ∨ Relation.TransGen (provenanceSupport kb.graph) revoked q) :
    ¬ BeliefSupported kb (deactivateProof active revoked) fact := by
  exact no_remaining_independent_proof_retracts_fact
    kb.graph active revoked fact hdepends

/-- An independent alternative proof preserves a belief after a competing
proof chain is revoked. -/
theorem independent_support_preserves_belief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked alternative : ProofId}
    (hvalid : ProofValid kb.graph active alternative)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport kb.graph) revoked alternative) :
    BeliefSupported kb (deactivateProof active revoked) (kb.graph.fact alternative) := by
  exact alternative_proof_preserves_fact
    kb.graph active hvalid hneq hnotdep

end NarrativeDynamics
