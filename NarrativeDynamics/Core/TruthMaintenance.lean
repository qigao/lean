import NarrativeDynamics.Core.Provenance

namespace NarrativeDynamics

universe uP uA uE uO uL uI uC

/-- Deactivate exactly one proof identifier while leaving every other active
proof unchanged. -/
def deactivateProof {ProofId : Type uP}
    (active : ProofId → Prop) (revoked : ProofId) : ProofId → Prop :=
  fun p => active p ∧ p ≠ revoked

/-- A proof is valid when it is active and every strict provenance ancestor
that it depends on is also active. This declarative semantics captures cascade
invalidation without committing to an incremental maintenance algorithm. -/
def ProofValid
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop) (p : ProofId) : Prop :=
  active p ∧
    ∀ ancestor,
      Relation.TransGen (provenanceSupport g) ancestor p → active ancestor

/-- A fact is currently supported when at least one valid proof concludes it. -/
def FactSupported
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (fact : HNode Agent Event Object Location Institution Concept) : Prop :=
  ∃ p, ProofValid g active p ∧ g.fact p = fact

/-- Revoking a proof invalidates every proof that transitively depends on it. -/
theorem deactivation_invalidates_descendant
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked descendant : ProofId}
    (path : Relation.TransGen (provenanceSupport g) revoked descendant) :
    ¬ ProofValid g (deactivateProof active revoked) descendant := by
  intro hvalid
  have hrevoked : deactivateProof active revoked revoked := hvalid.2 revoked path
  exact hrevoked.2 rfl

/-- An active alternative proof survives revocation when it is neither the
revoked proof itself nor transitively dependent on that proof. -/
theorem alternative_proof_survives_deactivation
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked alternative : ProofId}
    (hvalid : ProofValid g active alternative)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport g) revoked alternative) :
    ProofValid g (deactivateProof active revoked) alternative := by
  constructor
  · exact ⟨hvalid.1, hneq⟩
  · intro ancestor path
    have hactive : active ancestor := hvalid.2 ancestor path
    have hancestor : ancestor ≠ revoked := by
      intro heq
      subst ancestor
      exact hnotdep path
    exact ⟨hactive, hancestor⟩

/-- An independent alternative proof keeps its conclusion supported after a
competing proof is revoked. -/
theorem alternative_proof_preserves_fact
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    {revoked alternative : ProofId}
    (hvalid : ProofValid g active alternative)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport g) revoked alternative) :
    FactSupported g (deactivateProof active revoked) (g.fact alternative) := by
  refine ⟨alternative, ?_, rfl⟩
  exact alternative_proof_survives_deactivation g active hvalid hneq hnotdep

/-- If every proof of a fact is either the revoked proof itself or transitively
depends on it, then revocation removes all valid support for that fact. -/
theorem no_remaining_independent_proof_retracts_fact
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (revoked : ProofId)
    (fact : HNode Agent Event Object Location Institution Concept)
    (hdepends : ∀ q, g.fact q = fact →
      q = revoked ∨ Relation.TransGen (provenanceSupport g) revoked q) :
    ¬ FactSupported g (deactivateProof active revoked) fact := by
  intro hsupported
  rcases hsupported with ⟨q, hvalid, hfact⟩
  rcases hdepends q hfact with hq | hpath
  · subst q
    exact hvalid.1.2 rfl
  · exact deactivation_invalidates_descendant g active hpath hvalid

end NarrativeDynamics
