import NarrativeDynamics.Core.EpistemicBelief

namespace NarrativeDynamics

universe uP uA uE uO uL uI uC

/-- A candidate interpretation links one observed/evidential fact to one
hypothesis through an explicit provenance proof path, together with Bayesian
likelihoods used to update confidence in that hypothesis. -/
structure InterpretationCandidate
    (ProofId : Type uP)
    (Agent : Type uA) (Event : Type uE) (Object : Type uO)
    (Location : Type uL) (Institution : Type uI) (Concept : Type uC) where
  evidence : HNode Agent Event Object Location Institution Concept
  hypothesis : HNode Agent Event Object Location Institution Concept
  evidenceProof : ProofId
  attributionProof : ProofId
  likelihoodH : ℝ
  likelihoodNotH : ℝ

/-- A candidate interpretation is admissible only when both its evidence and
its attribution proof are currently valid, the proof records conclude the
claimed facts, and the evidence proof is a strict provenance ancestor of the
attribution proof. -/
structure InterpretationAdmissible
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept) : Prop where
  hypothesisMatches : c.hypothesis = s.hypothesis
  evidenceValid : ProofValid s.kb.graph s.active c.evidenceProof
  evidenceMatches : s.kb.graph.fact c.evidenceProof = c.evidence
  attributionValid : ProofValid s.kb.graph s.active c.attributionProof
  attributionMatches : s.kb.graph.fact c.attributionProof = c.hypothesis
  grounded : Relation.TransGen
    (provenanceSupport s.kb.graph) c.evidenceProof c.attributionProof

/-- Applying a grounded interpretation changes only Bayesian confidence. The
symbolic proof path is already present and admissibility is required as a proof
argument, so unsupported interpretations cannot be applied through this API. -/
noncomputable def applyInterpretation
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (_h : InterpretationAdmissible s c) :
    EpistemicBeliefState ProofId Agent Event Object Location Institution Concept :=
  bayesUpdateBelief s c.likelihoodH c.likelihoodNotH

/-- Admissibility exposes a currently valid proof of the claimed evidence. -/
theorem admissible_interpretation_has_valid_evidence
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : InterpretationAdmissible s c) :
    ProofValid s.kb.graph s.active c.evidenceProof ∧
      s.kb.graph.fact c.evidenceProof = c.evidence := by
  exact ⟨h.evidenceValid, h.evidenceMatches⟩

/-- Positive evidence strictly raises confidence when applied through a
provenance-grounded interpretation candidate. -/
theorem grounded_positive_interpretation_increases_confidence
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : InterpretationAdmissible s c)
    (hprior0 : 0 < s.confidence)
    (hprior1 : s.confidence < 1)
    (hnot : 0 ≤ c.likelihoodNotH)
    (hevidence : c.likelihoodNotH < c.likelihoodH) :
    s.confidence < (applyInterpretation s c h).confidence := by
  simpa [applyInterpretation] using
    (positive_evidence_increases_belief_confidence
      s hprior0 hprior1 hnot hevidence)

/-- Because admissible attribution is explicitly downstream of its evidence
proof, revoking that evidence invalidates the attribution proof by truth
maintenance. -/
theorem revoking_evidence_invalidates_attribution
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : InterpretationAdmissible s c) :
    ¬ ProofValid s.kb.graph
        (deactivateProof s.active c.evidenceProof)
        c.attributionProof := by
  exact deactivation_invalidates_descendant
    s.kb.graph s.active h.grounded

end NarrativeDynamics
