import NarrativeDynamics.Core.Interpretation

open NarrativeDynamics

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : InterpretationAdmissible s c) :
    ProofValid s.kb.graph s.active c.evidenceProof ∧
      s.kb.graph.fact c.evidenceProof = c.evidence := by
  exact admissible_interpretation_has_valid_evidence s c h

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : InterpretationAdmissible s c)
    (hprior0 : 0 < s.confidence)
    (hprior1 : s.confidence < 1)
    (hnot : 0 ≤ c.likelihoodNotH)
    (hevidence : c.likelihoodNotH < c.likelihoodH) :
    s.confidence < (applyInterpretation s c h).confidence := by
  exact grounded_positive_interpretation_increases_confidence
    s c h hprior0 hprior1 hnot hevidence

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : InterpretationAdmissible s c) :
    ¬ ProofValid s.kb.graph
        (deactivateProof s.active c.evidenceProof)
        c.attributionProof := by
  exact revoking_evidence_invalidates_attribution s c h
