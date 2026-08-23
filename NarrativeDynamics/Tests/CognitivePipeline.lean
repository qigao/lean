import NarrativeDynamics.Core.CognitivePipeline

open NarrativeDynamics

example
    {ProofId Agent Event Object Location Institution Concept Node : Type*}
    (w : WorldGraph Agent Event Node)
    (hw : WorldInvariant w)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (event : Event)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (hno : ¬ infoReachable w.info (w.eventNode event) (w.agentNode s.kb.owner)) :
    ¬ PerceptualInterpretationAdmissible w s event c := by
  exact no_info_path_blocks_perceptual_interpretation w hw s event c hno

example
    {ProofId Agent Event Object Location Institution Concept Node : Type*}
    (w : WorldGraph Agent Event Node)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (event : Event)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : PerceptualInterpretationAdmissible w s event c)
    (hprior0 : 0 < s.confidence)
    (hprior1 : s.confidence < 1)
    (hnot : 0 ≤ c.likelihoodNotH)
    (hevidence : c.likelihoodNotH < c.likelihoodH) :
    s.confidence < (applyPerceptualInterpretation w s event c h).confidence := by
  exact positive_perceptual_interpretation_increases_confidence
    w s event c h hprior0 hprior1 hnot hevidence

example
    {ProofId Agent Event Object Location Institution Concept Node : Type*}
    (w : WorldGraph Agent Event Node)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (event : Event)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : PerceptualInterpretationAdmissible w s event c) :
    ¬ ProofValid s.kb.graph
        (deactivateProof s.active c.evidenceProof)
        c.attributionProof := by
  exact revoking_perceptual_evidence_invalidates_attribution w s event c h
