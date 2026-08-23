import NarrativeDynamics.Core.EpistemicBelief

open NarrativeDynamics

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (hprior0 : 0 < s.confidence)
    (hprior1 : s.confidence < 1)
    {likelihoodH likelihoodNotH : ℝ}
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH) :
    s.confidence < (bayesUpdateBelief s likelihoodH likelihoodNotH).confidence := by
  exact positive_evidence_increases_belief_confidence s hprior0 hprior1 hnot hevidence

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (threshold : ℝ)
    (hheld : BeliefHeld threshold s)
    (hprior0 : 0 < s.confidence)
    (hprior1 : s.confidence < 1)
    {likelihoodH likelihoodNotH : ℝ}
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH) :
    BeliefHeld threshold (bayesUpdateBelief s likelihoodH likelihoodNotH) := by
  exact positive_evidence_preserves_held_belief
    s threshold hheld hprior0 hprior1 hnot hevidence

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (threshold : ℝ) (revoked : ProofId)
    (hdepends : ∀ q, s.kb.graph.fact q = s.hypothesis →
      q = revoked ∨ Relation.TransGen (provenanceSupport s.kb.graph) revoked q) :
    ¬ BeliefHeld threshold (revokeBeliefEvidence s revoked) := by
  exact revoking_all_support_retracts_held_belief s threshold revoked hdepends

example
    {ProofId Agent Event Object Location Institution Concept : Type*}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (threshold : ℝ)
    {revoked alternative : ProofId}
    (hheld : threshold < s.confidence)
    (hvalid : ProofValid s.kb.graph s.active alternative)
    (hfact : s.kb.graph.fact alternative = s.hypothesis)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport s.kb.graph) revoked alternative) :
    BeliefHeld threshold (revokeBeliefEvidence s revoked) := by
  exact independent_support_preserves_held_belief
    s threshold hheld hvalid hfact hneq hnotdep
