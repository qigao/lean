import NarrativeDynamics.Core.Belief
import NarrativeDynamics.Core.BeliefSupport

namespace NarrativeDynamics

universe uP uA uE uO uL uI uC

/-- A probabilistic belief state couples a private symbolic support KB with a
Bayesian confidence for one hypothesis. Symbolic support and numerical
confidence are intentionally separate coordinates. -/
structure EpistemicBeliefState
    (ProofId : Type uP)
    (Agent : Type uA) (Event : Type uE) (Object : Type uO)
    (Location : Type uL) (Institution : Type uI) (Concept : Type uC) where
  kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept
  active : ProofId → Prop
  hypothesis : HNode Agent Event Object Location Institution Concept
  confidence : ℝ

/-- An agent actively holds a belief only when the hypothesis has at least one
valid symbolic proof and its Bayesian confidence exceeds the agent's threshold. -/
def BeliefHeld
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (threshold : ℝ)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept) : Prop :=
  BeliefSupported s.kb s.active s.hypothesis ∧ threshold < s.confidence

/-- Bayesian evidence updates confidence but does not silently alter the
symbolic provenance graph or its active-proof set. -/
noncomputable def bayesUpdateBelief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (likelihoodH likelihoodNotH : ℝ) :
    EpistemicBeliefState ProofId Agent Event Object Location Institution Concept :=
  { s with confidence := bayesPosterior s.confidence likelihoodH likelihoodNotH }

/-- Revoking evidence changes symbolic support while preserving the stored
numerical confidence. The belief may therefore cease to be held even though a
previously computed probability remains high. -/
def revokeBeliefEvidence
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (revoked : ProofId) :
    EpistemicBeliefState ProofId Agent Event Object Location Institution Concept :=
  { s with active := deactivateProof s.active revoked }

/-- Positive Bayesian evidence strictly increases the confidence coordinate. -/
theorem positive_evidence_increases_belief_confidence
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (hprior0 : 0 < s.confidence)
    (hprior1 : s.confidence < 1)
    {likelihoodH likelihoodNotH : ℝ}
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH) :
    s.confidence < (bayesUpdateBelief s likelihoodH likelihoodNotH).confidence := by
  simpa [bayesUpdateBelief] using
    (bayesPosterior_gt_prior_of_positive_evidence
      hprior0 hprior1 hnot hevidence)

/-- A currently held belief remains held after positive evidence: symbolic
support is unchanged and the numerical confidence strictly increases. -/
theorem positive_evidence_preserves_held_belief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (threshold : ℝ)
    (hheld : BeliefHeld threshold s)
    (hprior0 : 0 < s.confidence)
    (hprior1 : s.confidence < 1)
    {likelihoodH likelihoodNotH : ℝ}
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH) :
    BeliefHeld threshold (bayesUpdateBelief s likelihoodH likelihoodNotH) := by
  constructor
  · exact hheld.1
  · exact lt_trans hheld.2
      (positive_evidence_increases_belief_confidence
        s hprior0 hprior1 hnot hevidence)

/-- Removing every proof path for the hypothesis retracts the held belief,
regardless of the stored confidence value. -/
theorem revoking_all_support_retracts_held_belief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (threshold : ℝ) (revoked : ProofId)
    (hdepends : ∀ q, s.kb.graph.fact q = s.hypothesis →
      q = revoked ∨ Relation.TransGen (provenanceSupport s.kb.graph) revoked q) :
    ¬ BeliefHeld threshold (revokeBeliefEvidence s revoked) := by
  have hnot :
      ¬ BeliefSupported s.kb (deactivateProof s.active revoked) s.hypothesis :=
    revoking_all_support_retracts_belief
      s.kb s.active revoked s.hypothesis hdepends
  intro hheld
  exact hnot hheld.1

/-- Independent symbolic support can keep a threshold-satisfying belief held
after another evidence chain is revoked. -/
theorem independent_support_preserves_held_belief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (threshold : ℝ)
    {revoked alternative : ProofId}
    (hconfidence : threshold < s.confidence)
    (hvalid : ProofValid s.kb.graph s.active alternative)
    (hfact : s.kb.graph.fact alternative = s.hypothesis)
    (hneq : alternative ≠ revoked)
    (hnotdep : ¬ Relation.TransGen (provenanceSupport s.kb.graph) revoked alternative) :
    BeliefHeld threshold (revokeBeliefEvidence s revoked) := by
  have hsuppAlt := independent_support_preserves_belief
    s.kb s.active hvalid hneq hnotdep
  have hsupp :
      BeliefSupported s.kb (deactivateProof s.active revoked) s.hypothesis := by
    simpa [hfact] using hsuppAlt
  exact ⟨hsupp, hconfidence⟩

end NarrativeDynamics
