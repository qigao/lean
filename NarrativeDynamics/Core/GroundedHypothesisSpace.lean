import NarrativeDynamics.Core.CognitivePipeline
import NarrativeDynamics.Core.HypothesisCompetition

namespace NarrativeDynamics

open scoped BigOperators

universe uH uP uA uE uO uL uI uC uN

/-- A finite family of competing interpretations for one agent and one
objective event. Every candidate is grounded in the same admitted observation
fact/proof, but may conclude a different hypothesis with a different prior and
likelihood model. -/
structure GroundedHypothesisSpace
    (ι : Type uH)
    (ProofId : Type uP)
    (Agent : Type uA) (Event : Type uE) (Object : Type uO)
    (Location : Type uL) (Institution : Type uI) (Concept : Type uC)
    (Node : Type uN)
    (w : WorldGraph Agent Event Node)
    (event : Event) where
  kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept
  active : ProofId → Prop
  commonEvidence : HNode Agent Event Object Location Institution Concept
  commonEvidenceProof : ProofId
  hypothesis : ι → HNode Agent Event Object Location Institution Concept
  prior : ι → ℝ
  candidate : ι → InterpretationCandidate ProofId Agent Event Object Location Institution Concept
  evidenceMatches : ∀ i, (candidate i).evidence = commonEvidence
  evidenceProofMatches : ∀ i, (candidate i).evidenceProof = commonEvidenceProof
  admissible : ∀ i, PerceptualInterpretationAdmissible w
    { kb := kb
      active := active
      hypothesis := hypothesis i
      confidence := prior i }
    event (candidate i)

namespace GroundedHypothesisSpace

/-- The single-hypothesis epistemic state used to validate candidate `i`. -/
def state
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) :
    EpistemicBeliefState ProofId Agent Event Object Location Institution Concept :=
  { kb := g.kb
    active := g.active
    hypothesis := g.hypothesis i
    confidence := g.prior i }

/-- Forget provenance only after grounding has been checked, yielding the
finite Bayesian competition induced by the candidate priors and likelihoods. -/
noncomputable def competition
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event) :
    HypothesisCompetition ι :=
  { prior := g.prior
    likelihood := fun i => (g.candidate i).likelihoodH }

end GroundedHypothesisSpace

/-- Every member of a grounded competition refers to the same admitted
observation fact and the same root evidence proof. -/
theorem grounded_candidate_shares_common_evidence
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) :
    (g.candidate i).evidence = g.commonEvidence ∧
      (g.candidate i).evidenceProof = g.commonEvidenceProof := by
  exact ⟨g.evidenceMatches i, g.evidenceProofMatches i⟩

/-- Each grounded candidate inherits the information-path guarantee of the
common objective observation. -/
theorem grounded_candidate_requires_info_path
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hw : WorldInvariant w) (i : ι) :
    infoReachable w.info (w.eventNode event) (w.agentNode g.kb.owner) := by
  exact admissible_observation_has_info_path
    w hw g.kb g.active event (g.candidate i).evidenceProof
    (g.admissible i).observation

/-- Once grounded candidates are projected to a finite Bayesian competition,
their posterior probabilities normalize whenever the common observation has
positive evidence mass. -/
theorem grounded_posterior_sum_one
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hmass : 0 < evidenceMass g.competition) :
    (∑ i, posterior g.competition i) = 1 := by
  exact posterior_sum_one g.competition hmass

/-- Because every attribution proof is downstream of the same observation
proof, revoking that common evidence invalidates every competing attribution. -/
theorem grounded_common_evidence_revocation_invalidates_attribution
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) :
    ¬ ProofValid g.kb.graph
        (deactivateProof g.active g.commonEvidenceProof)
        (g.candidate i).attributionProof := by
  have h := revoking_perceptual_evidence_invalidates_attribution
    w (g.state i) event (g.candidate i) (g.admissible i)
  rw [g.evidenceProofMatches i] at h
  simpa [GroundedHypothesisSpace.state] using h

end NarrativeDynamics
