import NarrativeDynamics.Core.ObservationAdmission
import NarrativeDynamics.Core.Interpretation

namespace NarrativeDynamics

universe uP uA uE uO uL uI uC uN

/-- A perceptual interpretation is admissible only when the objective world is
well formed, the candidate's evidence proof is a legitimate raw observation
of the named event by this KB owner, and the candidate is already a grounded
provenance interpretation inside the agent's private belief state. -/
structure PerceptualInterpretationAdmissible
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (event : Event)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept) : Prop where
  worldInvariant : WorldInvariant w
  observation : ObservationEvidenceAdmissible w s.kb s.active event c.evidenceProof
  interpretation : InterpretationAdmissible s c

/-- No permitted information path blocks the whole perceptual interpretation
pipeline: the event cannot be admitted as raw evidence, hence cannot ground an
interpretation through this API. -/
theorem no_info_path_blocks_perceptual_interpretation
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (hw : WorldInvariant w)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (event : Event)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (hno : ¬ infoReachable w.info (w.eventNode event) (w.agentNode s.kb.owner)) :
    ¬ PerceptualInterpretationAdmissible w s event c := by
  intro h
  exact hno (admissible_observation_has_info_path
    w hw s.kb s.active event c.evidenceProof h.observation)

/-- Apply the already-grounded interpretation after the world-to-observation
admission gate has also been proved. This composition adds no new epistemic
side effects beyond the existing Bayesian interpretation update. -/
noncomputable def applyPerceptualInterpretation
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (event : Event)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : PerceptualInterpretationAdmissible w s event c) :
    EpistemicBeliefState ProofId Agent Event Object Location Institution Concept :=
  applyInterpretation s c h.interpretation

/-- Positive evidence admitted through the full perception pipeline strictly
raises the hypothesis confidence. -/
theorem positive_perceptual_interpretation_increases_confidence
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
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
  simpa [applyPerceptualInterpretation] using
    (grounded_positive_interpretation_increases_confidence
      s c h.interpretation hprior0 hprior1 hnot hevidence)

/-- Revoking the observation proof that grounds a perceptual interpretation
invalidates its downstream attribution proof by truth maintenance. -/
theorem revoking_perceptual_evidence_invalidates_attribution
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (s : EpistemicBeliefState ProofId Agent Event Object Location Institution Concept)
    (event : Event)
    (c : InterpretationCandidate ProofId Agent Event Object Location Institution Concept)
    (h : PerceptualInterpretationAdmissible w s event c) :
    ¬ ProofValid s.kb.graph
        (deactivateProof s.active c.evidenceProof)
        c.attributionProof := by
  exact revoking_evidence_invalidates_attribution s c h.interpretation

end NarrativeDynamics
