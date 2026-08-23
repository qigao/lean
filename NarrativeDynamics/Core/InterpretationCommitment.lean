import NarrativeDynamics.Core.GroundedBeliefUpdate

namespace NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

/-- Commitment to one particular interpretation is stronger than proposition-
level belief support: the candidate's own attribution proof must remain valid,
and its normalized posterior coordinate must exceed the decision threshold. -/
def InterpretationCommittedAt
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (active : ProofId → Prop) (threshold : ℝ)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (i : ι) : Prop :=
  ProofValid g.kb.graph active (g.candidate i).attributionProof ∧
    threshold < posterior g.competition i

/-- In the original grounded active-proof state, a posterior above threshold
commits the agent to that candidate interpretation because grounding already
supplies a valid attribution proof. -/
theorem grounded_posterior_above_threshold_commits_interpretation
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι)
    (habove : threshold < posterior g.competition i) :
    InterpretationCommittedAt g.active threshold g i := by
  exact ⟨(g.admissible i).interpretation.attributionValid, habove⟩

/-- A committed candidate entails proposition-level held belief: its own valid
attribution proof supports the corresponding hypothesis, and posterior
confidence exceeds the same threshold. The converse need not hold because an
independent proof may support the proposition after this attribution fails. -/
theorem committed_interpretation_implies_held_belief
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι)
    (hcommit : InterpretationCommittedAt g.active threshold g i) :
    BeliefHeld threshold (posteriorBeliefState g i) := by
  have hfact :
      g.kb.graph.fact (g.candidate i).attributionProof = g.hypothesis i := by
    calc
      g.kb.graph.fact (g.candidate i).attributionProof
          = (g.candidate i).hypothesis :=
        (g.admissible i).interpretation.attributionMatches
      _ = (g.state i).hypothesis :=
        (g.admissible i).interpretation.hypothesisMatches
      _ = g.hypothesis i := rfl
  have hsupported : BeliefSupported g.kb g.active (g.hypothesis i) := by
    have h := valid_proof_supports_belief
      g.kb g.active (g.candidate i).attributionProof hcommit.1
    rw [hfact] at h
    exact h
  constructor
  · simpa [posteriorBeliefState, GroundedHypothesisSpace.state] using hsupported
  · simpa [posteriorBeliefState] using hcommit.2

/-- Numerical insufficiency blocks commitment even if the attribution proof is
symbolically valid. -/
theorem posterior_below_threshold_blocks_commitment
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι)
    (hbelow : posterior g.competition i ≤ threshold) :
    ¬ InterpretationCommittedAt g.active threshold g i := by
  intro hcommit
  exact (not_lt_of_ge hbelow) hcommit.2

/-- Retracting the common observation proof breaks every interpretation-
specific commitment grounded in it, independently of the numerical posterior. -/
theorem revoking_common_evidence_breaks_interpretation_commitment
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ) (i : ι) :
    ¬ InterpretationCommittedAt
        (deactivateProof g.active g.commonEvidenceProof)
        threshold g i := by
  intro hcommit
  exact (grounded_common_evidence_revocation_invalidates_attribution g i)
    hcommit.1

end NarrativeDynamics
