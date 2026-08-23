import NarrativeDynamics.Core.GroundedEpistemicGoal

open NarrativeDynamics

universe uH uP uA uE uO uL uI uC uN

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hmass : 0 < evidenceMass g.competition)
    (value : ℝ) :
    groundedExpectedInstrumentality g (fun _ => value) = value := by
  exact grounded_expectedInstrumentality_constant g hmass value

example
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ)
    (lower upper : ℝ)
    (hprior : ∀ i, 0 ≤ g.prior i)
    (hlikelihood : ∀ i, 0 ≤ (g.candidate i).likelihoodH)
    (hmass : 0 < evidenceMass g.competition)
    (hlower : ∀ i, lower ≤ instrumentality i)
    (hupper : ∀ i, instrumentality i ≤ upper) :
    lower ≤ groundedExpectedInstrumentality g instrumentality ∧
      groundedExpectedInstrumentality g instrumentality ≤ upper := by
  exact grounded_expectedInstrumentality_bounds
    g instrumentality lower upper hprior hlikelihood hmass hlower hupper

example
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event) :
    GroundedGoalEvaluationAdmissible g.active g := by
  exact grounded_goal_evaluation_admissible g

example
    {ι : Type uH} [Nonempty ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event) :
    ¬ GroundedGoalEvaluationAdmissible
        (deactivateProof g.active g.commonEvidenceProof) g := by
  exact revoking_common_evidence_blocks_grounded_goal_evaluation g
