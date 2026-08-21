import NarrativeDynamics.Core.InterpretationSelection
import NarrativeDynamics.Core.Drive

namespace NarrativeDynamics

open scoped BigOperators

universe uH uP uA uE uO uL uI uC uN

/-- Expected goal instrumentality under one finite, provenance-grounded
posterior over competing interpretations. The semantic goal receives no direct
belief reward: each posterior coordinate only weights the agent's learned
hypothesis-conditioned instrumentality prediction. -/
noncomputable def groundedExpectedInstrumentality
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ) : ℝ :=
  ∑ i, posterior g.competition i * instrumentality i

/-- Goal score after propagating a grounded finite posterior through learned
hypothesis-conditioned instrumentality. -/
noncomputable def groundedEpistemicGoalScore
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (pressure cost risk : ℝ)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : ι → ℝ) : ℝ :=
  goalScore pressure (groundedExpectedInstrumentality g instrumentality) cost risk

/-- A grounded goal evaluation is symbolically licensed only while every
hypothesis-specific attribution proof remains valid in the chosen active-proof
state. -/
def GroundedGoalEvaluationAdmissible
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (active : ProofId → Prop)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event) : Prop :=
  ∀ i, ProofValid g.kb.graph active (g.candidate i).attributionProof

/-- A normalized posterior preserves a hypothesis-independent learned
instrumentality value. -/
theorem grounded_expectedInstrumentality_constant
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hmass : 0 < evidenceMass g.competition)
    (value : ℝ) :
    groundedExpectedInstrumentality g (fun _ => value) = value := by
  classical
  unfold groundedExpectedInstrumentality
  rw [← Finset.sum_mul]
  rw [grounded_posterior_sum_one g hmass]
  exact one_mul value

/-- A normalized, nonnegative grounded posterior makes expected
instrumentality a convex combination: it remains inside every common
pointwise lower/upper bound. -/
theorem grounded_expectedInstrumentality_bounds
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
  classical
  have hnonneg : ∀ i, 0 ≤ posterior g.competition i :=
    fun i => grounded_posterior_nonneg g hprior hlikelihood hmass i
  constructor
  · calc
      lower = groundedExpectedInstrumentality g (fun _ => lower) :=
        (grounded_expectedInstrumentality_constant g hmass lower).symm
      _ ≤ groundedExpectedInstrumentality g instrumentality := by
        unfold groundedExpectedInstrumentality
        refine Finset.sum_le_sum ?_
        intro i hi
        exact mul_le_mul_of_nonneg_left (hlower i) (hnonneg i)
  · calc
      groundedExpectedInstrumentality g instrumentality ≤
          groundedExpectedInstrumentality g (fun _ => upper) := by
        unfold groundedExpectedInstrumentality
        refine Finset.sum_le_sum ?_
        intro i hi
        exact mul_le_mul_of_nonneg_left (hupper i) (hnonneg i)
      _ = upper := grounded_expectedInstrumentality_constant g hmass upper

/-- The original grounded active-proof state licenses finite goal evaluation. -/
theorem grounded_goal_evaluation_admissible
    {ι : Type uH} {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event) :
    GroundedGoalEvaluationAdmissible g.active g := by
  intro i
  exact (g.admissible i).interpretation.attributionValid

/-- With at least one competing hypothesis, retracting the common observation
proof invalidates every attribution and therefore blocks the grounded goal
evaluation as a whole. -/
theorem revoking_common_evidence_blocks_grounded_goal_evaluation
    {ι : Type uH} [Nonempty ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event) :
    ¬ GroundedGoalEvaluationAdmissible
        (deactivateProof g.active g.commonEvidenceProof) g := by
  classical
  intro hall
  let i : ι := Classical.choice (inferInstance : Nonempty ι)
  exact (grounded_common_evidence_revocation_invalidates_attribution g i) (hall i)

end NarrativeDynamics
