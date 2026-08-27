from __future__ import annotations

from dataclasses import replace
import unittest

import narrative_dynamics
import narrative_dynamics.narrative as narrative
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.analysis import analyze_narrative, derive_analysis_scope
from narrative_dynamics.narrative.decision import EvidenceAccess

from tests.narrative_test_support import make_health_model, make_test_domain, make_test_story


_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.trust import (
        NarrativeAnalysisArtifact,
        TrustedDomainBinding,
        TrustedDomainPin,
        TrustedDomainPolicy,
    )
except ImportError as error:
    _IMPORT_ERROR = error


_EXPECTED_PUBLIC_API = {
    # Canonical IR.
    "GENERIC_NARRATIVE_SCHEMA_VERSION",
    "EntityRef",
    "StateCellRef",
    "TypedValue",
    "Entity",
    "NarrativeEvent",
    "Observation",
    "Proposition",
    "Claim",
    "Reception",
    "ActionOption",
    "Decision",
    "GenericNarrative",
    # Domain declarations, transitions, and validation.
    "EntityTypeSpec",
    "ValueTypeSpec",
    "ParameterSpec",
    "StateVariableSpec",
    "StateEffectSpec",
    "EventTypeSpec",
    "ActionTypeSpec",
    "DecisionTypeSpec",
    "StateDeltaOp",
    "StateDelta",
    "SemanticHookBinding",
    "DomainSpec",
    "validate_narrative",
    # Replay.
    "EpistemicEvidence",
    "EpistemicCellView",
    "EpistemicState",
    "objective_state",
    "direct_state",
    "epistemic_state",
    # Uncertain belief.
    "BeliefMass",
    "BeliefLikelihood",
    "BeliefDistribution",
    "BeliefUpdateStep",
    "UncertainBeliefCellView",
    "UncertainBeliefState",
    "UncertainBeliefModelSpec",
    "UncertainBeliefResolutionError",
    "uncertain_epistemic_state",
    # Decision mechanisms.
    "EvidenceAccess",
    "DecisionCellView",
    "DecisionContext",
    "DecisionChoice",
    "DecisionModelSpec",
    "DecisionResult",
    "EpistemicResolutionError",
    "DecisionResolutionError",
    "require_resolved_cell",
    "run_decision_model",
    # Intentional decision.
    "GoalSpec",
    "GoalModelSpec",
    "GoalState",
    "ChoiceModelSpec",
    "IntentionalDecisionModelSpec",
    "IntentionalDecisionResult",
    "IntentionalDecisionResolutionError",
    "GoalResolutionError",
    "ChoiceResolutionError",
    "run_intentional_decision",
    # World transition.
    "ActionEffectSpec",
    "ActionTransitionSpec",
    "WorldTransitionModelSpec",
    "ActionIntent",
    "WorldState",
    "ActionTransitionRecord",
    "WorldStepResult",
    "WorldTransitionError",
    "WorldTransitionConflictError",
    "world_state_from_story",
    "advance_world_step",
    # Runtime observation projection.
    "ObservationCapabilitySpec",
    "ObserverProjectionSpec",
    "ObservationProjectionModelSpec",
    "ObservationFact",
    "ProjectedObservation",
    "ObservationProjectionResult",
    "ObservationProjectionError",
    "project_world_observations",
    # Runtime percept admission.
    "RuntimeEpistemicEvidence",
    "RuntimePerceptView",
    "RuntimeEvidenceBatch",
    "RuntimeEvidenceLedger",
    "RuntimePerceptAdmissionResult",
    "RuntimePerceptAdmissionError",
    "runtime_evidence_ledger_from_story",
    "admit_world_percepts",
    # Runtime cognition replay.
    "RuntimeEpistemicCellView",
    "RuntimeEpistemicState",
    "RuntimeEpistemicResolutionError",
    "RuntimeBeliefModelSpec",
    "RuntimeBeliefUpdateStep",
    "RuntimeUncertainBeliefCellView",
    "RuntimeUncertainBeliefState",
    "RuntimeBeliefResolutionError",
    "runtime_epistemic_state",
    "runtime_uncertain_belief_state",
    # Runtime intentional selection.
    "RuntimeIntentionalDecisionModelSpec",
    "RuntimeIntentionalDecisionResult",
    "RuntimeIntentionalDecisionResolutionError",
    "run_runtime_intentional_decision",
    # Runtime reactive selection.
    "ReactiveCueView",
    "RuntimeReactiveCueSnapshot",
    "RuntimeReactiveDecisionContext",
    "RuntimeReactiveDecisionModelSpec",
    "RuntimeReactiveDecisionResult",
    "RuntimeReactiveDecisionResolutionError",
    "run_runtime_reactive_decision",
    # Runtime planning selection.
    "PlanningHiddenState",
    "PlanningObservation",
    "PlanningBeliefState",
    "RuntimePlanningBeliefContext",
    "PlanningTransitionContext",
    "PlanningObservationContext",
    "PlanningRewardContext",
    "PlanningValueRecord",
    "PlanningBeliefUpdate",
    "RuntimePlanningDecisionModelSpec",
    "RuntimePlanningDecisionResult",
    "RuntimePlanningDecisionResolutionError",
    "run_runtime_planning_decision",
    # Runtime generic decision dispatch.
    "RuntimeDecisionModelSpec",
    "RuntimeDecisionDispatchResult",
    "RuntimeDecisionDispatchError",
    "run_runtime_decision",
    # Multi-step simulation.
    "RuntimeAgentSpec",
    "SimulationModelSpec",
    "SimulationState",
    "SimulationAgentStep",
    "SimulationStepResult",
    "SimulationTrajectory",
    "SimulationError",
    "SimulationStepError",
    "SimulationTrajectoryError",
    "simulation_state_from_story",
    "simulate_step",
    "simulate_trajectory",
    # Analysis.
    "AnalysisScope",
    "TriggerRef",
    "AgentEvolutionView",
    "EvolutionSnapshot",
    "NarrativeTrajectory",
    "MechanismPairwiseComparison",
    "MechanismComparison",
    "NarrativeAnalysis",
    "derive_analysis_scope",
    "build_trajectory",
    "analyze_narrative",
    # Interventions.
    "Intervention",
    "Divergence",
    "CounterfactualResult",
    "generate_minimal_interventions",
    "evaluate_counterfactual",
    "first_divergence",
    # Compiler provenance and deterministic compilation.
    "SourceDocument",
    "SourceSpan",
    "SourceBundle",
    "ExtractorIdentity",
    "CandidateEntityRef",
    "CandidateStateCellRef",
    "CandidateValue",
    "CandidateEntity",
    "CandidateEvent",
    "CandidateObservation",
    "CandidateProposition",
    "CandidateClaim",
    "CandidateReception",
    "CandidateActionOption",
    "CandidateDecision",
    "CandidateBundle",
    "ResolutionRecord",
    "CompilationDiagnostic",
    "CompilationResult",
    "compile_candidates",
    # External trust and analysis lineage.
    "TrustedDomainPin",
    "TrustedDomainPolicy",
    "TrustedDomainBinding",
    "NarrativeAnalysisArtifact",
}


def pin_for(domain):
    return TrustedDomainPin(
        domain.domain_id,
        domain.version,
        domain.content_hash,
        {hook.name: hook.implementation_hash for hook in domain.semantic_hooks},
    )


class NarrativeTrustTests(unittest.TestCase):
    def require_trust(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"narrative trust boundary is missing: {_IMPORT_ERROR}")

    def test_policy_and_lineage(self) -> None:
        self.require_trust()
        domain = make_test_domain()
        policy = TrustedDomainPolicy("research", "1", (pin_for(domain),))
        binding = policy.bind(domain)
        self.assertIsInstance(binding, TrustedDomainBinding)
        self.assertEqual(binding.pin.domain_spec_hash, domain.content_hash)

        with self.assertRaisesRegex(ValueError, "trusted domain"):
            policy.bind(replace(domain, version="2"))

        story = make_test_story()
        model = make_health_model("epistemic", EvidenceAccess.EPISTEMIC)
        analysis = analyze_narrative(story, domain, "d1", model)
        scope = derive_analysis_scope(story, domain, "d1")
        common = dict(
            canonical_narrative_hash=story.content_hash,
            domain_spec_hash=domain.content_hash,
            decision_model_identity=model.content_hash,
            analysis_scope_hash=stable_content_hash(scope.to_dict()),
            intervention_manifest_hash=stable_content_hash(()),
            analysis_payload_hash=stable_content_hash(analysis.to_dict()),
        )
        first = NarrativeAnalysisArtifact(
            **common,
            compilation_artifact_hash="sha256:" + "1" * 64,
        )
        second = NarrativeAnalysisArtifact(
            **common,
            compilation_artifact_hash="sha256:" + "2" * 64,
        )
        self.assertEqual(first.analysis_identity_hash, second.analysis_identity_hash)
        self.assertNotEqual(first.lineage_dict(), second.lineage_dict())

    def test_exact_public_surface_and_root_isolation(self) -> None:
        actual = set(getattr(narrative, "__all__", ()))
        self.assertEqual(actual, _EXPECTED_PUBLIC_API)
        for name in sorted(_EXPECTED_PUBLIC_API):
            self.assertTrue(hasattr(narrative, name), name)
            self.assertFalse(hasattr(narrative_dynamics, name), name)


if __name__ == "__main__":
    unittest.main()
