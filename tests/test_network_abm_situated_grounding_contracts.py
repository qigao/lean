from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_grounding_contracts import (
    SituatedGroundedClaim,
    SituatedGroundingEvidence,
    SituatedGroundingEvidenceKind,
    SituatedGroundingModality,
    SituatedGroundingPolarity,
    SituatedGroundingPredicate,
    SituatedGroundingProviderIdentity,
    SituatedGroundingTemporalScope,
    SituatedSemanticGroundingArtifact,
    SituatedSemanticGroundingModel,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import perception_model
from tests.test_network_abm_situated_percept_memory_cognition import recall_model
from tests.test_network_abm_situated_percept_social_cognition import bound_social_model


def grounding_model() -> SituatedSemanticGroundingModel:
    cognition = cognitive_office_model()
    memory = recall_model({}, cognition=cognition, perception=perception_model())
    social = bound_social_model(memory)
    return SituatedSemanticGroundingModel(
        "office-grounding",
        "1",
        memory,
        social,
        ("alice", "bob", "memo"),
        (
            SituatedGroundingPredicate(
                "restructuring-status",
                ("approved", "denied"),
                social_topic_id="restructuring",
            ),
        ),
        maximum_evidence_items=8,
        maximum_claims=4,
    )


def exact_tell_evidence() -> SituatedGroundingEvidence:
    return SituatedGroundingEvidence(
        "percept:bob:alice-tell",
        SituatedGroundingEvidenceKind.PERCEPT,
        "bob",
        "alice-tell",
        "sha256:" + "1" * 64,
        3,
        (ObservationChannel.AUDITORY,),
        SituatedPerceptFidelity.EXACT,
        "Alice said the restructuring was approved.",
        actor_agent_id="alice",
        event_kind=SituatedActionKind.TELL,
        place_id="corridor",
        outcome="told",
        details=(EvidenceFact("message", "The restructuring is approved."),),
    )


def claim() -> SituatedGroundedClaim:
    return SituatedGroundedClaim(
        "memo",
        "restructuring-status",
        "approved",
        SituatedGroundingPolarity.AFFIRMED,
        SituatedGroundingModality.ASSERTED,
        SituatedGroundingTemporalScope.PRESENT,
        "alice",
        0.85,
        ("percept:bob:alice-tell",),
    )


class SituatedGroundingContractTests(unittest.TestCase):
    def test_model_binds_exact_private_memory_social_vocabulary(self):
        model = grounding_model()
        self.assertEqual(model.subject_ids, ("alice", "bob", "memo"))
        self.assertEqual(model.predicates[0].value_ids, ("approved", "denied"))
        self.assertEqual(model.predicates[0].social_topic_id, "restructuring")
        self.assertEqual(model.to_dict()["percept_memory_model_hash"], model.percept_memory_model.content_hash)

        with self.assertRaisesRegex(ValueError, "world entity"):
            replace(model, subject_ids=model.subject_ids + ("invented-person",))
        with self.assertRaisesRegex(ValueError, "social topic"):
            replace(
                model,
                predicates=(SituatedGroundingPredicate("status", ("approved",), "missing"),),
            )
        with self.assertRaisesRegex(ValueError, "topic symbol"):
            replace(
                model,
                predicates=(SituatedGroundingPredicate("status", ("approved", "invented"), "restructuring"),),
            )

    def test_evidence_enforces_sanitized_fidelity_and_memory_identity(self):
        evidence = exact_tell_evidence()
        self.assertEqual(evidence.details[0].value, "The restructuring is approved.")
        self.assertIsNone(evidence.memory_id)

        with self.assertRaisesRegex(ValueError, "detected percept"):
            replace(evidence, fidelity=SituatedPerceptFidelity.DETECTED)
        with self.assertRaisesRegex(ValueError, "memory evidence requires"):
            replace(evidence, evidence_kind=SituatedGroundingEvidenceKind.MEMORY)
        memory = replace(
            evidence,
            evidence_id="memory:bob:alice-tell",
            evidence_kind=SituatedGroundingEvidenceKind.MEMORY,
            memory_id="memory:bob:alice-tell",
        )
        self.assertEqual(memory.memory_id, memory.evidence_id)

    def test_claim_and_artifact_are_canonical_and_content_addressed(self):
        model = grounding_model()
        evidence = exact_tell_evidence()
        accepted = claim()
        provider = SituatedGroundingProviderIdentity("test-provider", "1", "fixture-model")
        artifact = SituatedSemanticGroundingArtifact(
            model.model_id,
            model.content_hash,
            "request-1",
            "bob",
            provider,
            "sha256:" + "2" * 64,
            "sha256:" + "3" * 64,
            (evidence,),
            (accepted,),
        )
        self.assertTrue(artifact.content_hash.startswith("sha256:"))
        self.assertEqual(artifact.claims[0].evidence_ids, (evidence.evidence_id,))
        self.assertEqual(artifact.to_dict()["provider"], provider.to_dict())

        with self.assertRaisesRegex(ValueError, "confidence"):
            replace(accepted, confidence=1.1)
        with self.assertRaisesRegex(ValueError, "evidence"):
            replace(artifact, claims=(replace(accepted, evidence_ids=("not-private",)),))
        with self.assertRaisesRegex(ValueError, "observer"):
            replace(artifact, observer_agent_id="alice")

    def test_provider_identity_and_predicate_reject_loose_values(self):
        with self.assertRaisesRegex(ValueError, "provider id"):
            SituatedGroundingProviderIdentity("", "1", "model")
        with self.assertRaisesRegex(ValueError, "unique"):
            SituatedGroundingPredicate("status", ("approved", "approved"))
        with self.assertRaisesRegex(ValueError, "at least one"):
            SituatedGroundingPredicate("status", ())


if __name__ == "__main__":
    unittest.main()
