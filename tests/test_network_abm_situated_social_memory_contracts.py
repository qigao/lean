from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
    SituatedMemoryCognitiveModel,
    initialize_situated_memory_cognition,
)
from narrative_dynamics.abm.situated_memory_contracts import standard_situated_memory_policy
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedClaimTopic,
    SituatedConsolidatedClaim,
    SituatedSocialEvidence,
    SituatedSocialEvidenceKind,
    SituatedSocialMemoryModel,
    SituatedSocialMemoryPolicy,
    SituatedSourceRelationship,
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from tests.situated_cognition_fixtures import cognitive_office_model


def memory_cognition_model():
    cognition = cognitive_office_model()
    return SituatedMemoryCognitiveModel(
        "office-memory-cognition",
        "1",
        cognition,
        standard_situated_memory_policy(),
        tuple(SituatedAgentRecallPolicy(item.agent_id) for item in cognition.agents),
    )


def social_model():
    return SituatedSocialMemoryModel(
        "office-social-memory",
        "1",
        memory_cognition_model(),
        (
            SituatedClaimTopic("restructuring", ("denied", "approved")),
        ),
        SituatedSocialMemoryPolicy(
            initial_source_trust=0.5,
            confirmation_rate=0.2,
            contradiction_rate=0.4,
            confirmation_affinity_delta=0.1,
            contradiction_affinity_delta=0.2,
            max_unresolved_age_rounds=3,
            max_active_claims=2,
        ),
    )


class SituatedSocialMemoryContractTests(unittest.TestCase):
    def test_topics_policy_and_model_are_canonical_and_exact(self):
        model = social_model()

        self.assertEqual(model.topics[0].symbol_ids, ("approved", "denied"))
        self.assertEqual(model.content_hash, social_model().content_hash)
        with self.assertRaisesRegex(ValueError, "one social topic"):
            replace(
                model,
                topics=model.topics + (SituatedClaimTopic("duplicate", ("approved", "denied")),),
            )
        with self.assertRaisesRegex(ValueError, "cognitive observation symbol"):
            replace(model, topics=(SituatedClaimTopic("unknown", ("missing", "approved")),))
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            replace(model.policy, initial_source_trust=1.1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            replace(model.policy, max_active_claims=0)

    def test_checkpoint_initializes_every_directed_relationship_without_history(self):
        model = social_model()
        cognition = model.memory_cognitive_model.cognitive_model
        story = initialize_situated_story(
            cognition.world_model,
            initialize_situated_world(cognition.world_model),
        )
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent("wait", "alice", SituatedActionKind.WAIT),
        ))
        cognitive_state = initialize_situated_memory_cognition(
            model.memory_cognitive_model,
            story,
        )

        state = initialize_situated_social_memory(model, cognitive_state)

        agent_count = len(cognition.agents)
        self.assertEqual(len(state.relationships), agent_count * (agent_count - 1))
        self.assertTrue(state.checkpoint)
        self.assertIsNone(state.parent_state_hash)
        self.assertEqual(state.round_index, 1)
        self.assertEqual(state.cognitive_state_hash, cognitive_state.content_hash)
        self.assertEqual(state.claims, ())
        self.assertEqual(state.processed_evidence_ids, ())
        self.assertTrue(all(item.trust == 0.5 for item in state.relationships))
        self.assertTrue(all(item.affinity == 0.0 for item in state.relationships))
        self.assertEqual(
            tuple((item.observer_agent_id, item.source_agent_id) for item in state.relationships),
            tuple(sorted(
                (observer.agent_id, source.agent_id)
                for observer in cognition.agents
                for source in cognition.agents
                if observer.agent_id != source.agent_id
            )),
        )

    def test_relationship_claim_and_evidence_reject_invalid_identity_or_history(self):
        relationship = SituatedSourceRelationship("bob", "alice", 0.5, 0.0, 0, 0)
        with self.assertRaisesRegex(ValueError, "must differ"):
            replace(relationship, source_agent_id="bob")

        claim = SituatedConsolidatedClaim(
            "claim-1",
            "bob",
            "alice",
            "restructuring",
            "approved",
            ("event-1",),
            ("memory-1",),
            1,
            1,
            1,
            SituatedClaimStatus.ACTIVE,
        )
        self.assertEqual(claim.support_count, 1)
        with self.assertRaisesRegex(ValueError, "support count"):
            replace(claim, support_count=2)
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            replace(claim, last_round=0)

        testimony = SituatedSocialEvidence(
            "evidence-1",
            SituatedSocialEvidenceKind.TESTIMONY,
            "bob",
            "restructuring",
            "approved",
            1,
            "event-1",
            source_agent_id="alice",
            memory_id="memory-1",
        )
        self.assertEqual(testimony.source_agent_id, "alice")
        with self.assertRaisesRegex(ValueError, "requires a distinct source"):
            replace(testimony, source_agent_id="bob")
        verification = replace(
            testimony,
            evidence_id="verification-1",
            kind=SituatedSocialEvidenceKind.VERIFICATION,
            source_agent_id=None,
            memory_id=None,
        )
        self.assertIsNone(verification.source_agent_id)
        with self.assertRaisesRegex(ValueError, "cannot name a source"):
            replace(verification, source_agent_id="alice")


if __name__ == "__main__":
    unittest.main()
