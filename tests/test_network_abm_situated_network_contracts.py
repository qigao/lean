from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import ObservationChannel
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_memory_cognition_contracts import SituatedAgentRecallPolicy
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    standard_situated_percept_memory_policy,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_story import initialize_situated_story
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkAccessEdge,
    SituatedNetworkAgentNode,
    SituatedNetworkEmergenceMetrics,
    SituatedNetworkRelationshipEdge,
    SituatedNetworkRoundResult,
    SituatedNetworkRuntimeModel,
    SituatedNetworkRuntimeState,
    SituatedNetworkSnapshot,
    SituatedNetworkTrajectory,
    SituatedNetworkTransmission,
)
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import perception_model
from tests.test_network_abm_situated_social_memory_contracts import social_model


def digest(label: str) -> str:
    return "sha256:" + label.encode("utf-8").hex().ljust(64, "0")[:64]


def memory_model():
    cognition = cognitive_office_model()
    return SituatedPerceptMemoryCognitiveModel(
        "office-percept-memory-cognition",
        "1",
        perception_model(),
        cognition,
        standard_situated_percept_memory_policy(),
        tuple(SituatedAgentRecallPolicy(agent.agent_id) for agent in cognition.agents),
    )


def runtime_model():
    return SituatedNetworkRuntimeModel(
        "office-network", "1", memory_model(), social_model(), "approved", 0.7, 0.5
    )


def snapshot_fixture(*, access_edges=None, round_index: int = 1):
    access_edges = (
        SituatedNetworkAccessEdge("alice", "bob", 1.0, 5.0, True),
    ) if access_edges is None else access_edges
    return SituatedNetworkSnapshot(
        "office-network",
        runtime_model().content_hash,
        round_index,
        digest("story"),
        digest("cognitive"),
        digest("social"),
        (
            SituatedNetworkAgentNode("bob", "recipient", "records", 0.4, 1),
            SituatedNetworkAgentNode("alice", "source", "lobby", 0.8, 0),
        ),
        (
            SituatedNetworkRelationshipEdge("alice", "bob", 0.6, 0.1, 1, 0, True),
        ),
        access_edges,
        () if round_index == 0 else (
            SituatedNetworkTransmission(
                "tell-1", "alice", "bob", SituatedPerceptFidelity.DETECTED,
                (ObservationChannel.AUDITORY,),
            ),
        ),
        0 if round_index == 0 else 1,
    )


def metrics_fixture(snapshot):
    return SituatedNetworkEmergenceMetrics(
        snapshot.content_hash, snapshot.round_index,
        2, 2, 1, 0.5, 0.6, 0.04, 1, 0.6, 1,
        1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    )


class SituatedNetworkContractTests(unittest.TestCase):
    def test_runtime_model_rejects_missing_common_hypothesis(self):
        with self.assertRaisesRegex(ValueError, "tracked hypothesis"):
            SituatedNetworkRuntimeModel(
                "office-network", "1", memory_model(), social_model(),
                "missing", 0.7, 0.5,
            )

    def test_runtime_model_requires_exact_cognitive_world_and_roster_binding(self):
        model = runtime_model()
        self.assertEqual(model.content_hash, runtime_model().content_hash)
        with self.assertRaisesRegex(ValueError, "exact cognitive model"):
            social = social_model()
            divergent_cognition = replace(
                social.memory_cognitive_model.cognitive_model, version="other"
            )
            divergent_memory = replace(
                social.memory_cognitive_model, cognitive_model=divergent_cognition
            )
            replace(model, social_memory_model=replace(social, memory_cognitive_model=divergent_memory))
        with self.assertRaisesRegex(ValueError, "threshold"):
            replace(model, adoption_threshold=1.1)

    def test_snapshot_rejects_duplicate_directed_access_pair(self):
        edge = SituatedNetworkAccessEdge("alice", "bob", 1.0, 5.0, True)
        with self.assertRaisesRegex(ValueError, "access pairs must be unique"):
            snapshot_fixture(access_edges=(edge, edge))

    def test_snapshot_canonicalizes_network_identity_and_rejects_duplicate_transmission(self):
        snapshot = snapshot_fixture()
        self.assertEqual(tuple(node.agent_id for node in snapshot.nodes), ("alice", "bob"))
        self.assertEqual(snapshot.content_hash, snapshot_fixture().content_hash)
        with self.assertRaisesRegex(ValueError, "transmission identities must be unique"):
            replace(snapshot, transmissions=snapshot.transmissions * 2)

    def test_snapshot_counts_latest_tell_events_without_private_payloads(self):
        snapshot = snapshot_fixture()

        self.assertEqual(snapshot.latest_tell_event_count, 1)
        with self.assertRaisesRegex(ValueError, "latest tell event count"):
            replace(snapshot, latest_tell_event_count=0)
        with self.assertRaisesRegex(ValueError, "round zero"):
            replace(
                snapshot_fixture(round_index=0),
                transmissions=(),
                latest_tell_event_count=1,
            )

    def test_metrics_allow_all_inactive_edges_to_have_all_edge_mean_trust(self):
        snapshot = snapshot_fixture()

        inactive = replace(
            metrics_fixture(snapshot),
            active_relationship_edge_count=0,
            mean_relationship_trust=0.6,
        )

        self.assertEqual(inactive.mean_relationship_trust, 0.6)

    def test_node_edges_and_transmissions_reject_invalid_private_or_physical_projection(self):
        with self.assertRaisesRegex(ValueError, "probability"):
            SituatedNetworkAgentNode("alice", "source", "lobby", -0.1, 0)
        with self.assertRaisesRegex(ValueError, "must differ"):
            SituatedNetworkRelationshipEdge("alice", "alice", 0.5, 0.0, 0, 0, False)
        with self.assertRaisesRegex(ValueError, "auditory"):
            SituatedNetworkAccessEdge("alice", "bob", 1.0, -0.1, True)
        with self.assertRaisesRegex(ValueError, "distinct"):
            SituatedNetworkTransmission(
                "tell-1", "alice", "alice", SituatedPerceptFidelity.EXACT,
                (ObservationChannel.AUDITORY,),
            )

    def test_transmission_rejects_actor_private_channels(self):
        for channel in (ObservationChannel.SELF, ObservationChannel.INSPECTION):
            with self.subTest(channel=channel), self.assertRaisesRegex(
                ValueError, "visual or auditory"
            ):
                SituatedNetworkTransmission(
                    "tell-1", "alice", "bob", SituatedPerceptFidelity.EXACT,
                    (channel,),
                )

    def test_runtime_state_rejects_forged_round_zero_parent(self):
        model, story, cognitive_state, social_state, snapshot, metrics = self._initial_runtime_values()
        with self.assertRaisesRegex(ValueError, "initial round"):
            SituatedNetworkRuntimeState(
                "office-network", model.content_hash, 0, digest("forged"),
                story, cognitive_state, social_state, snapshot, metrics,
            )

    def test_runtime_state_rejects_cross_subsystem_hash_mismatches(self):
        model, story, cognitive_state, social_state, _, _ = self._initial_runtime_values()

        mismatched_cognition = replace(cognitive_state, story_hash=digest("unrelated-story"))
        snapshot = self._snapshot_for(model, story, mismatched_cognition, social_state)
        with self.assertRaisesRegex(ValueError, "cognitive state must bind exact story"):
            SituatedNetworkRuntimeState(
                "office-network", model.content_hash, 0, None,
                story, mismatched_cognition, social_state, snapshot, metrics_fixture(snapshot),
            )

        mismatched_social = replace(social_state, cognitive_state_hash=digest("unrelated-cognition"))
        snapshot = self._snapshot_for(model, story, cognitive_state, mismatched_social)
        with self.assertRaisesRegex(ValueError, "social state must bind exact cognitive state"):
            SituatedNetworkRuntimeState(
                "office-network", model.content_hash, 0, None,
                story, cognitive_state, mismatched_social, snapshot, metrics_fixture(snapshot),
            )

    def test_runtime_result_and_trajectory_require_exact_parent_hash_chain(self):
        model, story, cognitive_state, social_state, snapshot, metrics = self._initial_runtime_values()
        state = SituatedNetworkRuntimeState(
            "office-network", model.content_hash, 0, None,
            story, cognitive_state, social_state, snapshot, metrics,
        )
        with TemporaryDirectory() as temporary:
            advanced = simulate_situated_percept_social_cognitive_round(
                f"{temporary}/runtime.sqlite3",
                model.percept_memory_model,
                model.social_memory_model,
                story,
                cognitive_state,
                social_state,
            )
        next_snapshot = replace(
            snapshot,
            round_index=1,
            story_hash=advanced.next_story.content_hash,
            cognitive_state_hash=advanced.next_cognitive_state.content_hash,
            social_state_hash=advanced.next_social_state.content_hash,
        )
        next_metrics = replace(metrics_fixture(next_snapshot), round_index=1)
        next_state = SituatedNetworkRuntimeState(
            "office-network", model.content_hash, 1, state.content_hash,
            advanced.next_story, advanced.next_cognitive_state, advanced.next_social_state,
            next_snapshot, next_metrics,
        )
        result = SituatedNetworkRoundResult("office-network", model.content_hash, state, next_state)
        trajectory = SituatedNetworkTrajectory("office-network", model.content_hash, state, (result,), next_state)
        self.assertEqual(trajectory.content_hash, SituatedNetworkTrajectory(
            "office-network", model.content_hash, state, (result,), next_state
        ).content_hash)
        with self.assertRaisesRegex(ValueError, "exact parent"):
            SituatedNetworkRoundResult(
                "office-network", model.content_hash, state,
                replace(next_state, parent_state_hash=digest("forged")),
            )
        with self.assertRaisesRegex(ValueError, "exact chain"):
            replace(trajectory, final_state=state)

    @staticmethod
    def _snapshot_for(model, story, cognitive_state, social_state):
        return replace(
            snapshot_fixture(round_index=0),
            model_hash=model.content_hash,
            story_hash=story.content_hash,
            cognitive_state_hash=cognitive_state.content_hash,
            social_state_hash=social_state.content_hash,
        )

    @classmethod
    def _initial_runtime_values(cls):
        model = runtime_model()
        story = initialize_situated_story(
            model.percept_memory_model.cognitive_model.world_model,
            initialize_situated_world(model.percept_memory_model.cognitive_model.world_model),
            perception_model=model.percept_memory_model.perception_model,
        )
        cognitive_state = initialize_situated_percept_memory_cognition(
            model.percept_memory_model, story
        )
        social_state = initialize_situated_social_memory(
            model.social_memory_model, cognitive_state
        )
        snapshot = cls._snapshot_for(model, story, cognitive_state, social_state)
        metrics = metrics_fixture(snapshot)
        return model, story, cognitive_state, social_state, snapshot, metrics


if __name__ == "__main__":
    unittest.main()
