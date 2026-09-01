from dataclasses import replace
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    standard_situated_percept_memory_policy,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from narrative_dynamics.abm.situated_memory_cognition_contracts import SituatedAgentRecallPolicy
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import (
    advance_situated_story,
    initialize_situated_story,
)
from narrative_dynamics.abm.situated_network import (
    initialize_situated_network_runtime,
    measure_situated_network_emergence,
    project_situated_network_snapshot,
    simulate_situated_network_round,
    simulate_situated_network_runtime,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import (
    SECRET,
    initial_state,
    perception_model,
)
from tests.test_network_abm_situated_social_memory_contracts import social_model


def runtime_model():
    base_cognition = cognitive_office_model()
    world = replace(
        base_cognition.world_model,
        agents=tuple(
            item for item in base_cognition.world_model.agents
            if item.agent_id in {"alice", "bob"}
        ),
    )
    cognition = replace(
        base_cognition,
        world_model=world,
        agents=tuple(
            item for item in base_cognition.agents if item.agent_id in {"alice", "bob"}
        ),
    )
    base_perception = perception_model()
    perception = replace(
        base_perception,
        world_model=world,
        agent_profiles=tuple(
            item for item in base_perception.agent_profiles
            if item.agent_id in {"alice", "bob"}
        ),
    )
    percept_memory = SituatedPerceptMemoryCognitiveModel(
        "office-percept-memory-cognition",
        "1",
        perception,
        cognition,
        standard_situated_percept_memory_policy(),
        tuple(SituatedAgentRecallPolicy(item.agent_id) for item in cognition.agents),
    )
    base_social = social_model()
    social = replace(
        base_social,
        memory_cognitive_model=replace(
            base_social.memory_cognitive_model,
            cognitive_model=cognition,
            agents=tuple(
                item for item in base_social.memory_cognitive_model.agents
                if item.agent_id in {"alice", "bob"}
            ),
        ),
    )
    return SituatedNetworkRuntimeModel(
        "office-network", "1", percept_memory, social, "approved", 0.7, 0.5
    )


def tell_case(*, door_open: bool, auditory_intensity: float = 60.0):
    model = runtime_model()
    if auditory_intensity != 60.0:
        perception = replace(
            model.percept_memory_model.perception_model,
            signal_profiles=tuple(
                replace(item, auditory_intensity=auditory_intensity)
                if item.kind is SituatedActionKind.TELL
                else item
                for item in model.percept_memory_model.perception_model.signal_profiles
            ),
        )
        model = replace(
            model,
            percept_memory_model=replace(
                model.percept_memory_model,
                perception_model=perception,
            ),
        )
    story = initialize_situated_story(
        model.percept_memory_model.cognitive_model.world_model,
        initial_state(model.percept_memory_model.perception_model, door_open=door_open),
        perception_model=model.percept_memory_model.perception_model,
    )
    story = advance_situated_story(
        model.percept_memory_model.cognitive_model.world_model,
        story,
        (SituatedActionIntent("alice-tell", "alice", SituatedActionKind.TELL, message=SECRET),),
    )
    cognitive_state = initialize_situated_percept_memory_cognition(
        model.percept_memory_model, story
    )
    social_state = initialize_situated_social_memory(model.social_memory_model, cognitive_state)
    return model, story, cognitive_state, social_state


def initial_runtime_case():
    model = runtime_model()
    story = initialize_situated_story(
        model.percept_memory_model.cognitive_model.world_model,
        initial_state(model.percept_memory_model.perception_model, door_open=True),
        perception_model=model.percept_memory_model.perception_model,
    )
    cognitive_state = initialize_situated_percept_memory_cognition(
        model.percept_memory_model, story
    )
    social_state = initialize_situated_social_memory(
        model.social_memory_model, cognitive_state
    )
    return model, story, cognitive_state, social_state


class SituatedNetworkRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.database = f"{self.temporary.name}/network.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def test_atomic_round_keeps_story_cognition_social_snapshot_and_metrics_synchronized(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            model, story, cognitive_state, social_state
        )

        with patch(
            "narrative_dynamics.abm.situated_network."
            "simulate_situated_percept_social_cognitive_round",
            wraps=simulate_situated_percept_social_cognitive_round,
        ) as simulated:
            result = simulate_situated_network_round(self.database, model, initial)

        self.assertEqual(simulated.call_count, 1)
        self.assertEqual(result.prior_state, initial)
        self.assertEqual(result.next_state.round_index, initial.round_index + 1)
        self.assertEqual(result.next_state.parent_state_hash, initial.content_hash)
        self.assertEqual(
            result.next_state.story.content_hash,
            result.next_state.snapshot.story_hash,
        )
        self.assertEqual(
            result.next_state.cognitive_state.content_hash,
            result.next_state.snapshot.cognitive_state_hash,
        )
        self.assertEqual(
            result.next_state.social_state.content_hash,
            result.next_state.snapshot.social_state_hash,
        )
        self.assertEqual(
            result.next_state.metrics.snapshot_hash,
            result.next_state.snapshot.content_hash,
        )

    def test_runtime_returns_the_exact_parent_linked_state_chain(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            model, story, cognitive_state, social_state
        )

        trajectory = simulate_situated_network_runtime(
            self.database, model, initial, round_count=2
        )

        state = initial
        for item in trajectory.rounds:
            self.assertEqual(item.prior_state, state)
            self.assertEqual(item.next_state.parent_state_hash, state.content_hash)
            state = item.next_state
        self.assertEqual(trajectory.final_state, state)

    def test_trajectory_rejects_a_broken_parent_chain(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            model, story, cognitive_state, social_state
        )
        valid_trajectory = simulate_situated_network_runtime(
            self.database, model, initial, round_count=1
        )
        unrelated_state = replace(
            valid_trajectory.final_state,
            parent_state_hash="sha256:" + "f" * 64,
        )

        with self.assertRaisesRegex(ValueError, "exact chain"):
            replace(valid_trajectory, final_state=unrelated_state)

    def test_runtime_requires_a_positive_integer_round_count(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            model, story, cognitive_state, social_state
        )

        for round_count in (0, -1, 1.0, True):
            with self.subTest(round_count=round_count):
                with self.assertRaisesRegex(ValueError, "positive round count"):
                    simulate_situated_network_runtime(
                        self.database, model, initial, round_count=round_count
                    )


class SituatedNetworkProjectionTests(unittest.TestCase):
    def test_closed_door_projects_detected_tell_without_secret_payload(self):
        runtime_model, closed_story, cognitive_state, social_state = tell_case(door_open=False)

        snapshot = project_situated_network_snapshot(
            runtime_model, closed_story, cognitive_state, social_state
        )

        bob = next(item for item in snapshot.transmissions if item.observer_agent_id == "bob")
        self.assertIs(bob.fidelity, SituatedPerceptFidelity.DETECTED)
        serialized = json.dumps(snapshot.to_dict(), sort_keys=True)
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn('"message"', serialized)

    def test_open_door_projects_exact_tell_and_hand_checked_metrics(self):
        runtime_model, open_story, cognitive_state, social_state = tell_case(door_open=True)

        snapshot = project_situated_network_snapshot(
            runtime_model, open_story, cognitive_state, social_state
        )
        metrics = measure_situated_network_emergence(runtime_model, snapshot, social_state)

        self.assertEqual(metrics.population_size, 2)
        self.assertEqual(metrics.latest_tell_event_count, 1)
        self.assertEqual(metrics.exact_transmission_count, 1)

    def test_no_observer_tell_counts_event_without_transmission(self):
        runtime_model, story, cognitive_state, social_state = tell_case(
            door_open=True, auditory_intensity=10.0
        )

        snapshot = project_situated_network_snapshot(
            runtime_model, story, cognitive_state, social_state
        )
        metrics = measure_situated_network_emergence(runtime_model, snapshot, social_state)

        self.assertEqual(snapshot.transmissions, ())
        self.assertEqual(metrics.latest_tell_event_count, 1)
        self.assertEqual(metrics.transmission_count, 0)
        self.assertEqual(metrics.reached_observer_count, 0)

    def test_snapshot_projects_all_directed_access_and_excludes_actor_self_percept(self):
        runtime_model, story, cognitive_state, social_state = tell_case(door_open=True)

        snapshot = project_situated_network_snapshot(
            runtime_model, story, cognitive_state, social_state
        )

        self.assertEqual(
            {(item.source_agent_id, item.observer_agent_id) for item in snapshot.access_edges},
            {("alice", "bob"), ("bob", "alice")},
        )
        self.assertEqual(
            {(item.source_agent_id, item.observer_agent_id) for item in snapshot.relationship_edges},
            {("alice", "bob"), ("bob", "alice")},
        )
        self.assertEqual(
            {(item.source_agent_id, item.observer_agent_id) for item in snapshot.transmissions},
            {("alice", "bob")},
        )

    def test_round_zero_metrics_have_no_latest_round_counts(self):
        network_model = runtime_model()
        story = initialize_situated_story(
            network_model.percept_memory_model.cognitive_model.world_model,
            initial_state(network_model.percept_memory_model.perception_model, door_open=True),
            perception_model=network_model.percept_memory_model.perception_model,
        )
        cognitive_state = initialize_situated_percept_memory_cognition(
            network_model.percept_memory_model, story
        )
        social_state = initialize_situated_social_memory(
            network_model.social_memory_model, cognitive_state
        )

        snapshot = project_situated_network_snapshot(
            network_model, story, cognitive_state, social_state
        )
        metrics = measure_situated_network_emergence(network_model, snapshot, social_state)

        self.assertEqual(snapshot.transmissions, ())
        self.assertEqual(
            (
                metrics.latest_tell_event_count,
                metrics.transmission_count,
                metrics.reached_observer_count,
                metrics.exact_transmission_count,
                metrics.detected_transmission_count,
                metrics.identified_transmission_count,
            ),
            (0, 0, 0, 0, 0, 0),
        )

    def test_co_located_agents_have_direct_access_edges_and_metric_pairs(self):
        network_model = runtime_model()
        world_state = initial_state(
            network_model.percept_memory_model.perception_model, door_open=True
        )
        co_located = replace(
            world_state,
            agents=tuple(
                replace(item, place_id="records") if item.agent_id == "bob" else item
                for item in world_state.agents
            ),
        )
        story = initialize_situated_story(
            network_model.percept_memory_model.cognitive_model.world_model,
            co_located,
            perception_model=network_model.percept_memory_model.perception_model,
        )
        cognitive_state = initialize_situated_percept_memory_cognition(
            network_model.percept_memory_model, story
        )
        social_state = initialize_situated_social_memory(
            network_model.social_memory_model, cognitive_state
        )

        snapshot = project_situated_network_snapshot(
            network_model, story, cognitive_state, social_state
        )
        metrics = measure_situated_network_emergence(
            network_model, snapshot, social_state
        )

        self.assertTrue(all(item.direct_interaction for item in snapshot.access_edges))
        self.assertEqual(metrics.direct_interaction_pair_count, 2)

    def test_all_inactive_relationships_keep_all_edge_mean_trust(self):
        runtime_model, story, cognitive_state, social_state = tell_case(door_open=True)
        runtime_model = replace(runtime_model, relationship_trust_threshold=0.9)

        snapshot = project_situated_network_snapshot(
            runtime_model, story, cognitive_state, social_state
        )
        metrics = measure_situated_network_emergence(runtime_model, snapshot, social_state)

        self.assertEqual(metrics.active_relationship_edge_count, 0)
        self.assertEqual(metrics.mean_relationship_trust, 0.5)

    def test_projection_rejects_cognitive_state_not_bound_to_story(self):
        runtime_model, story, cognitive_state, social_state = tell_case(door_open=True)

        with self.assertRaisesRegex(ValueError, "exact current story"):
            project_situated_network_snapshot(
                runtime_model,
                story,
                replace(cognitive_state, story_hash="sha256:" + "0" * 64),
                social_state,
            )

    def test_metrics_reject_social_state_from_another_model_hash(self):
        runtime_model, story, cognitive_state, social_state = tell_case(door_open=True)
        snapshot = project_situated_network_snapshot(
            runtime_model, story, cognitive_state, social_state
        )
        other_social_state = replace(social_state, model_hash="sha256:" + "f" * 64)
        forged_snapshot = replace(
            snapshot, social_state_hash=other_social_state.content_hash
        )

        with self.assertRaisesRegex(ValueError, "exact social model"):
            measure_situated_network_emergence(
                runtime_model, forged_snapshot, other_social_state
            )


if __name__ == "__main__":
    unittest.main()
