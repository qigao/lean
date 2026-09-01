from dataclasses import replace
import json
import sqlite3
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
from narrative_dynamics.abm import situated_percept_memory
from narrative_dynamics.abm.situated_percept_memory import (
    SituatedPerceptMemoryConflictError,
    ingest_situated_percept_story,
    list_situated_percept_memories,
    set_situated_percept_memory_active,
)
from narrative_dynamics.abm.situated_perception import project_situated_percepts
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
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRoundResult,
    SituatedNetworkTrajectory,
)
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


def logical_percept_memory_rows(database):
    connection = sqlite3.connect(database)
    try:
        return tuple(
            connection.execute(
                "SELECT * FROM percept_memory_records "
                "ORDER BY agent_id, memory_id"
            ).fetchall()
        )
    finally:
        connection.close()


class SituatedNetworkRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.database = f"{self.temporary.name}/network.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def test_atomic_round_keeps_story_cognition_social_snapshot_and_metrics_synchronized(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
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
        self.assertEqual(
            result.next_state.memory_store_hash,
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            ),
        )

    def test_initialization_binds_a_path_independent_logical_memory_checkpoint(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        other_database = f"{self.temporary.name}/other-layout.sqlite3"

        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        other = initialize_situated_network_runtime(
            other_database, model, story, cognitive_state, social_state
        )

        self.assertEqual(
            initial.memory_store_hash,
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            ),
        )
        self.assertEqual(initial.memory_store_hash, other.memory_store_hash)
        self.assertEqual(initial.content_hash, other.content_hash)

    def test_logical_memory_hash_ignores_rowids_and_insertion_order(self):
        model, story, _, _ = tell_case(door_open=True)
        other_database = f"{self.temporary.name}/reverse.sqlite3"

        for agent_id in ("alice", "bob"):
            ingest_situated_percept_story(
                self.database,
                model.percept_memory_model.perception_model,
                story,
                agent_id,
                model.percept_memory_model.memory_policy,
            )
        for agent_id in ("bob", "alice"):
            ingest_situated_percept_story(
                other_database,
                model.percept_memory_model.perception_model,
                story,
                agent_id,
                model.percept_memory_model.memory_policy,
            )

        self.assertEqual(
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            ),
            situated_percept_memory.hash_situated_percept_memory_store(
                other_database
            ),
        )

    def test_logical_memory_hash_rejects_non_binary_active_values(self):
        model, story, _, _ = tell_case(door_open=True)
        ingest_situated_percept_story(
            self.database,
            model.percept_memory_model.perception_model,
            story,
            "bob",
            model.percept_memory_model.memory_policy,
        )
        legal_hash = (
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            )
        )
        self.assertEqual(
            len(list_situated_percept_memories(self.database, "bob")),
            2,
        )

        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE percept_memory_records SET active = 2"
            )
            connection.commit()
            visible_count = connection.execute(
                "SELECT COUNT(*) FROM percept_memory_records WHERE active = 1"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(visible_count, 0)
        with self.assertRaisesRegex(
            SituatedPerceptMemoryConflictError,
            "active",
        ):
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            )

        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE percept_memory_records SET active = 0"
            )
            connection.commit()
        finally:
            connection.close()
        inactive_hash = (
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            )
        )
        inactive_memories = list_situated_percept_memories(
            self.database,
            "bob",
            include_inactive=True,
        )
        self.assertNotEqual(inactive_hash, legal_hash)
        self.assertEqual(len(inactive_memories), 2)
        self.assertTrue(all(not item.active for item in inactive_memories))

    def test_successful_round_publishes_the_exact_next_memory_checkpoint(self):
        model, story, cognitive_state, social_state = tell_case(door_open=True)
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )

        result = simulate_situated_network_round(self.database, model, initial)

        self.assertNotEqual(
            result.next_state.memory_store_hash,
            initial.memory_store_hash,
        )
        self.assertEqual(
            result.next_state.memory_store_hash,
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            ),
        )

    def test_external_memory_activation_rejects_the_stale_runtime_state(self):
        model, story, cognitive_state, social_state = tell_case(door_open=True)
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        advanced = simulate_situated_network_round(self.database, model, initial)
        bob_memory = list_situated_percept_memories(self.database, "bob")[0]
        set_situated_percept_memory_active(
            self.database, "bob", bob_memory.memory_id, False
        )

        with self.assertRaisesRegex(ValueError, "memory store hash"):
            simulate_situated_network_round(
                self.database, model, advanced.next_state
            )

    def test_same_explicit_state_cannot_advance_a_different_store_checkpoint(self):
        model, story, cognitive_state, social_state = tell_case(door_open=True)
        other_database = f"{self.temporary.name}/other-checkpoint.sqlite3"
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        initialize_situated_network_runtime(
            other_database, model, story, cognitive_state, social_state
        )
        ingest_situated_percept_story(
            other_database,
            model.percept_memory_model.perception_model,
            story,
            "bob",
            model.percept_memory_model.memory_policy,
        )

        with self.assertRaisesRegex(ValueError, "memory store hash"):
            simulate_situated_network_round(other_database, model, initial)

    def test_mid_round_ingestion_conflict_leaves_original_store_unchanged(self):
        model, story, cognitive_state, social_state = tell_case(door_open=True)
        conflicting_policy = replace(
            model.percept_memory_model.memory_policy,
            version="conflicting-checkpoint",
        )
        ingest_situated_percept_story(
            self.database,
            model.percept_memory_model.perception_model,
            story,
            "bob",
            conflicting_policy,
        )
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        before_rows = logical_percept_memory_rows(self.database)
        before_hash = (
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            )
        )
        before_alice = list_situated_percept_memories(
            self.database, "alice", include_inactive=True
        )
        before_bob = list_situated_percept_memories(
            self.database, "bob", include_inactive=True
        )

        with self.assertRaises(SituatedPerceptMemoryConflictError):
            simulate_situated_network_round(self.database, model, initial)

        self.assertEqual(logical_percept_memory_rows(self.database), before_rows)
        self.assertEqual(
            situated_percept_memory.hash_situated_percept_memory_store(
                self.database
            ),
            before_hash,
        )
        self.assertEqual(
            list_situated_percept_memories(
                self.database, "alice", include_inactive=True
            ),
            before_alice,
        )
        self.assertEqual(
            list_situated_percept_memories(
                self.database, "bob", include_inactive=True
            ),
            before_bob,
        )

    def test_initialization_accepts_an_exact_nonzero_checkpoint_root(self):
        model, story, cognitive_state, social_state = tell_case(door_open=True)

        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )

        self.assertEqual(initial.round_index, 1)
        self.assertIsNone(initial.parent_state_hash)
        self.assertTrue(initial.checkpoint)
        self.assertIs(initial.to_dict()["checkpoint"], True)

    def test_nonzero_parentless_noncheckpoint_state_is_rejected(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        result = simulate_situated_network_round(self.database, model, initial)

        with self.assertRaisesRegex(ValueError, "exact parent"):
            replace(
                result.next_state,
                parent_state_hash=None,
                checkpoint=False,
            )

    def test_checkpoint_marker_requires_exact_cognitive_and_social_checkpoints(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        result = simulate_situated_network_round(self.database, model, initial)

        with self.assertRaisesRegex(ValueError, "cognitive and social checkpoints"):
            replace(
                result.next_state,
                parent_state_hash=None,
                checkpoint=True,
            )

    def test_round_rejects_snapshot_that_disagrees_with_authoritative_cognition(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        forged_snapshot = replace(
            initial.snapshot,
            nodes=tuple(
                replace(item, tracked_belief_probability=0.9)
                if item.agent_id == "alice"
                else item
                for item in initial.snapshot.nodes
            ),
        )
        forged_metrics = measure_situated_network_emergence(
            model, forged_snapshot, social_state
        )
        forged_state = replace(
            initial, snapshot=forged_snapshot, metrics=forged_metrics
        )

        with self.assertRaisesRegex(ValueError, "authoritative snapshot"):
            simulate_situated_network_round(self.database, model, forged_state)

    def test_round_rejects_metrics_that_disagree_with_authoritative_measurement(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )
        forged_metrics = replace(
            initial.metrics,
            active_relationship_edge_count=0,
        )
        forged_state = replace(initial, metrics=forged_metrics)

        with self.assertRaisesRegex(ValueError, "authoritative metrics"):
            simulate_situated_network_round(self.database, model, forged_state)

    def test_runtime_returns_the_exact_parent_linked_state_chain(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
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
            self.database, model, story, cognitive_state, social_state
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

    def test_round_and_trajectory_reject_a_two_branch_underlying_splice(self):
        model, _, _, _ = initial_runtime_case()

        def branch(kind, database):
            story = initialize_situated_story(
                model.percept_memory_model.cognitive_model.world_model,
                initial_state(
                    model.percept_memory_model.perception_model,
                    door_open=True,
                ),
                perception_model=model.percept_memory_model.perception_model,
            )
            story = advance_situated_story(
                model.percept_memory_model.cognitive_model.world_model,
                story,
                (
                    SituatedActionIntent(
                        f"alice-{kind.value}",
                        "alice",
                        kind,
                        message="branch testimony"
                        if kind is SituatedActionKind.TELL
                        else None,
                    ),
                ),
            )
            cognition = initialize_situated_percept_memory_cognition(
                model.percept_memory_model, story
            )
            social = initialize_situated_social_memory(
                model.social_memory_model, cognition
            )
            checkpoint = initialize_situated_network_runtime(
                database, model, story, cognition, social
            )
            return checkpoint, simulate_situated_network_round(
                database, model, checkpoint
            )

        prior_a, _ = branch(
            SituatedActionKind.WAIT,
            f"{self.temporary.name}/branch-a.sqlite3",
        )
        _, round_b = branch(
            SituatedActionKind.TELL,
            f"{self.temporary.name}/branch-b.sqlite3",
        )
        spliced_next = replace(
            round_b.next_state,
            parent_state_hash=prior_a.content_hash,
        )

        with self.assertRaisesRegex(ValueError, "underlying branch"):
            SituatedNetworkRoundResult(
                model.model_id,
                model.content_hash,
                prior_a,
                spliced_next,
            )

        forged_round = object.__new__(SituatedNetworkRoundResult)
        object.__setattr__(forged_round, "model_id", model.model_id)
        object.__setattr__(forged_round, "model_hash", model.content_hash)
        object.__setattr__(forged_round, "prior_state", prior_a)
        object.__setattr__(forged_round, "next_state", spliced_next)
        with self.assertRaisesRegex(ValueError, "underlying branch"):
            SituatedNetworkTrajectory(
                model.model_id,
                model.content_hash,
                prior_a,
                (forged_round,),
                spliced_next,
            )

    def test_runtime_requires_a_positive_integer_round_count(self):
        model, story, cognitive_state, social_state = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database, model, story, cognitive_state, social_state
        )

        for round_count in (0, -1, 1.0, True):
            with self.subTest(round_count=round_count):
                with self.assertRaisesRegex(ValueError, "positive round count"):
                    simulate_situated_network_runtime(
                        self.database, model, initial, round_count=round_count
                    )


class SituatedNetworkProjectionTests(unittest.TestCase):
    def test_closed_door_anonymous_detected_percept_is_not_a_v19_transmission(self):
        runtime_model, closed_story, cognitive_state, social_state = tell_case(door_open=False)
        projection = project_situated_percepts(
            runtime_model.percept_memory_model.perception_model,
            closed_story.rounds[-1],
        )
        bob_percept = next(
            item
            for item in projection.percepts
            if item.agent_id == "bob"
        )

        snapshot = project_situated_network_snapshot(
            runtime_model, closed_story, cognitive_state, social_state
        )
        metrics = measure_situated_network_emergence(
            runtime_model, snapshot, social_state
        )

        self.assertIs(bob_percept.fidelity, SituatedPerceptFidelity.DETECTED)
        self.assertIsNone(bob_percept.actor_agent_id)
        self.assertIsNone(bob_percept.kind)
        self.assertEqual(snapshot.transmissions, ())
        self.assertEqual(metrics.latest_tell_event_count, 1)
        self.assertEqual(metrics.transmission_count, 0)
        self.assertEqual(metrics.detected_transmission_count, 0)
        serialized = json.dumps(snapshot.to_dict(), sort_keys=True)
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn('"message"', serialized)

    def test_open_door_transmission_fields_equal_the_disclosing_v15_tell_percept(self):
        runtime_model, open_story, cognitive_state, social_state = tell_case(door_open=True)
        projection = project_situated_percepts(
            runtime_model.percept_memory_model.perception_model,
            open_story.rounds[-1],
        )
        bob_percept = next(
            item
            for item in projection.percepts
            if item.agent_id == "bob"
        )

        snapshot = project_situated_network_snapshot(
            runtime_model, open_story, cognitive_state, social_state
        )
        metrics = measure_situated_network_emergence(runtime_model, snapshot, social_state)
        transmission = snapshot.transmissions[0]

        self.assertIs(bob_percept.kind, SituatedActionKind.TELL)
        self.assertIsNotNone(bob_percept.actor_agent_id)
        self.assertEqual(
            (
                transmission.event_id,
                transmission.source_agent_id,
                transmission.observer_agent_id,
                transmission.fidelity,
                transmission.channels,
            ),
            (
                bob_percept.source_event_id,
                bob_percept.actor_agent_id,
                bob_percept.agent_id,
                bob_percept.fidelity,
                bob_percept.channels,
            ),
        )
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
