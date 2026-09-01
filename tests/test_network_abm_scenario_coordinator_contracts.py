from __future__ import annotations

import dataclasses
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.scenario_coordinator_contracts import (
    SCENARIO_AGENT_STATE_VIEW_SCHEMA,
    SCENARIO_CHECKPOINT_SCHEMA,
    SCENARIO_COMMAND_REQUEST_SCHEMA,
    SCENARIO_COMMAND_RESULT_SCHEMA,
    SCENARIO_FORK_REQUEST_SCHEMA,
    SCENARIO_FORK_RESULT_SCHEMA,
    SCENARIO_PUBLIC_STATE_VIEW_SCHEMA,
    SCENARIO_RUN_VIEW_SCHEMA,
    ScenarioAgentStateView,
    ScenarioCheckpoint,
    ScenarioCommandCapability,
    ScenarioCommandKind,
    ScenarioCommandReason,
    ScenarioCommandRequest,
    ScenarioCommandResult,
    ScenarioForkRequest,
    ScenarioForkResult,
    ScenarioPublicStateView,
    ScenarioRunStatus,
    ScenarioRunView,
)
from narrative_dynamics.abm.scenario_state_store import (
    InMemoryScenarioStateStore,
    ScenarioStateStore,
)
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationOutputAudience,
)
from narrative_dynamics.abm.situated_network import (
    initialize_situated_network_runtime,
)
from tests.test_network_abm_situated_network import initial_runtime_case


SCENARIO_HASH = "sha256:" + "1" * 64
STATE_HASH = "sha256:" + "2" * 64
CAPABILITY_HASH = "sha256:" + "3" * 64
REQUEST_HASH = "sha256:" + "4" * 64
CHECKPOINT_HASH = "sha256:" + "5" * 64
OUTPUT_HASH = "sha256:" + "6" * 64


class ScenarioCoordinatorContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        model, story, cognitive_state, social_state = initial_runtime_case()
        cls.initial = initialize_situated_network_runtime(
            f"{cls.temporary.name}/state.sqlite3",
            model,
            story,
            cognitive_state,
            social_state,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_enum_and_schema_literals(self) -> None:
        # Mutation caught: transport lifecycle status vocabulary drifts.
        self.assertEqual(
            tuple(item.value for item in ScenarioRunStatus),
            ("created", "running", "paused", "stopped", "completed"),
        )
        # Mutation caught: the closed command vocabulary gains or loses a command.
        self.assertEqual(
            tuple(item.value for item in ScenarioCommandKind),
            ("start", "pause", "resume", "step", "stop", "checkpoint"),
        )
        self.assertEqual(
            tuple(item.value for item in ScenarioCommandReason),
            (
                "accepted",
                "unauthorized",
                "run_mismatch",
                "scenario_mismatch",
                "epoch_mismatch",
                "stale_state",
                "invalid_status",
                "maximum_rounds",
            ),
        )
        self.assertEqual(
            (
                SCENARIO_COMMAND_REQUEST_SCHEMA,
                SCENARIO_COMMAND_RESULT_SCHEMA,
                SCENARIO_RUN_VIEW_SCHEMA,
                SCENARIO_PUBLIC_STATE_VIEW_SCHEMA,
                SCENARIO_AGENT_STATE_VIEW_SCHEMA,
                SCENARIO_CHECKPOINT_SCHEMA,
                SCENARIO_FORK_REQUEST_SCHEMA,
                SCENARIO_FORK_RESULT_SCHEMA,
            ),
            (
                "narrative-dynamics.scenario-command-request/v1",
                "narrative-dynamics.scenario-command-result/v1",
                "narrative-dynamics.scenario-run-view/v1",
                "narrative-dynamics.scenario-public-state-view/v1",
                "narrative-dynamics.scenario-agent-state-view/v1",
                "narrative-dynamics.scenario-checkpoint/v1",
                "narrative-dynamics.scenario-fork-request/v1",
                "narrative-dynamics.scenario-fork-result/v1",
            ),
        )

    def test_command_request_is_frozen_canonical_and_content_addressed(self) -> None:
        request = ScenarioCommandRequest(
            "command-1",
            "idem-1",
            "run-1",
            SCENARIO_HASH,
            1,
            STATE_HASH,
            "operator",
            ScenarioCommandKind.START,
        )

        # Mutation caught: enum objects leak into the transport representation.
        self.assertEqual(request.to_dict()["kind"], "start")
        # Mutation caught: frozen value equality depends on object identity.
        self.assertEqual(request, dataclasses.replace(request))
        # Mutation caught: the request loses its stable canonical content identity.
        self.assertRegex(request.content_hash, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(request.to_dict()["schema"], SCENARIO_COMMAND_REQUEST_SCHEMA)

    def test_command_request_enforces_checkpoint_shape(self) -> None:
        values = (
            "command-1",
            "idem-1",
            "run-1",
            SCENARIO_HASH,
            1,
            STATE_HASH,
            "operator",
        )
        # Mutation caught: non-checkpoint commands can smuggle a checkpoint identity.
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            ScenarioCommandRequest(
                *values,
                ScenarioCommandKind.START,
                "checkpoint-1",
            )
        # Mutation caught: checkpoint commands can omit their requested identity.
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            ScenarioCommandRequest(*values, ScenarioCommandKind.CHECKPOINT)

    def test_command_capability_rejects_noncanonical_or_duplicate_kinds(self) -> None:
        # Mutation caught: capability identity depends on caller tuple order.
        with self.assertRaisesRegex(ValueError, "sorted"):
            ScenarioCommandCapability(
                "operator",
                "run-1",
                (ScenarioCommandKind.STEP, ScenarioCommandKind.START),
            )
        # Mutation caught: duplicate allowlist entries produce ambiguous authority hashes.
        with self.assertRaisesRegex(ValueError, "unique"):
            ScenarioCommandCapability(
                "operator",
                "run-1",
                (ScenarioCommandKind.START, ScenarioCommandKind.START),
            )
        capability = ScenarioCommandCapability(
            "operator",
            "run-1",
            (ScenarioCommandKind.START, ScenarioCommandKind.STEP),
            can_fork=True,
            can_read_all_audit=True,
        )
        self.assertTrue(capability.can_fork)
        self.assertTrue(capability.can_read_all_audit)
        self.assertRegex(capability.content_hash, r"^sha256:[0-9a-f]{64}$")

    def test_agent_output_capability_requires_an_owner(self) -> None:
        # Mutation caught: an unowned Agent capability can read arbitrary private state.
        with self.assertRaisesRegex(ValueError, "owner"):
            SimulationAudienceCapability(SimulationOutputAudience.AGENT)

    def test_contracts_reject_malformed_hashes_and_negative_rounds(self) -> None:
        # Mutation caught: malformed expected-state values enter command identities.
        with self.assertRaisesRegex(ValueError, "hash"):
            ScenarioCommandRequest(
                "command-1",
                "idem-1",
                "run-1",
                SCENARIO_HASH,
                1,
                "not-a-hash",
                "operator",
                ScenarioCommandKind.START,
            )
        # Mutation caught: public query views accept a round before genesis.
        with self.assertRaisesRegex(ValueError, "round"):
            ScenarioPublicStateView(
                "run-1",
                SCENARIO_HASH,
                -1,
                STATE_HASH,
                (),
                (),
                (),
                self.initial.metrics,
            )

    def test_result_and_run_view_are_typed_canonical_values(self) -> None:
        result = ScenarioCommandResult(
            "command-1",
            "idem-1",
            REQUEST_HASH,
            CAPABILITY_HASH,
            "run-1",
            SCENARIO_HASH,
            1,
            ScenarioCommandKind.STEP,
            True,
            ScenarioCommandReason.ACCEPTED,
            ScenarioRunStatus.RUNNING,
            ScenarioRunStatus.RUNNING,
            STATE_HASH,
            self.initial.content_hash,
            self.initial.round_index,
            OUTPUT_HASH,
        )
        view = ScenarioRunView(
            "run-1",
            "stream-1",
            SCENARIO_HASH,
            1,
            ScenarioRunStatus.RUNNING,
            self.initial.round_index,
            self.initial.content_hash,
            2,
            (OUTPUT_HASH,),
            (CHECKPOINT_HASH,),
        )

        # Mutation caught: result and run status enums leak into wire dictionaries.
        self.assertEqual(result.to_dict()["reason"], "accepted")
        self.assertEqual(view.to_dict()["status"], "running")
        self.assertEqual(result.to_dict()["schema"], SCENARIO_COMMAND_RESULT_SCHEMA)
        self.assertEqual(view.to_dict()["schema"], SCENARIO_RUN_VIEW_SCHEMA)
        self.assertRegex(result.content_hash, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(view.content_hash, r"^sha256:[0-9a-f]{64}$")

    def test_public_and_agent_views_preserve_exact_typed_state(self) -> None:
        mind = next(
            item for item in self.initial.cognitive_state.minds
            if item.agent_id == "alice"
        )
        relationship = next(
            item for item in self.initial.social_state.relationships
            if item.observer_agent_id == "alice"
        )
        public = ScenarioPublicStateView(
            "run-1",
            SCENARIO_HASH,
            self.initial.round_index,
            self.initial.content_hash,
            (("bob", "open"), ("alice", "records")),
            (("records-open", True),),
            (("memo", "records", None),),
            self.initial.metrics,
        )
        agent = ScenarioAgentStateView(
            "run-1",
            SCENARIO_HASH,
            self.initial.round_index,
            self.initial.content_hash,
            "alice",
            "records",
            mind,
            (relationship,),
            (),
        )

        # Mutation caught: public physical identities retain nondeterministic input order.
        self.assertEqual(public.agent_places, (("alice", "records"), ("bob", "open")))
        # Mutation caught: private views copy or erase the exact typed mind value.
        self.assertIs(agent.mind, mind)
        self.assertEqual(agent.to_dict()["mind"], mind.to_dict())
        wrong_observer = dataclasses.replace(relationship, observer_agent_id="charlie")
        # Mutation caught: another observer's social graph leaks into an Agent view.
        with self.assertRaisesRegex(ValueError, "observer"):
            dataclasses.replace(agent, relationships=(wrong_observer,))

    def test_checkpoint_serializes_hashes_without_state_or_paths(self) -> None:
        checkpoint = ScenarioCheckpoint(
            "checkpoint-1",
            "run-1",
            SCENARIO_HASH,
            1,
            self.initial,
            7,
        )

        payload = checkpoint.to_dict()
        # Mutation caught: checkpoint identities embed private runtime trees or local paths.
        self.assertNotIn("state", payload)
        self.assertNotIn("path", payload)
        self.assertEqual(payload["state_hash"], self.initial.content_hash)
        self.assertEqual(payload["memory_store_hash"], self.initial.memory_store_hash)
        self.assertEqual(payload["schema"], SCENARIO_CHECKPOINT_SCHEMA)
        self.assertIs(checkpoint.state, self.initial)

        object.__setattr__(checkpoint, "state_hash", STATE_HASH)
        # Mutation caught: a checkpoint can publish a state hash unlike its embedded state.
        with self.assertRaisesRegex(ValueError, "embedded state"):
            checkpoint.to_dict()

        checkpoint = dataclasses.replace(checkpoint)
        object.__setattr__(checkpoint, "memory_store_hash", STATE_HASH)
        # Mutation caught: a checkpoint can publish a memory hash unlike its embedded state.
        with self.assertRaisesRegex(ValueError, "memory"):
            checkpoint.to_dict()

    def test_fork_contracts_bind_exact_checkpoint_and_child_state_hashes(self) -> None:
        request = ScenarioForkRequest(
            "fork-1",
            "idem-fork-1",
            "run-1",
            SCENARIO_HASH,
            1,
            CHECKPOINT_HASH,
            "run-2",
            "stream-2",
        )
        result = ScenarioForkResult(
            "fork-1",
            "idem-fork-1",
            request.content_hash,
            CAPABILITY_HASH,
            "run-1",
            "run-2",
            "stream-2",
            SCENARIO_HASH,
            2,
            CHECKPOINT_HASH,
            self.initial.content_hash,
        )

        # Mutation caught: fork artifacts omit the exact checkpoint/child identities.
        self.assertEqual(result.to_dict()["checkpoint_hash"], CHECKPOINT_HASH)
        self.assertEqual(result.to_dict()["child_state_hash"], self.initial.content_hash)
        self.assertEqual(request.to_dict()["schema"], SCENARIO_FORK_REQUEST_SCHEMA)
        self.assertEqual(result.to_dict()["schema"], SCENARIO_FORK_RESULT_SCHEMA)
        with self.assertRaisesRegex(ValueError, "hash"):
            dataclasses.replace(result, child_state_hash="not-a-hash")


class InMemoryScenarioStateStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        model, story, cognitive_state, social_state = initial_runtime_case()
        cls.initial = initialize_situated_network_runtime(
            f"{cls.temporary.name}/store.sqlite3",
            model,
            story,
            cognitive_state,
            social_state,
        )
        cls.advanced = dataclasses.replace(
            cls.initial,
            memory_store_hash="sha256:" + "7" * 64,
        )
        cls.later = dataclasses.replace(
            cls.initial,
            memory_store_hash="sha256:" + "8" * 64,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_protocol_is_runtime_checkable(self) -> None:
        # Mutation caught: the storage seam cannot be substituted structurally.
        self.assertIsInstance(InMemoryScenarioStateStore(), ScenarioStateStore)

    def test_compare_and_swap_preserves_exact_object_identity(self) -> None:
        store = InMemoryScenarioStateStore()
        store.initialize("run-1", self.initial)
        # Mutation caught: initialization copies or reconstructs the exact typed state.
        self.assertIs(store.load("run-1"), self.initial)
        store.compare_and_swap("run-1", self.initial.content_hash, self.advanced)
        # Mutation caught: successful CAS stores a copy instead of the supplied object.
        self.assertIs(store.load("run-1"), self.advanced)
        # Mutation caught: stale writers overwrite the current exact state.
        with self.assertRaisesRegex(ValueError, "stale"):
            store.compare_and_swap("run-1", self.initial.content_hash, self.later)
        self.assertIs(store.load("run-1"), self.advanced)

    def test_duplicate_initialize_and_unknown_load_leave_existing_state_unchanged(self) -> None:
        store = InMemoryScenarioStateStore()
        store.initialize("run-1", self.initial)
        # Mutation caught: duplicate initialization silently replaces an owned run.
        with self.assertRaisesRegex(ValueError, "already"):
            store.initialize("run-1", self.advanced)
        self.assertIs(store.load("run-1"), self.initial)
        # Mutation caught: an unknown run fabricates a default state or mutates the store.
        with self.assertRaisesRegex(KeyError, "unknown"):
            store.load("missing")
        self.assertIs(store.load("run-1"), self.initial)


if __name__ == "__main__":
    unittest.main()
