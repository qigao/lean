from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
import unittest
from unittest.mock import patch

from narrative_dynamics.abm import scenario_coordinator as coordinator_module
from narrative_dynamics.abm.scenario_checkpoint_store import (
    LocalScenarioCheckpointStore,
    _PhysicalFileOwnershipToken,
    _ScenarioCheckpointOwnedRestoreError,
)
from narrative_dynamics.abm.scenario_compiler import (
    compile_situated_scenario_package,
    initialize_compiled_scenario,
)
from narrative_dynamics.abm.scenario_coordinator import ScenarioCoordinator
from narrative_dynamics.abm.scenario_coordinator_contracts import (
    ScenarioCommandCapability,
    ScenarioCommandKind,
    ScenarioCommandReason,
    ScenarioCommandRequest,
    ScenarioForkRequest,
    ScenarioRunStatus,
)
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.scenario_state_store import InMemoryScenarioStateStore
from narrative_dynamics.abm.simulation_output import project_simulation_output
from narrative_dynamics.abm.simulation_output_bus import (
    SimulationDeliveryFailure,
    SimulationDeliveryReport,
    SimulationOutputBus,
)
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationCommandResultPayload,
    SimulationOutputAudience,
    SimulationOutputKind,
)
from narrative_dynamics.abm.situated_percept_memory import (
    hash_situated_percept_memory_store,
)
from narrative_dynamics.abm.situated_network import simulate_situated_network_round
from tests.scenario_package_fixtures import (
    mutate_json,
    refresh_manifest_hash,
    write_law_firm_package,
)


class ScenarioCoordinatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package_temporary = TemporaryDirectory()
        package_root = write_law_firm_package(
            Path(cls.package_temporary.name) / "law-firm"
        )
        mutate_json(
            package_root / "run.json",
            "/allowed_output_kinds",
            [kind.value for kind in SimulationOutputKind],
        )
        refresh_manifest_hash(package_root, "run", "run")
        cls.scenario = compile_situated_scenario_package(
            load_situated_scenario_package(package_root)
        )
        cls.all_command_kinds = tuple(
            sorted(ScenarioCommandKind, key=lambda kind: kind.value)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.package_temporary.cleanup()

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "memory.sqlite3"
        self.command_number = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def coordinator(
        self,
        *,
        scenario=None,
        state_store=None,
        publisher=None,
        checkpoint_store=None,
        database=None,
        run_id="law-firm-run",
        stream_id="law-firm-stream",
    ) -> ScenarioCoordinator:
        return ScenarioCoordinator.create(
            self.database if database is None else database,
            self.scenario if scenario is None else scenario,
            run_id=run_id,
            stream_id=stream_id,
            state_store=state_store,
            publisher=publisher,
            checkpoint_store=checkpoint_store,
        )

    def capability(
        self,
        *kinds: ScenarioCommandKind,
        authority_id: str = "operator",
        run_id: str = "law-firm-run",
        can_read_all_audit: bool = False,
        can_fork: bool = False,
    ) -> ScenarioCommandCapability:
        allowed = self.all_command_kinds if not kinds else tuple(
            sorted(kinds, key=lambda kind: kind.value)
        )
        return ScenarioCommandCapability(
            authority_id,
            run_id,
            allowed,
            can_fork=can_fork,
            can_read_all_audit=can_read_all_audit,
        )

    def request(
        self,
        coordinator: ScenarioCoordinator,
        kind: ScenarioCommandKind,
        *,
        command_id: str | None = None,
        idempotency_key: str | None = None,
        run_id: str = "law-firm-run",
        scenario_hash: str | None = None,
        coordinator_epoch: int = 1,
        expected_state_hash: str | None = None,
        authority_id: str = "operator",
        requested_checkpoint_id: str | None = None,
    ) -> ScenarioCommandRequest:
        self.command_number += 1
        suffix = str(self.command_number)
        return ScenarioCommandRequest(
            command_id or f"command-{suffix}",
            idempotency_key or f"idem-{suffix}",
            run_id,
            coordinator.run_view().scenario_hash
            if scenario_hash is None
            else scenario_hash,
            coordinator_epoch,
            coordinator.state.content_hash
            if expected_state_hash is None
            else expected_state_hash,
            authority_id,
            kind,
            requested_checkpoint_id
            if requested_checkpoint_id is not None
            else (f"checkpoint-{suffix}" if kind is ScenarioCommandKind.CHECKPOINT else None),
        )

    def submit(
        self,
        coordinator: ScenarioCoordinator,
        kind: ScenarioCommandKind,
        capability: ScenarioCommandCapability | None = None,
    ):
        return coordinator.submit_command(
            self.request(coordinator, kind),
            self.capability() if capability is None else capability,
        )

    def start(self, coordinator: ScenarioCoordinator):
        return self.submit(coordinator, ScenarioCommandKind.START)

    def failed_projector_attempt(self, name: str):
        database = self.root / f"{name}.sqlite3"
        coordinator = ScenarioCoordinator.create(
            database,
            self.scenario,
            run_id="law-firm-run",
            stream_id=f"{name}-stream",
        )
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        with patch(
            "narrative_dynamics.abm.scenario_coordinator.project_simulation_output",
            side_effect=RuntimeError("projector seam failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "projector seam failed"):
                coordinator.submit_command(request, self.capability())
        return coordinator, request

    def checkpointed_coordinator(self, *, interval: int = 2):
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                checkpoint_interval=interval,
            ),
        )
        store = LocalScenarioCheckpointStore(self.root / "checkpoints")
        coordinator = self.coordinator(
            scenario=scenario,
            checkpoint_store=store,
        )
        self.start(coordinator)
        for _ in range(interval):
            self.submit(coordinator, ScenarioCommandKind.STEP)
        checkpoint_hash = coordinator.run_view().checkpoint_hashes[-1]
        return coordinator, store, store.load(checkpoint_hash)

    def fork_request(
        self,
        coordinator: ScenarioCoordinator,
        checkpoint_hash: str,
        *,
        fork_id: str = "fork-1",
        idempotency_key: str = "fork-idem-1",
        source_run_id: str = "law-firm-run",
        scenario_hash: str | None = None,
        source_epoch: int = 1,
        child_run_id: str = "law-firm-child",
        child_stream_id: str = "law-firm-child-stream",
    ) -> ScenarioForkRequest:
        return ScenarioForkRequest(
            fork_id,
            idempotency_key,
            source_run_id,
            coordinator.run_view().scenario_hash
            if scenario_hash is None
            else scenario_hash,
            source_epoch,
            checkpoint_hash,
            child_run_id,
            child_stream_id,
        )

    def test_create_initializes_exact_created_run_and_delegates_scoped_queries(self) -> None:
        coordinator = self.coordinator()
        view = coordinator.run_view()
        initial_state = coordinator.state

        # Mutation caught: creation starts or advances work instead of owning genesis.
        self.assertIs(view.status, ScenarioRunStatus.CREATED)
        self.assertEqual(view.round_index, 0)
        self.assertEqual(view.next_sequence, 1)
        self.assertEqual(view.output_batch_hashes, ())
        self.assertEqual(view.checkpoint_hashes, ())
        self.assertIs(coordinator.state, initial_state)
        self.assertEqual(
            coordinator.public_state_view().state_hash,
            coordinator.state.content_hash,
        )
        alice = coordinator.agent_state_view(
            "alice",
            SimulationAudienceCapability(SimulationOutputAudience.AGENT, "alice"),
        )
        self.assertEqual(alice.agent_id, "alice")
        with self.assertRaisesRegex(PermissionError, "not authorized"):
            coordinator.agent_state_view(
                "alice",
                SimulationAudienceCapability(
                    SimulationOutputAudience.AGENT,
                    "bob",
                ),
            )
        self.assertIs(
            coordinator.network_state(
                SimulationAudienceCapability(SimulationOutputAudience.INTERNAL)
            ),
            coordinator.state.snapshot,
        )

    def test_network_state_and_run_view_are_projected_under_one_state_lock(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        capability = self.capability()
        audience = SimulationAudienceCapability(SimulationOutputAudience.ANALYST)
        atomic_network_state = coordinator.network_state_with_run_view

        network_projection_entered = Event()
        release_network_projection = Event()
        run_projection_entered = Event()
        release_run_projection = Event()
        transition_entered = Event()
        read_result = []
        read_errors = []
        command_errors = []
        original_network_projector = coordinator_module.project_scenario_network_state
        original_run_projector = coordinator_module.project_scenario_run_view
        original_submit = coordinator._submit_command

        def project_network(state, exact_audience):
            snapshot = original_network_projector(state, exact_audience)
            network_projection_entered.set()
            if not release_network_projection.wait(2):
                raise RuntimeError("network projection barrier timed out")
            return snapshot

        def project_run_view(**values):
            run = original_run_projector(**values)
            run_projection_entered.set()
            if not release_run_projection.wait(2):
                raise RuntimeError("run projection barrier timed out")
            return run

        def submit(exact_request, exact_capability):
            transition_entered.set()
            return original_submit(exact_request, exact_capability)

        def read_pair() -> None:
            try:
                read_result.append(atomic_network_state(audience))
            except Exception as error:  # pragma: no cover - asserted below
                read_errors.append(error)

        def transition() -> None:
            try:
                coordinator.submit_command(request, capability)
            except Exception as error:  # pragma: no cover - asserted below
                command_errors.append(error)

        with patch.object(
            coordinator_module,
            "project_scenario_network_state",
            side_effect=project_network,
        ), patch.object(
            coordinator_module,
            "project_scenario_run_view",
            side_effect=project_run_view,
        ), patch.object(coordinator, "_submit_command", side_effect=submit):
            reader = Thread(target=read_pair)
            reader.start()
            self.assertTrue(network_projection_entered.wait(2))
            commander = Thread(target=transition)
            commander.start()
            release_network_projection.set()
            self.assertTrue(run_projection_entered.wait(2))
            self.assertFalse(transition_entered.is_set())
            release_run_projection.set()
            reader.join(2)
            commander.join(2)

        self.assertFalse(reader.is_alive())
        self.assertFalse(commander.is_alive())
        self.assertEqual(read_errors, [])
        self.assertEqual(command_errors, [])
        snapshot, run = read_result[0]
        self.assertEqual(snapshot.round_index, 0)
        self.assertEqual(run.round_index, 0)
        self.assertEqual(run.state_hash, request.expected_state_hash)
        self.assertEqual(coordinator.run_view().round_index, 1)
        self.assertNotEqual(coordinator.run_view().state_hash, run.state_hash)

    def test_creation_rejects_wrong_scenario_and_duplicate_state_store_run(self) -> None:
        with self.assertRaisesRegex(TypeError, "CompiledSituatedScenario"):
            ScenarioCoordinator.create(
                self.database,
                object(),
                run_id="law-firm-run",
                stream_id="law-firm-stream",
            )

        state_store = InMemoryScenarioStateStore()
        first = self.coordinator(state_store=state_store)
        second_database = self.root / "second.sqlite3"
        with self.assertRaisesRegex(ValueError, "already initialized"):
            ScenarioCoordinator.create(
                second_database,
                self.scenario,
                run_id="law-firm-run",
                stream_id="second-stream",
                state_store=state_store,
            )
        self.assertEqual(first.state.round_index, 0)

    def test_control_lifecycle_is_synchronous_and_exact_duplicates_return_same_object(self) -> None:
        coordinator = self.coordinator()
        operator = self.capability()
        start_request = self.request(coordinator, ScenarioCommandKind.START)

        started = coordinator.submit_command(start_request, operator)

        # Mutation caught: start launches an implicit round or duplicate execution.
        self.assertTrue(started.accepted)
        self.assertIs(started.reason, ScenarioCommandReason.ACCEPTED)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.RUNNING)
        self.assertEqual(coordinator.state.round_index, 0)
        self.assertIs(coordinator.submit_command(start_request, operator), started)
        self.assertIs(coordinator.command_result(start_request.command_id, operator), started)

        paused = self.submit(coordinator, ScenarioCommandKind.PAUSE)
        self.assertTrue(paused.accepted)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.PAUSED)
        resumed = self.submit(coordinator, ScenarioCommandKind.RESUME)
        self.assertTrue(resumed.accepted)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.RUNNING)
        stopped = self.submit(coordinator, ScenarioCommandKind.STOP)
        self.assertTrue(stopped.accepted)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.STOPPED)
        self.assertEqual(coordinator.state.round_index, 0)
        self.assertEqual(coordinator.run_view().output_batch_hashes, ())

    def test_stop_is_legal_directly_from_created_and_paused(self) -> None:
        created = self.coordinator()
        stopped_created = self.submit(created, ScenarioCommandKind.STOP)
        self.assertTrue(stopped_created.accepted)
        self.assertIs(created.run_view().status, ScenarioRunStatus.STOPPED)

        paused_database = self.root / "paused-stop.sqlite3"
        paused = ScenarioCoordinator.create(
            paused_database,
            self.scenario,
            run_id="law-firm-run",
            stream_id="paused-stop-stream",
        )
        self.start(paused)
        self.submit(paused, ScenarioCommandKind.PAUSE)
        stopped_paused = self.submit(paused, ScenarioCommandKind.STOP)
        self.assertTrue(stopped_paused.accepted)
        self.assertIs(paused.run_view().status, ScenarioRunStatus.STOPPED)

    def test_complete_illegal_status_matrix_returns_typed_receipts_without_run_changes(self) -> None:
        cases = (
            ((), ScenarioCommandKind.PAUSE),
            ((), ScenarioCommandKind.RESUME),
            ((), ScenarioCommandKind.STEP),
            ((ScenarioCommandKind.START,), ScenarioCommandKind.START),
            ((ScenarioCommandKind.START,), ScenarioCommandKind.RESUME),
            (
                (ScenarioCommandKind.START, ScenarioCommandKind.PAUSE),
                ScenarioCommandKind.START,
            ),
            (
                (ScenarioCommandKind.START, ScenarioCommandKind.PAUSE),
                ScenarioCommandKind.PAUSE,
            ),
        )

        for prelude, illegal in cases:
            with self.subTest(prelude=prelude, illegal=illegal):
                database = self.root / f"illegal-{len(list(self.root.glob('illegal-*')))}.sqlite3"
                coordinator = ScenarioCoordinator.create(
                    database,
                    self.scenario,
                    run_id="law-firm-run",
                    stream_id=f"stream-{self.command_number}",
                )
                for kind in prelude:
                    self.assertTrue(self.submit(coordinator, kind).accepted)
                before = coordinator.run_view()
                request = self.request(coordinator, illegal)
                result = coordinator.submit_command(request, self.capability())
                self.assertFalse(result.accepted)
                self.assertIs(result.reason, ScenarioCommandReason.INVALID_STATUS)
                self.assertEqual(coordinator.run_view(), before)
                self.assertIs(
                    coordinator.command_result(request.command_id, self.capability()),
                    result,
                )

    def test_stopped_is_terminal_and_rejected_receipt_is_exactly_idempotent(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        self.submit(coordinator, ScenarioCommandKind.STOP)
        before = coordinator.run_view()

        for kind in (
            ScenarioCommandKind.START,
            ScenarioCommandKind.PAUSE,
            ScenarioCommandKind.RESUME,
            ScenarioCommandKind.STEP,
            ScenarioCommandKind.STOP,
        ):
            with self.subTest(kind=kind):
                request = self.request(coordinator, kind)
                result = coordinator.submit_command(request, self.capability())
                self.assertFalse(result.accepted)
                self.assertIs(result.reason, ScenarioCommandReason.INVALID_STATUS)
                self.assertIs(
                    coordinator.submit_command(request, self.capability()),
                    result,
                )
                self.assertEqual(coordinator.run_view(), before)

    def test_validation_rejections_have_stable_precedence_and_no_run_mutation(self) -> None:
        coordinator = self.coordinator()
        before = coordinator.run_view()
        other_hash = "sha256:" + "f" * 64
        cases = (
            (
                self.request(coordinator, ScenarioCommandKind.START),
                self.capability(ScenarioCommandKind.STEP),
                ScenarioCommandReason.UNAUTHORIZED,
            ),
            (
                self.request(
                    coordinator,
                    ScenarioCommandKind.START,
                    authority_id="intruder",
                ),
                self.capability(),
                ScenarioCommandReason.UNAUTHORIZED,
            ),
            (
                self.request(
                    coordinator,
                    ScenarioCommandKind.START,
                    run_id="other-run",
                ),
                self.capability(),
                ScenarioCommandReason.RUN_MISMATCH,
            ),
            (
                self.request(
                    coordinator,
                    ScenarioCommandKind.START,
                    scenario_hash=other_hash,
                ),
                self.capability(),
                ScenarioCommandReason.SCENARIO_MISMATCH,
            ),
            (
                self.request(
                    coordinator,
                    ScenarioCommandKind.START,
                    coordinator_epoch=2,
                ),
                self.capability(),
                ScenarioCommandReason.EPOCH_MISMATCH,
            ),
            (
                self.request(
                    coordinator,
                    ScenarioCommandKind.START,
                    expected_state_hash=other_hash,
                ),
                self.capability(),
                ScenarioCommandReason.STALE_STATE,
            ),
        )

        for request, capability, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                result = coordinator.submit_command(request, capability)
                self.assertFalse(result.accepted)
                self.assertIs(result.reason, expected_reason)
                self.assertEqual(coordinator.run_view(), before)
                self.assertIs(
                    coordinator.command_result(request.command_id, capability),
                    result,
                )

    def test_idempotency_and_command_id_collisions_raise_without_audit_or_mutation(self) -> None:
        coordinator = self.coordinator()
        operator = self.capability()
        original = self.request(
            coordinator,
            ScenarioCommandKind.PAUSE,
            command_id="shared-command",
            idempotency_key="shared-key",
        )
        rejected = coordinator.submit_command(original, operator)
        before = coordinator.run_view()

        changed_request = replace(original, command_id="different-command")
        with self.assertRaisesRegex(ValueError, "idempotency key"):
            coordinator.submit_command(changed_request, operator)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                "different-command",
                self.capability(can_read_all_audit=True),
            )

        changed_capability = self.capability(
            ScenarioCommandKind.PAUSE,
            can_read_all_audit=True,
        )
        with self.assertRaisesRegex(ValueError, "idempotency key"):
            coordinator.submit_command(original, changed_capability)

        reused_command = replace(original, idempotency_key="different-key")
        with self.assertRaisesRegex(ValueError, "command id"):
            coordinator.submit_command(reused_command, operator)

        self.assertEqual(coordinator.run_view(), before)
        self.assertIs(coordinator.command_result("shared-command", operator), rejected)

    def test_failed_attempt_rejects_different_request_under_same_idempotency_key(self) -> None:
        coordinator, failed = self.failed_projector_attempt("failed-idempotency")
        changed = replace(failed, command_id="changed-command")
        before = coordinator.run_view()

        # Mutation caught: execution failure erases the key and lets another command reuse it.
        with self.assertRaisesRegex(ValueError, "idempotency key"):
            coordinator.submit_command(changed, self.capability())

        self.assertEqual(coordinator.run_view(), before)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                failed.command_id,
                self.capability(can_read_all_audit=True),
            )

    def test_failed_attempt_rejects_different_key_under_same_command_id(self) -> None:
        coordinator, failed = self.failed_projector_attempt("failed-command-id")
        changed = replace(failed, idempotency_key="changed-key")
        before = coordinator.run_view()

        # Mutation caught: execution failure erases the command ID collision boundary.
        with self.assertRaisesRegex(ValueError, "command id"):
            coordinator.submit_command(changed, self.capability())

        self.assertEqual(coordinator.run_view(), before)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                failed.command_id,
                self.capability(can_read_all_audit=True),
            )

    def test_failed_attempt_rejects_changed_capability_but_exact_retry_succeeds(self) -> None:
        coordinator, failed = self.failed_projector_attempt("failed-capability")
        narrower = self.capability(ScenarioCommandKind.STEP)

        # Mutation caught: retry identity ignores the separately supplied capability hash.
        with self.assertRaisesRegex(ValueError, "idempotency key"):
            coordinator.submit_command(failed, narrower)

        retried = coordinator.submit_command(failed, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(coordinator.state.round_index, 1)

    def test_command_audit_is_private_to_authority_or_host_capability(self) -> None:
        coordinator = self.coordinator()
        operator = self.capability()
        request = self.request(coordinator, ScenarioCommandKind.START)
        result = coordinator.submit_command(request, operator)

        with self.assertRaisesRegex(PermissionError, "not authorized"):
            coordinator.command_result(
                request.command_id,
                self.capability(authority_id="other"),
            )
        host = self.capability(
            authority_id="host",
            can_read_all_audit=True,
        )
        self.assertIs(coordinator.command_result(request.command_id, host), result)

    def test_command_audit_hides_known_and_unknown_ids_from_foreign_authorities(self) -> None:
        coordinator = self.coordinator()
        request = self.request(coordinator, ScenarioCommandKind.START)
        coordinator.submit_command(request, self.capability())
        foreign = self.capability(authority_id="foreign")

        errors = []
        for command_id in (request.command_id, "unknown-command"):
            with self.subTest(command_id=command_id):
                with self.assertRaises(PermissionError) as raised:
                    coordinator.command_result(command_id, foreign)
                errors.append(str(raised.exception))

        # Mutation caught: looking up the ID before authorization reveals existence.
        self.assertEqual(
            errors,
            [
                "scenario command audit is not authorized",
                "scenario command audit is not authorized",
            ],
        )
        host = self.capability(authority_id="host", can_read_all_audit=True)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result("unknown-command", host)
        with self.assertRaisesRegex(PermissionError, "not authorized"):
            coordinator.command_result("unknown-command", self.capability())
        wrong_run_host = self.capability(
            authority_id="host",
            run_id="other-run",
            can_read_all_audit=True,
        )
        for command_id in (request.command_id, "unknown-command"):
            with self.subTest(wrong_run_command_id=command_id):
                with self.assertRaisesRegex(PermissionError, "not authorized"):
                    coordinator.command_result(command_id, wrong_run_host)

    def test_concurrent_real_steps_serialize_and_keep_sqlite_state_coherent(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        first_request = self.request(coordinator, ScenarioCommandKind.STEP)
        second_request = self.request(coordinator, ScenarioCommandKind.STEP)
        real_transition = coordinator_module.simulate_situated_network_round
        first_transition_entered = Event()
        release_first_transition = Event()
        second_transition_entered = Event()
        second_attempting = Event()
        invocation_lock = Lock()
        invocation_count = 0
        results = {}
        errors = []

        def controlled_transition(*args, **kwargs):
            nonlocal invocation_count
            with invocation_lock:
                invocation_count += 1
                invocation = invocation_count
            if invocation == 1:
                first_transition_entered.set()
                if not release_first_transition.wait(5):
                    raise RuntimeError("test transition release timed out")
            else:
                second_transition_entered.set()
            return real_transition(*args, **kwargs)

        def submit(label, request, *, attempting=None) -> None:
            if attempting is not None:
                attempting.set()
            try:
                results[label] = coordinator.submit_command(
                    request,
                    self.capability(),
                )
            except Exception as error:
                errors.append(error)

        with patch.object(
            coordinator_module,
            "simulate_situated_network_round",
            side_effect=controlled_transition,
        ):
            first = Thread(target=submit, args=("first", first_request))
            first.start()
            self.assertTrue(first_transition_entered.wait(5))
            second = Thread(
                target=submit,
                args=("second", second_request),
                kwargs={"attempting": second_attempting},
            )
            second.start()
            self.assertTrue(second_attempting.wait(5))
            entered_while_first_was_active = second_transition_entered.wait(0.5)
            release_first_transition.set()
            first.join(5)
            second.join(5)

        # Mutation caught: a Boolean guard rejects/races instead of serializing threads.
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(entered_while_first_was_active)
        self.assertEqual(errors, [])
        self.assertTrue(results["first"].accepted)
        self.assertFalse(results["second"].accepted)
        self.assertIs(results["second"].reason, ScenarioCommandReason.STALE_STATE)
        self.assertEqual(coordinator.state.round_index, 1)
        self.assertEqual(
            coordinator.state.memory_store_hash,
            hash_situated_percept_memory_store(self.database),
        )

    def test_cross_thread_query_waits_for_a_coherent_committed_step(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        real_transition = coordinator_module.simulate_situated_network_round
        transition_entered = Event()
        release_transition = Event()
        reader_attempting = Event()
        reader_done = Event()
        observed = []

        def controlled_transition(*args, **kwargs):
            transition_entered.set()
            if not release_transition.wait(5):
                raise RuntimeError("test transition release timed out")
            return real_transition(*args, **kwargs)

        def read_view() -> None:
            reader_attempting.set()
            observed.append(coordinator.run_view())
            reader_done.set()

        with patch.object(
            coordinator_module,
            "simulate_situated_network_round",
            side_effect=controlled_transition,
        ):
            writer = Thread(
                target=coordinator.submit_command,
                args=(request, self.capability()),
            )
            writer.start()
            self.assertTrue(transition_entered.wait(5))
            reader = Thread(target=read_view)
            reader.start()
            self.assertTrue(reader_attempting.wait(5))
            completed_before_commit = reader_done.wait(0.5)
            release_transition.set()
            writer.join(5)
            reader.join(5)

        # Mutation caught: unlocked queries can observe the pre-step half of an operation.
        self.assertFalse(completed_before_commit)
        self.assertFalse(writer.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].round_index, 1)
        self.assertEqual(observed[0].state_hash, coordinator.state.content_hash)

    def test_external_bus_callback_query_does_not_deadlock_step_publication(self) -> None:
        bus = SimulationOutputBus()
        step_publish_attempted = Event()

        class SignalingPublisher:
            def publish(self, batch):
                step_publish_attempted.set()
                return bus.publish(batch)

        coordinator = self.coordinator(publisher=SignalingPublisher())
        self.start(coordinator)
        external_database = self.root / "external-publication.sqlite3"
        external_initial = initialize_compiled_scenario(
            external_database,
            self.scenario,
        )
        external_round = simulate_situated_network_round(
            external_database,
            self.scenario.runtime_model,
            external_initial,
        )
        external_batch = project_simulation_output(
            self.scenario,
            external_round,
            stream_id="external-publication",
            first_sequence=1,
        )
        callback_entered = Event()
        allow_callback_query = Event()
        callback_query_done = Event()
        callback_lock = Lock()
        callback_count = 0
        observed = []
        external_reports = []
        step_results = []
        errors = []

        def callback(_) -> None:
            nonlocal callback_count
            with callback_lock:
                callback_count += 1
                call_number = callback_count
            if call_number != 1:
                return
            callback_entered.set()
            if not allow_callback_query.wait(5):
                raise RuntimeError("test callback query release timed out")
            observed.append(coordinator.run_view())
            callback_query_done.set()

        bus.subscribe(
            "coordinator-query",
            tuple(SimulationOutputKind),
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            callback,
        )

        def publish_external() -> None:
            try:
                external_reports.append(bus.publish(external_batch))
            except Exception as error:
                errors.append(error)

        step_request = self.request(coordinator, ScenarioCommandKind.STEP)

        def step() -> None:
            try:
                step_results.append(
                    coordinator.submit_command(step_request, self.capability())
                )
            except Exception as error:
                errors.append(error)

        external = Thread(target=publish_external, daemon=True)
        external.start()
        self.assertTrue(callback_entered.wait(5))
        stepping = Thread(target=step, daemon=True)
        stepping.start()
        self.assertTrue(step_publish_attempted.wait(5))
        allow_callback_query.set()
        external.join(5)
        stepping.join(5)

        # Mutation caught: holding the state lock while waiting for the bus forms
        # bus -> callback query -> coordinator / coordinator -> bus lock inversion.
        self.assertFalse(external.is_alive())
        self.assertFalse(stepping.is_alive())
        self.assertTrue(callback_query_done.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(len(external_reports), 1)
        self.assertEqual(external_reports[0].failures, ())
        self.assertEqual(len(step_results), 1)
        self.assertTrue(step_results[0].accepted)
        self.assertEqual(coordinator.last_delivery_report.failures, ())
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].round_index, 1)
        self.assertEqual(observed[0].state_hash, coordinator.state.content_hash)
        self.assertEqual(
            coordinator.state.memory_store_hash,
            hash_situated_percept_memory_store(self.database),
        )

    def test_second_writer_waits_until_publication_report_is_committed(self) -> None:
        bus = SimulationOutputBus()
        coordinator = self.coordinator(publisher=bus)
        callback_entered = Event()
        release_callback = Event()
        second_attempting = Event()
        second_done = Event()
        first_results = []
        second_results = []
        errors = []

        def callback(_) -> None:
            callback_entered.set()
            if not release_callback.wait(5):
                raise RuntimeError("test publication release timed out")

        bus.subscribe(
            "publication-gate",
            tuple(SimulationOutputKind),
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            callback,
        )
        self.start(coordinator)
        first_request = self.request(coordinator, ScenarioCommandKind.STEP)

        def first_step() -> None:
            try:
                first_results.append(
                    coordinator.submit_command(first_request, self.capability())
                )
            except Exception as error:
                errors.append(error)

        first = Thread(target=first_step)
        first.start()
        self.assertTrue(callback_entered.wait(5))
        self.assertEqual(coordinator.state.round_index, 1)
        second_request = self.request(coordinator, ScenarioCommandKind.PAUSE)

        def second_write() -> None:
            second_attempting.set()
            try:
                second_results.append(
                    coordinator.submit_command(second_request, self.capability())
                )
            except Exception as error:
                errors.append(error)
            finally:
                second_done.set()

        second = Thread(target=second_write)
        second.start()
        self.assertTrue(second_attempting.wait(5))
        completed_during_publication = second_done.wait(0.5)
        release_callback.set()
        first.join(5)
        second.join(5)

        # Mutation caught: releasing operation ownership before callbacks lets a
        # second writer overtake delivery-report publication.
        self.assertFalse(completed_during_publication)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(first_results), 1)
        self.assertEqual(len(second_results), 1)
        self.assertTrue(first_results[0].accepted)
        self.assertTrue(second_results[0].accepted)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.PAUSED)
        self.assertEqual(
            coordinator.last_delivery_report.batch_hash,
            first_results[0].output_batch_hash,
        )

    def test_one_running_step_advances_one_round_and_emits_canonical_command_record(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)

        result = coordinator.submit_command(request, self.capability())

        # Mutation caught: one command loops, skips, or binds output to the wrong state.
        self.assertTrue(result.accepted)
        self.assertEqual(coordinator.state.round_index, 1)
        self.assertEqual(result.next_state_hash, coordinator.state.content_hash)
        self.assertIsNotNone(result.output_batch_hash)
        view = coordinator.output_view(
            result.output_batch_hash,
            SimulationAudienceCapability(SimulationOutputAudience.INTERNAL),
        )
        self.assertEqual(view.first_sequence, 1)
        self.assertEqual(view.next_state_hash, coordinator.state.content_hash)
        self.assertEqual(view.records[0].kind, SimulationOutputKind.COMMAND_RESULT)
        payload = view.records[0].payload
        self.assertIsInstance(payload, SimulationCommandResultPayload)
        self.assertEqual(payload.command_id, request.command_id)
        self.assertTrue(payload.accepted)
        self.assertEqual(payload.reason_code, ScenarioCommandReason.ACCEPTED.value)
        self.assertEqual(
            set(view.records[0].source_artifact_hashes),
            {
                request.content_hash,
                view.prior_state_hash,
                view.next_state_hash,
                view.round_result_hash,
            },
        )
        self.assertEqual(
            tuple(record.sequence for record in view.records),
            tuple(range(view.first_sequence, view.last_sequence + 1)),
        )
        self.assertEqual(coordinator.run_view().next_sequence, view.last_sequence + 1)

    def test_paused_step_advances_once_and_remains_paused(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        self.submit(coordinator, ScenarioCommandKind.PAUSE)

        stepped = self.submit(coordinator, ScenarioCommandKind.STEP)

        self.assertTrue(stepped.accepted)
        self.assertEqual(coordinator.state.round_index, 1)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.PAUSED)
        self.assertIs(stepped.prior_status, ScenarioRunStatus.PAUSED)
        self.assertIs(stepped.next_status, ScenarioRunStatus.PAUSED)

    def test_maximum_round_completes_run_and_later_step_reports_maximum_rounds(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(self.scenario.run_policy, maximum_rounds=1),
        )
        coordinator = self.coordinator(scenario=scenario)
        self.start(coordinator)

        accepted = self.submit(coordinator, ScenarioCommandKind.STEP)
        before = coordinator.run_view()
        later_request = self.request(coordinator, ScenarioCommandKind.STEP)
        later = coordinator.submit_command(later_request, self.capability())

        self.assertTrue(accepted.accepted)
        self.assertIs(accepted.next_status, ScenarioRunStatus.COMPLETED)
        self.assertIs(before.status, ScenarioRunStatus.COMPLETED)
        self.assertFalse(later.accepted)
        self.assertIs(later.reason, ScenarioCommandReason.MAXIMUM_ROUNDS)
        self.assertEqual(coordinator.run_view(), before)
        for kind in (
            ScenarioCommandKind.START,
            ScenarioCommandKind.PAUSE,
            ScenarioCommandKind.RESUME,
            ScenarioCommandKind.STOP,
        ):
            result = self.submit(coordinator, kind)
            self.assertFalse(result.accepted)
            self.assertIs(result.reason, ScenarioCommandReason.INVALID_STATUS)
            self.assertEqual(coordinator.run_view(), before)

    def test_command_only_policy_produces_valid_single_record_batch(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                allowed_output_kinds=(SimulationOutputKind.COMMAND_RESULT.value,),
            ),
        )
        coordinator = self.coordinator(scenario=scenario)
        self.start(coordinator)

        result = self.submit(coordinator, ScenarioCommandKind.STEP)
        view = coordinator.output_view(
            result.output_batch_hash,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
        )

        # Mutation caught: the coordinator trusts the base projector's empty-candidate failure.
        self.assertEqual(len(view.records), 1)
        self.assertIs(view.records[0].kind, SimulationOutputKind.COMMAND_RESULT)
        self.assertEqual(view.first_sequence, 1)
        self.assertEqual(view.last_sequence, 1)

    def test_one_record_budget_still_permits_start_and_first_command_only_step(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                maximum_rounds=1,
                maximum_output_records=1,
                allowed_output_kinds=(SimulationOutputKind.COMMAND_RESULT.value,),
            ),
        )
        coordinator = self.coordinator(scenario=scenario)

        started = self.start(coordinator)
        stepped = self.submit(coordinator, ScenarioCommandKind.STEP)

        # Mutation caught: tying command attempts directly to output capacity blocks step.
        self.assertTrue(started.accepted)
        self.assertTrue(stepped.accepted)
        self.assertEqual(coordinator.state.round_index, 1)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.COMPLETED)

    def test_command_history_cannot_preempt_the_full_declared_round_budget(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                maximum_rounds=3,
                maximum_output_records=3,
                allowed_output_kinds=(SimulationOutputKind.COMMAND_RESULT.value,),
            ),
        )
        coordinator = self.coordinator(scenario=scenario)

        self.start(coordinator)
        results = [
            self.submit(coordinator, ScenarioCommandKind.STEP)
            for _ in range(3)
        ]

        # Mutation caught: a smaller attempt cap rejects a legal final declared step.
        self.assertTrue(all(result.accepted for result in results))
        self.assertEqual(coordinator.state.round_index, 3)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.COMPLETED)

    def test_policy_without_base_or_command_record_has_stable_atomic_failure(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                allowed_output_kinds=(SimulationOutputKind.STORY_PROGRESS.value,),
            ),
        )
        coordinator = self.coordinator(scenario=scenario)
        self.start(coordinator)
        before = coordinator.run_view()
        memory_hash = hash_situated_percept_memory_store(self.database)
        request = self.request(coordinator, ScenarioCommandKind.STEP)

        with self.assertRaisesRegex(
            ValueError,
            "^simulation output projection produced no supported records$",
        ):
            coordinator.submit_command(request, self.capability())

        self.assertEqual(hash_situated_percept_memory_store(self.database), memory_hash)
        self.assertEqual(coordinator.run_view(), before)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                request.command_id,
                self.capability(can_read_all_audit=True),
            )

    def test_output_limit_counts_optional_command_and_all_retained_records(self) -> None:
        command_plus_metrics = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                allowed_output_kinds=(
                    SimulationOutputKind.COMMAND_RESULT.value,
                    SimulationOutputKind.NETWORK_METRICS.value,
                ),
                maximum_output_records=3,
            ),
        )
        coordinator = self.coordinator(scenario=command_plus_metrics)
        self.start(coordinator)
        first = self.submit(coordinator, ScenarioCommandKind.STEP)
        first_view = coordinator.output_view(
            first.output_batch_hash,
            SimulationAudienceCapability(SimulationOutputAudience.INTERNAL),
        )
        self.assertEqual(
            tuple(record.kind for record in first_view.records),
            (
                SimulationOutputKind.COMMAND_RESULT,
                SimulationOutputKind.NETWORK_METRICS,
            ),
        )
        before = coordinator.run_view()
        memory_hash = hash_situated_percept_memory_store(self.database)
        request = self.request(coordinator, ScenarioCommandKind.STEP)

        with self.assertRaisesRegex(ValueError, "maximum output records"):
            coordinator.submit_command(request, self.capability())

        self.assertEqual(hash_situated_percept_memory_store(self.database), memory_hash)
        self.assertEqual(coordinator.run_view(), before)

    def test_command_attempt_history_limit_bounds_invalid_control_spam_and_preserves_cached_results(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                maximum_rounds=1,
                maximum_output_records=1,
            ),
        )
        coordinator = self.coordinator(scenario=scenario)
        operator = self.capability()
        first_request = self.request(coordinator, ScenarioCommandKind.PAUSE)
        first = coordinator.submit_command(first_request, operator)
        second_request = self.request(coordinator, ScenarioCommandKind.RESUME)
        second = coordinator.submit_command(second_request, operator)
        # The derived bound is 1 output + 1 round + 6 closed command kinds = 8.
        for _ in range(6):
            rejected = coordinator.submit_command(
                self.request(coordinator, ScenarioCommandKind.STEP),
                operator,
            )
            self.assertFalse(rejected.accepted)
        before = coordinator.run_view()

        # Mutation caught: rejected controls can grow attempt/audit history without bound.
        with self.assertRaisesRegex(
            ValueError,
            "^scenario command history limit exceeded$",
        ):
            coordinator.submit_command(
                self.request(coordinator, ScenarioCommandKind.STEP),
                operator,
            )

        self.assertEqual(coordinator.run_view(), before)
        self.assertIs(coordinator.submit_command(first_request, operator), first)
        self.assertIs(coordinator.command_result(second_request.command_id, operator), second)
        with self.assertRaisesRegex(ValueError, "idempotency key"):
            coordinator.submit_command(
                replace(first_request, command_id="changed-at-capacity"),
                operator,
            )

    def test_exact_failed_attempt_retry_remains_available_at_history_capacity(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                maximum_rounds=1,
                maximum_output_records=2,
                allowed_output_kinds=(SimulationOutputKind.COMMAND_RESULT.value,),
            ),
        )
        coordinator = self.coordinator(scenario=scenario)
        self.start(coordinator)
        failed = self.request(coordinator, ScenarioCommandKind.STEP)
        with patch(
            "narrative_dynamics.abm.scenario_coordinator.project_simulation_output",
            side_effect=RuntimeError("projector seam failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "projector seam failed"):
                coordinator.submit_command(failed, self.capability())
        # Fill the remaining 7 slots in the derived 2 + 1 + 6 attempt bound.
        for _ in range(7):
            rejected = coordinator.submit_command(
                self.request(coordinator, ScenarioCommandKind.START),
                self.capability(),
            )
            self.assertFalse(rejected.accepted)
        before = coordinator.run_view()

        with self.assertRaisesRegex(
            ValueError,
            "^scenario command history limit exceeded$",
        ):
            coordinator.submit_command(
                self.request(coordinator, ScenarioCommandKind.PAUSE),
                self.capability(),
            )
        self.assertEqual(coordinator.run_view(), before)

        # Mutation caught: the capacity gate rejects an exact registered failed retry.
        retried = coordinator.submit_command(failed, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(coordinator.state.round_index, 1)

    def test_callback_observes_committed_state_and_history_and_failure_is_operational(self) -> None:
        bus = SimulationOutputBus()
        coordinator = self.coordinator(publisher=bus)
        observed = []

        def callback(view) -> None:
            observed.append(
                (
                    coordinator.state.content_hash,
                    coordinator.run_view().output_batch_hashes,
                    coordinator.output_view(
                        view.source_batch_hash,
                        SimulationAudienceCapability(
                            SimulationOutputAudience.INTERNAL
                        ),
                    ).source_batch_hash,
                )
            )
            raise RuntimeError("private callback detail")

        bus.subscribe(
            "failing-observer",
            tuple(SimulationOutputKind),
            SimulationAudienceCapability(SimulationOutputAudience.INTERNAL),
            callback,
        )
        self.start(coordinator)

        result = self.submit(coordinator, ScenarioCommandKind.STEP)

        self.assertTrue(result.accepted)
        self.assertEqual(len(observed), 1)
        state_hash, batch_hashes, observed_batch_hash = observed[0]
        self.assertEqual(state_hash, result.next_state_hash)
        self.assertEqual(batch_hashes, (result.output_batch_hash,))
        self.assertEqual(observed_batch_hash, result.output_batch_hash)
        report = coordinator.last_delivery_report
        self.assertIsNotNone(report)
        self.assertEqual(report.failures[0].code, "callback_error")
        self.assertNotIn("private callback detail", repr(report))

    def test_callback_command_reentrancy_is_stable_and_does_not_rollback_step(self) -> None:
        bus = SimulationOutputBus()
        coordinator = self.coordinator(publisher=bus)
        errors = []

        def callback(_) -> None:
            request = self.request(coordinator, ScenarioCommandKind.PAUSE)
            try:
                coordinator.submit_command(request, self.capability())
            except RuntimeError as error:
                errors.append(str(error))
                raise

        bus.subscribe(
            "reentrant",
            tuple(SimulationOutputKind),
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            callback,
        )
        self.start(coordinator)

        result = self.submit(coordinator, ScenarioCommandKind.STEP)

        self.assertTrue(result.accepted)
        self.assertEqual(errors, ["reentrant scenario command submission"])
        self.assertEqual(coordinator.state.round_index, 1)
        self.assertIs(coordinator.run_view().status, ScenarioRunStatus.RUNNING)
        self.assertEqual(
            coordinator.last_delivery_report.failures[0].code,
            "callback_error",
        )

    def test_publisher_exception_becomes_redacted_report_and_exact_retry_does_not_republish(self) -> None:
        class RaisingPublisher:
            def __init__(self) -> None:
                self.calls = 0

            def publish(self, _) -> None:
                self.calls += 1
                raise RuntimeError("private publisher failure C:\\secret\\memory.sqlite3")

        publisher = RaisingPublisher()
        coordinator = self.coordinator(publisher=publisher)
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)

        # Mutation caught: a post-commit publisher exception escapes as command failure.
        accepted = coordinator.submit_command(request, self.capability())

        self.assertTrue(accepted.accepted)
        self.assertEqual(coordinator.state.round_index, 1)
        expected = SimulationDeliveryReport(
            accepted.output_batch_hash,
            (),
            (
                SimulationDeliveryFailure(
                    "scenario-output-publisher",
                    "callback_error",
                ),
            ),
        )
        self.assertEqual(coordinator.last_delivery_report, expected)
        self.assertNotIn("private publisher failure", repr(expected))
        self.assertNotIn("secret", repr(expected.to_dict()))
        self.assertIs(
            coordinator.submit_command(request, self.capability()),
            accepted,
        )
        self.assertEqual(publisher.calls, 1)

    def test_invalid_or_wrong_batch_publisher_report_is_normalized_after_commit(self) -> None:
        private_hash = "sha256:" + "f" * 64
        cases = (
            (
                "invalid-report",
                "private invalid publisher return C:\\secret",
            ),
            (
                "wrong-batch-report",
                SimulationDeliveryReport(
                    private_hash,
                    ("private-subscriber",),
                    (),
                ),
            ),
        )

        for name, returned in cases:
            with self.subTest(name=name):
                database = self.root / f"{name}.sqlite3"

                class FixedPublisher:
                    def publish(self, _):
                        return returned

                coordinator = ScenarioCoordinator.create(
                    database,
                    self.scenario,
                    run_id="law-firm-run",
                    stream_id=f"{name}-stream",
                    publisher=FixedPublisher(),
                )
                self.start(coordinator)
                accepted = self.submit(coordinator, ScenarioCommandKind.STEP)

                # Mutation caught: arbitrary publisher output becomes operational state.
                self.assertTrue(accepted.accepted)
                self.assertEqual(coordinator.state.round_index, 1)
                self.assertEqual(
                    coordinator.last_delivery_report,
                    SimulationDeliveryReport(
                        accepted.output_batch_hash,
                        (),
                        (
                            SimulationDeliveryFailure(
                                "scenario-output-publisher",
                                "callback_error",
                            ),
                        ),
                    ),
                )
                self.assertNotIn("private", repr(coordinator.last_delivery_report))
                self.assertNotIn(private_hash, repr(coordinator.last_delivery_report))

    def test_valid_publisher_report_is_preserved_by_exact_identity(self) -> None:
        class ValidPublisher:
            def __init__(self) -> None:
                self.report = None

            def publish(self, batch):
                self.report = SimulationDeliveryReport(
                    batch.content_hash,
                    ("subscriber",),
                    (),
                )
                return self.report

        publisher = ValidPublisher()
        coordinator = self.coordinator(publisher=publisher)
        self.start(coordinator)

        accepted = self.submit(coordinator, ScenarioCommandKind.STEP)

        # Mutation caught: normalization rebuilds a valid report and loses identity.
        self.assertTrue(accepted.accepted)
        self.assertIs(coordinator.last_delivery_report, publisher.report)

    def test_projector_failure_restores_real_sqlite_and_all_coordinator_fields_then_retries(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        before = coordinator.run_view()
        memory_hash = hash_situated_percept_memory_store(self.database)

        with patch(
            "narrative_dynamics.abm.scenario_coordinator.project_simulation_output",
            side_effect=RuntimeError("projector seam failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "projector seam failed"):
                coordinator.submit_command(request, self.capability())

        self.assertEqual(hash_situated_percept_memory_store(self.database), memory_hash)
        self.assertEqual(coordinator.run_view(), before)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                request.command_id,
                self.capability(can_read_all_audit=True),
            )

        retried = coordinator.submit_command(request, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(coordinator.state.round_index, 1)

    def test_state_store_cas_failure_restores_real_sqlite_and_retries_same_request(self) -> None:
        store = InMemoryScenarioStateStore()
        coordinator = self.coordinator(state_store=store)
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        before = coordinator.run_view()
        prior_state = coordinator.state
        memory_hash = hash_situated_percept_memory_store(self.database)

        with patch.object(
            store,
            "compare_and_swap",
            side_effect=RuntimeError("CAS seam failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "CAS seam failed"):
                coordinator.submit_command(request, self.capability())

        self.assertIs(store.load("law-firm-run"), prior_state)
        self.assertEqual(hash_situated_percept_memory_store(self.database), memory_hash)
        self.assertEqual(coordinator.run_view(), before)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                request.command_id,
                self.capability(can_read_all_audit=True),
            )

        retried = coordinator.submit_command(request, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(coordinator.state.round_index, 1)

    def test_output_limit_seam_failure_restores_real_sqlite_and_retries_same_request(self) -> None:
        coordinator = self.coordinator()
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        before = coordinator.run_view()
        memory_hash = hash_situated_percept_memory_store(self.database)

        with patch(
            "narrative_dynamics.abm.scenario_coordinator."
            "ScenarioCoordinator._enforce_output_record_limit",
            side_effect=ValueError("scenario run exceeded maximum output records"),
        ):
            with self.assertRaisesRegex(ValueError, "maximum output records"):
                coordinator.submit_command(request, self.capability())

        self.assertEqual(hash_situated_percept_memory_store(self.database), memory_hash)
        self.assertEqual(coordinator.run_view(), before)
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                request.command_id,
                self.capability(can_read_all_audit=True),
            )

        retried = coordinator.submit_command(request, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(coordinator.state.round_index, 1)

    def test_manual_checkpoint_is_exact_and_idempotent(self) -> None:
        store = LocalScenarioCheckpointStore(self.root / "manual-checkpoints")
        coordinator = self.coordinator(checkpoint_store=store)
        request = self.request(
            coordinator,
            ScenarioCommandKind.CHECKPOINT,
            requested_checkpoint_id="opening-state",
        )

        created = coordinator.submit_command(request, self.capability())

        self.assertTrue(created.accepted)
        self.assertIsNotNone(created.checkpoint_hash)
        self.assertEqual(coordinator.run_view().checkpoint_hashes, (created.checkpoint_hash,))
        checkpoint = store.load(created.checkpoint_hash)
        self.assertIs(checkpoint.state, coordinator.state)
        self.assertEqual(checkpoint.next_sequence, 1)
        self.assertIs(coordinator.submit_command(request, self.capability()), created)
        self.assertEqual(coordinator.run_view().checkpoint_hashes, (created.checkpoint_hash,))

    def test_checkpoint_command_requires_configured_store_without_caching_failure(self) -> None:
        coordinator = self.coordinator()
        request = self.request(coordinator, ScenarioCommandKind.CHECKPOINT)

        with self.assertRaisesRegex(RuntimeError, "checkpoint store is not configured"):
            coordinator.submit_command(request, self.capability())

        self.assertEqual(coordinator.run_view().checkpoint_hashes, ())
        with self.assertRaisesRegex(KeyError, "unknown scenario command"):
            coordinator.command_result(
                request.command_id,
                self.capability(can_read_all_audit=True),
            )

    def test_manual_checkpoint_rejects_reserved_automatic_id_before_store_mutation(self) -> None:
        store = LocalScenarioCheckpointStore(self.root / "reserved-checkpoints")
        coordinator = self.coordinator(checkpoint_store=store)
        request = self.request(
            coordinator,
            ScenarioCommandKind.CHECKPOINT,
            requested_checkpoint_id="law-firm-run-round-2",
        )

        with patch.object(
            store,
            "create",
            side_effect=AssertionError("reserved ID reached checkpoint store"),
        ) as create:
            with self.assertRaisesRegex(ValueError, "reserved for automatic checkpoints"):
                coordinator.submit_command(request, self.capability())

        create.assert_not_called()
        self.assertEqual(coordinator.run_view().checkpoint_hashes, ())
        self.assertEqual(tuple((self.root / "reserved-checkpoints").iterdir()), ())

    def test_interval_checkpoint_marks_exact_round_batch_once(self) -> None:
        coordinator, store, checkpoint = self.checkpointed_coordinator(interval=2)

        view = coordinator.output_view(
            coordinator.run_view().output_batch_hashes[-1],
            SimulationAudienceCapability(SimulationOutputAudience.INTERNAL),
        )
        self.assertTrue(view.checkpoint)
        self.assertEqual(len(coordinator.run_view().checkpoint_hashes), 1)
        self.assertEqual(checkpoint.checkpoint_id, "law-firm-run-round-2")
        self.assertEqual(checkpoint.state, coordinator.state)
        self.assertEqual(checkpoint.next_sequence, coordinator.run_view().next_sequence)
        self.assertIs(store.load(checkpoint.content_hash), checkpoint)

    def test_automatic_checkpoint_creation_failure_rolls_back_entire_step(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(self.scenario.run_policy, checkpoint_interval=2),
        )
        store = LocalScenarioCheckpointStore(self.root / "create-failure-checkpoints")
        coordinator = self.coordinator(scenario=scenario, checkpoint_store=store)
        self.start(coordinator)
        self.submit(coordinator, ScenarioCommandKind.STEP)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        before = coordinator.run_view()
        before_memory_hash = hash_situated_percept_memory_store(self.database)

        with patch.object(store, "create", side_effect=RuntimeError("checkpoint seam failed")):
            with self.assertRaisesRegex(RuntimeError, "checkpoint seam failed"):
                coordinator.submit_command(request, self.capability())

        self.assertEqual(coordinator.run_view(), before)
        self.assertEqual(hash_situated_percept_memory_store(self.database), before_memory_hash)
        self.assertEqual(tuple((self.root / "create-failure-checkpoints").iterdir()), ())
        retried = coordinator.submit_command(request, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(len(coordinator.run_view().checkpoint_hashes), 1)

    def test_post_checkpoint_cas_failure_discards_artifact_and_rolls_back(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(self.scenario.run_policy, checkpoint_interval=1),
        )
        store = LocalScenarioCheckpointStore(self.root / "cas-checkpoints")
        state_store = InMemoryScenarioStateStore()
        coordinator = self.coordinator(
            scenario=scenario,
            checkpoint_store=store,
            state_store=state_store,
        )
        self.start(coordinator)
        request = self.request(coordinator, ScenarioCommandKind.STEP)
        before = coordinator.run_view()
        before_state = coordinator.state
        before_memory_hash = hash_situated_percept_memory_store(self.database)

        with patch.object(
            state_store,
            "compare_and_swap",
            side_effect=RuntimeError("CAS after checkpoint failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "CAS after checkpoint failed"):
                coordinator.submit_command(request, self.capability())

        self.assertEqual(coordinator.run_view(), before)
        self.assertIs(coordinator.state, before_state)
        self.assertEqual(hash_situated_percept_memory_store(self.database), before_memory_hash)
        self.assertEqual(tuple((self.root / "cas-checkpoints").iterdir()), ())
        retried = coordinator.submit_command(request, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(len(coordinator.run_view().checkpoint_hashes), 1)

    def test_fork_restores_paused_independent_child_with_fresh_identity(self) -> None:
        coordinator, _, checkpoint = self.checkpointed_coordinator()
        request = self.fork_request(coordinator, checkpoint.content_hash)
        capability = self.capability(can_fork=True)
        child_database = self.root / "child.sqlite3"

        child, result = coordinator.fork(request, capability, child_database)

        self.assertIs(child.run_view().status, ScenarioRunStatus.PAUSED)
        self.assertIs(child.state, checkpoint.state)
        self.assertEqual(child.run_view().next_sequence, 1)
        self.assertEqual(
            child.run_view().coordinator_epoch,
            coordinator.run_view().coordinator_epoch + 1,
        )
        self.assertEqual(child.run_view().parent_checkpoint_hash, checkpoint.content_hash)
        self.assertEqual(result.child_state_hash, checkpoint.state.content_hash)
        self.assertEqual(
            hash_situated_percept_memory_store(child_database),
            checkpoint.memory_store_hash,
        )
        repeated_child, repeated_result = coordinator.fork(
            request,
            capability,
            child_database,
        )
        self.assertIs(repeated_child, child)
        self.assertIs(repeated_result, result)

        self.submit(coordinator, ScenarioCommandKind.STEP)
        parent_after_step = coordinator.state
        child_capability = self.capability(run_id="law-firm-child")
        for number in range(2):
            child_request = ScenarioCommandRequest(
                f"child-command-{number}",
                f"child-idem-{number}",
                "law-firm-child",
                child.run_view().scenario_hash,
                child.run_view().coordinator_epoch,
                child.state.content_hash,
                "operator",
                ScenarioCommandKind.STEP,
            )
            self.assertTrue(child.submit_command(child_request, child_capability).accepted)

        self.assertIs(coordinator.state, parent_after_step)
        self.assertNotEqual(child.state.content_hash, coordinator.state.content_hash)
        self.assertEqual(
            hash_situated_percept_memory_store(self.database),
            coordinator.state.memory_store_hash,
        )
        self.assertEqual(
            hash_situated_percept_memory_store(child_database),
            child.state.memory_store_hash,
        )

    def test_fork_child_publisher_routes_output_without_entering_identity_or_retry(self) -> None:
        class Publisher:
            def __init__(self, subscription_id: str) -> None:
                self.subscription_id = subscription_id
                self.batches = []

            def publish(self, batch):
                self.batches.append(batch)
                return SimulationDeliveryReport(
                    batch.content_hash,
                    (self.subscription_id,),
                    (),
                )

        coordinator, _, checkpoint = self.checkpointed_coordinator()
        request = self.fork_request(coordinator, checkpoint.content_hash)
        capability = self.capability(can_fork=True)
        target = self.root / "routed-child.sqlite3"
        first_publisher = Publisher("child-route")
        child, result = coordinator.fork(
            request,
            capability,
            target,
            child_publisher=first_publisher,
        )
        retry_publisher = Publisher("must-not-rebind")
        retried_child, retried_result = coordinator.fork(
            request,
            capability,
            target,
            child_publisher=retry_publisher,
        )

        self.assertIs(retried_child, child)
        self.assertIs(retried_result, result)
        self.assertEqual(result.request_hash, request.content_hash)
        child_request = ScenarioCommandRequest(
            "routed-command",
            "routed-command-key",
            "law-firm-child",
            child.run_view().scenario_hash,
            child.run_view().coordinator_epoch,
            child.state.content_hash,
            "operator",
            ScenarioCommandKind.STEP,
        )
        child.submit_command(
            child_request,
            self.capability(run_id="law-firm-child"),
        )
        self.assertEqual(len(first_publisher.batches), 1)
        self.assertEqual(retry_publisher.batches, [])
        self.assertEqual(
            child.last_delivery_report.delivered_subscription_ids,
            ("child-route",),
        )

    def test_fork_validates_child_publisher_before_restore_or_attempt_registration(self) -> None:
        coordinator, _, checkpoint = self.checkpointed_coordinator()
        request = self.fork_request(coordinator, checkpoint.content_hash)
        capability = self.capability(can_fork=True)
        target = self.root / "publisher-validation-child.sqlite3"

        with self.assertRaisesRegex(TypeError, "child publisher must provide publish"):
            coordinator.fork(
                request,
                capability,
                target,
                child_publisher=object(),
            )

        self.assertFalse(target.exists())
        child, result = coordinator.fork(request, capability, target)
        self.assertTrue(target.exists())
        self.assertEqual(result.request_hash, request.content_hash)
        self.assertIsNotNone(child)

    def test_fork_rejects_unauthorized_unknown_or_mismatched_source(self) -> None:
        coordinator, _, checkpoint = self.checkpointed_coordinator()
        request = self.fork_request(coordinator, checkpoint.content_hash)
        allowed = self.capability(can_fork=True)

        with self.assertRaisesRegex(PermissionError, "fork is not authorized"):
            coordinator.fork(request, self.capability(), self.root / "unauthorized.sqlite3")
        unknown = replace(request, fork_id="unknown", idempotency_key="unknown", checkpoint_hash="sha256:" + "f" * 64)
        with self.assertRaisesRegex(KeyError, "unknown scenario checkpoint"):
            coordinator.fork(unknown, allowed, self.root / "unknown.sqlite3")

        other_hash = "sha256:" + "e" * 64
        cases = (
            (replace(request, fork_id="run", idempotency_key="run", source_run_id="other-run"), "source run"),
            (replace(request, fork_id="scenario", idempotency_key="scenario", scenario_hash=other_hash), "scenario"),
            (replace(request, fork_id="epoch", idempotency_key="epoch", source_epoch=2), "source epoch"),
        )
        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    coordinator.fork(changed, allowed, self.root / f"{message}.sqlite3")

    def test_fork_identity_collisions_and_duplicate_children_are_rejected(self) -> None:
        coordinator, _, checkpoint = self.checkpointed_coordinator()
        allowed = self.capability(can_fork=True)
        request = self.fork_request(coordinator, checkpoint.content_hash)
        coordinator.fork(request, allowed, self.root / "first-child.sqlite3")

        changed_key = replace(request, child_run_id="changed-child")
        with self.assertRaisesRegex(ValueError, "fork idempotency key was reused"):
            coordinator.fork(changed_key, allowed, self.root / "changed.sqlite3")
        changed_fork_id = replace(
            request,
            idempotency_key="another-key",
            child_run_id="another-child",
            child_stream_id="another-stream",
        )
        with self.assertRaisesRegex(ValueError, "fork id was reused"):
            coordinator.fork(changed_fork_id, allowed, self.root / "another.sqlite3")

        duplicate_run = self.fork_request(
            coordinator,
            checkpoint.content_hash,
            fork_id="duplicate-run",
            idempotency_key="duplicate-run",
            child_stream_id="fresh-stream",
        )
        with self.assertRaisesRegex(ValueError, "child run already exists"):
            coordinator.fork(duplicate_run, allowed, self.root / "duplicate-run.sqlite3")
        duplicate_stream = self.fork_request(
            coordinator,
            checkpoint.content_hash,
            fork_id="duplicate-stream",
            idempotency_key="duplicate-stream",
            child_run_id="fresh-child",
        )
        with self.assertRaisesRegex(ValueError, "child stream already exists"):
            coordinator.fork(duplicate_stream, allowed, self.root / "duplicate-stream.sqlite3")

    def test_invalid_fork_spam_is_not_retained_and_authorized_history_is_bounded(self) -> None:
        scenario = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                maximum_rounds=1,
                maximum_output_records=1,
            ),
        )
        store = LocalScenarioCheckpointStore(self.root / "bounded-fork-checkpoints")
        coordinator = self.coordinator(
            scenario=scenario,
            checkpoint_store=store,
        )
        checkpoint_request = self.request(
            coordinator,
            ScenarioCommandKind.CHECKPOINT,
            requested_checkpoint_id="fork-base",
        )
        checkpoint_result = coordinator.submit_command(
            checkpoint_request,
            self.capability(),
        )
        checkpoint_hash = checkpoint_result.checkpoint_hash
        allowed = self.capability(can_fork=True)

        for number in range(4):
            unauthorized = self.fork_request(
                coordinator,
                checkpoint_hash,
                fork_id=f"unauthorized-{number}",
                idempotency_key=f"unauthorized-{number}",
                child_run_id=f"unauthorized-child-{number}",
                child_stream_id=f"unauthorized-stream-{number}",
            )
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                coordinator.fork(
                    unauthorized,
                    self.capability(),
                    self.root / f"unauthorized-{number}.sqlite3",
                )

        invalid_requests = (
            replace(
                self.fork_request(coordinator, checkpoint_hash),
                fork_id="invalid-run",
                idempotency_key="invalid-run",
                source_run_id="other-run",
            ),
            replace(
                self.fork_request(coordinator, checkpoint_hash),
                fork_id="invalid-epoch",
                idempotency_key="invalid-epoch",
                source_epoch=2,
            ),
            replace(
                self.fork_request(coordinator, checkpoint_hash),
                fork_id="invalid-checkpoint",
                idempotency_key="invalid-checkpoint",
                checkpoint_hash="sha256:" + "f" * 64,
            ),
        )
        for number, invalid in enumerate(invalid_requests):
            with self.subTest(invalid=invalid.fork_id):
                with self.assertRaises((ValueError, KeyError)):
                    coordinator.fork(
                        invalid,
                        allowed,
                        self.root / f"invalid-{number}.sqlite3",
                    )

        retained = []
        # Independent fork attempts use the same finite 1 + 1 + 6 derived bound.
        for number in range(8):
            request = self.fork_request(
                coordinator,
                checkpoint_hash,
                fork_id=f"bounded-{number}",
                idempotency_key=f"bounded-{number}",
                child_run_id=f"bounded-child-{number}",
                child_stream_id=f"bounded-stream-{number}",
            )
            retained.append(
                (
                    request,
                    coordinator.fork(
                        request,
                        allowed,
                        self.root / f"bounded-{number}.sqlite3",
                    ),
                )
            )

        overflow = self.fork_request(
            coordinator,
            checkpoint_hash,
            fork_id="bounded-overflow",
            idempotency_key="bounded-overflow",
            child_run_id="bounded-overflow-child",
            child_stream_id="bounded-overflow-stream",
        )
        with self.assertRaisesRegex(
            ValueError,
            "^scenario fork history limit exceeded$",
        ):
            coordinator.fork(
                overflow,
                allowed,
                self.root / "bounded-overflow.sqlite3",
            )

        first_request, (first_child, first_result) = retained[0]
        cached_child, cached_result = coordinator.fork(
            first_request,
            allowed,
            self.root / "ignored-at-capacity.sqlite3",
        )
        self.assertIs(cached_child, first_child)
        self.assertIs(cached_result, first_result)
        with self.assertRaisesRegex(ValueError, "idempotency key was reused"):
            coordinator.fork(
                replace(first_request, child_run_id="collision-at-capacity"),
                allowed,
                self.root / "collision-at-capacity.sqlite3",
            )

    def test_fork_restore_failures_return_no_child_and_are_not_cached(self) -> None:
        coordinator, store, checkpoint = self.checkpointed_coordinator()
        allowed = self.capability(can_fork=True)

        existing_request = self.fork_request(coordinator, checkpoint.content_hash)
        existing_target = self.root / "existing-child.sqlite3"
        existing_target.write_bytes(b"private-existing")
        with self.assertRaisesRegex(ValueError, "restore target already exists"):
            coordinator.fork(existing_request, allowed, existing_target)
        self.assertEqual(existing_target.read_bytes(), b"private-existing")
        existing_target.unlink()
        child, _ = coordinator.fork(existing_request, allowed, existing_target)
        self.assertTrue(existing_target.exists())
        self.assertIs(child.run_view().status, ScenarioRunStatus.PAUSED)

        failed_request = self.fork_request(
            coordinator,
            checkpoint.content_hash,
            fork_id="restore-failure",
            idempotency_key="restore-failure",
            child_run_id="restore-child",
            child_stream_id="restore-stream",
        )
        failed_target = self.root / "restore-failure.sqlite3"

        with patch.object(
            store,
            "_restore_owned",
            side_effect=RuntimeError("restore seam failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "restore seam failed"):
                coordinator.fork(failed_request, allowed, failed_target)
        self.assertFalse(failed_target.exists())
        retried, _ = coordinator.fork(failed_request, allowed, failed_target)
        self.assertIs(retried.run_view().status, ScenarioRunStatus.PAUSED)

    def test_preclaimed_owned_restore_failure_is_left_for_caller_cleanup(self) -> None:
        coordinator, store, checkpoint = self.checkpointed_coordinator()
        request = self.fork_request(
            coordinator,
            checkpoint.content_hash,
            fork_id="preclaimed-owned-failure",
            idempotency_key="preclaimed-owned-failure",
            child_run_id="preclaimed-owned-child",
            child_stream_id="preclaimed-owned-stream",
        )
        target = self.root / "preclaimed-owned-child.sqlite3"
        target.write_bytes(b"")
        target_token = _PhysicalFileOwnershipToken.from_stat(
            target.stat(follow_symlinks=False)
        )

        with patch.object(
            store,
            "_restore_preclaimed_owned",
            side_effect=_ScenarioCheckpointOwnedRestoreError(
                "preclaimed restore failure",
                target_token,
            ),
        ), patch.object(
            store,
            "_cleanup_restored_target",
            side_effect=AssertionError("caller-owned target must not be cleaned"),
        ):
            with self.assertRaisesRegex(RuntimeError, "^preclaimed restore failure$"):
                coordinator.fork(
                    request,
                    self.capability(can_fork=True),
                    target,
                    child_database_ownership_token=target_token,
                )

        self.assertEqual(target.read_bytes(), b"")
        self.assertEqual(
            _PhysicalFileOwnershipToken.from_stat(
                target.stat(follow_symlinks=False)
            ),
            target_token,
        )
        target.unlink()
        retried, _ = coordinator.fork(
            request,
            self.capability(can_fork=True),
            target,
        )
        self.assertIs(retried.run_view().status, ScenarioRunStatus.PAUSED)

    def test_fork_cleanup_failure_is_visible_and_path_redacted(self) -> None:
        coordinator, store, checkpoint = self.checkpointed_coordinator()
        request = self.fork_request(
            coordinator,
            checkpoint.content_hash,
            fork_id="cleanup-failure",
            idempotency_key="cleanup-failure",
            child_run_id="cleanup-child",
            child_stream_id="cleanup-stream",
        )
        allowed = self.capability(can_fork=True)
        target = self.root / "private-cleanup-child.sqlite3"
        real_unlink = Path.unlink

        def fail_child_cleanup(path, *args, **kwargs):
            if path == target:
                raise OSError(f"private child path {path}")
            return real_unlink(path, *args, **kwargs)

        with patch(
            "narrative_dynamics.abm.scenario_coordinator."
            "hash_situated_percept_memory_store",
            side_effect=RuntimeError("private post-restore failure"),
        ), patch.object(
            Path,
            "unlink",
            autospec=True,
            side_effect=fail_child_cleanup,
        ):
            with self.assertRaisesRegex(RuntimeError, "fork cleanup failed") as raised:
                coordinator.fork(request, allowed, target)

        self.assertNotIn(str(target), str(raised.exception))
        self.assertNotIn(str(self.root), str(raised.exception))

    def test_fork_cleanup_never_unlinks_substituted_valid_child_database(self) -> None:
        coordinator, _, checkpoint = self.checkpointed_coordinator()
        request = self.fork_request(
            coordinator,
            checkpoint.content_hash,
            fork_id="substituted-child",
            idempotency_key="substituted-child",
            child_run_id="substituted-child-run",
            child_stream_id="substituted-child-stream",
        )
        allowed = self.capability(can_fork=True)
        target = self.root / "substituted-child.sqlite3"
        foreign = self.root / "foreign-child.sqlite3"
        foreign_bytes = None

        def substitute_then_fail(path) -> str:
            nonlocal foreign_bytes
            source = destination = None
            try:
                source = sqlite3.connect(path)
                destination = sqlite3.connect(foreign)
                source.backup(destination)
            finally:
                if destination is not None:
                    destination.close()
                if source is not None:
                    source.close()
            with foreign.open("ab") as stream:
                stream.write(b"unrelated-foreign-child-data")
            foreign_bytes = foreign.read_bytes()
            os.replace(foreign, path)
            raise RuntimeError("post-restore seam failed")

        with patch(
            "narrative_dynamics.abm.scenario_coordinator."
            "hash_situated_percept_memory_store",
            side_effect=substitute_then_fail,
        ):
            with self.assertRaisesRegex(RuntimeError, "ownership") as raised:
                coordinator.fork(request, allowed, target)

        self.assertEqual(target.read_bytes(), foreign_bytes)
        self.assertNotIn(str(target), str(raised.exception))
        self.assertNotIn(str(self.root), str(raised.exception))

        target.unlink()
        child, _ = coordinator.fork(request, allowed, target)
        self.assertIs(child.run_view().status, ScenarioRunStatus.PAUSED)


if __name__ == "__main__":
    unittest.main()
