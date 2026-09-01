from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from narrative_dynamics.abm.scenario_compiler import (
    compile_situated_scenario_package,
)
from narrative_dynamics.abm.scenario_coordinator import ScenarioCoordinator
from narrative_dynamics.abm.scenario_coordinator_contracts import (
    ScenarioCommandCapability,
    ScenarioCommandKind,
    ScenarioCommandReason,
    ScenarioCommandRequest,
    ScenarioRunStatus,
)
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.scenario_state_store import InMemoryScenarioStateStore
from narrative_dynamics.abm.simulation_output_bus import SimulationOutputBus
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationCommandResultPayload,
    SimulationOutputAudience,
    SimulationOutputKind,
)
from narrative_dynamics.abm.situated_percept_memory import (
    hash_situated_percept_memory_store,
)
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
    ) -> ScenarioCoordinator:
        return ScenarioCoordinator.create(
            self.database,
            self.scenario if scenario is None else scenario,
            run_id="law-firm-run",
            stream_id="law-firm-stream",
            state_store=state_store,
            publisher=publisher,
        )

    def capability(
        self,
        *kinds: ScenarioCommandKind,
        authority_id: str = "operator",
        run_id: str = "law-firm-run",
        can_read_all_audit: bool = False,
    ) -> ScenarioCommandCapability:
        allowed = self.all_command_kinds if not kinds else tuple(
            sorted(kinds, key=lambda kind: kind.value)
        )
        return ScenarioCommandCapability(
            authority_id,
            run_id,
            allowed,
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
            coordinator.command_result("different-command", operator)

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
            coordinator.command_result(request.command_id, self.capability())

    def test_output_limit_counts_optional_command_and_all_retained_records(self) -> None:
        command_plus_metrics = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                allowed_output_kinds=(
                    SimulationOutputKind.COMMAND_RESULT.value,
                    SimulationOutputKind.NETWORK_METRICS.value,
                ),
                maximum_output_records=1,
            ),
        )
        coordinator = self.coordinator(scenario=command_plus_metrics)
        self.start(coordinator)
        before = coordinator.run_view()
        memory_hash = hash_situated_percept_memory_store(self.database)
        request = self.request(coordinator, ScenarioCommandKind.STEP)

        with self.assertRaisesRegex(ValueError, "maximum output records"):
            coordinator.submit_command(request, self.capability())

        self.assertEqual(hash_situated_percept_memory_store(self.database), memory_hash)
        self.assertEqual(coordinator.run_view(), before)

        metrics_only = replace(
            command_plus_metrics,
            run_policy=replace(
                command_plus_metrics.run_policy,
                allowed_output_kinds=(SimulationOutputKind.NETWORK_METRICS.value,),
            ),
        )
        second_database = self.root / "metrics.sqlite3"
        second = ScenarioCoordinator.create(
            second_database,
            metrics_only,
            run_id="law-firm-run",
            stream_id="metrics-stream",
        )
        self.start(second)
        first = self.submit(second, ScenarioCommandKind.STEP)
        self.assertTrue(first.accepted)
        self.assertEqual(second.run_view().next_sequence, 2)
        metrics_view = second.output_view(
            first.output_batch_hash,
            SimulationAudienceCapability(SimulationOutputAudience.INTERNAL),
        )
        self.assertEqual(
            tuple(record.kind for record in metrics_view.records),
            (SimulationOutputKind.NETWORK_METRICS,),
        )
        second_request = self.request(second, ScenarioCommandKind.STEP)
        second_before = second.run_view()
        second_memory_hash = hash_situated_percept_memory_store(second_database)
        with self.assertRaisesRegex(ValueError, "maximum output records"):
            second.submit_command(second_request, self.capability())
        self.assertEqual(
            hash_situated_percept_memory_store(second_database),
            second_memory_hash,
        )
        self.assertEqual(second.run_view(), second_before)

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
            coordinator.command_result(request.command_id, self.capability())

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
            coordinator.command_result(request.command_id, self.capability())

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
            coordinator.command_result(request.command_id, self.capability())

        retried = coordinator.submit_command(request, self.capability())
        self.assertTrue(retried.accepted)
        self.assertEqual(coordinator.state.round_index, 1)


if __name__ == "__main__":
    unittest.main()
