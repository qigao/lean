"""Synchronous authority for one exact compiled scenario run."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from threading import Condition, Lock, RLock, get_ident

from narrative_dynamics.abm.scenario_checkpoint_store import (
    LocalScenarioCheckpointStore,
    _PhysicalFileOwnershipToken,
    _ScenarioCheckpointOwnedRestoreError,
    _ScenarioCheckpointRestoreTargetExistsError,
)
from narrative_dynamics.abm.scenario_compiler import initialize_compiled_scenario
from narrative_dynamics.abm.scenario_coordinator_contracts import (
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
from narrative_dynamics.abm.scenario_package_contracts import (
    CompiledSituatedScenario,
)
from narrative_dynamics.abm.scenario_queries import (
    project_scenario_agent_state,
    project_scenario_network_state,
    project_scenario_output_view,
    project_scenario_public_state,
    project_scenario_run_view,
)
from narrative_dynamics.abm.scenario_state_store import (
    InMemoryScenarioStateStore,
    ScenarioStateStore,
)
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
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputRecord,
    SimulationOutputView,
    output_record_sort_key,
)
from narrative_dynamics.abm.situated_network import simulate_situated_network_round
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRoundResult,
    SituatedNetworkRuntimeState,
    SituatedNetworkSnapshot,
)
from narrative_dynamics.abm.situated_percept_memory import (
    hash_situated_percept_memory_store,
)


_EMPTY_PROJECTION_ERROR = "simulation output projection produced no supported records"
_OUTPUT_LIMIT_ERROR = "scenario run exceeded maximum output records"
_COMMAND_HISTORY_LIMIT_ERROR = "scenario command history limit exceeded"
_FORK_HISTORY_LIMIT_ERROR = "scenario fork history limit exceeded"
_PUBLISHER_FAILURE_SUBSCRIPTION_ID = "scenario-output-publisher"


def _derived_attempt_history_limit(scenario: CompiledSituatedScenario) -> int:
    """Bound one attempt map without coupling it to retained output alone.

    The output budget and declared round budget provide finite scenario-specific
    capacity.  One additional slot per closed command kind leaves bounded lifecycle
    and audit headroom.  Because both budgets are positive, the formula always
    admits mandatory ``start`` plus every declared ``step``.
    """

    return (
        scenario.run_policy.maximum_output_records
        + scenario.run_policy.maximum_rounds
        + len(ScenarioCommandKind)
    )


def _non_empty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _database_path(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scenario coordinator database path must be non-empty")
    return value


def _copy_sqlite_database(source_path: str, destination_path: str) -> None:
    source = destination = None
    try:
        source = sqlite3.connect(source_path)
        destination = sqlite3.connect(destination_path)
        source.backup(destination)
    except sqlite3.Error as error:
        raise RuntimeError("scenario coordinator database backup failed") from error
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()


def _cleanup_fork_database(
    checkpoint_store: LocalScenarioCheckpointStore,
    path: Path,
    ownership_token: _PhysicalFileOwnershipToken,
) -> None:
    try:
        checkpoint_store._cleanup_restored_target(path, ownership_token)
    except Exception:
        raise RuntimeError(
            "scenario fork cleanup failed: ownership changed"
        ) from None


class ScenarioCoordinator:
    """Serialize control and exact V19 transitions for one local run."""

    def __init__(
        self,
        database_path: str,
        scenario: CompiledSituatedScenario,
        *,
        run_id: str,
        stream_id: str,
        state_store: ScenarioStateStore,
        publisher: object,
        checkpoint_store: LocalScenarioCheckpointStore | None,
        coordinator_epoch: int,
        parent_checkpoint_hash: str | None,
        status: ScenarioRunStatus = ScenarioRunStatus.CREATED,
        next_sequence: int = 1,
    ) -> None:
        self._state_lock = RLock()
        self._operation_condition = Condition(Lock())
        self._database_path = database_path
        self._scenario = scenario
        self._run_id = run_id
        self._stream_id = stream_id
        self._state_store = state_store
        self._publisher = publisher
        self._checkpoint_store = checkpoint_store
        self._coordinator_epoch = coordinator_epoch
        self._parent_checkpoint_hash = parent_checkpoint_hash
        self._status = status
        self._next_sequence = next_sequence
        self._output_batches: tuple[SimulationOutputBatch, ...] = ()
        self._outputs_by_hash: dict[str, SimulationOutputBatch] = {}
        self._checkpoints: tuple[ScenarioCheckpoint, ...] = ()
        self._results_by_command_id: dict[
            str, tuple[str, ScenarioCommandResult]
        ] = {}
        self._idempotency_results: dict[
            str, tuple[str, str, ScenarioCommandResult]
        ] = {}
        self._attempts_by_idempotency_key: dict[
            str, tuple[str, str, str]
        ] = {}
        self._attempts_by_command_id: dict[str, str] = {}
        self._operation_owner_thread_id: int | None = None
        self._operation_name: str | None = None
        self._operation_phase: str | None = None
        self._command_history_limit = _derived_attempt_history_limit(scenario)
        self._fork_history_limit = _derived_attempt_history_limit(scenario)
        self._last_delivery_report: SimulationDeliveryReport | None = None
        self._fork_attempts_by_idempotency_key: dict[
            str, tuple[str, str, str]
        ] = {}
        self._fork_attempts_by_fork_id: dict[str, str] = {}
        self._fork_results: dict[
            str, tuple["ScenarioCoordinator", ScenarioForkResult]
        ] = {}
        self._child_run_ids: set[str] = set()
        self._child_stream_ids: set[str] = set()

    @classmethod
    def create(
        cls,
        database_path: str | Path,
        scenario: CompiledSituatedScenario,
        *,
        run_id: str,
        stream_id: str,
        state_store: ScenarioStateStore | None = None,
        publisher: object | None = None,
        checkpoint_store: LocalScenarioCheckpointStore | None = None,
        coordinator_epoch: int = 1,
        parent_checkpoint_hash: str | None = None,
    ) -> "ScenarioCoordinator":
        if not isinstance(scenario, CompiledSituatedScenario):
            raise TypeError(
                "scenario coordinator creation requires CompiledSituatedScenario"
            )
        exact_database_path = _database_path(database_path)
        exact_run_id = _non_empty_text(run_id, label="scenario coordinator run id")
        exact_stream_id = _non_empty_text(
            stream_id,
            label="scenario coordinator stream id",
        )
        exact_epoch = _positive_integer(
            coordinator_epoch,
            label="scenario coordinator epoch",
        )
        if parent_checkpoint_hash is not None:
            ScenarioRunView(
                exact_run_id,
                exact_stream_id,
                scenario.content_hash,
                exact_epoch,
                ScenarioRunStatus.CREATED,
                0,
                scenario.initial_cognitive_state.content_hash,
                1,
                (),
                (),
                parent_checkpoint_hash,
            )
        exact_store = (
            InMemoryScenarioStateStore() if state_store is None else state_store
        )
        if not isinstance(exact_store, ScenarioStateStore):
            raise TypeError("scenario coordinator state store must implement ScenarioStateStore")
        exact_publisher = SimulationOutputBus() if publisher is None else publisher
        if not callable(getattr(exact_publisher, "publish", None)):
            raise TypeError("scenario coordinator publisher must provide publish")
        if checkpoint_store is not None and not all(
            callable(getattr(checkpoint_store, name, None))
            for name in (
                "create",
                "load",
                "restore",
                "discard",
                "_restore_owned",
                "_cleanup_restored_target",
                "_verify_restored_target",
            )
        ):
            raise TypeError(
                "scenario coordinator checkpoint store must provide the local checkpoint lifecycle"
            )

        initial_state = initialize_compiled_scenario(exact_database_path, scenario)
        exact_store.initialize(exact_run_id, initial_state)
        return cls(
            exact_database_path,
            scenario,
            run_id=exact_run_id,
            stream_id=exact_stream_id,
            state_store=exact_store,
            publisher=exact_publisher,
            checkpoint_store=checkpoint_store,
            coordinator_epoch=exact_epoch,
            parent_checkpoint_hash=parent_checkpoint_hash,
        )

    @property
    def state(self) -> SituatedNetworkRuntimeState:
        with self._state_lock:
            return self._state_store.load(self._run_id)

    @property
    def last_delivery_report(self) -> SimulationDeliveryReport | None:
        with self._state_lock:
            return self._last_delivery_report

    def run_view(self) -> ScenarioRunView:
        with self._state_lock:
            return project_scenario_run_view(
                run_id=self._run_id,
                stream_id=self._stream_id,
                scenario_hash=self._scenario.content_hash,
                coordinator_epoch=self._coordinator_epoch,
                status=self._status,
                state=self.state,
                next_sequence=self._next_sequence,
                output_batches=self._output_batches,
                checkpoints=self._checkpoints,
                parent_checkpoint_hash=self._parent_checkpoint_hash,
            )

    def public_state_view(self) -> ScenarioPublicStateView:
        with self._state_lock:
            return project_scenario_public_state(
                self._run_id,
                self._scenario.content_hash,
                self.state,
            )

    def agent_state_view(
        self,
        agent_id: str,
        capability: SimulationAudienceCapability,
    ) -> ScenarioAgentStateView:
        with self._state_lock:
            return project_scenario_agent_state(
                self._run_id,
                self._scenario.content_hash,
                self.state,
                agent_id,
                capability,
            )

    def network_state(
        self,
        capability: SimulationAudienceCapability,
    ) -> SituatedNetworkSnapshot:
        with self._state_lock:
            return project_scenario_network_state(self.state, capability)

    def output_view(
        self,
        batch_hash: str,
        capability: SimulationAudienceCapability,
    ) -> SimulationOutputView:
        with self._state_lock:
            try:
                batch = self._outputs_by_hash[batch_hash]
            except (KeyError, TypeError):
                raise KeyError("unknown scenario output batch") from None
            return project_scenario_output_view(batch, capability)

    def command_result(
        self,
        command_id: str,
        capability: ScenarioCommandCapability,
    ) -> ScenarioCommandResult:
        if not isinstance(capability, ScenarioCommandCapability):
            raise TypeError(
                "scenario command audit requires ScenarioCommandCapability"
            )
        with self._state_lock:
            if capability.run_id != self._run_id:
                raise PermissionError("scenario command audit is not authorized")
            try:
                retained = self._results_by_command_id.get(command_id)
            except TypeError:
                retained = None
            if capability.can_read_all_audit:
                if retained is None:
                    raise KeyError("unknown scenario command") from None
                return retained[1]
            if retained is None or capability.authority_id != retained[0]:
                raise PermissionError("scenario command audit is not authorized")
            return retained[1]

    def submit_command(
        self,
        request: ScenarioCommandRequest,
        capability: ScenarioCommandCapability,
    ) -> ScenarioCommandResult:
        if not isinstance(request, ScenarioCommandRequest):
            raise TypeError("scenario command requires ScenarioCommandRequest")
        if not isinstance(capability, ScenarioCommandCapability):
            raise TypeError("scenario command requires ScenarioCommandCapability")
        self._begin_operation("submit_command")
        try:
            with self._state_lock:
                result, publication_batch = self._submit_command(
                    request,
                    capability,
                )
            if publication_batch is not None:
                self._mark_publication_phase()
                report = self._publish(publication_batch)
                with self._state_lock:
                    self._last_delivery_report = report
            return result
        finally:
            self._end_operation()

    def fork(
        self,
        request: ScenarioForkRequest,
        capability: ScenarioCommandCapability,
        child_database_path: str | Path,
    ) -> tuple["ScenarioCoordinator", ScenarioForkResult]:
        if not isinstance(request, ScenarioForkRequest):
            raise TypeError("scenario fork requires ScenarioForkRequest")
        if not isinstance(capability, ScenarioCommandCapability):
            raise TypeError("scenario fork requires ScenarioCommandCapability")
        self._begin_operation("fork")
        try:
            with self._state_lock:
                return self._fork(request, capability, child_database_path)
        finally:
            self._end_operation()

    def _begin_operation(self, operation: str) -> None:
        thread_id = get_ident()
        with self._operation_condition:
            if self._operation_owner_thread_id == thread_id:
                raise RuntimeError("reentrant scenario command submission")
            while self._operation_owner_thread_id is not None:
                self._operation_condition.wait()
            self._operation_owner_thread_id = thread_id
            self._operation_name = operation
            self._operation_phase = "executing"

    def _mark_publication_phase(self) -> None:
        thread_id = get_ident()
        with self._operation_condition:
            if (
                self._operation_owner_thread_id != thread_id
                or self._operation_name != "submit_command"
                or self._operation_phase != "executing"
            ):
                raise RuntimeError("scenario coordinator operation ownership changed")
            self._operation_phase = "publishing"

    def _end_operation(self) -> None:
        thread_id = get_ident()
        with self._operation_condition:
            if self._operation_owner_thread_id != thread_id:
                raise RuntimeError("scenario coordinator operation ownership changed")
            self._operation_owner_thread_id = None
            self._operation_name = None
            self._operation_phase = None
            self._operation_condition.notify_all()

    def _fork(
        self,
        request: ScenarioForkRequest,
        capability: ScenarioCommandCapability,
        child_database_path: str | Path,
    ) -> tuple["ScenarioCoordinator", ScenarioForkResult]:
        attempt = self._fork_attempts_by_idempotency_key.get(
            request.idempotency_key
        )
        registered_attempt = attempt is not None
        if attempt is not None:
            fork_id, request_hash, capability_hash = attempt
            if (
                fork_id == request.fork_id
                and request_hash == request.content_hash
                and capability_hash == capability.content_hash
            ):
                cached = self._fork_results.get(request.idempotency_key)
                if cached is not None:
                    return cached
            else:
                raise ValueError("scenario fork idempotency key was reused")
        elif request.fork_id in self._fork_attempts_by_fork_id:
            raise ValueError("scenario fork id was reused")

        if not capability.can_fork or capability.run_id != self._run_id:
            raise PermissionError("scenario fork is not authorized")
        if request.source_run_id != self._run_id:
            raise ValueError("scenario fork source run does not match")
        if request.scenario_hash != self._scenario.content_hash:
            raise ValueError("scenario fork scenario does not match")
        if request.source_epoch != self._coordinator_epoch:
            raise ValueError("scenario fork source epoch does not match")
        if request.child_run_id == self._run_id:
            raise ValueError("scenario fork child run must differ from source run")
        if request.child_stream_id == self._stream_id:
            raise ValueError("scenario fork child stream must differ from source stream")
        if request.child_run_id in self._child_run_ids:
            raise ValueError("scenario fork child run already exists")
        if request.child_stream_id in self._child_stream_ids:
            raise ValueError("scenario fork child stream already exists")
        if self._checkpoint_store is None:
            raise RuntimeError("scenario checkpoint store is not configured")

        retained = next(
            (
                checkpoint
                for checkpoint in self._checkpoints
                if checkpoint.content_hash == request.checkpoint_hash
            ),
            None,
        )
        if retained is None:
            raise KeyError("unknown scenario checkpoint")
        checkpoint = self._checkpoint_store.load(request.checkpoint_hash)
        if checkpoint != retained:
            raise ValueError("scenario fork checkpoint metadata does not match")
        if checkpoint.run_id != request.source_run_id:
            raise ValueError("scenario fork checkpoint source run does not match")
        if checkpoint.scenario_hash != request.scenario_hash:
            raise ValueError("scenario fork checkpoint scenario does not match")
        if checkpoint.coordinator_epoch != request.source_epoch:
            raise ValueError("scenario fork checkpoint source epoch does not match")

        if not registered_attempt:
            if (
                len(self._fork_attempts_by_idempotency_key)
                >= self._fork_history_limit
            ):
                raise ValueError(_FORK_HISTORY_LIMIT_ERROR)
            self._fork_attempts_by_idempotency_key[request.idempotency_key] = (
                request.fork_id,
                request.content_hash,
                capability.content_hash,
            )
            self._fork_attempts_by_fork_id[request.fork_id] = (
                request.idempotency_key
            )

        child_path = _database_path(child_database_path)
        child_path_value = Path(child_path)
        child_ownership_token: _PhysicalFileOwnershipToken | None = None
        try:
            child_ownership_token = self._checkpoint_store._restore_owned(
                request.checkpoint_hash,
                child_path,
            )
            self._checkpoint_store._verify_restored_target(
                child_path,
                child_ownership_token,
            )
            if (
                hash_situated_percept_memory_store(child_path)
                != checkpoint.memory_store_hash
            ):
                raise ValueError("scenario fork restored memory hash does not match")
            self._checkpoint_store._verify_restored_target(
                child_path,
                child_ownership_token,
            )
            child_state_store = InMemoryScenarioStateStore()
            child_state_store.initialize(request.child_run_id, checkpoint.state)
            child = ScenarioCoordinator(
                child_path,
                self._scenario,
                run_id=request.child_run_id,
                stream_id=request.child_stream_id,
                state_store=child_state_store,
                publisher=SimulationOutputBus(),
                checkpoint_store=self._checkpoint_store,
                coordinator_epoch=self._coordinator_epoch + 1,
                parent_checkpoint_hash=checkpoint.content_hash,
                status=ScenarioRunStatus.PAUSED,
                next_sequence=1,
            )
            result = ScenarioForkResult(
                request.fork_id,
                request.idempotency_key,
                request.content_hash,
                capability.content_hash,
                self._run_id,
                request.child_run_id,
                request.child_stream_id,
                self._scenario.content_hash,
                self._coordinator_epoch + 1,
                checkpoint.content_hash,
                checkpoint.state.content_hash,
            )
            self._checkpoint_store._verify_restored_target(
                child_path,
                child_ownership_token,
            )
        except _ScenarioCheckpointRestoreTargetExistsError:
            raise
        except _ScenarioCheckpointOwnedRestoreError as error:
            _cleanup_fork_database(
                self._checkpoint_store,
                child_path_value,
                error.ownership_token,
            )
            raise RuntimeError(str(error)) from None
        except Exception:
            if child_ownership_token is not None:
                _cleanup_fork_database(
                    self._checkpoint_store,
                    child_path_value,
                    child_ownership_token,
                )
            raise

        retained_result = (child, result)
        self._fork_results[request.idempotency_key] = retained_result
        self._child_run_ids.add(request.child_run_id)
        self._child_stream_ids.add(request.child_stream_id)
        return retained_result

    def _submit_command(
        self,
        request: ScenarioCommandRequest,
        capability: ScenarioCommandCapability,
    ) -> tuple[ScenarioCommandResult, SimulationOutputBatch | None]:
        attempt = self._attempts_by_idempotency_key.get(request.idempotency_key)
        if attempt is not None:
            command_id, request_hash, capability_hash = attempt
            if (
                command_id == request.command_id
                and request_hash == request.content_hash
                and capability_hash == capability.content_hash
            ):
                cached = self._idempotency_results.get(request.idempotency_key)
                if cached is not None:
                    return cached[2], None
            else:
                raise ValueError("scenario command idempotency key was reused")
        elif request.command_id in self._attempts_by_command_id:
            raise ValueError("scenario command id was reused")
        else:
            if (
                len(self._attempts_by_idempotency_key)
                >= self._command_history_limit
            ):
                raise ValueError(_COMMAND_HISTORY_LIMIT_ERROR)
            self._attempts_by_idempotency_key[request.idempotency_key] = (
                request.command_id,
                request.content_hash,
                capability.content_hash,
            )
            self._attempts_by_command_id[request.command_id] = (
                request.idempotency_key
            )

        state = self.state
        rejection = self._rejection_reason(request, capability, state)
        if rejection is not None:
            result = self._result(
                request,
                capability,
                accepted=False,
                reason=rejection,
                prior_status=self._status,
                next_status=self._status,
                prior_state=state,
                next_state=state,
            )
            self._retain_result(request, capability, result)
            return result, None

        if request.kind is ScenarioCommandKind.STEP:
            return self._step(request, capability, state)
        if request.kind is ScenarioCommandKind.CHECKPOINT:
            return self._checkpoint(request, capability, state), None

        next_status = {
            ScenarioCommandKind.START: ScenarioRunStatus.RUNNING,
            ScenarioCommandKind.PAUSE: ScenarioRunStatus.PAUSED,
            ScenarioCommandKind.RESUME: ScenarioRunStatus.RUNNING,
            ScenarioCommandKind.STOP: ScenarioRunStatus.STOPPED,
        }[request.kind]
        result = self._result(
            request,
            capability,
            accepted=True,
            reason=ScenarioCommandReason.ACCEPTED,
            prior_status=self._status,
            next_status=next_status,
            prior_state=state,
            next_state=state,
        )
        self._status = next_status
        self._retain_result(request, capability, result)
        return result, None

    def _checkpoint(
        self,
        request: ScenarioCommandRequest,
        capability: ScenarioCommandCapability,
        state: SituatedNetworkRuntimeState,
    ) -> ScenarioCommandResult:
        if self._checkpoint_store is None:
            raise RuntimeError("scenario checkpoint store is not configured")
        checkpoint_id = request.requested_checkpoint_id
        if checkpoint_id is not None and re.fullmatch(
            rf"{re.escape(self._run_id)}-round-[1-9][0-9]*",
            checkpoint_id,
        ) is not None:
            raise ValueError(
                "scenario checkpoint id is reserved for automatic checkpoints"
            )
        checkpoint = self._build_checkpoint(
            checkpoint_id,
            state,
            self._next_sequence,
        )
        stored = self._checkpoint_store.create(checkpoint, self._database_path)
        result = self._result(
            request,
            capability,
            accepted=True,
            reason=ScenarioCommandReason.ACCEPTED,
            prior_status=self._status,
            next_status=self._status,
            prior_state=state,
            next_state=state,
            checkpoint_hash=stored.content_hash,
        )
        self._checkpoints = self._checkpoints + (stored,)
        self._retain_result(request, capability, result)
        return result

    def _build_checkpoint(
        self,
        checkpoint_id: str | None,
        state: SituatedNetworkRuntimeState,
        next_sequence: int,
    ) -> ScenarioCheckpoint:
        if checkpoint_id is None:
            raise ValueError("scenario checkpoint id must be provided")
        parent_hash = (
            self._checkpoints[-1].content_hash
            if self._checkpoints
            else self._parent_checkpoint_hash
        )
        return ScenarioCheckpoint(
            checkpoint_id,
            self._run_id,
            self._scenario.content_hash,
            self._coordinator_epoch,
            state,
            next_sequence,
            parent_hash,
        )

    def _rejection_reason(
        self,
        request: ScenarioCommandRequest,
        capability: ScenarioCommandCapability,
        state: SituatedNetworkRuntimeState,
    ) -> ScenarioCommandReason | None:
        if (
            capability.run_id != self._run_id
            or capability.authority_id != request.authority_id
            or request.kind not in capability.allowed_kinds
        ):
            return ScenarioCommandReason.UNAUTHORIZED
        if request.run_id != self._run_id:
            return ScenarioCommandReason.RUN_MISMATCH
        if request.scenario_hash != self._scenario.content_hash:
            return ScenarioCommandReason.SCENARIO_MISMATCH
        if request.coordinator_epoch != self._coordinator_epoch:
            return ScenarioCommandReason.EPOCH_MISMATCH
        if request.expected_state_hash != state.content_hash:
            return ScenarioCommandReason.STALE_STATE
        if (
            request.kind is ScenarioCommandKind.STEP
            and self._status is ScenarioRunStatus.COMPLETED
        ):
            return ScenarioCommandReason.MAXIMUM_ROUNDS
        if not self._status_allows(request.kind):
            return ScenarioCommandReason.INVALID_STATUS
        if (
            request.kind is ScenarioCommandKind.STEP
            and state.round_index >= self._scenario.run_policy.maximum_rounds
        ):
            return ScenarioCommandReason.MAXIMUM_ROUNDS
        return None

    def _status_allows(self, kind: ScenarioCommandKind) -> bool:
        if kind is ScenarioCommandKind.START:
            return self._status is ScenarioRunStatus.CREATED
        if kind is ScenarioCommandKind.PAUSE:
            return self._status is ScenarioRunStatus.RUNNING
        if kind is ScenarioCommandKind.RESUME:
            return self._status is ScenarioRunStatus.PAUSED
        if kind is ScenarioCommandKind.STEP:
            return self._status in (
                ScenarioRunStatus.RUNNING,
                ScenarioRunStatus.PAUSED,
            )
        if kind is ScenarioCommandKind.STOP:
            return self._status in (
                ScenarioRunStatus.CREATED,
                ScenarioRunStatus.RUNNING,
                ScenarioRunStatus.PAUSED,
            )
        return self._status is not ScenarioRunStatus.STOPPED

    def _step(
        self,
        request: ScenarioCommandRequest,
        capability: ScenarioCommandCapability,
        prior_state: SituatedNetworkRuntimeState,
    ) -> tuple[ScenarioCommandResult, SimulationOutputBatch]:
        with TemporaryDirectory(
            prefix="scenario-coordinator-v21-3-",
            ignore_cleanup_errors=True,
        ) as temporary:
            backup_path = str(Path(temporary) / "memory-backup.sqlite3")
            _copy_sqlite_database(self._database_path, backup_path)
            committed = False
            prepared_checkpoint: ScenarioCheckpoint | None = None
            try:
                round_result = simulate_situated_network_round(
                    self._database_path,
                    self._scenario.runtime_model,
                    prior_state,
                )
                batch = self._project_step_batch(request, round_result)
                self._enforce_output_record_limit(batch)

                next_state = round_result.next_state
                if (
                    self._checkpoint_store is not None
                    and next_state.round_index
                    % self._scenario.run_policy.checkpoint_interval
                    == 0
                ):
                    batch = replace(batch, checkpoint=True)
                    prepared_checkpoint = self._build_checkpoint(
                        f"{self._run_id}-round-{next_state.round_index}",
                        next_state,
                        batch.last_sequence + 1,
                    )
                    prepared_checkpoint = self._checkpoint_store.create(
                        prepared_checkpoint,
                        self._database_path,
                    )
                next_status = self._status
                if (
                    next_state.round_index
                    >= self._scenario.run_policy.maximum_rounds
                ):
                    next_status = ScenarioRunStatus.COMPLETED
                result = self._result(
                    request,
                    capability,
                    accepted=True,
                    reason=ScenarioCommandReason.ACCEPTED,
                    prior_status=self._status,
                    next_status=next_status,
                    prior_state=prior_state,
                    next_state=next_state,
                    output_batch_hash=batch.content_hash,
                    checkpoint_hash=(
                        None
                        if prepared_checkpoint is None
                        else prepared_checkpoint.content_hash
                    ),
                )

                next_outputs = self._output_batches + (batch,)
                next_outputs_by_hash = dict(self._outputs_by_hash)
                next_outputs_by_hash[batch.content_hash] = batch
                next_results = dict(self._results_by_command_id)
                next_results[request.command_id] = (
                    capability.authority_id,
                    result,
                )
                next_idempotency = dict(self._idempotency_results)
                next_idempotency[request.idempotency_key] = (
                    request.content_hash,
                    capability.content_hash,
                    result,
                )
                next_checkpoints = self._checkpoints + (
                    ()
                    if prepared_checkpoint is None
                    else (prepared_checkpoint,)
                )

                self._state_store.compare_and_swap(
                    self._run_id,
                    prior_state.content_hash,
                    next_state,
                )
                self._status = next_status
                self._output_batches = next_outputs
                self._outputs_by_hash = next_outputs_by_hash
                self._results_by_command_id = next_results
                self._idempotency_results = next_idempotency
                self._checkpoints = next_checkpoints
                self._next_sequence = batch.last_sequence + 1
                committed = True

                return result, batch
            except Exception:
                if not committed:
                    discard_error: Exception | None = None
                    if prepared_checkpoint is not None:
                        try:
                            self._checkpoint_store.discard(
                                prepared_checkpoint.content_hash
                            )
                        except Exception as error:
                            discard_error = error
                    _copy_sqlite_database(backup_path, self._database_path)
                    if discard_error is not None:
                        raise RuntimeError(
                            "scenario automatic checkpoint rollback failed"
                        ) from discard_error
                raise

    def _publish(self, batch: SimulationOutputBatch) -> SimulationDeliveryReport:
        try:
            report = self._publisher.publish(batch)
        except Exception:
            report = None
        if (
            not isinstance(report, SimulationDeliveryReport)
            or report.batch_hash != batch.content_hash
        ):
            return SimulationDeliveryReport(
                batch.content_hash,
                (),
                (
                    SimulationDeliveryFailure(
                        _PUBLISHER_FAILURE_SUBSCRIPTION_ID,
                        "callback_error",
                    ),
                ),
            )
        return report

    def _enforce_output_record_limit(self, batch: SimulationOutputBatch) -> None:
        retained_count = sum(len(item.records) for item in self._output_batches)
        if (
            retained_count + len(batch.records)
            > self._scenario.run_policy.maximum_output_records
        ):
            raise ValueError(_OUTPUT_LIMIT_ERROR)

    def _project_step_batch(
        self,
        request: ScenarioCommandRequest,
        round_result: SituatedNetworkRoundResult,
    ) -> SimulationOutputBatch:
        include_command = (
            SimulationOutputKind.COMMAND_RESULT.value
            in self._scenario.run_policy.allowed_output_kinds
        )
        base_batch: SimulationOutputBatch | None
        try:
            base_batch = project_simulation_output(
                self._scenario,
                round_result,
                stream_id=self._stream_id,
                first_sequence=self._next_sequence,
            )
        except ValueError as error:
            if not include_command or str(error) != _EMPTY_PROJECTION_ERROR:
                raise
            base_batch = None

        records = () if base_batch is None else base_batch.records
        if include_command:
            command_record = SimulationOutputRecord(
                self._stream_id,
                self._scenario.content_hash,
                self._next_sequence,
                round_result.next_state.round_index,
                round_result.next_state.content_hash,
                SimulationOutputKind.COMMAND_RESULT,
                SimulationOutputAudience.PUBLIC,
                None,
                (
                    request.content_hash,
                    round_result.prior_state.content_hash,
                    round_result.next_state.content_hash,
                    round_result.content_hash,
                ),
                SimulationCommandResultPayload(
                    request.command_id,
                    True,
                    ScenarioCommandReason.ACCEPTED.value,
                ),
            )
            records = (command_record,) + records
        if not records:
            raise ValueError(_EMPTY_PROJECTION_ERROR)

        ordered = tuple(sorted(records, key=output_record_sort_key))
        resequenced = tuple(
            replace(record, sequence=self._next_sequence + index)
            for index, record in enumerate(ordered)
        )
        return SimulationOutputBatch(
            self._stream_id,
            self._scenario.content_hash,
            round_result.prior_state.content_hash,
            round_result.next_state.content_hash,
            round_result.content_hash,
            self._next_sequence,
            self._next_sequence + len(resequenced) - 1,
            resequenced,
            checkpoint=False,
        )

    def _result(
        self,
        request: ScenarioCommandRequest,
        capability: ScenarioCommandCapability,
        *,
        accepted: bool,
        reason: ScenarioCommandReason,
        prior_status: ScenarioRunStatus,
        next_status: ScenarioRunStatus,
        prior_state: SituatedNetworkRuntimeState,
        next_state: SituatedNetworkRuntimeState,
        output_batch_hash: str | None = None,
        checkpoint_hash: str | None = None,
    ) -> ScenarioCommandResult:
        return ScenarioCommandResult(
            request.command_id,
            request.idempotency_key,
            request.content_hash,
            capability.content_hash,
            self._run_id,
            self._scenario.content_hash,
            self._coordinator_epoch,
            request.kind,
            accepted,
            reason,
            prior_status,
            next_status,
            prior_state.content_hash,
            next_state.content_hash,
            next_state.round_index,
            output_batch_hash,
            checkpoint_hash,
        )

    def _retain_result(
        self,
        request: ScenarioCommandRequest,
        capability: ScenarioCommandCapability,
        result: ScenarioCommandResult,
    ) -> None:
        self._results_by_command_id[request.command_id] = (
            capability.authority_id,
            result,
        )
        self._idempotency_results[request.idempotency_key] = (
            request.content_hash,
            capability.content_hash,
            result,
        )


__all__ = ("ScenarioCoordinator",)
