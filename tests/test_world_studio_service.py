from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from threading import Event, Lock

import pytest

from narrative_dynamics.abm import (
    LocalScenarioCheckpointStore,
    ScenarioCommandCapability,
    ScenarioCoordinator,
    ScenarioForkRequest,
)
from narrative_dynamics.studio import (
    InMemoryScenarioRunRegistry,
    ScenarioProjectWorkspace,
    StudioAuthorizationError,
    StudioCapability,
    StudioConflictError,
    StudioRunLifecycleError,
    StudioStaleStateError,
    WorldStudioService,
)


LAW_FIRM = Path("examples/law_firm_scenario").resolve()
ALL_PERMISSIONS = (
    "project.create",
    "project.read",
    "project.write",
    "scenario.compile",
    "run.create",
    "run.command",
    "run.fork",
    "run.read",
    "state.public",
    "state.agent",
    "state.network",
    "output.read",
    "command.read",
    "command.audit",
)


class LocalCoordinatorFactory:
    """Server-owned I/O seam that still creates real V21.3 coordinators."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._ownership_lock = Lock()
        self._owned: dict[str, ScenarioCoordinator | None] = {}
        self._created_instances: list[ScenarioCoordinator] = []
        self._aborted_creates = 0
        self._aborted_forks = 0

    @property
    def owned_run_ids(self) -> tuple[str, ...]:
        with self._ownership_lock:
            return tuple(sorted(self._owned))

    @property
    def created_instance_count(self) -> int:
        with self._ownership_lock:
            return len(self._created_instances)

    @property
    def aborted_create_count(self) -> int:
        with self._ownership_lock:
            return self._aborted_creates

    @property
    def aborted_fork_count(self) -> int:
        with self._ownership_lock:
            return self._aborted_forks

    def _own(
        self, run_id: str, coordinator: ScenarioCoordinator | None
    ) -> None:
        with self._ownership_lock:
            self._owned[run_id] = coordinator
            if coordinator is not None:
                self._created_instances.append(coordinator)

    def _discard(self, run_id: str) -> None:
        with self._ownership_lock:
            self._owned.pop(run_id, None)
        database = self._root / f"{run_id}.sqlite3"
        database.unlink(missing_ok=True)
        shutil.rmtree(
            self._root / f"{run_id}-checkpoints",
            ignore_errors=True,
        )

    def create(
        self,
        scenario,
        *,
        project_id: str,
        run_id: str,
        stream_id: str,
    ) -> ScenarioCoordinator:
        del project_id
        coordinator = ScenarioCoordinator.create(
            self._root / f"{run_id}.sqlite3",
            scenario,
            run_id=run_id,
            stream_id=stream_id,
            checkpoint_store=LocalScenarioCheckpointStore(
                self._root / f"{run_id}-checkpoints"
            ),
        )
        self._own(run_id, coordinator)
        return coordinator

    def fork(
        self,
        coordinator: ScenarioCoordinator,
        request: ScenarioForkRequest,
        capability: ScenarioCommandCapability,
    ):
        child, result = coordinator.fork(
            request,
            capability,
            self._root / f"{request.child_run_id}.sqlite3",
        )
        self._own(request.child_run_id, child)
        return child, result

    def abort_create(
        self,
        *,
        project_id: str,
        run_id: str,
        stream_id: str,
    ) -> None:
        del project_id, stream_id
        with self._ownership_lock:
            self._aborted_creates += 1
        self._discard(run_id)

    def abort_fork(
        self,
        coordinator: ScenarioCoordinator,
        request: ScenarioForkRequest,
    ) -> None:
        del coordinator
        with self._ownership_lock:
            self._aborted_forks += 1
        self._discard(request.child_run_id)


def _capability(
    *,
    project_ids: tuple[str, ...] = ("law-firm",),
    run_ids: tuple[str, ...] = ("run-1", "run-child"),
    agent_ids: tuple[str, ...] = ("alice",),
    permissions: tuple[str, ...] = ALL_PERMISSIONS,
) -> StudioCapability:
    return StudioCapability(
        "operator",
        project_ids=project_ids,
        run_ids=run_ids,
        agent_ids=agent_ids,
        permissions=permissions,
    )


def _service(
    tmp_path: Path,
    *,
    factory: LocalCoordinatorFactory | None = None,
    registry: InMemoryScenarioRunRegistry | None = None,
):
    workspace = ScenarioProjectWorkspace.open(
        tmp_path / "studio.sqlite3",
        import_roots=(LAW_FIRM.parent,),
        export_root=tmp_path,
    )
    registry = registry or InMemoryScenarioRunRegistry()
    service = WorldStudioService(
        workspace,
        registry,
        factory or LocalCoordinatorFactory(tmp_path),
        import_sources={"law-firm-fixture": LAW_FIRM},
    )
    return service, registry


def _create_and_import(service: WorldStudioService, capability: StudioCapability):
    created = service.invoke("project.create", {"project_id": "law-firm"}, capability)
    return service.invoke(
        "project.import",
        {
            "project_id": "law-firm",
            "source_id": "law-firm-fixture",
            "expected_revision": created["revision"],
            "expected_snapshot_hash": created["content_hash"],
        },
        capability,
    )


def _command(
    service: WorldStudioService,
    capability: StudioCapability,
    kind: str,
    number: int,
    *,
    requested_checkpoint_id: str | None = None,
):
    view = service.invoke("run.view", {"run_id": "run-1"}, capability)
    params = {
        "command_id": f"command-{number}",
        "idempotency_key": f"command-key-{number}",
        "run_id": "run-1",
        "scenario_hash": view["scenario_hash"],
        "coordinator_epoch": view["coordinator_epoch"],
        "expected_state_hash": view["state_hash"],
        "kind": kind,
    }
    if requested_checkpoint_id is not None:
        params["requested_checkpoint_id"] = requested_checkpoint_id
    return service.invoke("run.command", params, capability)


def test_capability_is_canonical_content_addressed_and_closed() -> None:
    capability = StudioCapability(
        "operator",
        project_ids=("z", "law-firm", "law-firm"),
        run_ids=("run-2", "run-1", "run-1"),
        agent_ids=("bob", "alice", "alice"),
        permissions=("state.public", "project.read", "state.public"),
    )
    reordered = StudioCapability(
        "operator",
        project_ids=("law-firm", "z"),
        run_ids=("run-1", "run-2"),
        agent_ids=("alice", "bob"),
        permissions=("project.read", "state.public"),
    )

    assert capability.project_ids == ("law-firm", "z")
    assert capability.run_ids == ("run-1", "run-2")
    assert capability.agent_ids == ("alice", "bob")
    assert capability.permissions == ("project.read", "state.public")
    assert capability.content_hash == reordered.content_hash
    assert hash(capability) == hash(reordered)
    with pytest.raises(ValueError, match="permission"):
        StudioCapability("operator", permissions=("filesystem.read",))


def test_project_and_scenario_surface_uses_real_workspace(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    capability = _capability()
    imported = _create_and_import(service, capability)

    snapshot = service.invoke(
        "project.snapshot", {"project_id": "law-firm"}, capability
    )
    report = service.invoke(
        "scenario.validate", {"project_id": "law-firm"}, capability
    )
    compiled = service.invoke(
        "scenario.compile",
        {
            "project_id": "law-firm",
            "expected_revision": snapshot["revision"],
            "expected_snapshot_hash": snapshot["content_hash"],
        },
        capability,
    )
    changed = service.invoke(
        "project.apply",
        {
            "operation_id": "operation-1",
            "idempotency_key": "operation-key-1",
            "project_id": "law-firm",
            "expected_revision": snapshot["revision"],
            "expected_snapshot_hash": snapshot["content_hash"],
            "document_role": "physical.world",
            "logical_id": None,
            "kind": "set_value",
            "pointer": "/places/0/label",
            "value": "Reception",
        },
        capability,
    )
    undone = service.invoke(
        "project.undo",
        {
            "project_id": "law-firm",
            "expected_revision": changed["next_snapshot"]["revision"],
            "expected_snapshot_hash": changed["next_snapshot"]["content_hash"],
        },
        capability,
    )
    redone = service.invoke(
        "project.redo",
        {
            "project_id": "law-firm",
            "expected_revision": undone["revision"],
            "expected_snapshot_hash": undone["content_hash"],
        },
        capability,
    )
    exported = service.invoke(
        "project.export",
        {"project_id": "law-firm", "target_name": "law-firm-release"},
        capability,
    )

    assert imported["scenario_id"] == "law-firm-case"
    assert snapshot == imported
    assert report["diagnostics"] == []
    assert compiled["scenario_hash"] == imported["compiled_scenario_hash"]
    assert changed["prior_revision"] == imported["revision"]
    assert undone["revision"] == changed["next_snapshot"]["revision"] + 1
    assert redone["revision"] == undone["revision"] + 1
    assert exported["project_id"] == "law-firm"
    assert (tmp_path / exported["target_name"]).is_dir()


def test_real_coordinator_lifecycle_and_scoped_queries(tmp_path: Path) -> None:
    service, registry = _service(tmp_path)
    capability = _capability()
    imported = _create_and_import(service, capability)
    created = service.invoke(
        "run.create",
        {
            "project_id": "law-firm",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
        },
        capability,
    )

    started = _command(service, capability, "start", 1)
    stepped = _command(service, capability, "step", 2)
    public = service.invoke("state.public", {"run_id": "run-1"}, capability)
    agent = service.invoke(
        "state.agent", {"run_id": "run-1", "agent_id": "alice"}, capability
    )
    network = service.invoke("state.network", {"run_id": "run-1"}, capability)
    output = service.invoke(
        "output.get",
        {"run_id": "run-1", "batch_hash": stepped["output_batch_hash"]},
        capability,
    )
    audit = service.invoke(
        "command.get", {"run_id": "run-1", "command_id": "command-2"}, capability
    )
    checkpointed = _command(
        service,
        capability,
        "checkpoint",
        3,
        requested_checkpoint_id="checkpoint-1",
    )
    parent = service.invoke("run.view", {"run_id": "run-1"}, capability)
    fork_params = {
        "fork_id": "fork-1",
        "idempotency_key": "fork-key-1",
        "source_run_id": "run-1",
        "scenario_hash": parent["scenario_hash"],
        "source_epoch": parent["coordinator_epoch"],
        "checkpoint_hash": checkpointed["checkpoint_hash"],
        "child_run_id": "run-child",
        "child_stream_id": "stream-child",
    }
    forked = service.invoke("run.fork", fork_params, capability)
    repeated_fork = service.invoke("run.fork", fork_params, capability)

    assert created["status"] == "created"
    assert started["accepted"] is True
    assert stepped["round_index"] == 1
    assert public["round_index"] == 1
    assert {item[0] for item in public["agent_places"]} >= {"alice", "bob"}
    assert agent["agent_id"] == "alice"
    assert agent["mind"]["agent_id"] == "alice"
    assert network["round_index"] == 1
    assert network["schema"] == "narrative-dynamics.scenario-network-state-view/v1"
    assert network["run_id"] == "run-1"
    assert network["scenario_hash"] == public["scenario_hash"]
    assert network["state_hash"] == public["state_hash"]
    assert output["source_batch_hash"] == stepped["output_batch_hash"]
    assert all(record["owner_agent_id"] is None for record in output["records"])
    assert audit == stepped
    assert checkpointed["checkpoint_hash"] in parent["checkpoint_hashes"]
    assert forked["child_run_id"] == "run-child"
    assert repeated_fork == forked
    assert service.invoke("run.view", {"run_id": "run-child"}, capability)[
        "parent_checkpoint_hash"
    ] == checkpointed["checkpoint_hash"]
    assert [item["run_id"] for item in service.invoke("run.list", {}, capability)] == [
        "run-1",
        "run-child",
    ]
    assert registry.resolve("run-child").run_view().run_id == "run-child"


def test_capabilities_are_narrowed_before_private_queries(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    full = _capability()
    imported = _create_and_import(service, full)
    service.invoke(
        "run.create",
        {
            "project_id": "law-firm",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
        },
        full,
    )
    _command(service, full, "start", 1)
    step = _command(service, full, "step", 2)

    public_output = service.invoke(
        "output.get",
        {"run_id": "run-1", "batch_hash": step["output_batch_hash"]},
        _capability(permissions=("output.read",)),
    )
    owner_output = service.invoke(
        "output.get",
        {
            "run_id": "run-1",
            "batch_hash": step["output_batch_hash"],
            "agent_id": "alice",
        },
        _capability(permissions=("output.read",), agent_ids=("alice",)),
    )

    assert all(item["audience"] == "public" for item in public_output["records"])
    assert all(
        item["audience"] == "public" or item["owner_agent_id"] == "alice"
        for item in owner_output["records"]
    )
    with pytest.raises(StudioAuthorizationError):
        service.invoke(
            "state.agent",
            {"run_id": "run-1", "agent_id": "bob"},
            _capability(
                agent_ids=("alice",),
                permissions=("state.agent",),
            ),
        )
    with pytest.raises(StudioAuthorizationError):
        service.invoke(
            "state.network",
            {"run_id": "run-1"},
            _capability(permissions=("state.public",)),
        )


def test_authorization_precedes_lookup_for_known_and_unknown_ids(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    full = _capability()
    imported = _create_and_import(service, full)
    service.invoke(
        "run.create",
        {
            "project_id": "law-firm",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
        },
        full,
    )
    denied = _capability(run_ids=(), permissions=("run.read",))

    errors = []
    for run_id in ("run-1", "unknown-run"):
        with pytest.raises(StudioAuthorizationError) as caught:
            service.invoke("run.view", {"run_id": run_id}, denied)
        errors.append((type(caught.value), caught.value.args))
    assert errors[0] == errors[1]

    project_denied = _capability(project_ids=(), permissions=("project.read",))
    project_errors = []
    for project_id in ("law-firm", "unknown-project"):
        with pytest.raises(StudioAuthorizationError) as caught:
            service.invoke("project.snapshot", {"project_id": project_id}, project_denied)
        project_errors.append((type(caught.value), caught.value.args))
    assert project_errors[0] == project_errors[1]


def test_stale_run_create_rejects_before_factory_io_and_registration(
    tmp_path: Path,
) -> None:
    service, registry = _service(tmp_path)
    capability = _capability()
    imported = _create_and_import(service, capability)

    with pytest.raises(StudioStaleStateError):
        service.invoke(
            "run.create",
            {
                "project_id": "law-firm",
                "run_id": "run-1",
                "stream_id": "stream-1",
                "expected_revision": imported["revision"] - 1,
                "expected_snapshot_hash": imported["content_hash"],
            },
            capability,
        )

    assert not (tmp_path / "run-1.sqlite3").exists()
    with pytest.raises(KeyError):
        registry.resolve("run-1")


def test_registry_is_no_overwrite_and_lists_stably(tmp_path: Path) -> None:
    service, registry = _service(tmp_path)
    capability = _capability(run_ids=("run-z", "run-a"))
    imported = _create_and_import(service, capability)
    for run_id in ("run-z", "run-a"):
        service.invoke(
            "run.create",
            {
                "project_id": "law-firm",
                "run_id": run_id,
                "stream_id": f"stream-{run_id}",
                "expected_revision": imported["revision"],
                "expected_snapshot_hash": imported["content_hash"],
            },
            capability,
        )

    assert [view.run_id for view in registry.list_for(capability)] == ["run-a", "run-z"]
    with pytest.raises(StudioConflictError):
        service.invoke(
            "run.create",
            {
                "project_id": "law-firm",
                "run_id": "run-a",
                "stream_id": "another-stream",
                "expected_revision": imported["revision"],
                "expected_snapshot_hash": imported["content_hash"],
            },
            capability,
        )


def test_command_identity_reuse_is_an_application_conflict(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    capability = _capability()
    imported = _create_and_import(service, capability)
    view = service.invoke(
        "run.create",
        {
            "project_id": "law-firm",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
        },
        capability,
    )
    request = {
        "command_id": "command-1",
        "idempotency_key": "shared-key",
        "run_id": "run-1",
        "scenario_hash": view["scenario_hash"],
        "coordinator_epoch": view["coordinator_epoch"],
        "expected_state_hash": view["state_hash"],
        "kind": "pause",
    }
    rejected = service.invoke("run.command", request, capability)

    assert rejected["accepted"] is False
    assert service.invoke("run.command", request, capability) == rejected
    with pytest.raises(StudioConflictError):
        service.invoke(
            "run.command",
            {**request, "command_id": "command-2"},
            capability,
        )


class BlockingCreateFactory(LocalCoordinatorFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.first_entered = Event()
        self.second_entered = Event()
        self.release = Event()
        self._entry_lock = Lock()
        self._entries = 0

    def create(self, scenario, *, project_id: str, run_id: str, stream_id: str):
        with self._entry_lock:
            self._entries += 1
            entry = self._entries
        if entry == 1:
            self.first_entered.set()
            assert self.release.wait(5)
        else:
            self.second_entered.set()
        return super().create(
            scenario,
            project_id=project_id,
            run_id=run_id,
            stream_id=stream_id,
        )


class BlockingForkFactory(LocalCoordinatorFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.first_fork_entered = Event()
        self.second_fork_entered = Event()
        self.release_fork = Event()
        self._fork_entry_lock = Lock()
        self._fork_entries = 0

    def fork(self, coordinator, request, capability):
        with self._fork_entry_lock:
            self._fork_entries += 1
            entry = self._fork_entries
        if entry == 1:
            self.first_fork_entered.set()
            assert self.release_fork.wait(5)
        else:
            self.second_fork_entered.set()
        return super().fork(coordinator, request, capability)


class PartialCreateFailureFactory(LocalCoordinatorFactory):
    def create(self, scenario, *, project_id: str, run_id: str, stream_id: str):
        super().create(
            scenario,
            project_id=project_id,
            run_id=run_id,
            stream_id=stream_id,
        )
        raise RuntimeError("factory failed after acquiring the run database")


class PartialForkFailureFactory(LocalCoordinatorFactory):
    def fork(self, coordinator, request, capability):
        database = self._root / f"{request.child_run_id}.sqlite3"
        database.write_bytes(b"partial factory resource")
        self._own(request.child_run_id, None)
        raise RuntimeError("factory failed before coordinator fork")


class NoCheckpointFactory(LocalCoordinatorFactory):
    def create(self, scenario, *, project_id: str, run_id: str, stream_id: str):
        del project_id
        coordinator = ScenarioCoordinator.create(
            self._root / f"{run_id}.sqlite3",
            scenario,
            run_id=run_id,
            stream_id=stream_id,
        )
        self._own(run_id, coordinator)
        return coordinator


class PublishThenRaiseRegistry(InMemoryScenarioRunRegistry):
    """Exercises a registry boundary that reports failure after atomic publication."""

    def __init__(self, *run_ids: str) -> None:
        super().__init__()
        self._throw_run_ids = frozenset(run_ids)

    def commit(self, reservation, coordinator, *, outcome=None) -> None:
        super().commit(reservation, coordinator, outcome=outcome)
        if reservation.run_id in self._throw_run_ids:
            super().commit(reservation, coordinator, outcome=outcome)
            raise RuntimeError("registry transport failed after publication")


class ObservedWaitingRegistry(InMemoryScenarioRunRegistry):
    def __init__(self, target_run_id: str) -> None:
        super().__init__()
        self._target_run_id = target_run_id
        self._target_lock = Lock()
        self._target_reservations = 0
        self.second_reservation_entered = Event()

    def reserve(self, run_id: str, request_hash: str):
        if run_id == self._target_run_id:
            with self._target_lock:
                self._target_reservations += 1
                if self._target_reservations == 2:
                    self.second_reservation_entered.set()
        return super().reserve(run_id, request_hash)


class CleanupFailingCreateFactory(LocalCoordinatorFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.create_entered = Event()
        self.release_create = Event()
        self.create_calls = 0
        self.cleanup_calls = 0

    def create(self, scenario, *, project_id: str, run_id: str, stream_id: str):
        del scenario, project_id, stream_id
        with self._ownership_lock:
            self.create_calls += 1
        (self._root / f"{run_id}.sqlite3").write_bytes(b"partial create resource")
        self._own(run_id, None)
        self.create_entered.set()
        assert self.release_create.wait(5)
        raise RuntimeError("create failed after its side effect")

    def abort_create(self, *, project_id: str, run_id: str, stream_id: str) -> None:
        del project_id, run_id, stream_id
        with self._ownership_lock:
            self.cleanup_calls += 1
        raise RuntimeError("create cleanup failed")


class CleanupFailingForkFactory(LocalCoordinatorFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.fork_entered = Event()
        self.release_fork = Event()
        self.fork_calls = 0
        self.cleanup_calls = 0

    def fork(self, coordinator, request, capability):
        del coordinator, capability
        with self._ownership_lock:
            self.fork_calls += 1
        (self._root / f"{request.child_run_id}.sqlite3").write_bytes(
            b"partial fork resource"
        )
        self._own(request.child_run_id, None)
        self.fork_entered.set()
        assert self.release_fork.wait(5)
        raise RuntimeError("fork failed after its side effect")

    def abort_fork(self, coordinator, request) -> None:
        del coordinator, request
        with self._ownership_lock:
            self.cleanup_calls += 1
        raise RuntimeError("fork cleanup failed")


def _run_create_params(imported, *, run_id: str = "run-1", stream_id: str = "stream-1"):
    return {
        "project_id": "law-firm",
        "run_id": run_id,
        "stream_id": stream_id,
        "expected_revision": imported["revision"],
        "expected_snapshot_hash": imported["content_hash"],
    }


def test_concurrent_exact_run_create_converges_on_one_owned_coordinator(
    tmp_path: Path,
) -> None:
    factory = BlockingCreateFactory(tmp_path)
    service, registry = _service(tmp_path, factory=factory)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _run_create_params(imported)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.invoke, "run.create", params, capability)
        assert factory.first_entered.wait(5)
        second = executor.submit(service.invoke, "run.create", params, capability)
        factory.second_entered.wait(0.25)
        factory.release.set()
        results = (first.result(timeout=10), second.result(timeout=10))

    assert results[0] == results[1]
    assert factory.created_instance_count == 1
    assert factory.owned_run_ids == ("run-1",)
    assert registry.resolve("run-1").run_view().content_hash == results[0]["content_hash"]
    assert (tmp_path / "run-1.sqlite3").is_file()


def test_run_create_reconciles_registry_publication_before_commit_error(
    tmp_path: Path,
) -> None:
    registry = PublishThenRaiseRegistry("run-1")
    factory = LocalCoordinatorFactory(tmp_path)
    service, _ = _service(tmp_path, factory=factory, registry=registry)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _run_create_params(imported)

    created = service.invoke("run.create", params, capability)
    retry = service.invoke("run.create", params, capability)
    resolved = service.invoke("run.view", {"run_id": "run-1"}, capability)

    assert retry == created
    assert resolved["content_hash"] == created["content_hash"]
    assert factory.created_instance_count == 1
    assert factory.aborted_create_count == 0
    assert factory.owned_run_ids == ("run-1",)
    assert (tmp_path / "run-1.sqlite3").is_file()


def test_failed_create_cleanup_poisons_waiting_and_exact_retries(
    tmp_path: Path,
) -> None:
    registry = ObservedWaitingRegistry("run-1")
    factory = CleanupFailingCreateFactory(tmp_path)
    service, _ = _service(tmp_path, factory=factory, registry=registry)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _run_create_params(imported)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.invoke, "run.create", params, capability)
        assert factory.create_entered.wait(5)
        second = executor.submit(service.invoke, "run.create", params, capability)
        assert registry.second_reservation_entered.wait(5)
        factory.release_create.set()
        errors = (first.exception(timeout=5), second.exception(timeout=5))

    assert all(isinstance(error, StudioRunLifecycleError) for error in errors)
    with pytest.raises(StudioRunLifecycleError):
        service.invoke("run.create", params, capability)
    assert factory.create_calls == 1
    assert factory.cleanup_calls == 1
    assert factory.owned_run_ids == ("run-1",)
    assert (tmp_path / "run-1.sqlite3").read_bytes() == b"partial create resource"


def test_failed_run_create_aborts_factory_resource_and_releases_reservation(
    tmp_path: Path,
) -> None:
    failing = PartialCreateFailureFactory(tmp_path)
    service, registry = _service(tmp_path, factory=failing)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _run_create_params(imported)

    with pytest.raises(RuntimeError, match="factory failed"):
        service.invoke("run.create", params, capability)

    assert failing.owned_run_ids == ()
    assert not (tmp_path / "run-1.sqlite3").exists()
    retry_factory = LocalCoordinatorFactory(tmp_path)
    retry_service = WorldStudioService(
        ScenarioProjectWorkspace.open(
            tmp_path / "studio.sqlite3",
            import_roots=(LAW_FIRM.parent,),
            export_root=tmp_path,
        ),
        registry,
        retry_factory,
        import_sources={"law-firm-fixture": LAW_FIRM},
    )
    assert retry_service.invoke("run.create", params, capability)["run_id"] == "run-1"
    assert retry_factory.owned_run_ids == ("run-1",)


def _checkpointed_service(service: WorldStudioService, capability: StudioCapability, imported):
    service.invoke("run.create", _run_create_params(imported), capability)
    _command(service, capability, "start", 1)
    _command(service, capability, "step", 2)
    checkpointed = _command(
        service,
        capability,
        "checkpoint",
        3,
        requested_checkpoint_id="checkpoint-concurrency",
    )
    parent = service.invoke("run.view", {"run_id": "run-1"}, capability)
    return {
        "fork_id": "fork-concurrency",
        "idempotency_key": "fork-concurrency-key",
        "source_run_id": "run-1",
        "scenario_hash": parent["scenario_hash"],
        "source_epoch": parent["coordinator_epoch"],
        "checkpoint_hash": checkpointed["checkpoint_hash"],
        "child_run_id": "run-child",
        "child_stream_id": "stream-child",
    }


def test_concurrent_exact_fork_converges_on_one_registered_child(
    tmp_path: Path,
) -> None:
    factory = BlockingForkFactory(tmp_path)
    service, registry = _service(tmp_path, factory=factory)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _checkpointed_service(service, capability, imported)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.invoke, "run.fork", params, capability)
        assert factory.first_fork_entered.wait(5)
        second = executor.submit(service.invoke, "run.fork", params, capability)
        factory.second_fork_entered.wait(0.25)
        factory.release_fork.set()
        results = (first.result(timeout=10), second.result(timeout=10))

    assert results[0] == results[1]
    assert factory.created_instance_count == 2
    assert factory.owned_run_ids == ("run-1", "run-child")
    assert registry.resolve("run-child").run_view().parent_checkpoint_hash == params[
        "checkpoint_hash"
    ]
    assert (tmp_path / "run-child.sqlite3").is_file()


def test_run_fork_reconciles_registry_publication_before_commit_error(
    tmp_path: Path,
) -> None:
    registry = PublishThenRaiseRegistry("run-child")
    factory = LocalCoordinatorFactory(tmp_path)
    service, _ = _service(tmp_path, factory=factory, registry=registry)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _checkpointed_service(service, capability, imported)

    forked = service.invoke("run.fork", params, capability)
    retry = service.invoke("run.fork", params, capability)
    child = service.invoke("run.view", {"run_id": "run-child"}, capability)

    assert retry == forked
    assert child["parent_checkpoint_hash"] == forked["checkpoint_hash"]
    assert factory.created_instance_count == 2
    assert factory.aborted_fork_count == 0
    assert factory.owned_run_ids == ("run-1", "run-child")
    assert (tmp_path / "run-child.sqlite3").is_file()


def test_failed_fork_cleanup_poisons_waiting_and_exact_retries(
    tmp_path: Path,
) -> None:
    registry = ObservedWaitingRegistry("run-child")
    factory = CleanupFailingForkFactory(tmp_path)
    service, _ = _service(tmp_path, factory=factory, registry=registry)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _checkpointed_service(service, capability, imported)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.invoke, "run.fork", params, capability)
        assert factory.fork_entered.wait(5)
        second = executor.submit(service.invoke, "run.fork", params, capability)
        assert registry.second_reservation_entered.wait(5)
        factory.release_fork.set()
        errors = (first.exception(timeout=5), second.exception(timeout=5))

    assert all(isinstance(error, StudioRunLifecycleError) for error in errors)
    with pytest.raises(StudioRunLifecycleError):
        service.invoke("run.fork", params, capability)
    assert factory.fork_calls == 1
    assert factory.cleanup_calls == 1
    assert factory.owned_run_ids == ("run-1", "run-child")
    assert (tmp_path / "run-child.sqlite3").read_bytes() == b"partial fork resource"


def test_conflicting_fork_retry_is_rejected_without_replacing_child(
    tmp_path: Path,
) -> None:
    service, registry = _service(tmp_path)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _checkpointed_service(service, capability, imported)
    accepted = service.invoke("run.fork", params, capability)
    child = registry.resolve("run-child")

    with pytest.raises(StudioConflictError):
        service.invoke(
            "run.fork",
            {
                **params,
                "fork_id": "different-fork",
                "idempotency_key": "different-key",
            },
            capability,
        )

    assert registry.resolve("run-child") is child
    assert service.invoke("run.view", {"run_id": "run-child"}, capability)[
        "content_hash"
    ] == child.run_view().content_hash
    assert accepted["child_run_id"] == "run-child"


def test_failed_fork_aborts_partial_resource_and_releases_reservation(
    tmp_path: Path,
) -> None:
    failing = PartialForkFailureFactory(tmp_path)
    service, registry = _service(tmp_path, factory=failing)
    capability = _capability()
    imported = _create_and_import(service, capability)
    params = _checkpointed_service(service, capability, imported)

    with pytest.raises(RuntimeError, match="factory failed"):
        service.invoke("run.fork", params, capability)

    assert failing.owned_run_ids == ("run-1",)
    assert not (tmp_path / "run-child.sqlite3").exists()
    retry_factory = LocalCoordinatorFactory(tmp_path)
    retry_service = WorldStudioService(
        ScenarioProjectWorkspace.open(
            tmp_path / "studio.sqlite3",
            import_roots=(LAW_FIRM.parent,),
            export_root=tmp_path,
        ),
        registry,
        retry_factory,
        import_sources={"law-firm-fixture": LAW_FIRM},
    )
    result = retry_service.invoke("run.fork", params, capability)
    assert result["child_run_id"] == "run-child"
    assert retry_factory.owned_run_ids == ("run-child",)


def test_output_owner_authorization_precedes_known_or_unknown_run_lookup(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    full = _capability(run_ids=("run-1", "unknown-run"))
    imported = _create_and_import(service, full)
    service.invoke("run.create", _run_create_params(imported), full)
    denied = _capability(
        run_ids=("run-1", "unknown-run"),
        agent_ids=("alice",),
        permissions=("output.read",),
    )

    errors = []
    for run_id in ("run-1", "unknown-run"):
        with pytest.raises(StudioAuthorizationError) as caught:
            service.invoke(
                "output.get",
                {
                    "run_id": run_id,
                    "batch_hash": "sha256:" + "0" * 64,
                    "agent_id": "bob",
                },
                denied,
            )
        errors.append((type(caught.value), caught.value.args))
    assert errors[0] == errors[1]


def test_missing_checkpoint_store_is_a_stable_fork_lifecycle_error(
    tmp_path: Path,
) -> None:
    factory = NoCheckpointFactory(tmp_path)
    service, _ = _service(tmp_path, factory=factory)
    capability = _capability()
    imported = _create_and_import(service, capability)
    service.invoke("run.create", _run_create_params(imported), capability)
    parent = service.invoke("run.view", {"run_id": "run-1"}, capability)
    params = {
        "fork_id": "fork-no-store",
        "idempotency_key": "fork-no-store-key",
        "source_run_id": "run-1",
        "scenario_hash": parent["scenario_hash"],
        "source_epoch": parent["coordinator_epoch"],
        "checkpoint_hash": "sha256:" + "0" * 64,
        "child_run_id": "run-child",
        "child_stream_id": "stream-child",
    }

    with pytest.raises(StudioRunLifecycleError):
        service.invoke("run.fork", params, capability)
    assert factory.owned_run_ids == ("run-1",)
    assert not (tmp_path / "run-child.sqlite3").exists()


def test_project_edit_after_run_create_cannot_change_run_authority(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    capability = _capability()
    imported = _create_and_import(service, capability)
    created = service.invoke("run.create", _run_create_params(imported), capability)

    changed = service.invoke(
        "project.apply",
        {
            "operation_id": "post-run-edit",
            "idempotency_key": "post-run-edit-key",
            "project_id": "law-firm",
            "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
            "document_role": "physical.world",
            "logical_id": None,
            "kind": "set_value",
            "pointer": "/places/0/label",
            "value": "Edited after run creation",
        },
        capability,
    )
    after = service.invoke("run.view", {"run_id": "run-1"}, capability)

    assert changed["next_snapshot"]["revision"] == imported["revision"] + 1
    assert after["scenario_hash"] == created["scenario_hash"]
    assert after["state_hash"] == created["state_hash"]


def test_every_service_method_family_denies_before_authority_use(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    full = _capability(project_ids=("law-firm", "other"))
    imported = _create_and_import(service, full)
    service.invoke("run.create", _run_create_params(imported), full)
    _command(service, full, "start", 1)
    stepped = _command(service, full, "step", 2)
    checkpointed = _command(
        service,
        full,
        "checkpoint",
        3,
        requested_checkpoint_id="checkpoint-auth",
    )
    run = service.invoke("run.view", {"run_id": "run-1"}, full)
    denied = _capability(
        project_ids=("law-firm", "other"),
        run_ids=("run-1", "run-child"),
        agent_ids=("alice",),
        permissions=(),
    )
    operation = {
        "operation_id": "denied-operation",
        "idempotency_key": "denied-operation-key",
        "project_id": "law-firm",
        "expected_revision": imported["revision"],
        "expected_snapshot_hash": imported["content_hash"],
        "document_role": "physical.world",
        "logical_id": None,
        "kind": "set_value",
        "pointer": "/places/0/label",
        "value": "Denied",
    }
    history = {
        "project_id": "law-firm",
        "expected_revision": imported["revision"],
        "expected_snapshot_hash": imported["content_hash"],
    }
    command = {
        "command_id": "denied-command",
        "idempotency_key": "denied-command-key",
        "run_id": "run-1",
        "scenario_hash": run["scenario_hash"],
        "coordinator_epoch": run["coordinator_epoch"],
        "expected_state_hash": run["state_hash"],
        "kind": "pause",
    }
    fork = {
        "fork_id": "denied-fork",
        "idempotency_key": "denied-fork-key",
        "source_run_id": "run-1",
        "scenario_hash": run["scenario_hash"],
        "source_epoch": run["coordinator_epoch"],
        "checkpoint_hash": checkpointed["checkpoint_hash"],
        "child_run_id": "run-child",
        "child_stream_id": "stream-child",
    }
    cases = (
        ("project.create", {"project_id": "other"}),
        (
            "project.import",
            {
                **history,
                "source_id": "law-firm-fixture",
            },
        ),
        ("project.snapshot", {"project_id": "law-firm"}),
        ("project.apply", operation),
        ("project.undo", history),
        ("project.redo", history),
        ("project.export", {"project_id": "law-firm", "target_name": "denied"}),
        ("scenario.validate", {"project_id": "law-firm"}),
        (
            "scenario.compile",
            {
                **history,
            },
        ),
        (
            "run.create",
            _run_create_params(
                imported,
                run_id="run-child",
                stream_id="stream-child",
            ),
        ),
        ("run.command", command),
        ("run.fork", fork),
        ("run.view", {"run_id": "run-1"}),
        ("run.list", {}),
        ("state.public", {"run_id": "run-1"}),
        ("state.agent", {"run_id": "run-1", "agent_id": "alice"}),
        ("state.network", {"run_id": "run-1"}),
        (
            "output.get",
            {"run_id": "run-1", "batch_hash": stepped["output_batch_hash"]},
        ),
        ("command.get", {"run_id": "run-1", "command_id": "command-2"}),
    )

    for method, params in cases:
        with pytest.raises(StudioAuthorizationError):
            service.invoke(method, params, denied)
