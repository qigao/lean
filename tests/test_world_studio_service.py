from __future__ import annotations

from pathlib import Path

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

    def create(
        self,
        scenario,
        *,
        project_id: str,
        run_id: str,
        stream_id: str,
    ) -> ScenarioCoordinator:
        del project_id
        return ScenarioCoordinator.create(
            self._root / f"{run_id}.sqlite3",
            scenario,
            run_id=run_id,
            stream_id=stream_id,
            checkpoint_store=LocalScenarioCheckpointStore(
                self._root / f"{run_id}-checkpoints"
            ),
        )

    def fork(
        self,
        coordinator: ScenarioCoordinator,
        request: ScenarioForkRequest,
        capability: ScenarioCommandCapability,
    ):
        return coordinator.fork(
            request,
            capability,
            self._root / f"{request.child_run_id}.sqlite3",
        )


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


def _service(tmp_path: Path):
    workspace = ScenarioProjectWorkspace.open(
        tmp_path / "studio.sqlite3",
        import_roots=(LAW_FIRM.parent,),
        export_root=tmp_path,
    )
    registry = InMemoryScenarioRunRegistry()
    service = WorldStudioService(
        workspace,
        registry,
        LocalCoordinatorFactory(tmp_path),
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
        "scenario.compile", {"project_id": "law-firm"}, capability
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
    with pytest.raises(StudioConflictError):
        service.invoke(
            "run.command",
            {**request, "command_id": "command-2"},
            capability,
        )
