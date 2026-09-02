from __future__ import annotations

import json
from pathlib import Path
import os
import stat

import pytest
from starlette.testclient import TestClient

from narrative_dynamics.studio import STUDIO_PERMISSIONS, StudioCapability
from tools.run_world_studio import (
    LauncherSettings,
    _LocalCoordinatorFactory,
    build_application,
    settings_from_args,
)


LAW_FIRM = Path("examples/law_firm_scenario").resolve()


def settings(tmp_path: Path, **overrides) -> LauncherSettings:
    workspace = tmp_path / "workspace"
    export = tmp_path / "export"
    static = tmp_path / "static"
    for root in (workspace, export, static):
        root.mkdir(exist_ok=True)
    (static / "index.html").write_text("<!doctype html><title>Studio</title>", encoding="utf-8")
    values = {
        "workspace_root": workspace,
        "import_roots": (LAW_FIRM.parent,),
        "import_sources": (("law-firm-fixture", LAW_FIRM),),
        "export_root": export,
        "static_root": static,
        "bind_host": "127.0.0.1",
        "bind_port": 8765,
        "allowed_origins": ("http://127.0.0.1:8765",),
        "development_trust_all": True,
        "auth_token": None,
        "tls_certificate": None,
        "tls_private_key": None,
        "authority_id": "local-operator",
        "project_ids": ("law-firm",),
        "run_ids": ("run-parent", "run-child"),
        "agent_ids": ("alice",),
        "permissions": STUDIO_PERMISSIONS,
    }
    values.update(overrides)
    return LauncherSettings(**values)


def test_development_trust_and_non_loopback_policy_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development trust-all requires loopback"):
        settings(tmp_path, bind_host="0.0.0.0")
    with pytest.raises(ValueError, match="non-loopback binding requires explicit authentication and TLS"):
        settings(
            tmp_path,
            bind_host="192.0.2.10",
            development_trust_all=False,
            auth_token="bounded-secret",
        )
    with pytest.raises(ValueError, match="TLS certificate and private key"):
        settings(tmp_path, tls_certificate=tmp_path / "missing.pem")


def test_token_mode_uses_one_time_session_handoff_not_ambient_bearer(
    tmp_path: Path,
) -> None:
    app = build_application(settings(
        tmp_path,
        development_trust_all=False,
        auth_token="operator-secret",
    ))
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/session").status_code == 401
        bearer_rpc = client.post("/rpc", headers={"authorization": "Bearer operator-secret"}, json={
            "jsonrpc": "2.0", "id": "ambient", "method": "project.create",
            "params": {"project_id": "law-firm"},
        })
        assert bearer_rpc.status_code == 401

        login = client.post("/session", headers={"authorization": "Bearer operator-secret"})
        assert login.status_code == 200
        assert "HttpOnly" in login.headers["set-cookie"]
        assert "Secure" not in login.headers["set-cookie"]
        assert client.post("/rpc", json={
            "jsonrpc": "2.0", "id": "cookie", "method": "project.create",
            "params": {"project_id": "law-firm"},
        }).status_code == 200


def test_session_capacity_and_lifetime_are_explicit_launcher_limits(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    token_file = tmp_path / "token.txt"
    token_file.write_text("operator-secret", encoding="utf-8")
    parsed = settings_from_args([
        "--workspace-root", str(configured.workspace_root),
        "--import-root", str(LAW_FIRM.parent),
        "--import-source", f"law-firm-fixture={LAW_FIRM}",
        "--export-root", str(configured.export_root),
        "--static-root", str(configured.static_root),
        "--bind-host", "127.0.0.1", "--bind-port", "8765",
        "--origin", "http://127.0.0.1:8765",
        "--auth-token-file", str(token_file),
        "--authority-id", "local-operator",
        "--project-id", "law-firm", "--run-id", "run-parent",
        "--maximum-sessions", "7", "--session-lifetime-seconds", "45",
    ])

    assert parsed.server_limits.maximum_sessions == 7
    assert parsed.server_limits.session_lifetime_seconds == 45


def test_composition_routes_real_sqlite_coordinator_and_fork_output(tmp_path: Path) -> None:
    app = build_application(settings(tmp_path))
    request_number = 0

    def rpc(client: TestClient, method: str, params: dict):
        nonlocal request_number
        request_number += 1
        response = client.post("/rpc", json={
            "jsonrpc": "2.0", "id": f"rpc-{request_number}", "method": method, "params": params,
        })
        assert response.status_code == 200
        payload = response.json()
        assert "error" not in payload, payload
        return payload["result"]

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        session = client.get("/session").json()
        assert session["authority"]["run_ids"] == ["run-child", "run-parent"]
        created = rpc(client, "project.create", {"project_id": "law-firm"})
        imported = rpc(client, "project.import", {
            "project_id": "law-firm", "source_id": "law-firm-fixture",
            "expected_revision": created["revision"],
            "expected_snapshot_hash": created["content_hash"],
        })
        compiled = rpc(client, "scenario.compile", {
            "project_id": "law-firm", "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
        })
        parent = rpc(client, "run.create", {
            "project_id": "law-firm", "run_id": "run-parent", "stream_id": "stream-parent",
            "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
        })
        assert compiled["scenario_hash"] == parent["scenario_hash"]

        def command(run: dict, kind: str, number: int, checkpoint_id: str | None = None):
            params = {
                "command_id": f"command-{number}", "idempotency_key": f"command-key-{number}",
                "run_id": run["run_id"], "scenario_hash": run["scenario_hash"],
                "coordinator_epoch": run["coordinator_epoch"],
                "expected_state_hash": run["state_hash"], "kind": kind,
            }
            if checkpoint_id is not None:
                params["requested_checkpoint_id"] = checkpoint_id
            return rpc(client, "run.command", params)

        command(parent, "start", 1)
        parent = rpc(client, "run.view", {"run_id": "run-parent"})
        checkpoint = command(parent, "checkpoint", 2, "fork-base")
        parent = rpc(client, "run.view", {"run_id": "run-parent"})
        forked = rpc(client, "run.fork", {
            "fork_id": "fork-1", "idempotency_key": "fork-key-1",
            "source_run_id": "run-parent", "scenario_hash": parent["scenario_hash"],
            "source_epoch": parent["coordinator_epoch"],
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "child_run_id": "run-child", "child_stream_id": "stream-child",
        })
        assert forked["child_state_hash"] == parent["state_hash"]
        child = rpc(client, "run.view", {"run_id": "run-child"})

        with client.websocket_connect(
            "/v1/stream",
            subprotocols=["nd-jsonrpc-v1"],
            headers={"origin": "http://127.0.0.1:8765"},
        ) as stream:
            stream.send_json({
                "jsonrpc": "2.0", "id": "subscribe-child", "method": "stream.subscribe",
                "params": {
                    "subscription_id": "child-public", "run_id": "run-child",
                    "stream_id": "stream-child", "kinds": ["state.delta", "network.metrics"],
                    "audience": "public",
                },
            })
            assert stream.receive_json()["result"]["subscription_id"] == "child-public"
            stepped = command(child, "step", 3)
            notification = stream.receive_json()
            assert notification["method"] == "stream.output"
            assert notification["params"]["subscription_id"] == "child-public"
            assert notification["params"]["output"]["source_batch_hash"] == stepped["output_batch_hash"]

    assert (settings(tmp_path).workspace_root / "projects.sqlite3").is_file()
    assert (settings(tmp_path).workspace_root / "runs" / "run-child.sqlite3").is_file()


def _rpc(client: TestClient, method: str, params: dict, request_id: str = "test") -> dict:
    response = client.post("/rpc", json={
        "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
    })
    assert response.status_code == 200
    return response.json()


def _imported_project(client: TestClient) -> dict:
    created = _rpc(client, "project.create", {"project_id": "law-firm"}, "create")["result"]
    return _rpc(client, "project.import", {
        "project_id": "law-firm", "source_id": "law-firm-fixture",
        "expected_revision": created["revision"],
        "expected_snapshot_hash": created["content_hash"],
    }, "import")["result"]


def _create_run(client: TestClient, snapshot: dict, run_id: str, stream_id: str) -> dict:
    return _rpc(client, "run.create", {
        "project_id": "law-firm", "run_id": run_id, "stream_id": stream_id,
        "expected_revision": snapshot["revision"],
        "expected_snapshot_hash": snapshot["content_hash"],
    }, f"create-{run_id}")["result"]


def _checkpoint_run(client: TestClient, run: dict, number: int) -> tuple[dict, dict]:
    started = _rpc(client, "run.command", {
        "command_id": f"start-{number}", "idempotency_key": f"start-key-{number}",
        "run_id": run["run_id"], "scenario_hash": run["scenario_hash"],
        "coordinator_epoch": run["coordinator_epoch"],
        "expected_state_hash": run["state_hash"], "kind": "start",
    }, f"start-{number}")["result"]
    running = _rpc(client, "run.view", {"run_id": run["run_id"]}, f"view-{number}")["result"]
    checkpoint = _rpc(client, "run.command", {
        "command_id": f"checkpoint-{number}", "idempotency_key": f"checkpoint-key-{number}",
        "run_id": running["run_id"], "scenario_hash": running["scenario_hash"],
        "coordinator_epoch": running["coordinator_epoch"],
        "expected_state_hash": running["state_hash"], "kind": "checkpoint",
        "requested_checkpoint_id": f"checkpoint-{number}",
    }, f"checkpoint-{number}")["result"]
    assert started["accepted"] is True
    return running, checkpoint


def test_restart_rejects_parent_run_id_reuse_without_deleting_prior_artifacts(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    first = build_application(configured)
    with TestClient(first, base_url="http://127.0.0.1:8765") as client:
        imported = _imported_project(client)
        _create_run(client, imported, "run-parent", "stream-parent")

    database = configured.workspace_root / "runs" / "run-parent.sqlite3"
    checkpoints = configured.workspace_root / "runs" / "run-parent-checkpoints"
    before_database = database.read_bytes()
    before_files = {
        item.relative_to(checkpoints).as_posix(): item.read_bytes()
        for item in checkpoints.rglob("*") if item.is_file()
    }

    restarted = build_application(configured)
    with TestClient(restarted, base_url="http://127.0.0.1:8765") as client:
        snapshot = _rpc(client, "project.snapshot", {"project_id": "law-firm"}, "snapshot")["result"]
        rejected = _rpc(client, "run.create", {
            "project_id": "law-firm", "run_id": "run-parent", "stream_id": "stream-reused",
            "expected_revision": snapshot["revision"],
            "expected_snapshot_hash": snapshot["content_hash"],
        }, "reuse-parent")
        assert "error" in rejected and "result" not in rejected

    assert database.read_bytes() == before_database
    assert {
        item.relative_to(checkpoints).as_posix(): item.read_bytes()
        for item in checkpoints.rglob("*") if item.is_file()
    } == before_files


def test_restart_rejects_child_run_id_reuse_without_deleting_prior_artifacts(tmp_path: Path) -> None:
    configured = settings(tmp_path, run_ids=("run-parent", "run-child", "run-replacement"))
    first = build_application(configured)
    with TestClient(first, base_url="http://127.0.0.1:8765") as client:
        imported = _imported_project(client)
        parent = _create_run(client, imported, "run-parent", "stream-parent")
        running, checkpoint = _checkpoint_run(client, parent, 1)
        forked = _rpc(client, "run.fork", {
            "fork_id": "fork-original", "idempotency_key": "fork-original-key",
            "source_run_id": "run-parent", "scenario_hash": running["scenario_hash"],
            "source_epoch": running["coordinator_epoch"],
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "child_run_id": "run-child", "child_stream_id": "stream-child",
        }, "fork-original")
        assert "result" in forked

    child_database = configured.workspace_root / "runs" / "run-child.sqlite3"
    before_database = child_database.read_bytes()

    restarted = build_application(configured)
    with TestClient(restarted, base_url="http://127.0.0.1:8765") as client:
        snapshot = _rpc(client, "project.snapshot", {"project_id": "law-firm"}, "snapshot")["result"]
        replacement = _create_run(client, snapshot, "run-replacement", "stream-replacement")
        running, checkpoint = _checkpoint_run(client, replacement, 2)
        rejected = _rpc(client, "run.fork", {
            "fork_id": "fork-reused", "idempotency_key": "fork-reused-key",
            "source_run_id": "run-replacement", "scenario_hash": running["scenario_hash"],
            "source_epoch": running["coordinator_epoch"],
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "child_run_id": "run-child", "child_stream_id": "stream-reused",
        }, "fork-reused")
        assert "error" in rejected and "result" not in rejected

    assert child_database.read_bytes() == before_database


@pytest.mark.parametrize("run_id", ("bad:name", "trailing.", "trailing ", "CON", "nul.txt", "bad<id>"))
def test_local_run_ids_reject_platform_unsafe_filename_identities(run_id: str) -> None:
    with pytest.raises(ValueError, match="filename-safe"):
        _LocalCoordinatorFactory._run_id(run_id)


def test_parent_database_is_exclusively_reserved_before_sqlite_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = settings(tmp_path)
    claimed_checkpoints: list[tuple[int, int]] = []
    original = __import__(
        "tools.run_world_studio", fromlist=["initialize_situated_percept_memory"]
    ).initialize_situated_percept_memory

    def observed_initialize(database_path, *args, **kwargs):
        details = os.stat(database_path, follow_symlinks=False)
        assert stat.S_ISREG(details.st_mode)
        assert details.st_size == 0
        checkpoint_details = os.stat(
            configured.workspace_root / "runs" / "run-parent-checkpoints",
            follow_symlinks=False,
        )
        assert stat.S_ISDIR(checkpoint_details.st_mode)
        claimed_checkpoints.append(
            (checkpoint_details.st_dev, checkpoint_details.st_ino)
        )
        return original(database_path, *args, **kwargs)

    monkeypatch.setattr(
        "tools.run_world_studio.initialize_situated_percept_memory",
        observed_initialize,
    )
    app = build_application(configured)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        imported = _imported_project(client)
        created = _create_run(client, imported, "run-parent", "stream-parent")
        assert created["run_id"] == "run-parent"
    final_checkpoints = os.stat(
        configured.workspace_root / "runs" / "run-parent-checkpoints",
        follow_symlinks=False,
    )
    assert claimed_checkpoints == [(final_checkpoints.st_dev, final_checkpoints.st_ino)]


def test_fork_rejects_foreign_checkpoint_namespace_without_deleting_it(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    app = build_application(configured)
    child_checkpoints = configured.workspace_root / "runs" / "run-child-checkpoints"
    child_checkpoints.mkdir()
    sentinel = child_checkpoints / "foreign.txt"
    sentinel.write_bytes(b"foreign-checkpoint-tree")

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        imported = _imported_project(client)
        parent = _create_run(client, imported, "run-parent", "stream-parent")
        running, checkpoint = _checkpoint_run(client, parent, 3)
        rejected = _rpc(client, "run.fork", {
            "fork_id": "fork-foreign", "idempotency_key": "fork-foreign-key",
            "source_run_id": "run-parent", "scenario_hash": running["scenario_hash"],
            "source_epoch": running["coordinator_epoch"],
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "child_run_id": "run-child", "child_stream_id": "stream-child",
        }, "fork-foreign")
        assert "error" in rejected and "result" not in rejected

    assert sentinel.read_bytes() == b"foreign-checkpoint-tree"
    assert not (configured.workspace_root / "runs" / "run-child.sqlite3").exists()


def test_parent_create_rejects_dangling_database_symlink_without_removing_it(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    runs = configured.workspace_root / "runs"
    runs.mkdir()
    database = runs / "run-parent.sqlite3"
    foreign_target = tmp_path / "foreign-target.sqlite3"
    try:
        database.symlink_to(foreign_target)
    except OSError:
        pytest.skip("host cannot create a file symlink")

    app = build_application(configured)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        imported = _imported_project(client)
        rejected = _rpc(client, "run.create", {
            "project_id": "law-firm", "run_id": "run-parent", "stream_id": "stream-parent",
            "expected_revision": imported["revision"],
            "expected_snapshot_hash": imported["content_hash"],
        }, "dangling-parent")
        assert "error" in rejected and "result" not in rejected

    assert database.is_symlink()
    assert not foreign_target.exists()


@pytest.mark.parametrize("failing_verification", (1, 3))
def test_preclaimed_fork_failure_keeps_factory_cleanup_authority_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_verification: int,
) -> None:
    configured = settings(tmp_path)
    app = build_application(configured)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        imported = _imported_project(client)
        parent = _create_run(client, imported, "run-parent", "stream-parent")
        running, checkpoint = _checkpoint_run(client, parent, 4)

    factory = app.state.coordinator_factory
    parent_coordinator = factory._owned["run-parent"]
    checkpoint_store = parent_coordinator._checkpoint_store
    original_verify = checkpoint_store._verify_restored_target
    verification_count = 0

    def fail_after_verification(*args, **kwargs):
        nonlocal verification_count
        result = original_verify(*args, **kwargs)
        verification_count += 1
        if verification_count == failing_verification:
            raise ValueError(f"injected verification {failing_verification}")
        return result

    monkeypatch.setattr(
        checkpoint_store,
        "_verify_restored_target",
        fail_after_verification,
    )
    capability = StudioCapability(
        configured.authority_id,
        project_ids=configured.project_ids,
        run_ids=configured.run_ids,
        agent_ids=configured.agent_ids,
        permissions=configured.permissions,
    )
    params = {
        "fork_id": "fork-retry",
        "idempotency_key": "fork-retry-key",
        "source_run_id": "run-parent",
        "scenario_hash": running["scenario_hash"],
        "source_epoch": running["coordinator_epoch"],
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "child_run_id": "run-child",
        "child_stream_id": "stream-child",
    }

    with pytest.raises(
        ValueError,
        match=rf"^injected verification {failing_verification}$",
    ):
        app.state.studio_service.invoke("run.fork", params, capability)

    runs = configured.workspace_root / "runs"
    assert not (runs / "run-child.sqlite3").exists()
    assert not (runs / ".run-child.owner").exists()
    assert json.loads((runs / "run-metadata.json").read_text(encoding="utf-8")) == {
        "schema": "narrative-dynamics.local-run-metadata/v1",
        "run_ids": ["run-parent"],
    }
    assert "run-child" not in factory._claims
    assert "run-child" not in factory._owned

    monkeypatch.setattr(
        checkpoint_store,
        "_verify_restored_target",
        original_verify,
    )
    retried = app.state.studio_service.invoke("run.fork", params, capability)
    assert retried["child_run_id"] == "run-child"
    assert (runs / "run-child.sqlite3").is_file()
