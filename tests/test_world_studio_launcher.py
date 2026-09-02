from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from narrative_dynamics.studio import STUDIO_PERMISSIONS
from tools.run_world_studio import LauncherSettings, build_application


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
