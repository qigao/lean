from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from narrative_dynamics.studio import JsonRpcDispatcher, StudioCapability
from narrative_dynamics.studio.streaming import StudioOutputRouter
from narrative_dynamics.integrations.world_studio_server import (
    WORLD_STUDIO_PROTOCOL_VERSION,
    WorldStudioServerLimits,
    create_world_studio_asgi_app,
)
from tests.test_world_studio_service import _capability, _create_and_import, _service
from tests.test_world_studio_streaming import public_batch


def auth(_request) -> StudioCapability:
    return _capability()


def app_for(tmp_path: Path, **overrides):
    service, _ = _service(tmp_path)
    capability = _capability()
    _create_and_import(service, capability)
    values = {
        "dispatcher": JsonRpcDispatcher(service),
        "output_router": StudioOutputRouter(),
        "authenticate_http": auth,
        "authenticate_websocket": auth,
        "allowed_origins": ("https://studio.example",),
        "limits": WorldStudioServerLimits(),
    }
    values.update(overrides)
    return create_world_studio_asgi_app(**values), capability


def test_health_is_exact_and_rpc_preserves_request_id(tmp_path: Path) -> None:
    app, _ = app_for(tmp_path)
    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/rpc",
            json={
                "jsonrpc": "2.0",
                "id": "browser-7",
                "method": "project.snapshot",
                "params": {"project_id": "law-firm"},
            },
        )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "protocol_version": WORLD_STUDIO_PROTOCOL_VERSION,
    }
    assert response.status_code == 200
    assert response.json()["id"] == "browser-7"
    assert response.json()["result"]["project_id"] == "law-firm"


@pytest.mark.parametrize(
    ("change", "status", "code"),
    (
        ({"headers": {"content-type": "text/plain"}}, 415, "unsupported_media_type"),
        ({"content": b"x" * 65}, 413, "body_limit"),
    ),
)
def test_http_transport_limits_are_stable(
    tmp_path: Path, change: dict, status: int, code: str
) -> None:
    app, _ = app_for(
        tmp_path,
        limits=WorldStudioServerLimits(maximum_http_body_bytes=64),
    )
    request = {"content": b"{}", "headers": {"content-type": "application/json"}}
    request.update(change)
    with TestClient(app) as client:
        response = client.post("/rpc", **request)

    assert response.status_code == status
    assert response.json() == {"error": {"code": code}}


class FailingDispatcher:
    def parse_and_dispatch(self, raw: bytes, capability: StudioCapability) -> bytes:
        del raw, capability
        raise RuntimeError(r"token sk-secret C:\private\gateway.py")


class SlowDispatcher:
    def parse_and_dispatch(self, raw: bytes, capability: StudioCapability) -> bytes:
        del raw, capability
        time.sleep(0.1)
        return b'{"jsonrpc":"2.0","id":1,"result":{}}'


class ImmediateDispatcher:
    def parse_and_dispatch(self, raw: bytes, capability: StudioCapability) -> bytes:
        del raw, capability
        return b'{"jsonrpc":"2.0","id":1,"result":{}}'


@pytest.mark.parametrize(
    ("dispatcher", "authenticator", "limits", "status", "code"),
    (
        (FailingDispatcher(), auth, WorldStudioServerLimits(), 500, "internal_error"),
        (
            SlowDispatcher(),
            auth,
            WorldStudioServerLimits(request_timeout_seconds=0.01),
            504,
            "timeout",
        ),
        (FailingDispatcher(), lambda request: None, WorldStudioServerLimits(), 401, "unauthorized"),
        (
            FailingDispatcher(),
            lambda request: (_ for _ in ()).throw(RuntimeError("secret-token")),
            WorldStudioServerLimits(),
            401,
            "unauthorized",
        ),
    ),
)
def test_http_auth_timeout_and_raw_failures_are_redacted(
    tmp_path: Path,
    dispatcher,
    authenticator,
    limits: WorldStudioServerLimits,
    status: int,
    code: str,
) -> None:
    app, _ = app_for(
        tmp_path,
        dispatcher=dispatcher,
        authenticate_http=authenticator,
        limits=limits,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/rpc",
            content=b'{"jsonrpc":"2.0","id":1,"method":"run.list"}',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == status
    assert response.json() == {"error": {"code": code}}
    encoded = response.content.lower()
    assert b"secret" not in encoded
    assert b"private" not in encoded
    assert b"token" not in encoded


def test_http_authentication_is_inside_the_request_timeout(tmp_path: Path) -> None:
    async def slow_authentication(request):
        del request
        import asyncio

        await asyncio.sleep(0.4)
        return _capability()

    app, _ = app_for(
        tmp_path,
        dispatcher=ImmediateDispatcher(),
        authenticate_http=slow_authentication,
        limits=WorldStudioServerLimits(request_timeout_seconds=0.2),
    )
    with TestClient(app) as client:
        response = client.post(
            "/rpc",
            content=b'{"jsonrpc":"2.0","id":1,"method":"run.list"}',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 504
    assert response.json() == {"error": {"code": "timeout"}}


def test_websocket_controls_and_live_output_use_jsonrpc_envelopes(tmp_path: Path) -> None:
    router = StudioOutputRouter()
    app, _ = app_for(tmp_path, output_router=router)
    headers = {"origin": "https://studio.example"}
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "subscribe-1",
                    "method": "stream.subscribe",
                    "params": {
                        "subscription_id": "sub-1",
                        "run_id": "run-1",
                        "stream_id": "stream-1",
                        "kinds": ["command.result"],
                    },
                }
            )
            subscribed = websocket.receive_json()
            assert subscribed["id"] == "subscribe-1"

            batch = public_batch(1)
            router.publish("run-1", batch)
            notification = websocket.receive_json()
            assert set(notification) == {"jsonrpc", "method", "params"}
            assert notification["jsonrpc"] == "2.0"
            assert notification["method"] == "stream.output"
            assert notification["params"]["subscription_id"] == "sub-1"
            assert notification["params"]["output"]["source_batch_hash"] == batch.content_hash

            for request_id, method, params in (
                (
                    "ack-1",
                    "stream.acknowledge",
                    {
                        "subscription_id": "sub-1",
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": batch.content_hash,
                    },
                ),
                (
                    "resume-1",
                    "stream.resume",
                    {
                        "subscription_id": "sub-1",
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": batch.content_hash,
                    },
                ),
                ("unsubscribe-1", "stream.unsubscribe", {"subscription_id": "sub-1"}),
            ):
                websocket.send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    }
                )
                control_response = websocket.receive_json()
                assert control_response["id"] == request_id
                assert "result" in control_response


@pytest.mark.parametrize(
    ("headers", "subprotocols"),
    (
        ({"origin": "https://evil.example"}, ["nd-jsonrpc-v1"]),
        ({"origin": "https://studio.example"}, ["other-protocol"]),
        ({}, ["nd-jsonrpc-v1"]),
    ),
)
def test_websocket_origin_and_subprotocol_fail_closed(
    tmp_path: Path, headers: dict[str, str], subprotocols: list[str]
) -> None:
    app, _ = app_for(tmp_path)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as raised:
            with client.websocket_connect(
                "/v1/stream", headers=headers, subprotocols=subprotocols
            ):
                pass
    assert raised.value.code == 1008


def test_websocket_authentication_and_frame_limit_fail_closed(tmp_path: Path) -> None:
    denied, _ = app_for(
        tmp_path,
        authenticate_websocket=lambda websocket: None,
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(denied) as client:
        with pytest.raises(WebSocketDisconnect) as raised:
            with client.websocket_connect(
                "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
            ):
                pass
    assert raised.value.code == 1008

    bounded_root = tmp_path / "bounded"
    bounded_root.mkdir()
    bounded, _ = app_for(
        bounded_root,
        limits=WorldStudioServerLimits(maximum_websocket_frame_bytes=32),
    )
    with TestClient(bounded) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_text("x" * 33)
            with pytest.raises(WebSocketDisconnect) as raised:
                websocket.receive_json()
    assert raised.value.code == 1009


def test_websocket_subscription_capacity_is_a_jsonrpc_failure(tmp_path: Path) -> None:
    app, _ = app_for(
        tmp_path,
        limits=WorldStudioServerLimits(maximum_subscriptions_per_connection=1),
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            for index in (1, 2):
                websocket.send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": index,
                        "method": "stream.subscribe",
                        "params": {
                            "subscription_id": f"sub-{index}",
                            "run_id": "run-1",
                            "stream_id": "stream-1",
                            "kinds": [],
                        },
                    }
                )
                response = websocket.receive_json()
                if index == 1:
                    assert "result" in response
                else:
                    assert response == {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "error": {"code": -32017, "message": "Capacity limit"},
                    }


def test_websocket_cursor_fields_are_strict_jsonrpc_params(tmp_path: Path) -> None:
    app, _ = app_for(tmp_path)
    headers = {"origin": "https://studio.example"}
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "invalid-cursor",
                    "method": "stream.acknowledge",
                    "params": {
                        "subscription_id": "sub-1",
                        "stream_id": "stream-1",
                        "last_sequence": "1",
                        "last_batch_hash": "not-a-hash",
                    },
                }
            )
            response = websocket.receive_json()

    assert response == {
        "jsonrpc": "2.0",
        "id": "invalid-cursor",
        "error": {"code": -32602, "message": "Invalid params"},
    }


def test_websocket_connection_capacity_and_disconnect_release_subscriptions(
    tmp_path: Path,
) -> None:
    router = StudioOutputRouter()
    app, _ = app_for(
        tmp_path,
        output_router=router,
        limits=WorldStudioServerLimits(maximum_websocket_connections=1),
    )
    headers = {"origin": "https://studio.example"}
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "stream.subscribe",
        "params": {
            "subscription_id": "lease-1",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "kinds": [],
        },
    }
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as first:
            first.send_json(request)
            assert "result" in first.receive_json()
            with pytest.raises(WebSocketDisconnect) as raised:
                with client.websocket_connect(
                    "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
                ):
                    pass
            assert raised.value.code == 1013

        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as replacement:
            replacement.send_json(request)
            assert "result" in replacement.receive_json()


class SlowSubscribeRouter(StudioOutputRouter):
    def __init__(self) -> None:
        super().__init__()
        self._delay_next = True

    def subscribe(self, *args, **kwargs):
        if self._delay_next:
            self._delay_next = False
            time.sleep(0.2)
        return super().subscribe(*args, **kwargs)


def test_timed_out_control_cannot_leave_a_lease_after_disconnect(tmp_path: Path) -> None:
    router = SlowSubscribeRouter()
    app, _ = app_for(
        tmp_path,
        output_router=router,
        limits=WorldStudioServerLimits(request_timeout_seconds=0.05),
    )
    headers = {"origin": "https://studio.example"}
    request = {
        "jsonrpc": "2.0",
        "id": "slow-subscribe",
        "method": "stream.subscribe",
        "params": {
            "subscription_id": "late-lease",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "kinds": [],
        },
    }
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(request)
            assert websocket.receive_json()["error"]["code"] == -32603
        time.sleep(0.25)
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as replacement:
            replacement.send_json(request)
            assert "result" in replacement.receive_json()


def test_core_imports_do_not_load_optional_asgi_dependencies() -> None:
    script = (
        "import sys; import narrative_dynamics.abm; import narrative_dynamics.studio; "
        "assert 'starlette' not in sys.modules; assert 'hypercorn' not in sys.modules"
    )
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
