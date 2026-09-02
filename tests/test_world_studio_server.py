from __future__ import annotations

import asyncio
from collections import deque
import json
from pathlib import Path
import sys
import threading
import time

import anyio
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from narrative_dynamics.studio import JsonRpcDispatcher, StudioCapability
from narrative_dynamics.studio.streaming import StudioOutputRouter
from narrative_dynamics.integrations.world_studio_server import (
    WORLD_STUDIO_PROTOCOL_VERSION,
    WORLD_STUDIO_SESSION_COOKIE,
    WorldStudioServerLimits,
    create_world_studio_asgi_app,
)
from tests.test_world_studio_service import _capability, _create_and_import, _service
from tests.test_world_studio_streaming import (
    private_batch,
    public_batch,
    public_batch_range,
)


def receive_json_bounded(websocket, timeout: float = 1.0):
    async def receive_message():
        with anyio.fail_after(timeout):
            return await websocket._send_rx.receive()

    message = websocket.portal.call(receive_message)
    websocket._raise_on_close(message)
    if "text" in message:
        return json.loads(message["text"])
    return json.loads(message["bytes"].decode("utf-8"))


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
        session = client.get("/session")
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
    assert session.json() == {
        "schema": "narrative-dynamics.studio-session/v1",
        "authority": {
            "authority_id": "operator",
            "project_ids": ["law-firm"],
            "run_ids": ["run-1", "run-child"],
            "agent_ids": ["alice"],
            "permissions": list(_capability().permissions),
        },
    }
    assert session.headers["cache-control"] == "no-store"
    assert response.status_code == 200
    assert response.json()["id"] == "browser-7"
    assert response.json()["result"]["project_id"] == "law-firm"


def test_browser_session_handoff_is_opaque_and_authenticates_rpc_and_wss(
    tmp_path: Path,
) -> None:
    bearer = "production-bearer-secret"

    def authenticate(request):
        return _capability() if request.headers.get("authorization") == f"Bearer {bearer}" else None

    app, _ = app_for(
        tmp_path,
        authenticate_http=authenticate,
        authenticate_websocket=lambda websocket: None,
        allow_ambient_authentication=False,
    )
    with TestClient(app, base_url="https://studio.example") as anonymous:
        bearer_rpc = anonymous.post("/rpc", headers={"authorization": f"Bearer {bearer}"}, json={
            "jsonrpc": "2.0", "id": "bearer-rpc", "method": "project.snapshot",
            "params": {"project_id": "law-firm"},
        })
        assert bearer_rpc.status_code == 401
    with TestClient(app, base_url="https://studio.example") as client:
        denied = client.post("/session", headers={"authorization": "Bearer invalid"})
        assert denied.status_code == 401
        assert "set-cookie" not in denied.headers

        login = client.post("/session", headers={"authorization": f"Bearer {bearer}"})
        cookie = login.cookies.get(WORLD_STUDIO_SESSION_COOKIE)
        assert login.status_code == 200
        assert cookie and cookie != bearer and bearer not in login.text
        set_cookie = login.headers["set-cookie"]
        assert "HttpOnly" in set_cookie
        assert "Secure" in set_cookie
        assert "SameSite=strict" in set_cookie
        assert "Path=/" in set_cookie
        assert bearer not in set_cookie
        assert client.post("/session").status_code == 401

        session = client.get("/session")
        response = client.post("/rpc", json={
            "jsonrpc": "2.0", "id": "cookie-rpc", "method": "project.snapshot",
            "params": {"project_id": "law-firm"},
        })
        assert session.status_code == response.status_code == 200
        assert response.json()["id"] == "cookie-rpc"

        with client.websocket_connect(
            "wss://studio.example/v1/stream",
            headers={"origin": "https://studio.example"},
            subprotocols=["nd-jsonrpc-v1"],
        ) as websocket:
            websocket.send_json({
                "jsonrpc": "2.0", "id": "cookie-wss", "method": "stream.subscribe",
                "params": {
                    "subscription_id": "cookie-sub", "run_id": "run-1",
                    "stream_id": "stream-1", "kinds": [], "audience": "public",
                },
            })
            assert receive_json_bounded(websocket)["result"]["subscription_id"] == "cookie-sub"

        logout = client.delete("/session")
        assert logout.status_code == 204
        assert "Max-Age=0" in logout.headers["set-cookie"]
        assert client.get("/session").status_code == 401


def test_browser_sessions_expire_and_evict_oldest_at_capacity(tmp_path: Path) -> None:
    now = [10.0]
    bearer = "bounded-bearer"

    def authenticate(request):
        return _capability() if request.headers.get("authorization") == f"Bearer {bearer}" else None

    app, _ = app_for(
        tmp_path,
        authenticate_http=authenticate,
        authenticate_websocket=lambda websocket: None,
        allow_ambient_authentication=False,
        clock=lambda: now[0],
        limits=WorldStudioServerLimits(maximum_sessions=1, session_lifetime_seconds=5),
    )
    with (
        TestClient(app, base_url="https://studio.example") as first,
        TestClient(app, base_url="https://studio.example") as second,
    ):
        first_login = first.post("/session", headers={"authorization": f"Bearer {bearer}"})
        first_cookie = first_login.cookies.get(WORLD_STUDIO_SESSION_COOKIE)
        assert first_cookie
        assert first.get("/session").status_code == 200

        second_login = second.post("/session", headers={"authorization": f"Bearer {bearer}"})
        second_cookie = second_login.cookies.get(WORLD_STUDIO_SESSION_COOKIE)
        assert second_cookie and second_cookie != first_cookie
        assert first.get("/session").status_code == 401
        assert second.get("/session").status_code == 200

        now[0] = 15.0
        assert second.get("/session").status_code == 401


def _private_subscription_request(identity: str) -> dict:
    return {
        "jsonrpc": "2.0", "id": f"subscribe-{identity}", "method": "stream.subscribe",
        "params": {
            "subscription_id": identity, "run_id": "run-1", "stream_id": "stream-1",
            "kinds": ["percept.private"], "audience": "agent", "owner_agent_id": "alice",
        },
    }


def test_logout_revokes_live_private_websocket_and_releases_subscription(tmp_path: Path) -> None:
    bearer = "logout-bearer"
    router = StudioOutputRouter()

    def authenticate(request):
        return _capability() if request.headers.get("authorization") == f"Bearer {bearer}" else None

    app, _ = app_for(
        tmp_path,
        output_router=router,
        authenticate_http=authenticate,
        authenticate_websocket=lambda websocket: None,
        allow_ambient_authentication=False,
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app, base_url="https://studio.example") as client:
        assert client.post("/session", headers={"authorization": f"Bearer {bearer}"}).status_code == 200
        with client.websocket_connect(
            "wss://studio.example/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(_private_subscription_request("logout-private"))
            assert "result" in receive_json_bounded(websocket)
            assert client.delete("/session").status_code == 204
            with pytest.raises(WebSocketDisconnect) as raised:
                receive_json_bounded(websocket)
            assert raised.value.code == 1008

        assert client.post("/session", headers={"authorization": f"Bearer {bearer}"}).status_code == 200
        with client.websocket_connect(
            "wss://studio.example/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as replacement:
            replacement.send_json(_private_subscription_request("logout-private"))
            assert "result" in receive_json_bounded(replacement)


def test_logout_discards_a_private_subscription_that_finishes_after_revocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bearer = "racing-logout-bearer"
    router = StudioOutputRouter()
    entered = threading.Event()
    release = threading.Event()
    original_subscribe = router.subscribe

    def delayed_subscribe(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_subscribe(*args, **kwargs)

    monkeypatch.setattr(router, "subscribe", delayed_subscribe)

    def authenticate(request):
        return _capability() if request.headers.get("authorization") == f"Bearer {bearer}" else None

    app, _ = app_for(
        tmp_path,
        output_router=router,
        authenticate_http=authenticate,
        authenticate_websocket=lambda websocket: None,
        allow_ambient_authentication=False,
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app, base_url="https://studio.example") as client:
        assert client.post("/session", headers={"authorization": f"Bearer {bearer}"}).status_code == 200
        with client.websocket_connect(
            "wss://studio.example/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(_private_subscription_request("racing-private"))
            assert entered.wait(timeout=1)
            assert client.delete("/session").status_code == 204
            release.set()
            with pytest.raises(WebSocketDisconnect):
                receive_json_bounded(websocket)

        assert client.post("/session", headers={"authorization": f"Bearer {bearer}"}).status_code == 200
        with client.websocket_connect(
            "wss://studio.example/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as replacement:
            replacement.send_json({
                "jsonrpc": "2.0", "id": "public-rebind", "method": "stream.subscribe",
                "params": {
                    "subscription_id": "racing-private", "run_id": "run-1",
                    "stream_id": "stream-1", "kinds": [], "audience": "public",
                },
            })
            assert "result" in receive_json_bounded(replacement)


def test_expiry_revokes_live_private_websocket_before_private_delivery(tmp_path: Path) -> None:
    now = [10.0]
    bearer = "expiry-bearer"
    router = StudioOutputRouter()

    def authenticate(request):
        return _capability() if request.headers.get("authorization") == f"Bearer {bearer}" else None

    app, _ = app_for(
        tmp_path,
        output_router=router,
        authenticate_http=authenticate,
        authenticate_websocket=lambda websocket: None,
        allow_ambient_authentication=False,
        clock=lambda: now[0],
        limits=WorldStudioServerLimits(session_lifetime_seconds=5),
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app, base_url="https://studio.example") as client:
        assert client.post("/session", headers={"authorization": f"Bearer {bearer}"}).status_code == 200
        with client.websocket_connect(
            "wss://studio.example/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(_private_subscription_request("expiry-private"))
            assert "result" in receive_json_bounded(websocket)
            now[0] = 15.0
            router.publish("run-1", private_batch())
            with pytest.raises(WebSocketDisconnect) as raised:
                receive_json_bounded(websocket)
            assert raised.value.code == 1008


def test_websocket_subscription_requires_closed_authorized_audience_and_owner(
    tmp_path: Path,
) -> None:
    restricted = _capability(permissions=("output.read",))
    app, _ = app_for(
        tmp_path,
        authenticate_websocket=lambda websocket: restricted,
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            def subscribe(identity: str, **extra):
                params = {
                    "subscription_id": identity, "run_id": "run-1",
                    "stream_id": "stream-1", "kinds": [], **extra,
                }
                websocket.send_json({
                    "jsonrpc": "2.0", "id": identity,
                    "method": "stream.subscribe", "params": params,
                })
                return receive_json_bounded(websocket)

            assert subscribe("missing")["error"]["code"] == -32602
            assert subscribe("objective", audience="objective")["error"]["code"] == -32602
            assert subscribe("analyst", audience="analyst")["error"]["code"] == -32010
            assert subscribe("public-owner", audience="public", owner_agent_id="alice")["error"]["code"] == -32602
            assert subscribe("agent-ownerless", audience="agent")["error"]["code"] == -32602
            accepted = subscribe("agent-alice", audience="agent", owner_agent_id="alice")
            assert accepted["result"] == {
                "subscription_id": "agent-alice",
                "run_id": "run-1",
                "stream_id": "stream-1",
                "kinds": [],
                "audience": "agent",
                "owner_agent_id": "alice",
            }


def test_static_index_assets_csp_and_traversal_policy_are_exact(tmp_path: Path) -> None:
    static = tmp_path / "static"
    assets = static / "assets"
    assets.mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>Studio</title>", encoding="utf-8")
    (assets / "index-a1b2c3d4.js").write_text("export {};", encoding="utf-8")
    (assets / "unhashed.js").write_text("export {};", encoding="utf-8")
    authority = tmp_path / "authority"
    authority.mkdir()
    app, _ = app_for(authority, static_root=static)

    with TestClient(app) as client:
        index = client.get("/")
        studio = client.get("/studio/")
        deep_link = client.get("/studio/projects/law-firm")
        immutable = client.get("/studio/assets/index-a1b2c3d4.js")
        unhashed = client.get("/studio/assets/unhashed.js")
        unprefixed = client.get("/assets/index-a1b2c3d4.js")
        asset_namespace = client.get("/studio/assets")
        missing_asset = client.get("/studio/assets/missing-a1b2c3d4.js")
        traversal = client.get("/studio/%2e%2e/authority/projects.sqlite3")
        double_encoded_traversal = client.get("/studio/%252e%252e/authority/projects.sqlite3")

    for response in (
        index, studio, deep_link, immutable, unhashed, unprefixed, asset_namespace,
        missing_asset, traversal, double_encoded_traversal,
    ):
        csp = response.headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "connect-src 'self'" in csp
        assert "worker-src 'self' blob:" in csp
        assert "unsafe-inline" not in csp
        assert "unsafe-eval" not in csp
    assert index.status_code == studio.status_code == deep_link.status_code == 200
    assert index.text == studio.text == deep_link.text
    assert index.headers["cache-control"] == "no-cache"
    assert studio.headers["cache-control"] == "no-cache"
    assert deep_link.headers["cache-control"] == "no-cache"
    assert immutable.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert unhashed.headers["cache-control"] == "no-cache"
    assert (
        unprefixed.status_code == asset_namespace.status_code == missing_asset.status_code ==
        traversal.status_code == double_encoded_traversal.status_code == 404
    )


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


class BlockingOnceAuthenticator:
    def __init__(self, capability: StudioCapability) -> None:
        self._capability = capability
        self._block_next = True

    def __call__(self, request) -> StudioCapability:
        del request
        if self._block_next:
            self._block_next = False
            time.sleep(0.2)
        return self._capability


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


def test_blocking_sync_http_authentication_times_out_without_stalling_loop(
    tmp_path: Path,
) -> None:
    blocking = BlockingOnceAuthenticator(_capability())
    app, _ = app_for(
        tmp_path,
        dispatcher=ImmediateDispatcher(),
        authenticate_http=blocking,
        limits=WorldStudioServerLimits(request_timeout_seconds=0.05),
    )
    with TestClient(app) as client:
        started = time.monotonic()
        response = client.post(
            "/rpc",
            content=b'{"jsonrpc":"2.0","id":1,"method":"run.list"}',
            headers={"content-type": "application/json"},
        )
        elapsed = time.monotonic() - started

    assert response.status_code == 504
    assert response.json() == {"error": {"code": "timeout"}}
    assert elapsed < 0.15


def test_blocking_sync_websocket_auth_times_out_without_claiming_connection(
    tmp_path: Path,
) -> None:
    blocking = BlockingOnceAuthenticator(_capability())
    app, _ = app_for(
        tmp_path,
        authenticate_websocket=blocking,
        limits=WorldStudioServerLimits(
            request_timeout_seconds=0.05,
            maximum_websocket_connections=1,
        ),
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app) as client:
        started = time.monotonic()
        with pytest.raises(WebSocketDisconnect) as raised:
            with client.websocket_connect(
                "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
            ):
                pass
        elapsed = time.monotonic() - started
        assert raised.value.code == 1013
        assert elapsed < 0.15
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ):
            pass


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
                        "audience": "public",
                    },
                }
            )
            subscribed = receive_json_bounded(websocket)
            assert subscribed["id"] == "subscribe-1"

            batch = public_batch(1)
            router.publish("run-1", batch)
            notification = receive_json_bounded(websocket)
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
                control_response = receive_json_bounded(websocket)
                assert control_response["id"] == request_id
                assert "result" in control_response


def test_nonempty_resume_replays_only_as_ordered_id_free_notifications(
    tmp_path: Path,
) -> None:
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
                    "id": "subscribe-replay",
                    "method": "stream.subscribe",
                    "params": {
                        "subscription_id": "sub-replay",
                        "run_id": "run-1",
                        "stream_id": "stream-1",
                        "kinds": ["command.result"],
                        "audience": "public",
                    },
                }
            )
            assert "result" in receive_json_bounded(websocket)
            batches = tuple(public_batch_range(sequence, sequence) for sequence in (1, 2, 3))
            router.publish("run-1", batches[0])
            receive_json_bounded(websocket)
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "ack-replay",
                    "method": "stream.acknowledge",
                    "params": {
                        "subscription_id": "sub-replay",
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": batches[0].content_hash,
                    },
                }
            )
            assert "result" in receive_json_bounded(websocket)
            router.publish("run-1", batches[1])
            router.publish("run-1", batches[2])
            receive_json_bounded(websocket)
            receive_json_bounded(websocket)

            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-replay",
                    "method": "stream.resume",
                    "params": {
                        "subscription_id": "sub-replay",
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": batches[0].content_hash,
                    },
                }
            )
            messages = tuple(receive_json_bounded(websocket) for _ in range(3))

    responses = tuple(item for item in messages if "id" in item)
    notifications = tuple(item for item in messages if "id" not in item)
    assert len(responses) == 1
    assert responses[0]["id"] == "resume-replay"
    assert responses[0]["result"] == {
        "subscription_id": "sub-replay",
        "resumed_from_sequence": 1,
        "replayed_count": 2,
        "current_sequence": 3,
        "current_batch_hash": batches[2].content_hash,
    }
    assert tuple(item["method"] for item in notifications) == (
        "stream.output",
        "stream.output",
    )
    assert tuple(
        item["params"]["output"]["last_sequence"] for item in notifications
    ) == (2, 3)
    assert len(
        {item["params"]["output"]["source_batch_hash"] for item in notifications}
    ) == 2


def test_bounded_gateway_queue_delivers_every_retained_replay_notification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_send_text = WebSocket.send_text
    release_output = threading.Event()
    release_output.set()

    async def slow_output_send(self, data: str) -> None:
        if '"method":"stream.output"' in data and not release_output.is_set():
            await asyncio.to_thread(release_output.wait)
        await original_send_text(self, data)

    monkeypatch.setattr(WebSocket, "send_text", slow_output_send)
    router = StudioOutputRouter()
    app, _ = app_for(tmp_path, output_router=router)
    headers = {"origin": "https://studio.example"}
    batches = tuple(
        public_batch_range(sequence, sequence) for sequence in range(1, 37)
    )
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "subscribe-many",
                    "method": "stream.subscribe",
                    "params": {
                        "subscription_id": "sub-many",
                        "run_id": "run-1",
                        "stream_id": "stream-1",
                        "kinds": [],
                        "audience": "public",
                    },
                }
            )
            receive_json_bounded(websocket)
            for batch in batches:
                router.publish("run-1", batch)
                receive_json_bounded(websocket)
            release_output.clear()
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-many",
                    "method": "stream.resume",
                    "params": {
                        "subscription_id": "sub-many",
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": batches[0].content_hash,
                    },
                }
            )
            release_timer = threading.Timer(0.1, release_output.set)
            release_timer.daemon = True
            release_timer.start()
            messages = tuple(receive_json_bounded(websocket) for _ in range(36))
            release_timer.join(timeout=1)

    notifications = tuple(message for message in messages if "id" not in message)
    assert tuple(
        message["params"]["output"]["last_sequence"] for message in notifications
    ) == tuple(range(2, 37))


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
                receive_json_bounded(websocket)
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
                            "audience": "public",
                        },
                    }
                )
                response = receive_json_bounded(websocket)
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
            response = receive_json_bounded(websocket)

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
            "audience": "public",
        },
    }
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as first:
            first.send_json(request)
            assert "result" in receive_json_bounded(first)
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
            assert "result" in receive_json_bounded(replacement)


def test_websocket_disconnect_releases_active_slot_and_exact_rebind_resumes(
    tmp_path: Path,
) -> None:
    router = StudioOutputRouter()
    app, _ = app_for(
        tmp_path,
        output_router=router,
        limits=WorldStudioServerLimits(maximum_websocket_connections=1),
    )
    headers = {"origin": "https://studio.example"}
    subscribe = {
        "jsonrpc": "2.0",
        "id": "subscribe-reconnect",
        "method": "stream.subscribe",
        "params": {
            "subscription_id": "lease-reconnect",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "kinds": ["command.result"],
            "audience": "public",
        },
    }
    first = public_batch_range(1, 1)
    second = public_batch_range(2, 2)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as initial:
            initial.send_json(subscribe)
            assert "result" in receive_json_bounded(initial)
            router.publish("run-1", first)
            receive_json_bounded(initial)
            initial.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "ack-reconnect",
                    "method": "stream.acknowledge",
                    "params": {
                        "subscription_id": "lease-reconnect",
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": first.content_hash,
                    },
                }
            )
            assert "result" in receive_json_bounded(initial)

        router.publish("run-1", second)
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as replacement:
            replacement.send_json(subscribe)
            assert "result" in receive_json_bounded(replacement)
            replacement.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-reconnect",
                    "method": "stream.resume",
                    "params": {
                        "subscription_id": "lease-reconnect",
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": first.content_hash,
                    },
                }
            )
            messages = (receive_json_bounded(replacement), receive_json_bounded(replacement))

    response = next(item for item in messages if "id" in item)
    notification = next(item for item in messages if "id" not in item)
    assert response["result"]["replayed_count"] == 1
    assert notification["params"]["output"]["source_batch_hash"] == second.content_hash


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
            "audience": "public",
        },
    }
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(request)
            timed_out = receive_json_bounded(websocket)
            assert timed_out["id"] == "slow-subscribe"
            assert timed_out["error"]["code"] == -32603
        time.sleep(0.25)
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as replacement:
            replacement.send_json(request)
            assert "result" in receive_json_bounded(replacement)


def test_late_subscribe_cleanup_preserves_subsequently_created_lease(
    tmp_path: Path,
) -> None:
    router = SlowSubscribeRouter()
    app, _ = app_for(
        tmp_path,
        output_router=router,
        limits=WorldStudioServerLimits(request_timeout_seconds=0.05),
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            for subscription_id in ("late-a", "later-b"):
                websocket.send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": subscription_id,
                        "method": "stream.subscribe",
                        "params": {
                            "subscription_id": subscription_id,
                            "run_id": "run-1",
                            "stream_id": "stream-1",
                            "kinds": [],
                            "audience": "public",
                        },
                    }
                )
                response = receive_json_bounded(websocket)
                if subscription_id == "late-a":
                    assert response["id"] == "late-a"
                    assert response["error"]["code"] == -32603
                else:
                    assert "result" in response
            time.sleep(0.25)
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "remove-later-b",
                    "method": "stream.unsubscribe",
                    "params": {"subscription_id": "later-b"},
                }
            )
            assert "result" in receive_json_bounded(websocket)


class SlowControlRouter(StudioOutputRouter):
    def __init__(self, delayed_method: str) -> None:
        super().__init__()
        self._delayed_method = delayed_method
        self._delayed = False

    def arm(self) -> None:
        self._delayed = True

    def _delay(self, method: str, subscription_id: str) -> None:
        if self._delayed and method == self._delayed_method and subscription_id == "sub-a":
            self._delayed = False
            time.sleep(0.2)

    def acknowledge(self, subscription_id, *args, **kwargs):
        self._delay("acknowledge", subscription_id)
        return super().acknowledge(subscription_id, *args, **kwargs)

    def resume(self, subscription_id, *args, **kwargs):
        self._delay("resume", subscription_id)
        return super().resume(subscription_id, *args, **kwargs)

    def unsubscribe(self, subscription_id, *args, **kwargs):
        self._delay("unsubscribe", subscription_id)
        return super().unsubscribe(subscription_id, *args, **kwargs)


@pytest.mark.parametrize("method", ("acknowledge", "resume", "unsubscribe"))
def test_timed_out_control_cleanup_does_not_remove_other_leases(
    tmp_path: Path, method: str
) -> None:
    router = SlowControlRouter(method)
    app, _ = app_for(
        tmp_path,
        output_router=router,
        limits=WorldStudioServerLimits(request_timeout_seconds=0.05),
    )
    headers = {"origin": "https://studio.example"}
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            for subscription_id in ("sub-a", "sub-b"):
                websocket.send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": f"create-{subscription_id}",
                        "method": "stream.subscribe",
                        "params": {
                            "subscription_id": subscription_id,
                            "run_id": "run-1",
                            "stream_id": "stream-1",
                            "kinds": [],
                            "audience": "public",
                        },
                    }
                )
                assert "result" in receive_json_bounded(websocket)
            batch = public_batch_range(1, 1)
            router.publish("run-1", batch)
            receive_json_bounded(websocket)
            receive_json_bounded(websocket)
            router.arm()
            params = {"subscription_id": "sub-a"}
            if method != "unsubscribe":
                params.update(
                    {
                        "stream_id": "stream-1",
                        "last_sequence": 1,
                        "last_batch_hash": batch.content_hash,
                    }
                )
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": f"timeout-{method}",
                    "method": f"stream.{method}",
                    "params": params,
                }
            )
            timed_out = receive_json_bounded(websocket)
            assert timed_out["id"] == f"timeout-{method}"
            assert timed_out["error"]["code"] == -32603
            time.sleep(0.25)
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "remove-sub-b",
                    "method": "stream.unsubscribe",
                    "params": {"subscription_id": "sub-b"},
                }
            )
            assert "result" in receive_json_bounded(websocket)


def test_oversized_output_closes_connection_without_sending_or_leaking_lease(
    tmp_path: Path,
) -> None:
    router = StudioOutputRouter()
    app, _ = app_for(
        tmp_path,
        output_router=router,
        limits=WorldStudioServerLimits(maximum_websocket_frame_bytes=350),
    )
    headers = {"origin": "https://studio.example"}
    subscribe = {
        "jsonrpc": "2.0",
        "id": "oversized-subscribe",
        "method": "stream.subscribe",
        "params": {
            "subscription_id": "oversized",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "kinds": ["percept.private"],
            "audience": "agent",
            "owner_agent_id": "alice",
        },
    }
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as websocket:
            websocket.send_json(subscribe)
            assert "result" in receive_json_bounded(websocket)
            router.publish("run-1", private_batch())
            with pytest.raises(WebSocketDisconnect) as raised:
                receive_json_bounded(websocket)
            assert raised.value.code == 1009

        replacement_request = dict(subscribe)
        replacement_request["id"] = "replacement-subscribe"
        with client.websocket_connect(
            "/v1/stream", headers=headers, subprotocols=["nd-jsonrpc-v1"]
        ) as replacement:
            replacement.send_json(replacement_request)
            assert "result" in receive_json_bounded(replacement)


def test_accept_failure_releases_connection_capacity(tmp_path: Path) -> None:
    app, _ = app_for(
        tmp_path,
        limits=WorldStudioServerLimits(maximum_websocket_connections=1),
    )
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "scheme": "wss",
        "server": ("testserver", 443),
        "client": ("testclient", 50000),
        "root_path": "",
        "path": "/v1/stream",
        "raw_path": b"/v1/stream",
        "query_string": b"",
        "headers": [
            (b"origin", b"https://studio.example"),
            (b"sec-websocket-protocol", b"nd-jsonrpc-v1"),
        ],
        "subprotocols": ["nd-jsonrpc-v1"],
        "state": {},
    }

    async def exercise() -> list[dict]:
        first_events = deque(({"type": "websocket.connect"},))

        async def first_receive():
            return first_events.popleft()

        async def failing_send(message):
            if message["type"] == "websocket.accept":
                raise RuntimeError("accept transport failed")

        with pytest.raises(RuntimeError):
            await asyncio.wait_for(app(scope, first_receive, failing_send), timeout=1)

        second_events = deque(
            ({"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1000})
        )
        sent = []

        async def second_receive():
            return second_events.popleft()

        async def second_send(message):
            sent.append(message)

        await asyncio.wait_for(app(scope, second_receive, second_send), timeout=1)
        return sent

    sent = asyncio.run(exercise())

    assert any(message["type"] == "websocket.accept" for message in sent)


def test_sender_failure_releases_subscription_and_connection_capacity(
    tmp_path: Path,
) -> None:
    app, _ = app_for(
        tmp_path,
        limits=WorldStudioServerLimits(maximum_websocket_connections=1),
    )
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "scheme": "wss",
        "server": ("testserver", 443),
        "client": ("testclient", 50000),
        "root_path": "",
        "path": "/v1/stream",
        "raw_path": b"/v1/stream",
        "query_string": b"",
        "headers": [
            (b"origin", b"https://studio.example"),
            (b"sec-websocket-protocol", b"nd-jsonrpc-v1"),
        ],
        "subprotocols": ["nd-jsonrpc-v1"],
        "state": {},
    }
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "sender-failure",
            "method": "stream.subscribe",
            "params": {
                "subscription_id": "sender-lease",
                "run_id": "run-1",
                "stream_id": "stream-1",
                "kinds": [],
                "audience": "public",
            },
        },
        separators=(",", ":"),
    )

    async def exercise() -> list[dict]:
        first_events = deque(
            (
                {"type": "websocket.connect"},
                {"type": "websocket.receive", "text": request},
            )
        )

        async def first_receive():
            return first_events.popleft()

        async def failing_send(message):
            if message["type"] == "websocket.send":
                raise RuntimeError("sender transport failed")

        await asyncio.wait_for(app(scope, first_receive, failing_send), timeout=1)

        second_events = deque(
            (
                {"type": "websocket.connect"},
                {"type": "websocket.receive", "text": request},
                {"type": "websocket.disconnect", "code": 1000},
            )
        )
        sent = []

        async def second_receive():
            return second_events.popleft()

        async def second_send(message):
            sent.append(message)

        await asyncio.wait_for(app(scope, second_receive, second_send), timeout=1)
        return sent

    sent = asyncio.run(exercise())
    payloads = tuple(
        json.loads(message["text"])
        for message in sent
        if message["type"] == "websocket.send"
    )

    assert any(message["type"] == "websocket.accept" for message in sent)
    assert any("result" in payload for payload in payloads)


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
