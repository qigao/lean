"""Optional Starlette boundary for World Studio JSON-RPC and streaming."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import inspect
from pathlib import Path
import re
import secrets
from threading import RLock
from time import monotonic
from typing import Callable
from urllib.parse import unquote

from narrative_dynamics.studio.capabilities import StudioCapability
from narrative_dynamics.studio.jsonrpc import JsonRpcDispatcher
from narrative_dynamics.studio.streaming import (
    StudioOutputRouter,
    StudioStreamingService,
    studio_output_view_to_dict,
)


WORLD_STUDIO_PROTOCOL_VERSION = "narrative-dynamics.world-studio/v1"
WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL = "nd-jsonrpc-v1"
WORLD_STUDIO_SESSION_COOKIE = "world_studio_session"
_HASHED_ASSET = re.compile(r"-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$")
_CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "base-uri 'none'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "worker-src 'self' blob:",
    )
)


@dataclass(frozen=True)
class WorldStudioServerLimits:
    maximum_http_body_bytes: int = 1_048_576
    maximum_websocket_frame_bytes: int = 1_048_576
    request_timeout_seconds: float = 30.0
    maximum_websocket_connections: int = 128
    maximum_subscriptions_per_connection: int = 16
    maximum_sessions: int = 128
    session_lifetime_seconds: int = 3_600

    def __post_init__(self) -> None:
        for name in (
            "maximum_http_body_bytes",
            "maximum_websocket_frame_bytes",
            "maximum_websocket_connections",
            "maximum_subscriptions_per_connection",
            "maximum_sessions",
            "session_lifetime_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"World Studio {name.replace('_', ' ')} must be an integer")
            if value <= 0:
                raise ValueError(f"World Studio {name.replace('_', ' ')} must be positive")
        if (
            not isinstance(self.request_timeout_seconds, (int, float))
            or isinstance(self.request_timeout_seconds, bool)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("World Studio request timeout seconds must be positive")


def _origins(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("World Studio allowed origins must be a non-empty tuple")
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or value == "*"
        for value in values
    ):
        raise ValueError("World Studio allowed origins must be bounded exact origins")
    if len(set(values)) != len(values):
        raise ValueError("World Studio allowed origins must be unique")
    return tuple(sorted(values))


def create_world_studio_asgi_app(
    dispatcher,
    output_router: StudioOutputRouter,
    authenticate_http: Callable[[object], StudioCapability | None],
    authenticate_websocket: Callable[[object], StudioCapability | None],
    *,
    allowed_origins: tuple[str, ...],
    limits: WorldStudioServerLimits,
    static_root=None,
    allow_ambient_authentication: bool = True,
    clock: Callable[[], float] = monotonic,
):
    """Build the optional ASGI app; Hypercorn supplies HTTP/2 and TLS at deployment."""
    import asyncio
    import json

    from anyio import CancelScope
    from starlette.applications import Starlette
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import FileResponse, JSONResponse, Response
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    if not callable(getattr(dispatcher, "parse_and_dispatch", None)):
        raise TypeError("World Studio server requires a JSON-RPC dispatcher")
    if not isinstance(output_router, StudioOutputRouter):
        raise TypeError("World Studio server requires StudioOutputRouter")
    if not callable(authenticate_http) or not callable(authenticate_websocket):
        raise TypeError("World Studio authenticators must be callable")
    origins = _origins(allowed_origins)
    if not isinstance(limits, WorldStudioServerLimits):
        raise TypeError("World Studio server limits must be WorldStudioServerLimits")
    if not isinstance(allow_ambient_authentication, bool):
        raise TypeError("World Studio ambient authentication flag must be boolean")
    if not callable(clock):
        raise TypeError("World Studio session clock must be callable")
    resolved_static_root = None
    if static_root is not None:
        try:
            resolved_static_root = Path(static_root).resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError("World Studio static root is unavailable") from None
        if not resolved_static_root.is_dir():
            raise ValueError("World Studio static root must be a directory")
        if not (resolved_static_root / "index.html").is_file():
            raise ValueError("World Studio static root must contain index.html")

    connection_condition = asyncio.Condition()
    active_connections = 0
    next_connection = 0
    session_lock = RLock()
    sessions: OrderedDict[str, tuple[StudioCapability, float]] = OrderedDict()
    session_connections: dict[str, dict[str, Callable[[], None]]] = {}

    def transport_error(code: str, status: int):
        return JSONResponse({"error": {"code": code}}, status_code=status)

    def control_metadata(raw: bytes):
        def strict_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError
                result[key] = value
            return result

        try:
            request = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=strict_object,
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return None, None, None
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            return None, None, None
        request_id = request.get("id") if "id" in request else None
        if not (
            request_id is None
            or isinstance(request_id, str)
            or isinstance(request_id, int)
            and not isinstance(request_id, bool)
        ):
            request_id = None
        method = request.get("method")
        params = request.get("params", {})
        subscription_id = None
        if isinstance(params, dict):
            candidate = params.get("subscription_id")
            if (
                isinstance(candidate, str)
                and candidate
                and len(candidate) <= 256
                and "/" not in candidate
                and "\\" not in candidate
            ):
                subscription_id = candidate
        return request_id, method if isinstance(method, str) else None, subscription_id

    def invalidate_sessions(identities: tuple[str, ...]) -> tuple[Callable[[], None], ...]:
        callbacks: list[Callable[[], None]] = []
        with session_lock:
            for identity in identities:
                sessions.pop(identity, None)
                callbacks.extend(session_connections.pop(identity, {}).values())
        return tuple(callbacks)

    @staticmethod
    def signal_revocation(callbacks: tuple[Callable[[], None], ...]) -> None:
        for callback in callbacks:
            try:
                callback()
            except Exception:
                continue

    def purge_expired_sessions(now: float) -> None:
        with session_lock:
            expired = tuple(
                identity for identity, (_, deadline) in sessions.items()
                if deadline <= now
            )
        signal_revocation(invalidate_sessions(expired))

    def session_binding(host_object) -> tuple[str, StudioCapability, float] | None:
        session_id = host_object.cookies.get(WORLD_STUDIO_SESSION_COOKIE)
        if not isinstance(session_id, str) or not session_id:
            return None
        now = clock()
        purge_expired_sessions(now)
        with session_lock:
            retained = sessions.get(session_id)
            return None if retained is None else (session_id, retained[0], retained[1])

    def session_capability(host_object) -> StudioCapability | None:
        retained = session_binding(host_object)
        return None if retained is None else retained[1]

    def session_is_valid(session_id: str, deadline: float) -> bool:
        now = clock()
        purge_expired_sessions(now)
        with session_lock:
            retained = sessions.get(session_id)
            return retained is not None and retained[1] == deadline and deadline > now

    def bind_session_connection(
        session_id: str,
        deadline: float,
        connection_id: str,
        revoke: Callable[[], None],
    ) -> bool:
        now = clock()
        purge_expired_sessions(now)
        with session_lock:
            retained = sessions.get(session_id)
            if retained is None or retained[1] != deadline or deadline <= now:
                return False
            session_connections.setdefault(session_id, {})[connection_id] = revoke
            return True

    def unbind_session_connection(session_id: str, connection_id: str) -> None:
        with session_lock:
            connections = session_connections.get(session_id)
            if connections is None:
                return
            connections.pop(connection_id, None)
            if not connections:
                session_connections.pop(session_id, None)

    def create_session(capability: StudioCapability) -> str:
        now = clock()
        purge_expired_sessions(now)
        evicted: tuple[Callable[[], None], ...] = ()
        with session_lock:
            evicted_ids: list[str] = []
            while len(sessions) >= limits.maximum_sessions:
                identity, _ = sessions.popitem(last=False)
                evicted_ids.append(identity)
            callbacks: list[Callable[[], None]] = []
            for identity in evicted_ids:
                callbacks.extend(session_connections.pop(identity, {}).values())
            evicted = tuple(callbacks)
            for _ in range(8):
                session_id = secrets.token_urlsafe(32)
                if session_id not in sessions:
                    sessions[session_id] = (
                        capability,
                        now + float(limits.session_lifetime_seconds),
                    )
                    signal_revocation(evicted)
                    return session_id
        raise RuntimeError("World Studio session identity allocation failed")

    async def authenticate(
        callback,
        host_object,
        *,
        allow_ambient: bool = True,
        allow_session: bool = True,
    ):
        if allow_session:
            retained = session_capability(host_object)
            if retained is not None:
                return retained
        if not allow_ambient:
            return None
        try:
            if inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
                getattr(callback, "__call__", None)
            ):
                capability = callback(host_object)
            else:
                capability = await asyncio.to_thread(callback, host_object)
            if inspect.isawaitable(capability):
                capability = await capability
        except Exception:
            return None
        return capability if isinstance(capability, StudioCapability) else None

    async def health(request):
        del request
        return JSONResponse(
            {"status": "ok", "protocol_version": WORLD_STUDIO_PROTOCOL_VERSION}
        )

    def session_payload(capability: StudioCapability):
        return {
            "schema": "narrative-dynamics.studio-session/v1",
            "authority": {
                "authority_id": capability.authority_id,
                "project_ids": capability.project_ids,
                "run_ids": capability.run_ids,
                "agent_ids": capability.agent_ids,
                "permissions": capability.permissions,
            },
        }

    async def session(request):
        if request.method == "DELETE":
            session_id = request.cookies.get(WORLD_STUDIO_SESSION_COOKIE)
            if isinstance(session_id, str):
                signal_revocation(invalidate_sessions((session_id,)))
            response = Response(status_code=204, headers={"cache-control": "no-store"})
            response.delete_cookie(
                WORLD_STUDIO_SESSION_COOKIE,
                path="/",
                secure=request.url.scheme == "https",
                httponly=True,
                samesite="strict",
            )
            return response
        try:
            if request.method == "POST":
                capability = await asyncio.wait_for(
                    authenticate(
                        authenticate_http,
                        request,
                        allow_ambient=True,
                        allow_session=False,
                    ),
                    timeout=float(limits.request_timeout_seconds),
                )
            else:
                capability = await asyncio.wait_for(
                    authenticate(
                        authenticate_http,
                        request,
                        allow_ambient=allow_ambient_authentication,
                    ),
                    timeout=float(limits.request_timeout_seconds),
                )
        except asyncio.TimeoutError:
            return transport_error("timeout", 504)
        if capability is None:
            return transport_error("unauthorized", 401)
        response = JSONResponse(
            session_payload(capability), headers={"cache-control": "no-store"}
        )
        if request.method == "POST":
            response.set_cookie(
                WORLD_STUDIO_SESSION_COOKIE,
                create_session(capability),
                max_age=limits.session_lifetime_seconds,
                path="/",
                secure=request.url.scheme == "https",
                httponly=True,
                samesite="strict",
            )
        return response

    async def rpc(request):
        content_type = request.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            return transport_error("unsupported_media_type", 415)
        body = bytearray()
        try:
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > limits.maximum_http_body_bytes:
                    return transport_error("body_limit", 413)
        except Exception:
            return transport_error("invalid_body", 400)
        try:
            capability = await asyncio.wait_for(
                authenticate(
                    authenticate_http,
                    request,
                    allow_ambient=allow_ambient_authentication,
                ),
                timeout=float(limits.request_timeout_seconds),
            )
        except asyncio.TimeoutError:
            return transport_error("timeout", 504)
        if capability is None:
            return transport_error("unauthorized", 401)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    dispatcher.parse_and_dispatch, bytes(body), capability
                ),
                timeout=float(limits.request_timeout_seconds),
            )
        except asyncio.TimeoutError:
            return transport_error("timeout", 504)
        except Exception:
            return transport_error("internal_error", 500)
        if not isinstance(response, bytes):
            return transport_error("internal_error", 500)
        if not response:
            return Response(status_code=204)
        return Response(response, media_type="application/json", status_code=200)

    async def static(request):
        raw_path = request.scope.get("raw_path", b"")
        try:
            decoded_path = unquote(
                raw_path.decode("ascii", errors="strict"),
                errors="strict",
            )
        except (UnicodeDecodeError, UnicodeEncodeError):
            return Response(status_code=404)
        if "\\" in decoded_path or "\x00" in decoded_path or "%" in decoded_path or any(
            segment in {".", ".."} for segment in decoded_path.split("/")
        ):
            return Response(status_code=404)
        if decoded_path == "/":
            relative = ""
        elif decoded_path in {"/studio", "/studio/"}:
            relative = ""
        elif decoded_path.startswith("/studio/"):
            relative = decoded_path[len("/studio/") :]
        else:
            return Response(status_code=404)
        if not relative:
            target = resolved_static_root / "index.html"
            cache_control = "no-cache"
        else:
            try:
                target = (resolved_static_root / relative).resolve(strict=True)
            except (OSError, RuntimeError, ValueError):
                target = None
            if target is None or not target.is_relative_to(resolved_static_root) or not target.is_file():
                if relative == "assets" or relative.startswith("assets/"):
                    return Response(status_code=404)
                target = resolved_static_root / "index.html"
                cache_control = "no-cache"
            else:
                cache_control = (
                    "public, max-age=31536000, immutable"
                    if relative.startswith("assets/") and _HASHED_ASSET.search(target.name)
                    else "no-cache"
                )
        return FileResponse(target, headers={"cache-control": cache_control})

    async def stream(websocket):
        nonlocal active_connections, next_connection
        origin = websocket.headers.get("origin")
        offered = tuple(websocket.scope.get("subprotocols", ()))
        if origin not in origins or WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL not in offered:
            await websocket.close(code=1008)
            return
        cookie_binding = session_binding(websocket)
        if cookie_binding is not None:
            bound_session_id, capability, bound_session_deadline = cookie_binding
        else:
            bound_session_id = None
            bound_session_deadline = None
            try:
                capability = await asyncio.wait_for(
                    authenticate(
                        authenticate_websocket,
                        websocket,
                        allow_ambient=allow_ambient_authentication,
                        allow_session=False,
                    ),
                    timeout=float(limits.request_timeout_seconds),
                )
            except asyncio.TimeoutError:
                await websocket.close(code=1013)
                return
        if capability is None:
            await websocket.close(code=1008)
            return
        admitted = False
        async with connection_condition:
            if active_connections >= limits.maximum_websocket_connections:
                try:
                    await asyncio.wait_for(
                        connection_condition.wait_for(
                            lambda: active_connections < limits.maximum_websocket_connections
                        ),
                        timeout=min(0.05, float(limits.request_timeout_seconds)),
                    )
                except asyncio.TimeoutError:
                    pass
            if active_connections < limits.maximum_websocket_connections:
                active_connections += 1
                next_connection += 1
                connection_id = f"connection-{next_connection}"
                admitted = True
        if not admitted:
            await websocket.close(code=1013)
            return

        tasks: set[asyncio.Task] = set()
        control_tasks: set[asyncio.Task] = set()
        connection_revoked = False
        try:
            loop = asyncio.get_running_loop()
            output_queue: asyncio.Queue = asyncio.Queue(
                maxsize=max(
                    1,
                    output_router.limits.maximum_retained_batches
                    * limits.maximum_subscriptions_per_connection,
                )
            )
            send_lock = asyncio.Lock()
            session_revoked = asyncio.Event()

            def revoke_session() -> None:
                loop.call_soon_threadsafe(session_revoked.set)

            if (
                bound_session_id is not None
                and bound_session_deadline is not None
                and not bind_session_connection(
                    bound_session_id,
                    bound_session_deadline,
                    connection_id,
                    revoke_session,
                )
            ):
                session_revoked.set()

            class OutboundFrameTooLarge(RuntimeError):
                pass

            class SessionRevoked(RuntimeError):
                pass

            def on_output(subscription_id, view) -> None:
                def enqueue() -> None:
                    try:
                        output_queue.put_nowait((subscription_id, view))
                    except asyncio.QueueFull:
                        pass

                loop.call_soon_threadsafe(enqueue)

            streaming_service = StudioStreamingService(
                output_router, connection_id, on_output=on_output
            )
            streaming_dispatcher = JsonRpcDispatcher(streaming_service)
            owned_subscriptions: set[str] = set()

            async def send_payload(payload) -> None:
                if (
                    session_revoked.is_set()
                    or (
                        bound_session_id is not None
                        and bound_session_deadline is not None
                        and not session_is_valid(
                            bound_session_id, bound_session_deadline
                        )
                    )
                ):
                    discard_owned_subscriptions()
                    try:
                        await websocket.close(code=1008)
                    except Exception:
                        pass
                    raise SessionRevoked()
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(encoded) > limits.maximum_websocket_frame_bytes:
                    try:
                        await websocket.close(code=1009)
                    except Exception:
                        pass
                    raise OutboundFrameTooLarge()
                async with send_lock:
                    invalid_at_send = session_revoked.is_set() or (
                        bound_session_id is not None
                        and bound_session_deadline is not None
                        and not session_is_valid(
                            bound_session_id, bound_session_deadline
                        )
                    )
                    if not invalid_at_send:
                        await websocket.send_text(encoded.decode("utf-8"))
                if invalid_at_send:
                    discard_owned_subscriptions()
                    try:
                        await websocket.close(code=1008)
                    except Exception:
                        pass
                    raise SessionRevoked()

            async def output_sender() -> None:
                while True:
                    subscription_id, view = await output_queue.get()
                    await send_payload(
                        {
                            "jsonrpc": "2.0",
                            "method": "stream.output",
                            "params": {
                                "subscription_id": subscription_id,
                                "output": studio_output_view_to_dict(view),
                            },
                        }
                    )

            async def control_receiver() -> None:
                nonlocal owned_subscriptions
                try:
                    await receive_controls()
                except WebSocketDisconnect:
                    return

            def discard_owned_subscriptions() -> None:
                for subscription_id in tuple(owned_subscriptions):
                    try:
                        output_router.unsubscribe(
                            subscription_id,
                            capability=capability,
                            connection_id=connection_id,
                        )
                    except Exception:
                        pass
                    owned_subscriptions.discard(subscription_id)

            async def session_guard() -> None:
                if bound_session_id is None or bound_session_deadline is None:
                    await asyncio.Future()
                    return
                while not session_revoked.is_set():
                    if not session_is_valid(bound_session_id, bound_session_deadline):
                        session_revoked.set()
                        break
                    try:
                        await asyncio.wait_for(session_revoked.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue
                discard_owned_subscriptions()
                try:
                    await websocket.close(code=1008)
                except Exception:
                    pass

            async def receive_controls() -> None:
                nonlocal owned_subscriptions
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("bytes") is not None:
                        raw = message["bytes"]
                    else:
                        text = message.get("text", "")
                        try:
                            raw = text.encode("utf-8", errors="strict")
                        except UnicodeEncodeError:
                            await websocket.close(code=1007)
                            return
                    if len(raw) > limits.maximum_websocket_frame_bytes:
                        await websocket.close(code=1009)
                        return
                    if (
                        bound_session_id is not None
                        and bound_session_deadline is not None
                        and not session_is_valid(bound_session_id, bound_session_deadline)
                    ):
                        discard_owned_subscriptions()
                        await websocket.close(code=1008)
                        return
                    control_task = asyncio.create_task(
                        asyncio.to_thread(
                            streaming_dispatcher.parse_and_dispatch, raw, capability
                        )
                    )
                    control_tasks.add(control_task)

                    def finish_control(completed) -> None:
                        control_tasks.discard(completed)
                        if connection_revoked:
                            output_router.revoke_connection(connection_id)
                            if not control_tasks:
                                output_router.release_connection_revocation(connection_id)

                    control_task.add_done_callback(finish_control)
                    request_id, control_method, control_subscription_id = control_metadata(raw)
                    try:
                        response = await asyncio.wait_for(
                            asyncio.shield(control_task),
                            timeout=float(limits.request_timeout_seconds),
                        )
                    except asyncio.TimeoutError:
                        def finish_timed_out_control(
                            completed,
                            method=control_method,
                            subscription_id=control_subscription_id,
                        ) -> None:
                            try:
                                late_response = completed.result()
                                late_decoded = (
                                    json.loads(late_response) if late_response else None
                                )
                            except Exception:
                                return
                            result = (
                                late_decoded.get("result")
                                if isinstance(late_decoded, dict)
                                else None
                            )
                            if (
                                method == "stream.subscribe"
                                and subscription_id is not None
                                and isinstance(result, dict)
                                and result.get("subscription_id") == subscription_id
                                and "run_id" in result
                            ):
                                try:
                                    output_router.unsubscribe(
                                        subscription_id,
                                        capability=capability,
                                        connection_id=connection_id,
                                    )
                                except Exception:
                                    pass

                        control_task.add_done_callback(finish_timed_out_control)
                        await send_payload(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32603,
                                    "message": "Internal error",
                                },
                            }
                        )
                        continue
                    except Exception:
                        await send_payload(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32603,
                                    "message": "Internal error",
                                },
                            }
                        )
                        continue
                    if not response:
                        continue
                    try:
                        decoded = json.loads(response)
                    except Exception:
                        await send_payload(
                            {
                                "jsonrpc": "2.0",
                                "id": None,
                                "error": {
                                    "code": -32603,
                                    "message": "Internal error",
                                },
                            }
                        )
                        continue
                    result = decoded.get("result") if isinstance(decoded, dict) else None
                    if isinstance(result, dict) and "subscription_id" in result:
                        subscription_id = result["subscription_id"]
                        if result.get("unsubscribed") is True:
                            owned_subscriptions.discard(subscription_id)
                        elif "run_id" in result:
                            owned_subscriptions.add(subscription_id)
                            if (
                                len(owned_subscriptions)
                                > limits.maximum_subscriptions_per_connection
                            ):
                                try:
                                    output_router.unsubscribe(
                                        subscription_id,
                                        capability=capability,
                                        connection_id=connection_id,
                                    )
                                except Exception:
                                    pass
                                owned_subscriptions.discard(subscription_id)
                                decoded = {
                                    "jsonrpc": "2.0",
                                    "id": decoded.get("id"),
                                    "error": {
                                        "code": -32017,
                                        "message": "Capacity limit",
                                    },
                                }
                    await send_payload(decoded)

            await websocket.accept(subprotocol=WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL)
            sender = asyncio.create_task(output_sender())
            tasks.add(sender)
            receiver = asyncio.create_task(control_receiver())
            tasks.add(receiver)
            if bound_session_id is not None:
                guard = asyncio.create_task(session_guard())
                tasks.add(guard)
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            failures = await asyncio.gather(*done, return_exceptions=True)
            if any(
                isinstance(failure, Exception)
                and not isinstance(failure, (OutboundFrameTooLarge, SessionRevoked))
                for failure in failures
            ):
                try:
                    await websocket.close(code=1011)
                except Exception:
                    pass
        finally:
            # AnyIO cancellation repeats at await points, including task draining.
            # Finish releasing ownership and capacity before propagating it.
            with CancelScope(shield=True):
                try:
                    for task in tasks:
                        task.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                finally:
                    try:
                        session_invalid = (
                            bound_session_id is not None
                            and bound_session_deadline is not None
                            and not session_is_valid(bound_session_id, bound_session_deadline)
                        )
                        if session_invalid:
                            connection_revoked = True
                            output_router.revoke_connection(connection_id)
                            if not control_tasks:
                                output_router.release_connection_revocation(connection_id)
                        else:
                            output_router.unsubscribe_connection(connection_id)
                            if control_tasks:
                                connection_revoked = True
                                output_router.revoke_connection(connection_id)
                    finally:
                        if bound_session_id is not None:
                            unbind_session_connection(bound_session_id, connection_id)
                        async with connection_condition:
                            active_connections -= 1
                            connection_condition.notify_all()

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/session", session, methods=["GET", "POST", "DELETE"]),
        Route("/rpc", rpc, methods=["POST"]),
        WebSocketRoute("/v1/stream", stream),
    ]
    if resolved_static_root is not None:
        routes.append(Route("/{path:path}", static, methods=["GET", "HEAD"]))
    app = Starlette(routes=routes)

    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["content-security-policy"] = _CONTENT_SECURITY_POLICY
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=security_headers)
    return app


__all__ = (
    "WORLD_STUDIO_PROTOCOL_VERSION",
    "WORLD_STUDIO_SESSION_COOKIE",
    "WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL",
    "WorldStudioServerLimits",
    "create_world_studio_asgi_app",
)
