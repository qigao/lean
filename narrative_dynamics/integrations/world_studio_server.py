"""Optional Starlette boundary for World Studio JSON-RPC and streaming."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import re
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

    def __post_init__(self) -> None:
        for name in (
            "maximum_http_body_bytes",
            "maximum_websocket_frame_bytes",
            "maximum_websocket_connections",
            "maximum_subscriptions_per_connection",
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
):
    """Build the optional ASGI app; Hypercorn supplies HTTP/2 and TLS at deployment."""
    import asyncio
    import json

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

    connection_lock = asyncio.Lock()
    active_connections = 0
    next_connection = 0

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

    async def authenticate(callback, host_object):
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

    async def session(request):
        try:
            capability = await asyncio.wait_for(
                authenticate(authenticate_http, request),
                timeout=float(limits.request_timeout_seconds),
            )
        except asyncio.TimeoutError:
            return transport_error("timeout", 504)
        if capability is None:
            return transport_error("unauthorized", 401)
        return JSONResponse(
            {
                "schema": "narrative-dynamics.studio-session/v1",
                "authority": {
                    "authority_id": capability.authority_id,
                    "project_ids": capability.project_ids,
                    "run_ids": capability.run_ids,
                    "agent_ids": capability.agent_ids,
                    "permissions": capability.permissions,
                },
            },
            headers={"cache-control": "no-store"},
        )

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
                authenticate(authenticate_http, request),
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
        if "\\" in decoded_path or any(
            segment in {".", ".."} for segment in decoded_path.split("/")
        ):
            return Response(status_code=404)
        relative = request.path_params.get("path", "")
        if relative in {"", "studio", "studio/"}:
            target = resolved_static_root / "index.html"
            cache_control = "no-store"
        else:
            try:
                target = (resolved_static_root / relative).resolve(strict=True)
            except (OSError, RuntimeError):
                return Response(status_code=404)
            if not target.is_relative_to(resolved_static_root) or not target.is_file():
                return Response(status_code=404)
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
        try:
            capability = await asyncio.wait_for(
                authenticate(authenticate_websocket, websocket),
                timeout=float(limits.request_timeout_seconds),
            )
        except asyncio.TimeoutError:
            await websocket.close(code=1013)
            return
        if capability is None:
            await websocket.close(code=1008)
            return
        async with connection_lock:
            if active_connections >= limits.maximum_websocket_connections:
                await websocket.close(code=1013)
                return
            active_connections += 1
            next_connection += 1
            connection_id = f"connection-{next_connection}"

        tasks: set[asyncio.Task] = set()
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

            class OutboundFrameTooLarge(RuntimeError):
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
                    await websocket.send_text(encoded.decode("utf-8"))

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
                    control_task = asyncio.create_task(
                        asyncio.to_thread(
                            streaming_dispatcher.parse_and_dispatch, raw, capability
                        )
                    )
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
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            failures = await asyncio.gather(*done, return_exceptions=True)
            if any(
                isinstance(failure, Exception)
                and not isinstance(failure, OutboundFrameTooLarge)
                for failure in failures
            ):
                try:
                    await websocket.close(code=1011)
                except Exception:
                    pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                output_router.unsubscribe_connection(connection_id)
            finally:
                async with connection_lock:
                    active_connections -= 1

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/session", session, methods=["GET"]),
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
    "WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL",
    "WorldStudioServerLimits",
    "create_world_studio_asgi_app",
)
