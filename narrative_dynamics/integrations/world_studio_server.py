"""Optional Starlette boundary for World Studio JSON-RPC and streaming."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

from narrative_dynamics.studio.capabilities import StudioCapability
from narrative_dynamics.studio.jsonrpc import JsonRpcDispatcher
from narrative_dynamics.studio.streaming import (
    StudioOutputRouter,
    StudioStreamingService,
    studio_output_view_to_dict,
)


WORLD_STUDIO_PROTOCOL_VERSION = "narrative-dynamics.world-studio/v1"
WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL = "nd-jsonrpc-v1"


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
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route, WebSocketRoute
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

    connection_lock = asyncio.Lock()
    active_connections = 0
    next_connection = 0

    def transport_error(code: str, status: int):
        return JSONResponse({"error": {"code": code}}, status_code=status)

    async def authenticate(callback, host_object):
        try:
            capability = callback(host_object)
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

        loop = asyncio.get_running_loop()
        output_queue: asyncio.Queue = asyncio.Queue(
            maxsize=max(1, limits.maximum_subscriptions_per_connection * 2)
        )
        send_lock = asyncio.Lock()

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
            async with send_lock:
                await websocket.send_json(payload)

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

        await websocket.accept(subprotocol=WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL)
        sender = asyncio.create_task(output_sender())
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    raw = message["bytes"]
                else:
                    text = message.get("text", "")
                    try:
                        raw = text.encode("utf-8", errors="strict")
                    except UnicodeEncodeError:
                        await websocket.close(code=1007)
                        break
                if len(raw) > limits.maximum_websocket_frame_bytes:
                    await websocket.close(code=1009)
                    break
                control_task = asyncio.create_task(
                    asyncio.to_thread(
                        streaming_dispatcher.parse_and_dispatch, raw, capability
                    )
                )
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(control_task),
                        timeout=float(limits.request_timeout_seconds),
                    )
                except asyncio.TimeoutError:
                    control_task.add_done_callback(
                        lambda completed: output_router.unsubscribe_connection(
                            connection_id
                        )
                    )
                    await send_payload(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32603, "message": "Internal error"},
                        }
                    )
                    continue
                except Exception:
                    await send_payload(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32603, "message": "Internal error"},
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
                            "error": {"code": -32603, "message": "Internal error"},
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
                        if len(owned_subscriptions) > limits.maximum_subscriptions_per_connection:
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
                                "error": {"code": -32017, "message": "Capacity limit"},
                            }
                await send_payload(decoded)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            try:
                await sender
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            output_router.unsubscribe_connection(connection_id)
            async with connection_lock:
                active_connections -= 1

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/rpc", rpc, methods=["POST"]),
        WebSocketRoute("/v1/stream", stream),
    ]
    if static_root is not None:
        from starlette.staticfiles import StaticFiles

        routes.append(Mount("/", app=StaticFiles(directory=static_root, html=True)))
    return Starlette(routes=routes)


__all__ = (
    "WORLD_STUDIO_PROTOCOL_VERSION",
    "WORLD_STUDIO_WEBSOCKET_SUBPROTOCOL",
    "WorldStudioServerLimits",
    "create_world_studio_asgi_app",
)
