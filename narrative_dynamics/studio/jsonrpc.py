"""Strict JSON-RPC 2.0 dispatcher for the transport-neutral studio service."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TypeAlias

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.studio.capabilities import StudioCapability
from narrative_dynamics.studio.service import (
    JsonObject,
    StudioError,
    StudioInvalidParamsError,
    StudioMethodNotFoundError,
    WorldStudioService,
)


JsonRpcId: TypeAlias = None | int | str


@dataclass(frozen=True)
class JsonRpcLimits:
    maximum_bytes: int = 1_048_576
    maximum_depth: int = 64
    maximum_members: int = 20_000
    maximum_string_bytes: int = 262_144

    def __post_init__(self) -> None:
        for name in (
            "maximum_bytes",
            "maximum_depth",
            "maximum_members",
            "maximum_string_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"JSON-RPC {name.replace('_', ' ')} must be an integer")
            if value <= 0:
                raise ValueError(f"JSON-RPC {name.replace('_', ' ')} must be positive")


class _ParseViolation(ValueError):
    pass


_STANDARD_MESSAGES = {
    -32700: "Parse error",
    -32600: "Invalid Request",
    -32601: "Method not found",
    -32602: "Invalid params",
    -32603: "Internal error",
}

_APPLICATION_MESSAGES = {
    -32010: "Unauthorized",
    -32011: "Stale state",
    -32012: "Validation failed",
    -32013: "Invalid run lifecycle",
    -32014: "Not found",
    -32015: "Identity conflict",
    -32016: "History gap",
    -32017: "Capacity limit",
}


def _error(
    code: int,
    request_id: JsonRpcId,
    *,
    data: JsonObject | None = None,
) -> JsonObject:
    message = (
        _STANDARD_MESSAGES[code]
        if code in _STANDARD_MESSAGES
        else _APPLICATION_MESSAGES[code]
    )
    detail: JsonObject = {"code": code, "message": message}
    if data:
        detail["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": detail}


def _valid_id(value: object) -> bool:
    return value is None or isinstance(value, str) or (
        isinstance(value, int) and not isinstance(value, bool)
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _ParseViolation("duplicate JSON object member")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    del value
    raise _ParseViolation("non-finite JSON number")


def _validate_tree(value: object, limits: JsonRpcLimits, *, depth: int = 1) -> int:
    if depth > limits.maximum_depth:
        raise _ParseViolation("JSON nesting limit exceeded")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _ParseViolation("non-finite JSON number")
        return 0
    if isinstance(value, str):
        if len(value.encode("utf-8")) > limits.maximum_string_bytes:
            raise _ParseViolation("JSON string limit exceeded")
        return 0
    if isinstance(value, list):
        members = len(value)
        for item in value:
            members += _validate_tree(item, limits, depth=depth + 1)
            if members > limits.maximum_members:
                raise _ParseViolation("JSON member limit exceeded")
        return members
    if isinstance(value, dict):
        members = len(value)
        for key, item in value.items():
            if not isinstance(key, str):
                raise _ParseViolation("JSON object key must be text")
            if len(key.encode("utf-8")) > limits.maximum_string_bytes:
                raise _ParseViolation("JSON string limit exceeded")
            members += _validate_tree(item, limits, depth=depth + 1)
            if members > limits.maximum_members:
                raise _ParseViolation("JSON member limit exceeded")
        return members
    raise _ParseViolation("value is not JSON")


class JsonRpcDispatcher:
    def __init__(
        self,
        service: WorldStudioService,
        *,
        limits: JsonRpcLimits | None = None,
    ) -> None:
        if not isinstance(service, WorldStudioService):
            raise TypeError("JSON-RPC dispatcher requires WorldStudioService")
        self._service = service
        self._limits = limits or JsonRpcLimits()
        if not isinstance(self._limits, JsonRpcLimits):
            raise TypeError("JSON-RPC dispatcher limits must be JsonRpcLimits")

    @property
    def limits(self) -> JsonRpcLimits:
        return self._limits

    @staticmethod
    def _internal_error(method: str, request_id: JsonRpcId) -> JsonObject:
        diagnostic_id = stable_content_hash(
            {"code": "jsonrpc_internal_error", "method": method}
        )
        return _error(-32603, request_id, data={"diagnostic_id": diagnostic_id})

    def dispatch(
        self,
        request: JsonObject,
        capability: StudioCapability,
    ) -> JsonObject | None:
        try:
            _validate_tree(request, self._limits)
        except _ParseViolation:
            return _error(-32600, None)
        if not isinstance(request, dict):
            return _error(-32600, None)

        request_id: JsonRpcId = None
        if "id" in request and _valid_id(request["id"]):
            request_id = request["id"]
        allowed = {"jsonrpc", "id", "method", "params"}
        if (
            not set(request).issubset(allowed)
            or "jsonrpc" not in request
            or request.get("jsonrpc") != "2.0"
            or "method" not in request
            or not isinstance(request.get("method"), str)
            or not request["method"]
            or ("id" in request and not _valid_id(request["id"]))
            or ("params" in request and not isinstance(request["params"], dict))
        ):
            return _error(-32600, request_id)

        method = request["method"]
        notification = "id" not in request
        if notification and self._service.is_state_changing(method):
            return _error(-32600, None)
        params = request.get("params", {})
        try:
            result = self._service.invoke(method, params, capability)
        except StudioMethodNotFoundError:
            response = _error(-32601, request_id)
        except StudioInvalidParamsError:
            response = _error(-32602, request_id)
        except StudioError as error:
            response = _error(
                error.rpc_code,
                request_id,
                data=error.data,
            )
        except Exception:
            response = self._internal_error(method, request_id)
        else:
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        return None if notification else response

    @staticmethod
    def _encode(response: JsonObject) -> bytes:
        return json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def parse_and_dispatch(
        self,
        raw_utf8: bytes,
        capability: StudioCapability,
    ) -> bytes:
        if not isinstance(raw_utf8, bytes):
            raise TypeError("JSON-RPC payload must be bytes")
        if len(raw_utf8) > self._limits.maximum_bytes:
            return self._encode(_error(-32700, None))
        try:
            text = raw_utf8.decode("utf-8", errors="strict")
            request = json.loads(
                text,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
            _validate_tree(request, self._limits)
        except (UnicodeDecodeError, json.JSONDecodeError, _ParseViolation, ValueError):
            return self._encode(_error(-32700, None))
        response = self.dispatch(request, capability)
        if response is None:
            return b""
        try:
            return self._encode(response)
        except (TypeError, ValueError, OverflowError):
            method = request.get("method", "") if isinstance(request, dict) else ""
            request_id = request.get("id") if isinstance(request, dict) and _valid_id(request.get("id")) else None
            return self._encode(self._internal_error(method, request_id))


__all__ = ("JsonRpcDispatcher", "JsonRpcId", "JsonRpcLimits")
