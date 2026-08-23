from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import traceback
from collections.abc import Mapping

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.process_execution import (
    PROCESS_PROTOCOL_VERSION,
    _json_bytes,
    _model_run_payload,
)


_MAX_REQUEST_BYTES = 4 * 1024 * 1024


class _WorkerProtocolError(RuntimeError):
    pass


class _WorkerResourceError(RuntimeError):
    def __init__(self, resource_name: str, message: str) -> None:
        super().__init__(message)
        self.resource_name = resource_name


def _write_result(path: str, payload: Mapping[str, object]) -> None:
    data = _json_bytes(payload)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def _error_payload(
    *,
    kind: str,
    message: str,
    **fields: object,
) -> dict[str, object]:
    return {
        "protocol_version": PROCESS_PROTOCOL_VERSION,
        "status": "error",
        "error": {
            "kind": kind,
            "message": message,
            **fields,
        },
    }


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _WorkerProtocolError(f"{label} must be a mapping")
    return value


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _WorkerProtocolError(f"{label} must be a non-empty string")
    return value


def _required_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _WorkerProtocolError(f"{label} must be an integer")
    return value


def _optional_positive_int(
    value: object,
    *,
    label: str,
) -> int | None:
    if value is None:
        return None
    validated = _required_int(value, label=label)
    if validated <= 0:
        raise _WorkerProtocolError(f"{label} must be positive")
    return validated


def _load_factory(path: str):
    module_name, separator, attribute_path = path.partition(":")
    if not separator or not module_name or not attribute_path:
        raise _WorkerProtocolError(
            "model factory must use 'module:attribute' syntax"
        )
    module = importlib.import_module(module_name)
    value = module
    for attribute in attribute_path.split("."):
        try:
            value = getattr(value, attribute)
        except AttributeError as error:
            raise _WorkerProtocolError(
                f"model factory path {path!r} has no attribute {attribute!r}"
            ) from error
    if not callable(value):
        raise _WorkerProtocolError(f"model factory {path!r} is not callable")
    return value


def _apply_resource_limits(limits: Mapping[str, object]) -> None:
    max_memory_bytes = _optional_positive_int(
        limits.get("max_memory_bytes"),
        label="maximum memory bytes",
    )
    max_cpu_seconds = _optional_positive_int(
        limits.get("max_cpu_seconds"),
        label="maximum CPU seconds",
    )
    if max_memory_bytes is None and max_cpu_seconds is None:
        return

    try:
        import resource
    except ImportError as error:
        resource_name = (
            "memory" if max_memory_bytes is not None else "cpu"
        )
        raise _WorkerResourceError(
            resource_name,
            "requested process resource limits are unsupported on this platform",
        ) from error

    try:
        if max_memory_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_memory_bytes, max_memory_bytes),
            )
        if max_cpu_seconds is not None:
            current_soft, current_hard = resource.getrlimit(resource.RLIMIT_CPU)
            requested_hard = max_cpu_seconds + 1
            if (
                current_hard != resource.RLIM_INFINITY
                and max_cpu_seconds > current_hard
            ):
                raise _WorkerResourceError(
                    "cpu",
                    "requested CPU limit exceeds the process hard limit",
                )
            hard_limit = (
                requested_hard
                if current_hard == resource.RLIM_INFINITY
                else min(requested_hard, current_hard)
            )
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (max_cpu_seconds, hard_limit),
            )
    except _WorkerResourceError:
        raise
    except (OSError, ValueError) as error:
        resource_name = (
            "memory" if max_memory_bytes is not None else "cpu"
        )
        raise _WorkerResourceError(
            resource_name,
            f"failed to apply {resource_name} resource limit: {error}",
        ) from error


def _read_request() -> Mapping[str, object]:
    raw_request = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(raw_request) > _MAX_REQUEST_BYTES:
        raise _WorkerProtocolError(
            f"worker request exceeds {_MAX_REQUEST_BYTES} bytes"
        )
    try:
        decoded = json.loads(raw_request.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _WorkerProtocolError(f"worker request is invalid JSON: {error}") from error
    request = _required_mapping(decoded, label="worker request")
    protocol_version = _required_int(
        request.get("protocol_version"),
        label="worker protocol version",
    )
    if protocol_version != PROCESS_PROTOCOL_VERSION:
        raise _WorkerProtocolError(
            f"unsupported worker protocol version {protocol_version}"
        )
    return request


def _run_request(request: Mapping[str, object]) -> dict[str, object]:
    expected_name = _required_string(
        request.get("expected_model_name"),
        label="expected model name",
    )
    factory_path = _required_string(
        request.get("factory"),
        label="model factory path",
    )
    scenario_payload = _required_mapping(
        request.get("scenario"),
        label="scenario envelope",
    )
    parameters = _required_mapping(
        request.get("parameters"),
        label="model parameters",
    )
    seed = _required_int(request.get("seed"), label="model seed")
    limits = _required_mapping(
        request.get("limits"),
        label="process limits",
    )
    max_trace_bytes = _required_int(
        limits.get("max_trace_bytes"),
        label="maximum trace bytes",
    )
    if max_trace_bytes <= 0:
        raise _WorkerProtocolError("maximum trace bytes must be positive")

    _apply_resource_limits(limits)
    factory = _load_factory(factory_path)
    try:
        model = factory()
    except MemoryError as error:
        raise _WorkerResourceError(
            "memory",
            "model factory exceeded the memory resource limit",
        ) from error
    except Exception as error:
        raise _WorkerProtocolError(
            f"model factory {factory_path!r} failed: {error}"
        ) from error

    model_name = getattr(model, "name", None)
    if model_name != expected_name:
        raise _WorkerProtocolError(
            "factory-created model name does not match the declared name"
        )
    simulate = getattr(model, "simulate", None)
    if not callable(simulate):
        raise _WorkerProtocolError(
            "factory-created model must provide callable simulate()"
        )

    scenario = Scenario(
        id=_required_string(
            scenario_payload.get("id"),
            label="scenario id",
        ),
        payload=dict(
            _required_mapping(
                scenario_payload.get("payload"),
                label="scenario payload",
            )
        ),
    )

    try:
        run = simulate(
            scenario,
            dict(parameters),
            random.Random(seed),
        )
    except MemoryError:
        return _error_payload(
            kind="resource_limit",
            message="remote model exceeded its memory resource limit",
            resource="memory",
        )
    except Exception as error:
        return _error_payload(
            kind="remote_error",
            message=str(error),
            type=error.__class__.__name__,
            traceback=traceback.format_exc(),
        )

    if not isinstance(run, ModelRun):
        raise _WorkerProtocolError("remote model simulate() must return ModelRun")

    try:
        run_payload = _model_run_payload(run)
        trace_bytes = len(_json_bytes(run_payload))
    except MemoryError:
        return _error_payload(
            kind="resource_limit",
            message="trace serialization exceeded the memory resource limit",
            resource="memory",
        )
    except Exception as error:
        raise _WorkerProtocolError(
            f"remote model returned an invalid canonical trace: {error}"
        ) from error

    if trace_bytes > max_trace_bytes:
        return _error_payload(
            kind="trace_limit",
            message=f"remote model trace exceeded {max_trace_bytes} bytes",
            trace_bytes=trace_bytes,
            limit_bytes=max_trace_bytes,
        )

    return {
        "protocol_version": PROCESS_PROTOCOL_VERSION,
        "status": "ok",
        "run": run_payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    arguments = parser.parse_args(argv)

    try:
        request = _read_request()
        response = _run_request(request)
    except _WorkerResourceError as error:
        response = _error_payload(
            kind="resource_limit",
            message=str(error),
            resource=error.resource_name,
        )
    except _WorkerProtocolError as error:
        response = _error_payload(
            kind="protocol_error",
            message=str(error),
            traceback=traceback.format_exc(),
        )
    except MemoryError:
        response = _error_payload(
            kind="resource_limit",
            message="worker exceeded its memory resource limit",
            resource="memory",
        )
    except Exception as error:
        response = _error_payload(
            kind="protocol_error",
            message=f"unexpected worker failure: {error}",
            traceback=traceback.format_exc(),
        )

    _write_result(arguments.result, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
