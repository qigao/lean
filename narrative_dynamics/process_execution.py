from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

from narrative_dynamics.contracts import (
    ExecutionCapture,
    ModelRun,
    Scenario,
    TraceEvent,
)


PROCESS_PROTOCOL_VERSION = 1
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_RESULT_OVERHEAD_BYTES = 64 * 1024
_POLL_INTERVAL_SECONDS = 0.01
_TERMINATION_GRACE_SECONDS = 0.25


def _validated_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot have surrounding whitespace")
    return value


def _positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return validated


def _positive_int_or_none(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or None")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _positive_int(value: object, *, label: str) -> int:
    validated = _positive_int_or_none(value, label=label)
    assert validated is not None
    return validated


class CancellationToken:
    """Thread-safe cooperative cancellation signal owned by the caller."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class ModelExecutionError(RuntimeError):
    """Base class for typed failures at the external-model boundary."""

    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
        return_code: int | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.duration_seconds = duration_seconds


class ModelTimeout(ModelExecutionError):
    def __init__(
        self,
        timeout_seconds: float,
        **kwargs: object,
    ) -> None:
        super().__init__(
            f"model subprocess exceeded timeout of {timeout_seconds:g} seconds",
            **kwargs,
        )
        self.timeout_seconds = timeout_seconds


class ModelCancelled(ModelExecutionError):
    def __init__(
        self,
        message: str = "model subprocess execution was cancelled",
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)


class ModelOutputLimitExceeded(ModelExecutionError):
    def __init__(
        self,
        *,
        captured_bytes: int,
        limit_bytes: int,
        **kwargs: object,
    ) -> None:
        super().__init__(
            (
                "model subprocess stdout/stderr exceeded "
                f"{limit_bytes} bytes"
            ),
            **kwargs,
        )
        self.captured_bytes = captured_bytes
        self.limit_bytes = limit_bytes


class ModelTraceLimitExceeded(ModelExecutionError):
    def __init__(
        self,
        *,
        trace_bytes: int,
        limit_bytes: int,
        **kwargs: object,
    ) -> None:
        super().__init__(
            f"model subprocess trace exceeded {limit_bytes} bytes",
            **kwargs,
        )
        self.trace_bytes = trace_bytes
        self.limit_bytes = limit_bytes


class ModelResourceLimitExceeded(ModelExecutionError):
    def __init__(
        self,
        *,
        resource: str,
        message: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            message or f"model subprocess exceeded its {resource} resource limit",
            **kwargs,
        )
        self.resource = resource


class ModelRemoteError(ModelExecutionError):
    def __init__(
        self,
        *,
        remote_type: str,
        remote_message: str,
        remote_traceback: str,
        **kwargs: object,
    ) -> None:
        super().__init__(
            f"remote model raised {remote_type}: {remote_message}",
            **kwargs,
        )
        self.remote_type = remote_type
        self.remote_message = remote_message
        self.remote_traceback = remote_traceback


class ModelProtocolError(ModelExecutionError):
    pass


@dataclass(frozen=True)
class ProcessLimits:
    """Budgets enforced around one fresh worker process."""

    timeout_seconds: float = 30.0
    max_output_bytes: int = 1024 * 1024
    max_trace_bytes: int = 1024 * 1024
    max_memory_bytes: int | None = None
    max_cpu_seconds: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_float(
                self.timeout_seconds,
                label="process timeout seconds",
            ),
        )
        object.__setattr__(
            self,
            "max_output_bytes",
            _positive_int(
                self.max_output_bytes,
                label="maximum captured output bytes",
            ),
        )
        object.__setattr__(
            self,
            "max_trace_bytes",
            _positive_int(
                self.max_trace_bytes,
                label="maximum trace bytes",
            ),
        )
        object.__setattr__(
            self,
            "max_memory_bytes",
            _positive_int_or_none(
                self.max_memory_bytes,
                label="maximum memory bytes",
            ),
        )
        object.__setattr__(
            self,
            "max_cpu_seconds",
            _positive_int_or_none(
                self.max_cpu_seconds,
                label="maximum CPU seconds",
            ),
        )

    def manifest_identity(self) -> dict[str, object]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "max_trace_bytes": self.max_trace_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "max_cpu_seconds": self.max_cpu_seconds,
        }


@dataclass(frozen=True)
class ProcessExecutionResult:
    run: ModelRun
    capture: ExecutionCapture

    def __post_init__(self) -> None:
        if not isinstance(self.run, ModelRun):
            raise TypeError("process execution result run must be a ModelRun")
        if not isinstance(self.capture, ExecutionCapture):
            raise TypeError(
                "process execution result capture must be an ExecutionCapture"
            )


def _plain_value(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        plain: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            plain[key] = _plain_value(item, label=f"{label}.{key}")
        return plain
    if isinstance(value, (list, tuple)):
        return [
            _plain_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{label} values must be canonical scalars, mappings, lists, or tuples"
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _model_run_payload(run: ModelRun) -> dict[str, object]:
    if not isinstance(run, ModelRun):
        raise TypeError("worker model must return ModelRun")
    return {
        "events": [
            {
                "tick": event.tick,
                "kind": event.kind,
                "data": _plain_value(event.data, label="event data"),
            }
            for event in run.events
        ],
        "outcome": _plain_value(run.outcome, label="model outcome"),
    }


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelProtocolError(f"{label} must be a mapping")
    return value


def _required_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelProtocolError(f"{label} must be an integer")
    return value


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ModelProtocolError(f"{label} must be a string")
    return value


def _decode_model_run(value: object) -> ModelRun:
    payload = _required_mapping(value, label="worker run payload")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ModelProtocolError("worker run events must be a list")
    events: list[TraceEvent] = []
    for index, raw_event in enumerate(raw_events):
        event = _required_mapping(
            raw_event,
            label=f"worker event {index}",
        )
        data = _required_mapping(
            event.get("data"),
            label=f"worker event {index} data",
        )
        events.append(
            TraceEvent(
                tick=_required_int(
                    event.get("tick"),
                    label=f"worker event {index} tick",
                ),
                kind=_required_string(
                    event.get("kind"),
                    label=f"worker event {index} kind",
                ),
                data=dict(data),
            )
        )
    outcome = _required_mapping(
        payload.get("outcome"),
        label="worker outcome",
    )
    return ModelRun(events=tuple(events), outcome=dict(outcome))


def _validated_factory_path(value: object) -> str:
    path = _validated_text(value, label="subprocess model factory")
    module_name, separator, attribute_path = path.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(
            "subprocess model factory must use 'module:attribute' syntax"
        )
    parts = module_name.split(".") + attribute_path.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ValueError("subprocess model factory path is not a valid Python path")
    return path


def _stream_size(stream) -> int:
    return int(os.fstat(stream.fileno()).st_size)


def _read_stream(stream, *, max_bytes: int) -> str:
    stream.flush()
    stream.seek(0)
    data = stream.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def _capture_streams(stdout_stream, stderr_stream, *, max_bytes: int):
    stdout_size = _stream_size(stdout_stream)
    stderr_size = _stream_size(stderr_stream)
    stdout = _read_stream(stdout_stream, max_bytes=max_bytes)
    remaining = max(0, max_bytes - len(stdout.encode("utf-8")))
    stderr = _read_stream(stderr_stream, max_bytes=remaining)
    return stdout, stderr, stdout_size + stderr_size


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()

    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()

    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _kill_remaining_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill descendants left in the worker's dedicated POSIX process group."""

    if os.name != "posix":
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _resource_for_return_code(
    return_code: int,
    limits: ProcessLimits,
) -> str | None:
    if os.name != "posix" or return_code >= 0:
        return None
    signum = -return_code
    if (
        limits.max_cpu_seconds is not None
        and hasattr(signal, "SIGXCPU")
        and signum == signal.SIGXCPU
    ):
        return "cpu"
    if limits.max_memory_bytes is not None and signum in {
        signal.SIGKILL,
        signal.SIGSEGV,
        signal.SIGABRT,
    }:
        return "memory"
    if limits.max_cpu_seconds is not None and signum == signal.SIGKILL:
        return "cpu"
    return None


@dataclass(frozen=True)
class SubprocessModel:
    """Execute an importable model factory in one new Python process per run."""

    name: str
    factory: str
    version: str = "unversioned"
    implementation_revision: str = "unversioned"
    limits: ProcessLimits = field(default_factory=ProcessLimits)
    python_executable: str = field(
        default_factory=lambda: sys.executable,
        compare=False,
        repr=False,
    )
    worker_module: str = "narrative_dynamics.subprocess_worker"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validated_text(self.name, label="subprocess model name"),
        )
        object.__setattr__(
            self,
            "factory",
            _validated_factory_path(self.factory),
        )
        object.__setattr__(
            self,
            "version",
            _validated_text(self.version, label="subprocess model version"),
        )
        object.__setattr__(
            self,
            "implementation_revision",
            _validated_text(
                self.implementation_revision,
                label="subprocess model implementation revision",
            ),
        )
        if not isinstance(self.limits, ProcessLimits):
            raise TypeError("subprocess model limits must be ProcessLimits")
        object.__setattr__(
            self,
            "python_executable",
            _validated_text(
                self.python_executable,
                label="subprocess Python executable",
            ),
        )
        object.__setattr__(
            self,
            "worker_module",
            _validated_text(
                self.worker_module,
                label="subprocess worker module",
            ),
        )

    @property
    def lifecycle(self) -> str:
        return "fresh_process_per_run"

    def manifest_identity(self) -> dict[str, object]:
        source_type = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        return {
            "name": self.name,
            "version": self.version,
            "type": source_type,
            "implementation_revision": self.implementation_revision,
            "lifecycle": self.lifecycle,
            "executor": "python_subprocess",
            "factory": self.factory,
            "worker": {
                "module": self.worker_module,
                "protocol_version": PROCESS_PROTOCOL_VERSION,
                "python_hash_seed": "0",
            },
            "limits": self.limits.manifest_identity(),
        }

    def execute(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
        cancellation: CancellationToken | None = None,
    ) -> ProcessExecutionResult:
        if type(scenario) is not Scenario:
            raise TypeError(
                "subprocess execution currently requires the canonical Scenario type"
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("subprocess execution seed must be an integer")
        if cancellation is not None and not isinstance(
            cancellation,
            CancellationToken,
        ):
            raise TypeError("cancellation must be a CancellationToken or None")
        if cancellation is not None and cancellation.cancelled:
            raise ModelCancelled(
                "model subprocess execution was cancelled before start"
            )

        request = {
            "protocol_version": PROCESS_PROTOCOL_VERSION,
            "expected_model_name": self.name,
            "factory": self.factory,
            "scenario": {
                "id": scenario.id,
                "payload": _plain_value(
                    scenario.payload,
                    label="subprocess scenario payload",
                ),
            },
            "parameters": _plain_value(
                dict(parameters),
                label="subprocess parameters",
            ),
            "seed": seed,
            "limits": self.limits.manifest_identity(),
        }
        request_bytes = _json_bytes(request)
        if len(request_bytes) > _MAX_REQUEST_BYTES:
            raise ModelProtocolError(
                f"subprocess request exceeds {_MAX_REQUEST_BYTES} bytes"
            )

        started = time.monotonic()
        with tempfile.TemporaryDirectory(
            prefix="narrative-dynamics-process-"
        ) as directory:
            result_path = os.path.join(directory, "result.json")
            stdout_path = os.path.join(directory, "stdout.log")
            stderr_path = os.path.join(directory, "stderr.log")
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONHASHSEED"] = "0"

            with open(stdout_path, "w+b") as stdout_stream, open(
                stderr_path,
                "w+b",
            ) as stderr_stream:
                try:
                    process = subprocess.Popen(
                        [
                            self.python_executable,
                            "-m",
                            self.worker_module,
                            "--result",
                            result_path,
                        ],
                        stdin=subprocess.PIPE,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        env=environment,
                        start_new_session=(os.name == "posix"),
                    )
                except OSError as error:
                    raise ModelProtocolError(
                        f"failed to start model subprocess: {error}"
                    ) from error

                try:
                    assert process.stdin is not None
                    try:
                        process.stdin.write(request_bytes)
                        process.stdin.close()
                    except BrokenPipeError:
                        pass

                    deadline = started + self.limits.timeout_seconds
                    while process.poll() is None:
                        captured_bytes = (
                            _stream_size(stdout_stream)
                            + _stream_size(stderr_stream)
                        )
                        if captured_bytes > self.limits.max_output_bytes:
                            _terminate_process_tree(process)
                            duration = time.monotonic() - started
                            stdout, stderr, final_bytes = _capture_streams(
                                stdout_stream,
                                stderr_stream,
                                max_bytes=self.limits.max_output_bytes,
                            )
                            raise ModelOutputLimitExceeded(
                                captured_bytes=max(captured_bytes, final_bytes),
                                limit_bytes=self.limits.max_output_bytes,
                                stdout=stdout,
                                stderr=stderr,
                                return_code=process.returncode,
                                duration_seconds=duration,
                            )

                        if cancellation is not None and cancellation.cancelled:
                            _terminate_process_tree(process)
                            duration = time.monotonic() - started
                            stdout, stderr, _ = _capture_streams(
                                stdout_stream,
                                stderr_stream,
                                max_bytes=self.limits.max_output_bytes,
                            )
                            raise ModelCancelled(
                                stdout=stdout,
                                stderr=stderr,
                                return_code=process.returncode,
                                duration_seconds=duration,
                            )

                        if time.monotonic() >= deadline:
                            _terminate_process_tree(process)
                            duration = time.monotonic() - started
                            stdout, stderr, _ = _capture_streams(
                                stdout_stream,
                                stderr_stream,
                                max_bytes=self.limits.max_output_bytes,
                            )
                            raise ModelTimeout(
                                self.limits.timeout_seconds,
                                stdout=stdout,
                                stderr=stderr,
                                return_code=process.returncode,
                                duration_seconds=duration,
                            )
                        time.sleep(_POLL_INTERVAL_SECONDS)

                    duration = time.monotonic() - started
                    stdout, stderr, captured_bytes = _capture_streams(
                        stdout_stream,
                        stderr_stream,
                        max_bytes=self.limits.max_output_bytes,
                    )
                    if captured_bytes > self.limits.max_output_bytes:
                        raise ModelOutputLimitExceeded(
                            captured_bytes=captured_bytes,
                            limit_bytes=self.limits.max_output_bytes,
                            stdout=stdout,
                            stderr=stderr,
                            return_code=process.returncode,
                            duration_seconds=duration,
                        )

                    return_code = process.returncode
                    assert return_code is not None
                    if return_code != 0:
                        resource_name = _resource_for_return_code(
                            return_code,
                            self.limits,
                        )
                        if resource_name is not None:
                            raise ModelResourceLimitExceeded(
                                resource=resource_name,
                                stdout=stdout,
                                stderr=stderr,
                                return_code=return_code,
                                duration_seconds=duration,
                            )
                        raise ModelProtocolError(
                            (
                                "model worker exited with return code "
                                f"{return_code} without a valid result"
                            ),
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )

                    if not os.path.exists(result_path):
                        raise ModelProtocolError(
                            "model worker did not create a result envelope",
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    result_size = os.path.getsize(result_path)
                    maximum_result_bytes = (
                        self.limits.max_trace_bytes + _RESULT_OVERHEAD_BYTES
                    )
                    if result_size > maximum_result_bytes:
                        raise ModelProtocolError(
                            (
                                "model worker result envelope exceeded "
                                f"{maximum_result_bytes} bytes"
                            ),
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    with open(result_path, "rb") as result_stream:
                        raw_result = result_stream.read(maximum_result_bytes + 1)
                    try:
                        envelope = json.loads(raw_result.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ModelProtocolError(
                            f"model worker returned invalid JSON: {error}",
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        ) from error

                    envelope_mapping = _required_mapping(
                        envelope,
                        label="worker result envelope",
                    )
                    protocol_version = _required_int(
                        envelope_mapping.get("protocol_version"),
                        label="worker protocol version",
                    )
                    if protocol_version != PROCESS_PROTOCOL_VERSION:
                        raise ModelProtocolError(
                            (
                                "unsupported worker protocol version "
                                f"{protocol_version}"
                            ),
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    status = _required_string(
                        envelope_mapping.get("status"),
                        label="worker result status",
                    )
                    if status == "ok":
                        run = _decode_model_run(envelope_mapping.get("run"))
                        return ProcessExecutionResult(
                            run=run,
                            capture=ExecutionCapture(
                                isolated=True,
                                stdout=stdout,
                                stderr=stderr,
                                return_code=return_code,
                                duration_seconds=duration,
                            ),
                        )

                    if status != "error":
                        raise ModelProtocolError(
                            f"unsupported worker result status {status!r}",
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    error_payload = _required_mapping(
                        envelope_mapping.get("error"),
                        label="worker error payload",
                    )
                    kind = _required_string(
                        error_payload.get("kind"),
                        label="worker error kind",
                    )
                    message = _required_string(
                        error_payload.get("message"),
                        label="worker error message",
                    )
                    if kind == "remote_error":
                        raise ModelRemoteError(
                            remote_type=_required_string(
                                error_payload.get("type"),
                                label="remote error type",
                            ),
                            remote_message=message,
                            remote_traceback=_required_string(
                                error_payload.get("traceback"),
                                label="remote traceback",
                            ),
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    if kind == "protocol_error":
                        raise ModelProtocolError(
                            message,
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    if kind == "trace_limit":
                        raise ModelTraceLimitExceeded(
                            trace_bytes=_required_int(
                                error_payload.get("trace_bytes"),
                                label="trace byte count",
                            ),
                            limit_bytes=_required_int(
                                error_payload.get("limit_bytes"),
                                label="trace byte limit",
                            ),
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    if kind == "resource_limit":
                        raise ModelResourceLimitExceeded(
                            resource=_required_string(
                                error_payload.get("resource"),
                                label="resource limit name",
                            ),
                            message=message,
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    raise ModelProtocolError(
                        f"unsupported worker error kind {kind!r}: {message}",
                        stdout=stdout,
                        stderr=stderr,
                        return_code=return_code,
                        duration_seconds=duration,
                    )
                finally:
                    if process.poll() is None:
                        _terminate_process_tree(process)
                    _kill_remaining_process_group(process)


__all__ = [
    "CancellationToken",
    "ModelCancelled",
    "ModelExecutionError",
    "ModelOutputLimitExceeded",
    "ModelProtocolError",
    "ModelRemoteError",
    "ModelResourceLimitExceeded",
    "ModelTimeout",
    "ModelTraceLimitExceeded",
    "ProcessExecutionResult",
    "ProcessLimits",
    "SubprocessModel",
]
