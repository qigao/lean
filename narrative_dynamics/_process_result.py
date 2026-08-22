from __future__ import annotations

from collections.abc import Mapping

from narrative_dynamics import _process_execution_core as _core
from narrative_dynamics.contracts import ExecutionCapture


def _context(
    *,
    stdout: str,
    stderr: str,
    return_code: int,
    duration_seconds: float,
) -> dict[str, object]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "duration_seconds": duration_seconds,
    }


def decode_worker_result(
    envelope: object,
    *,
    stdout: str,
    stderr: str,
    return_code: int,
    duration_seconds: float,
) -> _core.ProcessExecutionResult:
    """Validate one worker envelope and map it to a result or typed failure."""

    context = _context(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        duration_seconds=duration_seconds,
    )
    envelope_mapping = _core._required_mapping(
        envelope,
        label="worker result envelope",
    )
    protocol_version = _core._required_int(
        envelope_mapping.get("protocol_version"),
        label="worker protocol version",
    )
    if protocol_version != _core.PROCESS_PROTOCOL_VERSION:
        raise _core.ModelProtocolError(
            f"unsupported worker protocol version {protocol_version}",
            **context,
        )

    status = _core._required_string(
        envelope_mapping.get("status"),
        label="worker result status",
    )
    if status == "ok":
        return _core.ProcessExecutionResult(
            run=_core._decode_model_run(envelope_mapping.get("run")),
            capture=ExecutionCapture(
                isolated=True,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                duration_seconds=duration_seconds,
            ),
        )

    if status != "error":
        raise _core.ModelProtocolError(
            f"unsupported worker result status {status!r}",
            **context,
        )

    error_payload = _core._required_mapping(
        envelope_mapping.get("error"),
        label="worker error payload",
    )
    kind = _core._required_string(
        error_payload.get("kind"),
        label="worker error kind",
    )
    message = _core._required_string(
        error_payload.get("message"),
        label="worker error message",
    )

    if kind == "remote_error":
        raise _core.ModelRemoteError(
            remote_type=_core._required_string(
                error_payload.get("type"),
                label="remote error type",
            ),
            remote_message=message,
            remote_traceback=_core._required_string(
                error_payload.get("traceback"),
                label="remote traceback",
            ),
            **context,
        )
    if kind == "protocol_error":
        raise _core.ModelProtocolError(message, **context)
    if kind == "trace_limit":
        raise _core.ModelTraceLimitExceeded(
            trace_bytes=_core._required_int(
                error_payload.get("trace_bytes"),
                label="trace byte count",
            ),
            limit_bytes=_core._required_int(
                error_payload.get("limit_bytes"),
                label="trace byte limit",
            ),
            **context,
        )
    if kind == "resource_limit":
        raise _core.ModelResourceLimitExceeded(
            resource=_core._required_string(
                error_payload.get("resource"),
                label="resource limit name",
            ),
            message=message,
            **context,
        )
    raise _core.ModelProtocolError(
        f"unsupported worker error kind {kind!r}: {message}",
        **context,
    )
