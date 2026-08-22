from __future__ import annotations

from collections.abc import Mapping
import json
import os
import subprocess
import tempfile
import time

from narrative_dynamics import _process_execution_core as _core
from narrative_dynamics.contracts import Scenario
from narrative_dynamics._process_result import decode_worker_result


def _capture(
    stdout_stream,
    stderr_stream,
    *,
    max_bytes: int,
) -> tuple[str, str, int]:
    return _core._capture_streams(
        stdout_stream,
        stderr_stream,
        max_bytes=max_bytes,
    )


def _failure_context(
    *,
    stdout: str,
    stderr: str,
    process: subprocess.Popen[bytes],
    started: float,
) -> dict[str, object]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": process.returncode,
        "duration_seconds": time.monotonic() - started,
    }


def execute_subprocess_model(
    source: _core.SubprocessModel,
    scenario: Scenario,
    parameters: Mapping[str, float],
    *,
    seed: int,
    cancellation: _core.CancellationToken | None,
) -> _core.ProcessExecutionResult:
    """Run one importable model factory under a fresh-process supervisor."""

    if type(scenario) is not Scenario:
        raise TypeError(
            "subprocess execution currently requires the canonical Scenario type"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("subprocess execution seed must be an integer")
    if cancellation is not None and not isinstance(
        cancellation,
        _core.CancellationToken,
    ):
        raise TypeError("cancellation must be a CancellationToken or None")
    if cancellation is not None and cancellation.cancelled:
        raise _core.ModelCancelled(
            "model subprocess execution was cancelled before start"
        )

    request = {
        "protocol_version": _core.PROCESS_PROTOCOL_VERSION,
        "expected_model_name": source.name,
        "factory": source.factory,
        "scenario": {
            "id": scenario.id,
            "payload": _core._plain_value(
                scenario.payload,
                label="subprocess scenario payload",
            ),
        },
        "parameters": _core._plain_value(
            dict(parameters),
            label="subprocess parameters",
        ),
        "seed": seed,
        "limits": source.limits.manifest_identity(),
    }
    request_bytes = _core._json_bytes(request)
    if len(request_bytes) > _core._MAX_REQUEST_BYTES:
        raise _core.ModelProtocolError(
            f"subprocess request exceeds {_core._MAX_REQUEST_BYTES} bytes"
        )

    with tempfile.TemporaryDirectory(
        prefix="narrative-dynamics-process-"
    ) as directory:
        request_path = os.path.join(directory, "request.json")
        result_path = os.path.join(directory, "result.json")
        stdout_path = os.path.join(directory, "stdout.log")
        stderr_path = os.path.join(directory, "stderr.log")
        with open(request_path, "wb") as request_output:
            request_output.write(request_bytes)

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONHASHSEED"] = "0"

        with open(request_path, "rb") as request_stream, open(
            stdout_path,
            "w+b",
        ) as stdout_stream, open(
            stderr_path,
            "w+b",
        ) as stderr_stream:
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    [
                        source.python_executable,
                        "-m",
                        source.worker_module,
                        "--result",
                        result_path,
                    ],
                    stdin=request_stream,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    env=environment,
                    start_new_session=(os.name == "posix"),
                )
            except OSError as error:
                raise _core.ModelProtocolError(
                    f"failed to start model subprocess: {error}"
                ) from error

            try:
                deadline = started + source.limits.timeout_seconds
                while process.poll() is None:
                    captured_bytes = (
                        _core._stream_size(stdout_stream)
                        + _core._stream_size(stderr_stream)
                    )
                    if captured_bytes > source.limits.max_output_bytes:
                        _core._terminate_process_tree(process)
                        stdout, stderr, final_bytes = _capture(
                            stdout_stream,
                            stderr_stream,
                            max_bytes=source.limits.max_output_bytes,
                        )
                        raise _core.ModelOutputLimitExceeded(
                            captured_bytes=max(captured_bytes, final_bytes),
                            limit_bytes=source.limits.max_output_bytes,
                            **_failure_context(
                                stdout=stdout,
                                stderr=stderr,
                                process=process,
                                started=started,
                            ),
                        )

                    if cancellation is not None and cancellation.cancelled:
                        _core._terminate_process_tree(process)
                        stdout, stderr, _ = _capture(
                            stdout_stream,
                            stderr_stream,
                            max_bytes=source.limits.max_output_bytes,
                        )
                        raise _core.ModelCancelled(
                            **_failure_context(
                                stdout=stdout,
                                stderr=stderr,
                                process=process,
                                started=started,
                            )
                        )

                    if time.monotonic() >= deadline:
                        _core._terminate_process_tree(process)
                        stdout, stderr, _ = _capture(
                            stdout_stream,
                            stderr_stream,
                            max_bytes=source.limits.max_output_bytes,
                        )
                        raise _core.ModelTimeout(
                            source.limits.timeout_seconds,
                            **_failure_context(
                                stdout=stdout,
                                stderr=stderr,
                                process=process,
                                started=started,
                            ),
                        )
                    time.sleep(_core._POLL_INTERVAL_SECONDS)

                duration = time.monotonic() - started
                stdout, stderr, captured_bytes = _capture(
                    stdout_stream,
                    stderr_stream,
                    max_bytes=source.limits.max_output_bytes,
                )
                if captured_bytes > source.limits.max_output_bytes:
                    raise _core.ModelOutputLimitExceeded(
                        captured_bytes=captured_bytes,
                        limit_bytes=source.limits.max_output_bytes,
                        stdout=stdout,
                        stderr=stderr,
                        return_code=process.returncode,
                        duration_seconds=duration,
                    )

                return_code = process.returncode
                assert return_code is not None
                if return_code != 0:
                    resource_name = _core._resource_for_return_code(
                        return_code,
                        source.limits,
                    )
                    if resource_name is not None:
                        raise _core.ModelResourceLimitExceeded(
                            resource=resource_name,
                            stdout=stdout,
                            stderr=stderr,
                            return_code=return_code,
                            duration_seconds=duration,
                        )
                    raise _core.ModelProtocolError(
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
                    raise _core.ModelProtocolError(
                        "model worker did not create a result envelope",
                        stdout=stdout,
                        stderr=stderr,
                        return_code=return_code,
                        duration_seconds=duration,
                    )
                maximum_result_bytes = (
                    source.limits.max_trace_bytes
                    + _core._RESULT_OVERHEAD_BYTES
                )
                if os.path.getsize(result_path) > maximum_result_bytes:
                    raise _core.ModelProtocolError(
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
                    raise _core.ModelProtocolError(
                        f"model worker returned invalid JSON: {error}",
                        stdout=stdout,
                        stderr=stderr,
                        return_code=return_code,
                        duration_seconds=duration,
                    ) from error

                return decode_worker_result(
                    envelope,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=return_code,
                    duration_seconds=duration,
                )
            finally:
                if process.poll() is None:
                    _core._terminate_process_tree(process)
                _core._kill_remaining_process_group(process)
