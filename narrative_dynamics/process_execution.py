from __future__ import annotations

from collections.abc import Mapping

from narrative_dynamics import _process_execution_core as _core
from narrative_dynamics.contracts import Scenario
from narrative_dynamics._process_supervisor import execute_subprocess_model


PROCESS_PROTOCOL_VERSION = _core.PROCESS_PROTOCOL_VERSION
CancellationToken = _core.CancellationToken
ModelCancelled = _core.ModelCancelled
ModelExecutionError = _core.ModelExecutionError
ModelOutputLimitExceeded = _core.ModelOutputLimitExceeded
ModelProtocolError = _core.ModelProtocolError
ModelRemoteError = _core.ModelRemoteError
ModelResourceLimitExceeded = _core.ModelResourceLimitExceeded
ModelTimeout = _core.ModelTimeout
ModelTraceLimitExceeded = _core.ModelTraceLimitExceeded
ProcessExecutionResult = _core.ProcessExecutionResult
ProcessLimits = _core.ProcessLimits


class SubprocessModel(_core.SubprocessModel):
    """Explicit fresh-process source with non-blocking file-backed requests."""

    def execute(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
        cancellation: CancellationToken | None = None,
    ) -> ProcessExecutionResult:
        return execute_subprocess_model(
            self,
            scenario,
            parameters,
            seed=seed,
            cancellation=cancellation,
        )


# Worker-only protocol helpers intentionally remain private but importable.
_json_bytes = _core._json_bytes
_model_run_payload = _core._model_run_payload

for _public_type in (
    CancellationToken,
    ModelCancelled,
    ModelExecutionError,
    ModelOutputLimitExceeded,
    ModelProtocolError,
    ModelRemoteError,
    ModelResourceLimitExceeded,
    ModelTimeout,
    ModelTraceLimitExceeded,
    ProcessExecutionResult,
    ProcessLimits,
):
    _public_type.__module__ = __name__

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
