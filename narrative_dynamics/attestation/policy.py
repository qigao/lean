from __future__ import annotations

from narrative_dynamics.attestation.implementation import (
    ImplementationAttestationUnavailable,
    implementation_attestation_identity as measured_implementation_identity,
    measure_implementation,
)


class ImplementationAttestationMismatch(RuntimeError):
    """Measured implementation identity does not satisfy a pinned contract."""

    def __init__(
        self,
        *,
        expected_hash: str,
        actual_hash: str | None,
        status: str,
        reason: str | None = None,
    ) -> None:
        if status not in {"mismatch", "unavailable"}:
            raise ValueError(
                "implementation mismatch status must be mismatch or unavailable"
            )
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.status = status
        self.reason = reason
        if status == "unavailable":
            message = "implementation attestation is unavailable"
            if reason:
                message += f": {reason}"
        else:
            message = (
                "implementation attestation mismatch: "
                f"expected {expected_hash}, measured {actual_hash}"
            )
        super().__init__(message)


def _expected_hash(source: object) -> str | None:
    contract = getattr(source, "contract", None)
    expected = getattr(contract, "expected_implementation_hash", None)
    if expected is None:
        expected = getattr(source, "expected_implementation_hash", None)
    return expected if isinstance(expected, str) else None


def implementation_attestation_identity(source: object) -> dict[str, object]:
    """Measure implementation identity and enforce an optional contract pin."""

    expected = _expected_hash(source)
    if expected is None:
        return measured_implementation_identity(source)

    try:
        measured = measure_implementation(source)
    except ImplementationAttestationUnavailable as error:
        raise ImplementationAttestationMismatch(
            expected_hash=expected,
            actual_hash=None,
            status="unavailable",
            reason=str(error),
        ) from error

    actual = measured.content_hash
    if actual != expected:
        raise ImplementationAttestationMismatch(
            expected_hash=expected,
            actual_hash=actual,
            status="mismatch",
        )

    return {
        **measured.manifest_identity(),
        "status": "measured",
        "expected_content_hash": expected,
        "verification": "matched",
    }


__all__ = [
    "ImplementationAttestationMismatch",
    "implementation_attestation_identity",
]
