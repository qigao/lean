from __future__ import annotations

from collections.abc import Mapping

from narrative_dynamics.contracts import ExecutionCapture, TraceEvent
from narrative_dynamics.schema_validation import (
    ModelSchemaViolation,
    RUNTIME_SCHEMA_DIALECT,
    validate_contract_events,
    validate_schema_value,
)


def _schema_attestation(schema: object, *, status: str) -> dict[str, object]:
    entry: dict[str, object] = {"status": status}
    if bool(getattr(schema, "specified", False)):
        entry.update(
            {
                "schema_name": getattr(schema, "name"),
                "schema_version": getattr(schema, "version"),
                "schema_hash": getattr(schema, "content_hash"),
                "dialect": getattr(schema, "dialect"),
            }
        )
    return entry


def validate_contract_result(
    source: object,
    events: tuple[TraceEvent, ...],
    outcome: Mapping[str, object],
    input_attestation: Mapping[str, object] | None,
    *,
    execution: ExecutionCapture | None,
) -> dict[str, object] | None:
    """Validate model-owned events and outcome before trusted trace creation."""

    event_attestation = validate_contract_events(
        source,
        events,
        input_attestation,
        execution=execution,
    )
    contract = getattr(source, "contract", None)
    outcome_schema = getattr(contract, "outcome_schema", None)
    if outcome_schema is None:
        return event_attestation

    specified = bool(getattr(outcome_schema, "specified", False))
    if specified:
        try:
            validate_schema_value(
                outcome_schema,
                outcome,
                boundary="outcome",
            )
        except ModelSchemaViolation as error:
            raise error.attach_execution(execution)

    if event_attestation is None and not specified:
        return None
    attestation = dict(event_attestation or {})
    if not attestation:
        attestation = {
            "dialect": RUNTIME_SCHEMA_DIALECT,
            "contract_hash": getattr(contract, "content_hash"),
        }
    attestation["outcome"] = _schema_attestation(
        outcome_schema,
        status="validated" if specified else "unspecified",
    )
    return attestation


__all__ = ["validate_contract_result"]
