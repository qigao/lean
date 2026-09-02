from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from narrative_dynamics.studio import (
    DraftOperationKind,
    SCENARIO_DIAGNOSTIC_REPORT_SCHEMA,
    SCENARIO_DRAFT_OPERATION_SCHEMA,
    SCENARIO_DRAFT_SNAPSHOT_SCHEMA,
    ScenarioDiagnostic,
    ScenarioDiagnosticReport,
    ScenarioDiagnosticSeverity,
    ScenarioDraftDocument,
    ScenarioDraftOperation,
    ScenarioDraftSnapshot,
    canonical_json_hash,
)


SNAPSHOT_HASH = "sha256:" + "1" * 64


def test_contract_enums_and_schema_names_are_closed() -> None:
    assert tuple(item.value for item in DraftOperationKind) == (
        "replace_document",
        "set_value",
        "insert_value",
        "remove_value",
        "move_value",
        "set_layout",
    )
    assert tuple(item.value for item in ScenarioDiagnosticSeverity) == (
        "error",
        "warning",
    )
    assert (
        SCENARIO_DRAFT_SNAPSHOT_SCHEMA
        == "narrative-dynamics.scenario-draft-snapshot/v1"
    )
    assert (
        SCENARIO_DRAFT_OPERATION_SCHEMA
        == "narrative-dynamics.scenario-draft-operation/v1"
    )
    assert (
        SCENARIO_DIAGNOSTIC_REPORT_SCHEMA
        == "narrative-dynamics.scenario-diagnostic-report/v1"
    )


def test_operation_is_frozen_canonical_and_content_addressed() -> None:
    operation = ScenarioDraftOperation(
        "op-1",
        "key-1",
        "project-1",
        1,
        SNAPSHOT_HASH,
        "physical.world",
        None,
        DraftOperationKind.SET_VALUE,
        pointer="/places/0/name",
        value="Records archive",
    )

    assert operation.to_dict()["kind"] == "set_value"
    assert operation.content_hash.startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        operation.pointer = "/changed"  # type: ignore[misc]


def test_documents_snapshots_and_diagnostics_are_canonical_and_detached() -> None:
    source = {"places": [{"place_id": "records", "tags": ["secure"]}]}
    document = ScenarioDraftDocument(
        "physical.world",
        None,
        source,
        canonical_json_hash(
            {"schema": "narrative-dynamics.scenario-document/v1", "value": source}
        ),
    )
    source["places"][0]["place_id"] = "changed"
    diagnostic = ScenarioDiagnostic(
        ScenarioDiagnosticSeverity.ERROR,
        "unknown_reference",
        "physical.world",
        None,
        "/passages/0/to_place_id",
        ("missing-place",),
    )
    report = ScenarioDiagnosticReport("project-1", 2, (diagnostic,))
    snapshot = ScenarioDraftSnapshot(
        "project-1",
        2,
        (document,),
        {"physical.world": {"records": {"x": 10, "y": 20}}},
        report.content_hash,
        None,
        scenario_id="scenario-1",
        version="1",
    )

    assert document.to_dict()["value"]["places"][0]["place_id"] == "records"
    assert snapshot.to_dict()["documents"][0]["role"] == "physical.world"
    assert snapshot.document_semantic_hash.startswith("sha256:")
    assert snapshot.layout_hash.startswith("sha256:")
    assert snapshot.content_hash.startswith("sha256:")
    assert report.to_dict()["diagnostics"][0]["message_key"] == "unknown_reference"


@pytest.mark.parametrize("revision", [0, -1, True])
def test_operations_reject_non_positive_integer_revisions(revision: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ScenarioDraftOperation(
            "op-1",
            "key-1",
            "project-1",
            revision,  # type: ignore[arg-type]
            SNAPSHOT_HASH,
            "physical.world",
            None,
            DraftOperationKind.REMOVE_VALUE,
            pointer="/places/0",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_snapshot_hash", "sha256:not-a-hash"),
        ("pointer", "places/0"),
        ("pointer", "/places/~2bad"),
        ("document_role", "../physical.world"),
        ("document_role", "physical/world"),
        ("logical_id", "../agent"),
        ("logical_id", "agents/alice"),
    ],
)
def test_operations_reject_malformed_identity_hashes_and_pointers(
    field: str, value: str
) -> None:
    arguments: dict[str, object] = {
        "operation_id": "op-1",
        "idempotency_key": "key-1",
        "project_id": "project-1",
        "expected_revision": 1,
        "expected_snapshot_hash": SNAPSHOT_HASH,
        "document_role": "physical.world",
        "logical_id": None,
        "kind": DraftOperationKind.SET_VALUE,
        "pointer": "/places/0/name",
        "value": "Records archive",
    }
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        ScenarioDraftOperation(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [object(), {"x": object()}, math.inf, math.nan])
def test_operations_reject_non_json_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ScenarioDraftOperation(
            "op-1",
            "key-1",
            "project-1",
            1,
            SNAPSHOT_HASH,
            "physical.world",
            None,
            DraftOperationKind.SET_VALUE,
            pointer="/places/0/name",
            value=value,
        )


def test_operations_reject_unknown_fields_and_oversized_payloads() -> None:
    with pytest.raises(TypeError):
        ScenarioDraftOperation(  # type: ignore[call-arg]
            "op-1",
            "key-1",
            "project-1",
            1,
            SNAPSHOT_HASH,
            "physical.world",
            None,
            DraftOperationKind.REMOVE_VALUE,
            pointer="/places/0",
            callback=lambda: None,
        )
    with pytest.raises(ValueError, match="payload"):
        ScenarioDraftOperation(
            "op-2",
            "key-2",
            "project-1",
            1,
            SNAPSHOT_HASH,
            "physical.world",
            None,
            DraftOperationKind.SET_VALUE,
            pointer="/places/0/name",
            value="x" * (16_777_216 + 1),
        )


def test_layout_operation_rejects_domain_document_addressing() -> None:
    with pytest.raises(ValueError, match="layout"):
        ScenarioDraftOperation(
            "op-1",
            "key-1",
            "project-1",
            1,
            SNAPSHOT_HASH,
            "physical.world",
            None,
            DraftOperationKind.SET_LAYOUT,
            pointer="/places/records",
            value={"x": 10, "y": 20},
        )
