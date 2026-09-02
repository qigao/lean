from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from narrative_dynamics.studio import (
    DraftOperationKind,
    SQLiteScenarioProjectStore,
    ScenarioDiagnosticReport,
    ScenarioDraftOperation,
    ScenarioDraftSnapshot,
    ScenarioProjectCapacityError,
    ScenarioProjectConflictError,
    ScenarioProjectStorageError,
    ScenarioProjectValidationError,
    ScenarioProjectWorkspace,
    ScenarioWorkspaceLimits,
)


LAW_FIRM = Path("examples/law_firm_scenario").resolve()


def _workspace(database: Path, **kwargs: object) -> ScenarioProjectWorkspace:
    return ScenarioProjectWorkspace.open(
        database,
        import_roots=(LAW_FIRM.parent,),
        **kwargs,
    )


def _imported(database: Path, project_id: str = "law-firm", **kwargs: object):
    workspace = _workspace(database, **kwargs)
    created = workspace.create_project(project_id)
    imported = workspace.import_package(
        project_id,
        LAW_FIRM,
        expected_revision=created.revision,
        expected_snapshot_hash=created.content_hash,
    )
    return workspace, imported


def _operation(snapshot, *, operation_id: str, idempotency_key: str, **kwargs: object):
    arguments: dict[str, object] = {
        "operation_id": operation_id,
        "idempotency_key": idempotency_key,
        "project_id": snapshot.project_id,
        "expected_revision": snapshot.revision,
        "expected_snapshot_hash": snapshot.content_hash,
        "document_role": "physical.world",
        "logical_id": None,
        "kind": DraftOperationKind.SET_VALUE,
        "pointer": "/places/0/label",
        "value": "Reception",
    }
    arguments.update(kwargs)
    return ScenarioDraftOperation(**arguments)  # type: ignore[arg-type]


def _document(snapshot, role: str, logical_id: str | None = None):
    return next(
        item
        for item in snapshot.documents
        if item.role == role and item.logical_id == logical_id
    )


def test_real_sqlite_workspace_import_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "studio.sqlite3"
    workspace = _workspace(database)
    created = workspace.create_project("law-firm")
    assert created.revision == 1
    assert created.documents == ()
    assert workspace.validate("law-firm").diagnostics[0].code == "manifest_missing"

    imported = workspace.import_package(
        "law-firm",
        LAW_FIRM,
        expected_revision=created.revision,
        expected_snapshot_hash=created.content_hash,
    )
    reopened = _workspace(database).snapshot("law-firm")

    assert reopened == imported
    assert imported.revision == 2
    assert imported.scenario_id == "law-firm-case"
    assert not workspace.validate("law-firm").has_errors


def test_exact_retry_is_idempotent_and_both_collision_directions_reject(
    tmp_path: Path,
) -> None:
    workspace, imported = _imported(tmp_path / "studio.sqlite3")
    operation = _operation(imported, operation_id="op-1", idempotency_key="key-1")

    accepted = workspace.apply(operation)
    assert workspace.apply(operation) == accepted

    with pytest.raises(ScenarioProjectConflictError, match="idempotency"):
        workspace.apply(
            _operation(
                accepted.next_snapshot,
                operation_id="op-2",
                idempotency_key="key-1",
                value="Different",
            )
        )
    with pytest.raises(ScenarioProjectConflictError, match="operation"):
        workspace.apply(
            _operation(
                accepted.next_snapshot,
                operation_id="op-1",
                idempotency_key="key-2",
                value="Different",
            )
        )


def test_stale_revision_and_hash_reject_without_mutation(tmp_path: Path) -> None:
    workspace, imported = _imported(tmp_path / "studio.sqlite3")
    stale_revision = _operation(imported, operation_id="op-1", idempotency_key="key-1")
    accepted = workspace.apply(stale_revision)

    with pytest.raises(ScenarioProjectConflictError, match="stale"):
        workspace.apply(
            _operation(imported, operation_id="op-2", idempotency_key="key-2")
        )
    wrong_hash = replace(
        _operation(
            accepted.next_snapshot, operation_id="op-3", idempotency_key="key-3"
        ),
        expected_snapshot_hash="sha256:" + "0" * 64,
    )
    with pytest.raises(ScenarioProjectConflictError, match="stale"):
        workspace.apply(wrong_hash)
    assert workspace.snapshot("law-firm") == accepted.next_snapshot


def test_json_operations_have_rfc6901_set_insert_remove_move_and_replace_semantics(
    tmp_path: Path,
) -> None:
    workspace, snapshot = _imported(tmp_path / "studio.sqlite3")

    snapshot = workspace.apply(
        _operation(snapshot, operation_id="op-set", idempotency_key="key-set")
    ).next_snapshot
    assert _document(snapshot, "physical.world").value["places"][0]["label"] == "Reception"

    snapshot = workspace.apply(
        _operation(
            snapshot,
            operation_id="op-insert",
            idempotency_key="key-insert",
            kind=DraftOperationKind.INSERT_VALUE,
            pointer="/places/1",
            value={"place_id": "copy-room", "label": "Copy room"},
        )
    ).next_snapshot
    assert _document(snapshot, "physical.world").value["places"][1]["place_id"] == "copy-room"

    snapshot = workspace.apply(
        _operation(
            snapshot,
            operation_id="op-move",
            idempotency_key="key-move",
            kind=DraftOperationKind.MOVE_VALUE,
            pointer="/places/3",
            from_pointer="/places/1",
            value=None,
        )
    ).next_snapshot
    assert _document(snapshot, "physical.world").value["places"][3]["place_id"] == "copy-room"

    snapshot = workspace.apply(
        _operation(
            snapshot,
            operation_id="op-remove",
            idempotency_key="key-remove",
            kind=DraftOperationKind.REMOVE_VALUE,
            pointer="/places/3",
            value=None,
        )
    ).next_snapshot
    assert len(_document(snapshot, "physical.world").value["places"]) == 3

    replacement = dict(_document(snapshot, "physical.world").to_dict()["value"])
    replacement["version"] = "2"
    snapshot = workspace.apply(
        _operation(
            snapshot,
            operation_id="op-replace",
            idempotency_key="key-replace",
            kind=DraftOperationKind.REPLACE_DOCUMENT,
            pointer="",
            value=replacement,
        )
    ).next_snapshot
    assert _document(snapshot, "physical.world").value["version"] == "2"


def test_move_rejects_object_members_instead_of_treating_them_as_array_elements(
    tmp_path: Path,
) -> None:
    workspace, imported = _imported(tmp_path / "studio.sqlite3")
    operation = _operation(
        imported,
        operation_id="op-object-move",
        idempotency_key="key-object-move",
        kind=DraftOperationKind.MOVE_VALUE,
        from_pointer="/places/0/label",
        pointer="/places/1/moved_label",
        value=None,
    )

    with pytest.raises(ScenarioProjectValidationError, match="array"):
        workspace.apply(operation)
    assert workspace.snapshot("law-firm") == imported


def test_layout_revision_changes_workspace_identity_only(tmp_path: Path) -> None:
    workspace, imported = _imported(tmp_path / "studio.sqlite3")
    result = workspace.apply(
        _operation(
            imported,
            operation_id="op-layout",
            idempotency_key="key-layout",
            document_role="layout",
            kind=DraftOperationKind.SET_LAYOUT,
            pointer="/physical.world",
            value={"records": {"x": 10, "y": 20}},
        )
    )

    assert result.next_snapshot.content_hash != imported.content_hash
    assert result.next_snapshot.layout_hash != imported.layout_hash
    assert result.next_snapshot.document_semantic_hash == imported.document_semantic_hash
    assert result.next_snapshot.compiled_scenario_hash == imported.compiled_scenario_hash


def test_undo_and_redo_create_increasing_revisions(tmp_path: Path) -> None:
    workspace, imported = _imported(tmp_path / "studio.sqlite3")
    changed = workspace.apply(
        _operation(imported, operation_id="op-1", idempotency_key="key-1")
    ).next_snapshot

    undone = workspace.undo("law-firm", changed.revision, changed.content_hash)
    assert undone.revision == changed.revision + 1
    assert _document(undone, "physical.world").value["places"][0]["label"] == "Public lobby"

    redone = workspace.redo("law-firm", undone.revision, undone.content_hash)
    assert redone.revision == undone.revision + 1
    assert _document(redone, "physical.world").value["places"][0]["label"] == "Reception"


def test_multiple_undo_and_redo_steps_preserve_the_linear_journal(tmp_path: Path) -> None:
    workspace, imported = _imported(tmp_path / "studio.sqlite3")
    first = workspace.apply(
        _operation(imported, operation_id="op-1", idempotency_key="key-1")
    ).next_snapshot
    second = workspace.apply(
        _operation(
            first,
            operation_id="op-2",
            idempotency_key="key-2",
            pointer="/places/1/label",
            value="Conference room",
        )
    ).next_snapshot

    undo_second = workspace.undo("law-firm", second.revision, second.content_hash)
    undo_first = workspace.undo(
        "law-firm", undo_second.revision, undo_second.content_hash
    )
    redo_first = workspace.redo(
        "law-firm", undo_first.revision, undo_first.content_hash
    )
    redo_second = workspace.redo(
        "law-firm", redo_first.revision, redo_first.content_hash
    )

    world = _document(redo_second, "physical.world").value
    assert world["places"][0]["label"] == "Reception"
    assert world["places"][1]["label"] == "Conference room"


def test_project_isolation_and_bounded_journal_with_retry_before_capacity(
    tmp_path: Path,
) -> None:
    limits = ScenarioWorkspaceLimits(accepted_operation_journal=1)
    workspace, first = _imported(tmp_path / "studio.sqlite3", limits=limits)
    second_created = workspace.create_project("second")
    second = workspace.import_package(
        "second",
        LAW_FIRM,
        expected_revision=second_created.revision,
        expected_snapshot_hash=second_created.content_hash,
    )
    operation = _operation(first, operation_id="op-1", idempotency_key="key-1")
    accepted = workspace.apply(operation)

    assert workspace.apply(operation) == accepted
    with pytest.raises(ScenarioProjectCapacityError, match="journal"):
        workspace.apply(
            _operation(
                accepted.next_snapshot,
                operation_id="op-2",
                idempotency_key="key-2",
            )
        )
    assert workspace.snapshot("second") == second


def test_sqlite_failure_rolls_back_revision_documents_and_idempotency(
    tmp_path: Path,
) -> None:
    database = tmp_path / "studio.sqlite3"
    workspace, imported = _imported(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TRIGGER reject_project_update BEFORE UPDATE ON projects "
            "BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END"
        )
    operation = _operation(imported, operation_id="op-1", idempotency_key="key-1")

    with pytest.raises(ScenarioProjectStorageError, match="storage") as raised:
        workspace.apply(operation)
    assert "injected" not in str(raised.value)
    assert str(database) not in str(raised.value)
    assert workspace.snapshot("law-firm") == imported


def test_injectable_document_limit_rejects_real_package_before_storage(
    tmp_path: Path,
) -> None:
    limits = ScenarioWorkspaceLimits(json_document_bytes=100)
    workspace = _workspace(tmp_path / "studio.sqlite3", limits=limits)
    created = workspace.create_project("law-firm")

    with pytest.raises(ValueError, match="permitted size"):
        workspace.import_package(
            "law-firm",
            LAW_FIRM,
            expected_revision=created.revision,
            expected_snapshot_hash=created.content_hash,
        )
    assert workspace.snapshot("law-firm") == created


def test_operation_limit_counts_typed_payload_not_identity_envelope(
    tmp_path: Path,
) -> None:
    limits = ScenarioWorkspaceLimits(operation_payload_bytes=128)
    workspace, imported = _imported(tmp_path / "studio.sqlite3", limits=limits)

    result = workspace.apply(
        _operation(imported, operation_id="long-operation-id", idempotency_key="long-key")
    )

    assert result.next_snapshot.revision == imported.revision + 1


def test_store_import_snapshot_rejects_cross_project_report_before_write(
    tmp_path: Path,
) -> None:
    database = tmp_path / "studio.sqlite3"
    workspace = _workspace(database)
    alpha = workspace.create_project("alpha")
    beta = workspace.create_project("beta")
    store = SQLiteScenarioProjectStore(database, ScenarioWorkspaceLimits())
    beta_report = ScenarioDiagnosticReport(
        "beta", 2, workspace.validate("beta").diagnostics
    )
    cross_project = ScenarioDraftSnapshot(
        "alpha", 2, (), {}, beta_report.content_hash, None
    )

    with pytest.raises(ValueError, match="project identities"):
        store.import_snapshot(
            cross_project,
            beta_report,
            alpha.revision,
            alpha.content_hash,
        )

    assert workspace.snapshot("alpha") == alpha
    assert workspace.snapshot("beta") == beta


def test_store_apply_rejects_cross_project_next_state_before_write(
    tmp_path: Path,
) -> None:
    database = tmp_path / "studio.sqlite3"
    workspace = _workspace(database)
    alpha = workspace.create_project("alpha")
    beta = workspace.create_project("beta")
    store = SQLiteScenarioProjectStore(database, ScenarioWorkspaceLimits())
    beta_report = ScenarioDiagnosticReport(
        "beta", 2, workspace.validate("beta").diagnostics
    )
    beta_next = ScenarioDraftSnapshot(
        "beta", 2, (), {}, beta_report.content_hash, None
    )
    operation = ScenarioDraftOperation(
        "op-cross-project",
        "key-cross-project",
        "alpha",
        alpha.revision,
        alpha.content_hash,
        "physical.world",
        None,
        DraftOperationKind.REMOVE_VALUE,
        pointer="/unused",
    )

    with pytest.raises(ValueError, match="project identities"):
        store.apply(operation, beta_next, beta_report)

    assert workspace.snapshot("alpha") == alpha
    assert workspace.snapshot("beta") == beta
