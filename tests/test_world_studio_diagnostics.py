from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from narrative_dynamics.abm.scenario_compiler import compile_situated_scenario_package
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.studio import (
    DraftOperationKind,
    ScenarioDraftOperation,
    ScenarioProjectConflictError,
    ScenarioProjectValidationError,
    ScenarioProjectWorkspace,
)


LAW_FIRM = Path("examples/law_firm_scenario").resolve()


def _imported(database: Path, export_root: Path | None = None):
    workspace = ScenarioProjectWorkspace.open(
        database,
        import_roots=(LAW_FIRM.parent,),
        export_root=export_root,
    )
    created = workspace.create_project("law-firm")
    imported = workspace.import_package(
        "law-firm",
        LAW_FIRM,
        expected_revision=created.revision,
        expected_snapshot_hash=created.content_hash,
    )
    return workspace, imported


def test_dangling_passage_target_is_one_sanitized_pointer_diagnostic_and_undo_compiles(
    tmp_path: Path,
) -> None:
    database = tmp_path / "private" / "studio.sqlite3"
    workspace, imported = _imported(database)
    bad_value = "private-host-value-that-must-not-leak"
    result = workspace.apply(
        ScenarioDraftOperation(
            "op-bad-passage",
            "key-bad-passage",
            "law-firm",
            imported.revision,
            imported.content_hash,
            "physical.world",
            None,
            DraftOperationKind.SET_VALUE,
            pointer="/passages/0/target_place_id",
            value=bad_value,
        )
    )

    report = result.diagnostic_report
    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert (
        diagnostic.severity.value,
        diagnostic.code,
        diagnostic.document_role,
        diagnostic.logical_id,
        diagnostic.pointer,
    ) == (
        "error",
        "unknown_reference",
        "physical.world",
        None,
        "/passages/0/target_place_id",
    )
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert bad_value not in serialized
    assert str(database) not in serialized
    with pytest.raises(ScenarioProjectValidationError) as raised:
        workspace.compile("law-firm")
    assert raised.value.diagnostic_report == report

    restored = workspace.undo(
        "law-firm", result.next_snapshot.revision, result.next_snapshot.content_hash
    )
    compiled = workspace.compile("law-firm")
    direct = compile_situated_scenario_package(load_situated_scenario_package(LAW_FIRM))
    assert restored.compiled_scenario_hash == compiled.content_hash
    assert compiled == direct
    assert compiled.content_hash == direct.content_hash


def test_export_is_direct_loader_compatible_and_no_clobber(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    workspace, imported = _imported(tmp_path / "studio.sqlite3", export_root)

    exported = workspace.export("law-firm", "law-firm-release")
    target = export_root / exported.target_name
    loaded = load_situated_scenario_package(target)
    compiled = compile_situated_scenario_package(loaded)

    assert target.is_dir()
    assert exported.snapshot_hash == imported.content_hash
    assert exported.compiled_scenario_hash == compiled.content_hash
    assert workspace.compile("law-firm") == compiled
    with pytest.raises(ScenarioProjectConflictError, match="exists"):
        workspace.export("law-firm", "law-firm-release")


@pytest.mark.parametrize("target_name", ["../escape", "nested/name", "nested\\name", "."])
def test_export_rejects_root_escape_and_path_like_targets(
    tmp_path: Path, target_name: str
) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    workspace, _ = _imported(tmp_path / "studio.sqlite3", export_root)

    with pytest.raises(ScenarioProjectValidationError, match="target"):
        workspace.export("law-firm", target_name)


def test_import_and_export_are_disabled_by_default(tmp_path: Path) -> None:
    workspace = ScenarioProjectWorkspace.open(tmp_path / "studio.sqlite3")
    created = workspace.create_project("law-firm")
    with pytest.raises(ScenarioProjectValidationError, match="disabled"):
        workspace.import_package(
            "law-firm",
            LAW_FIRM,
            expected_revision=created.revision,
            expected_snapshot_hash=created.content_hash,
        )
    with pytest.raises(ScenarioProjectValidationError, match="disabled"):
        workspace.export("law-firm", "release")


def test_export_rejects_symlink_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    try:
        os.symlink(actual, linked, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ValueError, match="export root"):
        ScenarioProjectWorkspace.open(
            tmp_path / "studio.sqlite3",
            export_root=linked,
        )


def test_publication_failure_preserves_prior_export_and_leaves_no_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    workspace, _ = _imported(tmp_path / "studio.sqlite3", export_root)
    first = workspace.export("law-firm", "first-release")

    def fail_publication(stage: Path, target: Path) -> None:
        raise OSError("private injected detail")

    monkeypatch.setattr(
        "narrative_dynamics.studio.workspace._publish_stage", fail_publication
    )
    with pytest.raises(ScenarioProjectValidationError, match="publication") as raised:
        workspace.export("law-firm", "second-release")

    assert "private injected detail" not in str(raised.value)
    assert load_situated_scenario_package(export_root / first.target_name)
    assert not (export_root / "second-release").exists()
    assert not list(export_root.glob(".second-release.stage-*"))
