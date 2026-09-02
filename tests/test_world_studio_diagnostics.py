from __future__ import annotations

import json
import hashlib
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
    canonical_json_hash,
)
import narrative_dynamics.studio.workspace as studio_workspace


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
    workspace_compiled = workspace.compile("law-firm")
    manifest_bytes = (target / "scenario.json").read_bytes()
    manifest = json.loads(manifest_bytes)

    assert target.is_dir()
    assert exported.target_name == compiled.package_hash.removeprefix("sha256:")
    assert exported.snapshot_hash == imported.content_hash
    assert exported.compiled_scenario_hash == compiled.content_hash
    assert workspace_compiled == compiled
    assert loaded.raw_manifest_hash == (
        "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    )
    assert loaded.raw_manifest_hash == canonical_json_hash(manifest)
    assert workspace_compiled.raw_manifest_hash == loaded.raw_manifest_hash
    assert workspace_compiled.raw_source_document_hashes == (
        loaded.raw_document_hashes
    )
    loaded_raw_hashes = {
        (role, logical_id): content_hash
        for role, logical_id, content_hash in loaded.raw_document_hashes
    }
    for locator in manifest["documents"]:
        raw_hash = "sha256:" + hashlib.sha256(
            (target / locator["path"]).read_bytes()
        ).hexdigest()
        logical_id = locator["role"]
        if locator["role"] == "agent":
            value = json.loads((target / locator["path"]).read_bytes())["value"]
            logical_id = value["agent_id"]
        assert raw_hash == locator["sha256"]
        assert raw_hash == loaded_raw_hashes[(locator["role"], logical_id)]
    with pytest.raises(ScenarioProjectConflictError, match="exists"):
        workspace.export("law-firm", "law-firm-release")


def test_export_uses_one_loaded_snapshot_across_a_forced_interleaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    workspace, imported = _imported(tmp_path / "studio.sqlite3", export_root)
    interleaved: dict[str, object] = {}

    def mutate_after_load(snapshot) -> None:
        interleaved["snapshot"] = workspace.apply(
            ScenarioDraftOperation(
                "op-export-interleave",
                "key-export-interleave",
                "law-firm",
                snapshot.revision,
                snapshot.content_hash,
                "physical.world",
                None,
                DraftOperationKind.SET_VALUE,
                pointer="/places/0/label",
                value="Interleaved reception",
            )
        ).next_snapshot

    monkeypatch.setattr(
        studio_workspace, "_export_interleave", mutate_after_load, raising=False
    )
    exported = workspace.export("law-firm", "interleaved-release")

    assert "snapshot" in interleaved
    current = interleaved["snapshot"]
    loaded = load_situated_scenario_package(export_root / exported.target_name)
    compiled = compile_situated_scenario_package(loaded)
    assert exported.snapshot_hash == imported.content_hash
    assert exported.snapshot_hash != current.content_hash
    assert exported.package_hash == compiled.package_hash
    assert exported.compiled_scenario_hash == compiled.content_hash


def test_identical_content_has_one_hash_only_address_across_caller_labels(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    workspace, _ = _imported(tmp_path / "studio.sqlite3", export_root)
    compiled = workspace.compile("law-firm")

    first = workspace.export("law-firm", "first-caller-label")

    assert first.target_name == compiled.package_hash.removeprefix("sha256:")
    with pytest.raises(ScenarioProjectConflictError, match="exists"):
        workspace.export("law-firm", "different-caller-label")
    assert tuple(path.name for path in export_root.iterdir()) == (first.target_name,)


def test_invalid_exact_snapshot_rejects_export_with_stable_report_and_no_publication(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    workspace, imported = _imported(tmp_path / "studio.sqlite3", export_root)
    bad_value = "private-invalid-export-value"
    invalid = workspace.apply(
        ScenarioDraftOperation(
            "op-invalid-export",
            "key-invalid-export",
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

    with pytest.raises(ScenarioProjectValidationError, match="cannot compile") as raised:
        workspace.export("law-firm", "invalid-caller-label")

    assert raised.value.diagnostic_report == invalid.diagnostic_report
    serialized = json.dumps(raised.value.diagnostic_report.to_dict(), sort_keys=True)
    assert bad_value not in serialized
    assert str(tmp_path) not in serialized
    assert not tuple(export_root.iterdir())
    assert not list(export_root.glob(".*.stage-*"))


@pytest.mark.parametrize(
    "target_name", ["../escape", "nested/name", "nested\\name", ".", "x" * 129]
)
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
    workspace, imported = _imported(tmp_path / "studio.sqlite3", export_root)
    first = workspace.export("law-firm", "first-release")
    workspace.apply(
        ScenarioDraftOperation(
            "op-second-export",
            "key-second-export",
            "law-firm",
            imported.revision,
            imported.content_hash,
            "physical.world",
            None,
            DraftOperationKind.SET_VALUE,
            pointer="/places/0/label",
            value="Changed before failed publication",
        )
    )

    def fail_publication(stage: Path, target: Path) -> None:
        raise OSError("private injected detail")

    monkeypatch.setattr(
        "narrative_dynamics.studio.workspace._publish_stage", fail_publication
    )
    with pytest.raises(ScenarioProjectValidationError, match="publication") as raised:
        workspace.export("law-firm", "second-release")

    assert "private injected detail" not in str(raised.value)
    assert load_situated_scenario_package(export_root / first.target_name)
    assert not list(export_root.glob(".second-release.stage-*"))


def test_collision_at_publication_preserves_the_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    workspace, _ = _imported(tmp_path / "studio.sqlite3", export_root)
    real_publish = studio_workspace._publish_stage
    collision: dict[str, Path] = {}

    def collide_then_publish(stage: Path, target: Path) -> None:
        target.mkdir()
        (target / "racer.txt").write_text("racing content", encoding="utf-8")
        collision["target"] = target
        real_publish(stage, target)

    monkeypatch.setattr(studio_workspace, "_publish_stage", collide_then_publish)
    with pytest.raises(ScenarioProjectConflictError, match="exists"):
        workspace.export("law-firm", "racing-release")

    target = collision["target"]
    assert (target / "racer.txt").read_text(encoding="utf-8") == "racing content"
    assert not list(export_root.glob(".*.stage-*"))
