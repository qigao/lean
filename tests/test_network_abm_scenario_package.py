from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.contracts import stable_content_hash


PACKAGE_SCHEMA = "narrative-dynamics.scenario-package/v1"
DOCUMENT_SCHEMA = "narrative-dynamics.scenario-document/v1"


def _document(value: dict[str, object]) -> dict[str, object]:
    return {"schema": DOCUMENT_SCHEMA, "value": value}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_minimal_package(root: Path, *, reverse: bool = False) -> Path:
    root.mkdir(parents=True)
    documents = [
        ("physical.world", "world", {"name": "world"}),
        ("physical.perception", "perception", {"name": "perception"}),
        ("physical.initial_state", "initial-state", {"name": "initial"}),
        ("social.relationships", "relationships", {"name": "relationships"}),
        ("agent", "ada", {"name": "Ada"}),
        ("story.outline", "outline", {"name": "outline"}),
        ("knowledge.catalog", "catalog", {"name": "catalog"}),
        ("knowledge.access", "access", {"name": "access"}),
        ("asset.catalog", "assets", {"name": "assets"}),
        (
            "run",
            "run",
            {
                "fallbacks": {
                    "physical.map": "auto_layout",
                    "social.institutions": "none",
                    "social.norms": "none",
                    "story.interventions": "none",
                }
            },
        ),
    ]
    manifest_documents: list[dict[str, str]] = []
    for index, (role, logical_id, value) in enumerate(documents):
        relative_path = f"document-{index}.json"
        document = _document(value)
        _write_json(root / relative_path, document)
        manifest_documents.append(
            {
                "role": role,
                "logical_id": logical_id,
                "relative_path": relative_path,
                "expected_hash": stable_content_hash(document),
            }
        )
    if reverse:
        manifest_documents.reverse()
    _write_json(
        root / "scenario-package.json",
        {
            "schema": PACKAGE_SCHEMA,
            "scenario_id": "demo",
            "version": "1.0",
            "documents": manifest_documents,
        },
    )
    return root


def replace_manifest_path(root: Path, role: str, relative_path: str) -> None:
    manifest_path = root / "scenario-package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for locator in manifest["documents"]:
        if locator["role"] == role:
            locator["relative_path"] = relative_path
            break
    _write_json(manifest_path, manifest)


def manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "scenario-package.json").read_text(encoding="utf-8"))


def write_manifest(root: Path, value: dict[str, object]) -> None:
    _write_json(root / "scenario-package.json", value)


def locator_for(root: Path, role: str) -> dict[str, str]:
    return next(item for item in manifest(root)["documents"] if item["role"] == role)


class ScenarioPackageLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_manifest_and_source_identity_ignore_root_and_document_order(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        first = write_minimal_package(self.root / "first", reverse=False)
        second = write_minimal_package(self.root / "second", reverse=True)

        self.assertEqual(
            load_situated_scenario_package(first),
            load_situated_scenario_package(second),
        )

    def test_loader_rejects_parent_escape_before_reading_document(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "escape")
        replace_manifest_path(root, "physical.world", "../secret.json")

        with self.assertRaisesRegex(ValueError, "path"):
            load_situated_scenario_package(root)

    def test_loader_rejects_duplicate_singleton_role(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "duplicate-role")
        value = manifest(root)
        duplicate = dict(locator_for(root, "physical.world"))
        duplicate["logical_id"] = "another-world"
        value["documents"].append(duplicate)
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "singleton"):
            load_situated_scenario_package(root)

    def test_loader_rejects_duplicate_agent_logical_id(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "duplicate-agent")
        value = manifest(root)
        value["documents"].append(dict(locator_for(root, "agent")))
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "unique"):
            load_situated_scenario_package(root)

    def test_loader_rejects_duplicate_document_json_keys(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "duplicate-json")
        locator = locator_for(root, "physical.world")
        (root / locator["relative_path"]).write_text(
            '{"schema":"narrative-dynamics.scenario-document/v1",'
            '"value":{"name":"world","name":"duplicate"}}',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            load_situated_scenario_package(root)

    def test_loader_rejects_absolute_document_path(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "absolute")
        replace_manifest_path(root, "physical.world", str((root / "document-0.json").resolve()))

        with self.assertRaisesRegex(ValueError, "path"):
            load_situated_scenario_package(root)

    def test_loader_rejects_symlink_escape_when_supported(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "symlink")
        outside = self.root / "outside.json"
        _write_json(outside, _document({"name": "outside"}))
        link = root / "escape.json"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"symlinks are unavailable: {error}")
        replace_manifest_path(root, "physical.world", "escape.json")

        with self.assertRaisesRegex(ValueError, "path"):
            load_situated_scenario_package(root)

    def test_loader_rejects_invalid_utf8(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "invalid-utf8")
        locator = locator_for(root, "physical.world")
        (root / locator["relative_path"]).write_bytes(b"\xff")

        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_situated_scenario_package(root)

    def test_loader_rejects_non_finite_document_number(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "non-finite")
        locator = locator_for(root, "physical.world")
        (root / locator["relative_path"]).write_text(
            '{"schema":"narrative-dynamics.scenario-document/v1",'
            '"value":{"temperature":NaN}}',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "constant"):
            load_situated_scenario_package(root)

    def test_loader_rejects_oversized_document(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "oversized")
        locator = locator_for(root, "physical.world")
        (root / locator["relative_path"]).write_bytes(b" " * (1024 * 1024 + 1))

        with self.assertRaisesRegex(ValueError, "size"):
            load_situated_scenario_package(root)

    def test_loader_rejects_oversized_tiled_map(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "oversized-map")
        value = manifest(root)
        map_path = root / "map.json"
        map_path.write_bytes(b" " * (16 * 1024 * 1024 + 1))
        value["documents"].append(
            {
                "role": "physical.map",
                "logical_id": "map",
                "relative_path": "map.json",
                "expected_hash": "sha256:" + "0" * 64,
            }
        )
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "size"):
            load_situated_scenario_package(root)

    def test_loader_rejects_missing_document(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "missing")
        (root / locator_for(root, "physical.world")["relative_path"]).unlink()

        with self.assertRaisesRegex(ValueError, "file"):
            load_situated_scenario_package(root)

    def test_loader_rejects_directory_document_path_without_local_path(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "directory-document")
        (root / "not-a-document").mkdir()
        replace_manifest_path(root, "physical.world", "not-a-document")

        with self.assertRaisesRegex(ValueError, "file") as error:
            load_situated_scenario_package(root)
        self.assertNotIn(str(root), str(error.exception))
        self.assertNotIn("not-a-document", str(error.exception))

    def test_loader_rejects_directory_manifest_without_local_path(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "directory-manifest")
        manifest_path = root / "scenario-package.json"
        manifest_path.unlink()
        manifest_path.mkdir()

        with self.assertRaisesRegex(ValueError, "manifest") as error:
            load_situated_scenario_package(root)
        self.assertNotIn(str(root), str(error.exception))
        self.assertNotIn("scenario-package.json", str(error.exception))

    def test_loader_rejects_wrong_document_schema(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "wrong-schema")
        locator = locator_for(root, "physical.world")
        _write_json(root / locator["relative_path"], {"schema": "wrong", "value": {}})

        with self.assertRaisesRegex(ValueError, "schema"):
            load_situated_scenario_package(root)

    def test_loader_rejects_wrong_expected_hash(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "wrong-hash")
        value = manifest(root)
        locator = next(item for item in value["documents"] if item["role"] == "physical.world")
        locator["expected_hash"] = "sha256:" + "0" * 64
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "hash"):
            load_situated_scenario_package(root)

    def test_undeclared_files_do_not_affect_source_identity(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        first = write_minimal_package(self.root / "declared")
        second = write_minimal_package(self.root / "extra")
        _write_json(second / "not-declared.json", {"not": "declared"})

        self.assertEqual(
            load_situated_scenario_package(first),
            load_situated_scenario_package(second),
        )

    def test_source_documents_are_recursively_immutable(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "frozen")
        locator = locator_for(root, "physical.world")
        document = _document({"nested": {"items": ["one"]}})
        _write_json(root / locator["relative_path"], document)
        value = manifest(root)
        next(item for item in value["documents"] if item["role"] == "physical.world")["expected_hash"] = stable_content_hash(document)
        write_manifest(root, value)

        source = load_situated_scenario_package(root)
        world = next(item for item in source.documents if item.role.value == "physical.world")
        with self.assertRaises(TypeError):
            world.value["nested"]["items"] = ()
        self.assertIsInstance(world.value["nested"]["items"], tuple)

    def test_optional_documents_require_run_fallbacks(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "no-fallbacks")
        locator = locator_for(root, "run")
        document = _document({"fallbacks": {}})
        _write_json(root / locator["relative_path"], document)
        value = manifest(root)
        next(item for item in value["documents"] if item["role"] == "run")["expected_hash"] = stable_content_hash(document)
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "fallback"):
            load_situated_scenario_package(root)

    def test_source_contract_rejects_missing_required_documents(self):
        from narrative_dynamics.abm.scenario_package_contracts import ScenarioPackageSource

        with self.assertRaisesRegex(ValueError, "required"):
            ScenarioPackageSource(
                scenario_id="demo",
                version="1.0",
                documents=(),
                manifest_hash="sha256:" + "0" * 64,
            )

    def test_source_contract_requires_fallbacks_for_missing_optional_documents(self):
        from narrative_dynamics.abm.scenario_package_contracts import (
            ScenarioDocumentRole,
            ScenarioPackageSource,
            ScenarioSourceDocument,
        )

        root = write_minimal_package(self.root / "source-fallbacks")
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
        loaded = load_situated_scenario_package(root)
        documents = tuple(
            ScenarioSourceDocument(
                role=document.role,
                logical_id=document.logical_id,
                schema=document.schema,
                value={"fallbacks": {}}
                if document.role is ScenarioDocumentRole.RUN
                else document.value,
            )
            for document in loaded.documents
        )

        with self.assertRaisesRegex(ValueError, "fallback"):
            ScenarioPackageSource(
                scenario_id=loaded.scenario_id,
                version=loaded.version,
                documents=documents,
                manifest_hash=loaded.manifest_hash,
            )
