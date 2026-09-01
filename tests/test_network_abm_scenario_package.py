from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PACKAGE_SCHEMA = "narrative-dynamics.scenario-package/v1"
DOCUMENT_SCHEMA = "narrative-dynamics.scenario-document/v1"


def _document(value: dict[str, object]) -> dict[str, object]:
    return {"schema": DOCUMENT_SCHEMA, "value": value}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _raw_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def nested_json(depth: int) -> object:
    value: object = None
    for _ in range(depth):
        value = [value]
    return value


def write_minimal_package(root: Path, *, reverse: bool = False) -> Path:
    root.mkdir(parents=True)
    documents = [
        ("physical.world", "world", {"name": "world"}),
        ("physical.perception", "perception", {"name": "perception"}),
        ("physical.initial_state", "initial-state", {"name": "initial"}),
        ("social.relationships", "relationships", {"name": "relationships"}),
        ("agent", "ada", {"agent_id": "ada", "name": "Ada"}),
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
    for index, (role, _logical_id, value) in enumerate(documents):
        relative_path = "run.json" if role == "run" else f"document-{index}.json"
        document = _document(value)
        _write_json(root / relative_path, document)
        manifest_documents.append(
            {
                "role": role,
                "path": relative_path,
                "sha256": _raw_hash(root / relative_path),
            }
        )
    if reverse:
        manifest_documents.reverse()
    _write_json(
        root / "scenario.json",
        {
            "schema": PACKAGE_SCHEMA,
            "scenario_id": "demo",
            "version": "1.0",
            "documents": manifest_documents,
        },
    )
    return root


def replace_manifest_path(root: Path, role: str, relative_path: str) -> None:
    manifest_path = root / "scenario.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for locator in manifest["documents"]:
        if locator["role"] == role:
            locator["path"] = relative_path
            break
    _write_json(manifest_path, manifest)


def manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "scenario.json").read_text(encoding="utf-8"))


def write_manifest(root: Path, value: dict[str, object]) -> None:
    _write_json(root / "scenario.json", value)


def locator_for(root: Path, role: str) -> dict[str, str]:
    return next(item for item in manifest(root)["documents"] if item["role"] == role)


def refresh_locator_hash(root: Path, role: str) -> None:
    value = manifest(root)
    locator = next(item for item in value["documents"] if item["role"] == role)
    locator["sha256"] = _raw_hash(root / locator["path"])
    write_manifest(root, value)


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

    def test_canonical_manifest_uses_scenario_run_path_and_sha256_keys(self):
        root = write_minimal_package(self.root / "canonical-layout")
        value = manifest(root)

        self.assertTrue((root / "scenario.json").is_file())
        self.assertFalse((root / "scenario-package.json").exists())
        self.assertEqual(locator_for(root, "run")["path"], "run.json")
        self.assertTrue((root / "run.json").is_file())
        self.assertTrue(all(
            set(locator) == {"role", "path", "sha256"}
            for locator in value["documents"]
        ))

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
        duplicate["path"] = "another-world.json"
        (root / "another-world.json").write_bytes(
            (root / locator_for(root, "physical.world")["path"]).read_bytes()
        )
        value["documents"].append(duplicate)
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "singleton"):
            load_situated_scenario_package(root)

    def test_loader_rejects_duplicate_agent_logical_id(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "duplicate-agent")
        value = manifest(root)
        duplicate = dict(locator_for(root, "agent"))
        duplicate["path"] = "duplicate-agent.json"
        (root / duplicate["path"]).write_bytes(
            (root / locator_for(root, "agent")["path"]).read_bytes()
        )
        duplicate["sha256"] = _raw_hash(root / duplicate["path"])
        value["documents"].append(duplicate)
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "unique"):
            load_situated_scenario_package(root)

    def test_loader_rejects_duplicate_manifest_paths(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "duplicate-path")
        value = manifest(root)
        value["documents"].append(dict(locator_for(root, "agent")))
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "path.*unique"):
            load_situated_scenario_package(root)

    def test_loader_rejects_distinct_locators_resolving_to_the_same_file(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "same-file")
        world = locator_for(root, "physical.world")
        value = manifest(root)
        value["documents"].append(
            {
                "role": "agent",
                "path": f"./{world['path']}",
                "sha256": world["sha256"],
            }
        )
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "same file"):
            load_situated_scenario_package(root)

    def test_loader_rejects_duplicate_document_json_keys(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "duplicate-json")
        locator = locator_for(root, "physical.world")
        (root / locator["path"]).write_text(
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
        (root / locator["path"]).write_bytes(b"\xff")

        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_situated_scenario_package(root)

    def test_loader_rejects_non_finite_document_number(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "non-finite")
        locator = locator_for(root, "physical.world")
        (root / locator["path"]).write_text(
            '{"schema":"narrative-dynamics.scenario-document/v1",'
            '"value":{"temperature":NaN}}',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            load_situated_scenario_package(root)

    def test_loader_rejects_oversized_document(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "oversized")
        locator = locator_for(root, "physical.world")
        (root / locator["path"]).write_bytes(b" " * (1024 * 1024 + 1))

        with self.assertRaisesRegex(ValueError, "size"):
            load_situated_scenario_package(root)

    def test_loader_bounds_oversized_document_read_to_limit_plus_one(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "bounded-oversized")
        target = root / locator_for(root, "physical.world")["path"]
        target.write_bytes(b" " * (1024 * 1024 + 4096))
        requested_sizes: list[int] = []
        original_open = Path.open

        class GuardedReader:
            def __init__(self, stream) -> None:
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def read(self, size: int = -1) -> bytes:
                if size < 0 or size > 1024 * 1024 + 1:
                    raise AssertionError("scenario loader attempted an unbounded read")
                requested_sizes.append(size)
                return self.stream.read(size)

        def guarded_open(path: Path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path == target:
                return GuardedReader(stream)
            return stream

        with mock.patch.object(Path, "open", guarded_open):
            with self.assertRaisesRegex(ValueError, "size"):
                load_situated_scenario_package(root)

        self.assertEqual(requested_sizes, [1024 * 1024 + 1])

    def test_loader_rejects_oversized_tiled_map(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "oversized-map")
        value = manifest(root)
        map_path = root / "map.tmj"
        map_path.write_bytes(b" " * (16 * 1024 * 1024 + 1))
        value["documents"].append(
            {
                "role": "physical.map",
                "path": "map.tmj",
                "sha256": "sha256:" + "0" * 64,
            }
        )
        write_manifest(root, value)

        with self.assertRaisesRegex(ValueError, "size"):
            load_situated_scenario_package(root)

    def test_loader_rejects_missing_document(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "missing")
        (root / locator_for(root, "physical.world")["path"]).unlink()

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
        manifest_path = root / "scenario.json"
        manifest_path.unlink()
        manifest_path.mkdir()

        with self.assertRaisesRegex(ValueError, "manifest") as error:
            load_situated_scenario_package(root)
        self.assertNotIn(str(root), str(error.exception))
        self.assertNotIn("scenario.json", str(error.exception))

    def test_loader_rejects_wrong_document_schema(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "wrong-schema")
        locator = locator_for(root, "physical.world")
        _write_json(root / locator["path"], {"schema": "wrong", "value": {}})

        with self.assertRaisesRegex(ValueError, "schema"):
            load_situated_scenario_package(root)

    def test_loader_rejects_wrong_expected_hash(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "wrong-hash")
        value = manifest(root)
        locator = next(item for item in value["documents"] if item["role"] == "physical.world")
        locator["sha256"] = "sha256:" + "0" * 64
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

    def test_raw_source_provenance_changes_without_changing_source_equality(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        first = write_minimal_package(self.root / "compact")
        second = write_minimal_package(self.root / "whitespace")
        world_path = second / locator_for(second, "physical.world")["path"]
        decoded = json.loads(world_path.read_text(encoding="utf-8"))
        world_path.write_text(json.dumps(decoded, indent=4) + "\n", encoding="utf-8")
        refresh_locator_hash(second, "physical.world")

        first_source = load_situated_scenario_package(first)
        second_source = load_situated_scenario_package(second)
        self.assertEqual(first_source, second_source)
        self.assertNotEqual(
            first_source.raw_manifest_hash,
            second_source.raw_manifest_hash,
        )
        self.assertNotEqual(
            first_source.raw_document_hashes,
            second_source.raw_document_hashes,
        )

    def test_loader_sanitizes_parser_limits_and_nonfinite_spellings(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        cases = {
            "deep": (
                '{"schema":"narrative-dynamics.scenario-document/v1",'
                '"value":{"nested":' + "[" * 1200 + "0" + "]" * 1200 + "}}"
            ),
            "overflow-exponent": (
                '{"schema":"narrative-dynamics.scenario-document/v1",'
                '"value":{"number":1e999}}'
            ),
            "nan": (
                '{"schema":"narrative-dynamics.scenario-document/v1",'
                '"value":{"number":NaN}}'
            ),
            "infinity": (
                '{"schema":"narrative-dynamics.scenario-document/v1",'
                '"value":{"number":Infinity}}'
            ),
            "negative-infinity": (
                '{"schema":"narrative-dynamics.scenario-document/v1",'
                '"value":{"number":-Infinity}}'
            ),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                root = write_minimal_package(self.root / name)
                path = root / locator_for(root, "physical.world")["path"]
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError) as raised:
                    load_situated_scenario_package(root)
                self.assertIsNone(raised.exception.__cause__)
                self.assertNotIn(str(root), str(raised.exception))
                self.assertNotIn(text, str(raised.exception))

    def test_loader_sanitizes_missing_root_failure_cause(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        missing = self.root / "secret-missing-root"
        with self.assertRaisesRegex(ValueError, "root") as raised:
            load_situated_scenario_package(missing)

        self.assertIsNone(raised.exception.__cause__)
        self.assertNotIn(str(missing), str(raised.exception))

    def test_source_document_rejects_excessive_nesting_without_recursion_leak(self):
        from narrative_dynamics.abm.scenario_package_contracts import (
            ScenarioDocumentRole,
            ScenarioSourceDocument,
        )

        with self.assertRaisesRegex(ValueError, "nesting") as raised:
            ScenarioSourceDocument(
                ScenarioDocumentRole.RUN,
                "run",
                DOCUMENT_SCHEMA,
                {"nested": nested_json(700)},
                "sha256:" + "0" * 64,
            )
        self.assertIsNone(raised.exception.__cause__)
        self.assertNotIsInstance(raised.exception, RecursionError)

    def test_loader_rejects_post_parse_excessive_nesting_without_source_leak(self):
        from narrative_dynamics.abm.scenario_package import (
            load_situated_scenario_package,
        )

        root = write_minimal_package(self.root / "deep-freeze")
        run_path = root / locator_for(root, "run")["path"]
        run_document = json.loads(run_path.read_text(encoding="utf-8"))
        run_document["value"]["nested"] = nested_json(700)
        _write_json(run_path, run_document)
        refresh_locator_hash(root, "run")
        with self.assertRaisesRegex(ValueError, "nesting") as raised:
            load_situated_scenario_package(root)
        self.assertIsNone(raised.exception.__cause__)
        self.assertNotIn(str(root), str(raised.exception))

    def test_source_documents_are_recursively_immutable(self):
        from narrative_dynamics.abm.scenario_package import load_situated_scenario_package

        root = write_minimal_package(self.root / "frozen")
        locator = locator_for(root, "physical.world")
        document = _document({"nested": {"items": ["one"]}})
        _write_json(root / locator["path"], document)
        value = manifest(root)
        next(item for item in value["documents"] if item["role"] == "physical.world")["sha256"] = _raw_hash(root / locator["path"])
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
        _write_json(root / locator["path"], document)
        value = manifest(root)
        next(item for item in value["documents"] if item["role"] == "run")["sha256"] = _raw_hash(root / locator["path"])
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
                raw_manifest_hash="sha256:" + "0" * 64,
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
                raw_content_hash=document.raw_content_hash,
            )
            for document in loaded.documents
        )

        with self.assertRaisesRegex(ValueError, "fallback"):
            ScenarioPackageSource(
                scenario_id=loaded.scenario_id,
                version=loaded.version,
                documents=documents,
                raw_manifest_hash=loaded.raw_manifest_hash,
            )
