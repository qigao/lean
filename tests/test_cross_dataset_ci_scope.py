from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests import test_cross_dataset_ci_boundary as boundary


class CrossDatasetScopeRegressionTests(unittest.TestCase):
    """Exercise the real Git boundary with offline, disposable repositories."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "full"
        self.root.mkdir()
        self.git(self.root, "init", "-q", "--initial-branch=main")
        self.write("README.md", "synthetic baseline\n")
        self.write("tests/old.txt", "synthetic old file\n")
        self.write("tests/mode.txt", "synthetic mode fixture\n")
        self.baseline_workflow = (
            "name: synthetic proof fixture\n\n"
            "jobs:\n"
            "  python-tests:\n"
            "    name: Python tests\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - name: Python numerical tests\n"
            "        run: python3 -m unittest discover -s tests -v\n"
        )
        self.write(".github/workflows/proof.yml", self.baseline_workflow)
        # Populate the old whitelist so its shallow fallback really passes,
        # rather than merely failing because a fixture file is absent.
        for name in (
            "__init__", "cross_dataset_authorization", "cross_dataset_candidates",
            "cross_dataset_capabilities", "cross_dataset_inference", "cross_dataset_ledger",
            "cross_dataset_prediction", "cross_dataset_privacy", "cross_dataset_release",
            "cross_dataset_reporting", "cross_dataset_search", "cross_dataset_source",
            "studies/cross_dataset_transfer_locked_final",
        ):
            self.write(f"narrative_dynamics/{name}.py", "# synthetic offline marker\n")
        self.base = self.commit()

    @staticmethod
    def git(root: Path, *arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments), cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    def write(self, path: str, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def commit(self) -> str:
        self.git(self.root, "add", "--all")
        self.git(
            self.root, "-c", "user.name=Scope fixture", "-c",
            "user.email=scope@example.invalid", "commit", "-qm", "synthetic fixture",
        )
        return self.git(self.root, "rev-parse", "HEAD")

    def shallow_clone(self) -> Path:
        clone = Path(self.temporary.name) / "shallow"
        self.git(
            self.root, "-c", "protocol.file.allow=always", "clone", "-q",
            "--depth=1", self.root.as_uri(), str(clone),
        )
        self.assertEqual(self.git(clone, "rev-parse", "--is-shallow-repository"), "true")
        self.assertEqual(
            self.git(clone, "rev-parse", "HEAD^{tree}"),
            self.git(self.root, "rev-parse", "HEAD^{tree}"),
        )
        return clone

    def paths(self, root: Path) -> tuple[str, ...]:
        with patch.dict(boundary.__dict__, _ROOT=root, _DESIGN_BASE=self.base):
            return boundary._phase_a_paths()

    def scope_result(self, root: Path) -> unittest.TestResult:
        result = unittest.TestResult()
        with patch.dict(boundary.__dict__, _ROOT=root, _DESIGN_BASE=self.base):
            for method in (
                "test_phase_a_has_no_real_source_or_network_workflow",
                "test_phase_a_adds_no_workflow_real_data_or_source_specific_module",
            ):
                boundary.CrossDatasetCiBoundaryTests(method).run(result)
        return result

    def test_scope_uses_actual_added_modified_deleted_and_renamed_paths(self) -> None:
        self.write("README.md", "synthetic changed file\n")
        self.git(self.root, "mv", "tests/old.txt", "tests/renamed.txt")
        self.write("tests/new.txt", "synthetic new file\n")
        self.commit()
        self.assertEqual(self.paths(self.root), (
            "README.md", "tests/new.txt", "tests/old.txt", "tests/renamed.txt",
        ))

    def test_scope_preserves_unusual_path_names(self) -> None:
        path = "tests/space ü.txt"
        self.write(path, "synthetic Unicode filename\n")
        self.commit()
        self.assertEqual(self.paths(self.root), (path,))

    def test_scope_detects_mode_only_change(self) -> None:
        self.git(self.root, "update-index", "--chmod=+x", "tests/mode.txt")
        self.git(
            self.root, "-c", "user.name=Scope fixture", "-c",
            "user.email=scope@example.invalid", "commit", "-qm", "synthetic mode change",
        )
        self.assertEqual(self.paths(self.root), ("tests/mode.txt",))

    def test_missing_base_in_real_shallow_clone_fails_closed(self) -> None:
        self.write("tests/new.txt", "synthetic new file\n")
        self.commit()
        clone = self.shallow_clone()
        with self.assertRaisesRegex(RuntimeError, "BASE_REVISION_UNAVAILABLE"):
            self.paths(clone)

    def test_present_base_without_ancestry_proof_fails_closed(self) -> None:
        self.write("tests/middle.txt", "synthetic middle commit\n")
        self.commit()
        self.write("tests/new.txt", "synthetic new file\n")
        self.commit()
        clone = self.shallow_clone()
        self.git(clone, "-c", "protocol.file.allow=always", "fetch", "-q", "--depth=1", "origin", self.base)
        self.git(clone, "cat-file", "-e", f"{self.base}^{{commit}}")
        with self.assertRaisesRegex(RuntimeError, "BASE_ANCESTRY_UNVERIFIED"):
            self.paths(clone)

    def test_unrelated_baseline_is_not_replaced_with_a_merge_base(self) -> None:
        self.write("tests/changed.txt", "synthetic main change\n")
        self.commit()
        self.git(self.root, "checkout", "-qb", "side", self.base)
        self.write("tests/side.txt", "synthetic diverged baseline\n")
        self.base = self.commit()
        self.git(self.root, "checkout", "-q", "main")
        with self.assertRaisesRegex(RuntimeError, "BASE_ANCESTRY_UNVERIFIED"):
            self.paths(self.root)

    def test_same_tree_cannot_hide_forbidden_files_in_shallow_clone(self) -> None:
        self.write(".github/workflows/extra.yml", "# inert synthetic workflow\n")
        self.write("fixtures/extra.csv", "synthetic,marker\n")
        self.write("narrative_dynamics/osf_fixture.py", "# synthetic marker only\n")
        self.commit()
        full = self.scope_result(self.root)
        self.assertFalse(full.wasSuccessful())
        self.assertFalse(full.errors)
        clone = self.shallow_clone()
        shallow = self.scope_result(clone)
        self.assertFalse(shallow.wasSuccessful())
        self.assertEqual(len(shallow.errors), 2)
        self.assertTrue(all("BASE_REVISION_UNAVAILABLE" in error for _, error in shallow.errors))
        self.git(clone, "-c", "protocol.file.allow=always", "fetch", "-q", "--unshallow", "origin")
        self.assertEqual(self.paths(clone), self.paths(self.root))
        hydrated = self.scope_result(clone)
        self.assertEqual(len(hydrated.failures), len(full.failures))
        self.assertFalse(hydrated.errors)

    def test_hydrated_valid_candidate_passes_both_scope_checks(self) -> None:
        self.write("tests/new.txt", "synthetic new file\n")
        self.commit()
        clone = self.shallow_clone()
        self.git(clone, "-c", "protocol.file.allow=always", "fetch", "-q", "--unshallow", "origin")
        self.assertEqual(self.paths(clone), ("tests/new.txt",))
        self.assertTrue(self.scope_result(clone).wasSuccessful())

    def test_only_history_checkout_change_is_permitted_in_existing_workflow(self) -> None:
        self.write(".github/workflows/proof.yml", self.baseline_workflow.replace(
            "      - uses: actions/checkout@v4\n",
            "      - uses: actions/checkout@v4\n        with:\n          fetch-depth: 0\n",
        ))
        self.commit()
        self.assertTrue(self.scope_result(self.root).wasSuccessful())

    def test_workflow_command_change_remains_forbidden(self) -> None:
        self.write(".github/workflows/proof.yml", self.baseline_workflow.replace(
            "      - uses: actions/checkout@v4\n",
            "      - uses: actions/checkout@v4\n        with:\n          fetch-depth: 0\n",
        ).replace("python3 -m unittest discover -s tests -v", "echo weakened gate"))
        self.commit()
        result = self.scope_result(self.root)
        self.assertFalse(result.wasSuccessful())
        self.assertFalse(result.errors)

    def test_workflow_trigger_addition_remains_forbidden(self) -> None:
        self.write(".github/workflows/proof.yml", self.baseline_workflow + "\non: workflow_dispatch\n")
        self.commit()
        result = self.scope_result(self.root)
        self.assertFalse(result.wasSuccessful())
        self.assertFalse(result.errors)


if __name__ == "__main__":
    unittest.main()
