from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch
import urllib.request

import narrative_dynamics
from narrative_dynamics.cross_dataset_source import DatasetSourceManifest


_DESIGN_BASE = "0378a40e934b3b241d6883df709aa056caec5f69"
_APPROVED_BASE = "c979eafe650a506bf30f78ab5b35078742b54087"
_ROOT = Path(__file__).resolve().parents[1]
# The only workflow delta permitted by this infrastructure correction.
# Compare the complete file against the frozen design-base version below.
_PYTHON_CHECKOUT = (
    "  python-tests:\n"
    "    name: Python tests\n"
    "    runs-on: ubuntu-latest\n"
    "    steps:\n"
    "      - uses: actions/checkout@v4\n"
)
_FROZEN_FILE_DIGESTS = {
    "narrative_dynamics/adapters/narrative_two_stage.py": (
        "657f509206f211ea2df61e267fa835754b1bed44b4431fe065f65267b0bf797b"
    ),
    "narrative_dynamics/studies/__init__.py": (
        "e0a07e40d46dcb1a63522d627734eee10a755788023c9527045d6e89e604d663"
    ),
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _git_revision_available(revision: str) -> bool:
    return (
        subprocess.run(
            ("git", "cat-file", "-e", f"{revision}^{{commit}}"),
            cwd=_ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _phase_a_paths() -> tuple[str, ...]:
    if not _git_revision_available(_DESIGN_BASE):
        raise RuntimeError(
            "BASE_REVISION_UNAVAILABLE: cannot prove Phase A changed-file scope"
        )
    try:
        _git("merge-base", "--is-ancestor", _DESIGN_BASE, "HEAD")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "BASE_ANCESTRY_UNVERIFIED: cannot prove Phase A changed-file scope"
        ) from exc
    # Compare the exact approved base, never a substituted merge base. NUL
    # delimiters preserve filenames; disabling renames keeps both endpoints.
    return tuple(
        path
        for path in _git(
            "diff", "--no-ext-diff", "--no-renames", "--name-only", "-z",
            _DESIGN_BASE, "HEAD", "--",
        ).split("\0")
        if path
    )


class CrossDatasetCiBoundaryTests(unittest.TestCase):
    def test_phase_a_paths_reject_missing_base(self) -> None:
        with patch(
            f"{__name__}._git_revision_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "BASE_REVISION_UNAVAILABLE"):
                _phase_a_paths()

    def test_phase_a_has_no_real_source_or_network_workflow(self) -> None:
        forbidden_patterns = (
            "osf.io/",
            "doi.org/",
            "requests.get(",
            "urllib.request",
            "workflow_dispatch:",
        )
        for path in _phase_a_paths():
            if not path.startswith(("narrative_dynamics/", ".github/workflows/")):
                continue
            text = (_ROOT / path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertFalse(
                    any(pattern in text for pattern in forbidden_patterns)
                )

    def test_phase_a_adds_no_workflow_real_data_or_source_specific_module(self) -> None:
        paths = _phase_a_paths()
        workflow_paths = tuple(path for path in paths if path.startswith(".github/workflows/"))
        proof_path = ".github/workflows/proof.yml"
        self.assertIn(workflow_paths, ((), (proof_path,)))
        if workflow_paths:
            # No new workflow, trigger, command, permission or job is allowed.
            # The existing Python checkout alone gains the required history.
            baseline = _git("show", f"{_DESIGN_BASE}:{proof_path}")
            self.assertEqual(baseline.count(_PYTHON_CHECKOUT), 1)
            expected = baseline.replace(
                _PYTHON_CHECKOUT,
                _PYTHON_CHECKOUT + "        with:\n          fetch-depth: 0\n",
            )
            self.assertFalse((_ROOT / proof_path).is_symlink())
            self.assertEqual((_ROOT / proof_path).read_bytes(), expected.encode("utf-8"))
            self.assertEqual(
                _git("ls-tree", _DESIGN_BASE, "--", proof_path).split()[:2],
                _git("ls-tree", "HEAD", "--", proof_path).split()[:2],
            )
        self.assertFalse(
            any(path.lower().endswith((".csv", ".tsv", ".parquet")) for path in paths)
        )
        runtime_modules = tuple(
            Path(path).name.lower()
            for path in paths
            if path.startswith("narrative_dynamics/")
        )
        source_specific_markers = ("osf", "doi", "feher_hare_source", "dataset_manifest.json")
        self.assertFalse(
            any(
                marker in module
                for module in runtime_modules
                for marker in source_specific_markers
            )
        )

    def test_frozen_adapter_and_studies_exports_are_byte_identical_to_base(self) -> None:
        for path, approved_digest in _FROZEN_FILE_DIGESTS.items():
            with self.subTest(path=path):
                current = (_ROOT / path).read_bytes()
                self.assertEqual(hashlib.sha256(current).hexdigest(), approved_digest)
                if _git_revision_available(_APPROVED_BASE):
                    approved = subprocess.run(
                        ("git", "show", f"{_APPROVED_BASE}:{path}"),
                        cwd=_ROOT,
                        check=True,
                        capture_output=True,
                    ).stdout
                    self.assertEqual(current, approved)

    def test_root_import_has_no_network_side_effect(self) -> None:
        def blocked(*_args, **_kwargs):
            raise AssertionError("import attempted network access")

        prefix = "narrative_dynamics.cross_dataset_"
        preserved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name.startswith(prefix)
        }
        try:
            for name in preserved_modules:
                sys.modules.pop(name)
            with patch.object(socket, "create_connection", blocked), patch.object(
                urllib.request,
                "urlopen",
                blocked,
            ):
                importlib.reload(narrative_dynamics)
        finally:
            for name in tuple(sys.modules):
                if name.startswith(prefix):
                    sys.modules.pop(name)
            sys.modules.update(preserved_modules)
            importlib.reload(narrative_dynamics)

    def test_z_root_import_preserves_loaded_transfer_module_identities(self) -> None:
        self.assertIs(
            narrative_dynamics.DatasetSourceManifest,
            DatasetSourceManifest,
        )


if __name__ == "__main__":
    unittest.main()
