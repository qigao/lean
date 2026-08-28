from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.two_stage_test_support import (
    build_synthetic_two_stage_checkout,
    git_blob_sha,
)

_SOURCE_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.studies.two_stage_source import (
        TwoStageSourceFile,
        TwoStageSourceManifest,
        verify_two_stage_snapshot,
    )
except Exception as error:
    _SOURCE_IMPORT_ERROR = error


def _files(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative.endswith("_game.csv"):
            task, participant, purpose = "magic_carpet", path.stem.removesuffix("_game"), "scientific_evidence"
        elif relative.endswith("_config.txt"):
            task, participant, purpose = "magic_carpet", path.stem.removesuffix("_config"), "transform_metadata"
        elif relative.endswith("_info.txt"):
            task, participant, purpose = "spaceship", path.stem.removesuffix("_info"), "transform_metadata"
        elif relative.endswith("_practice.csv"):
            continue
        elif relative.startswith("results/spaceship/choices/") and relative.endswith(".csv"):
            task, participant, purpose = "spaceship", path.stem, "scientific_evidence"
        else:
            continue
        rows.append(
            TwoStageSourceFile(
                path=relative,
                git_blob_sha=git_blob_sha(path),
                task_variant=task,
                source_participant_id=participant,
                purpose=purpose,
            )
        )
    return tuple(rows)


class TwoStageSourceTests(unittest.TestCase):
    def require_source(self) -> None:
        self.assertIsNone(
            _SOURCE_IMPORT_ERROR,
            f"two-stage source boundary is missing: {_SOURCE_IMPORT_ERROR}",
        )

    def _manifest(self, root: Path, revision: str, *, reverse: bool = False):
        self.require_source()
        values = _files(root)
        if reverse:
            values = tuple(reversed(values))
        return TwoStageSourceManifest(
            name="synthetic-two-stage-source",
            version="1",
            repository="test/synthetic-two-stage",
            revision=revision,
            license_reference="test-only",
            files=values,
        )

    def test_source_manifest_is_content_hashed_and_order_canonical(self):
        self.require_source()
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(Path(tmp))
            left = self._manifest(root, revision)
            right = self._manifest(root, revision, reverse=True)
            self.assertEqual(left.files, right.files)
            self.assertEqual(left.content_hash, right.content_hash)

    def test_snapshot_requires_exact_repository_revision(self):
        self.require_source()
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(Path(tmp))
            manifest = TwoStageSourceManifest(
                name="bad-revision",
                version="1",
                repository="test/synthetic-two-stage",
                revision="0" * 40,
                license_reference="test-only",
                files=_files(root),
            )
            with self.assertRaises(ValueError):
                verify_two_stage_snapshot(root, manifest)
            self.assertNotEqual(revision, "0" * 40)

    def test_snapshot_requires_every_declared_path_and_git_blob_identity(self):
        self.require_source()
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(Path(tmp))
            manifest = self._manifest(root, revision)
            target = root / manifest.files[0].path
            original = target.read_bytes()
            target.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                verify_two_stage_snapshot(root, manifest)

    def test_unlisted_main_task_file_cannot_silently_enter_scientific_evidence(self):
        self.require_source()
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(Path(tmp))
            files = list(_files(root))
            files = [item for item in files if item.source_participant_id != "m000"]
            manifest = TwoStageSourceManifest(
                name="missing-main-file",
                version="1",
                repository="test/synthetic-two-stage",
                revision=revision,
                license_reference="test-only",
                files=tuple(files),
            )
            with self.assertRaises(ValueError):
                verify_two_stage_snapshot(root, manifest)

    def test_magic_game_and_spaceship_main_file_patterns_are_distinct(self):
        self.require_source()
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(Path(tmp))
            manifest = self._manifest(root, revision)
            evidence = [item for item in manifest.files if item.purpose == "scientific_evidence"]
            self.assertTrue(all(item.path.endswith("_game.csv") for item in evidence if item.task_variant == "magic_carpet"))
            self.assertTrue(all(
                item.path.endswith(".csv") and not item.path.endswith("_practice.csv") and not item.path.endswith("_game.csv")
                for item in evidence
                if item.task_variant == "spaceship"
            ))

    def test_metadata_only_source_identity_is_not_an_eligible_participant(self):
        self.require_source()
        metadata = TwoStageSourceFile(
            path="results/spaceship/choices/meta_only_info.txt",
            git_blob_sha="a" * 40,
            task_variant="spaceship",
            source_participant_id="meta_only",
            purpose="transform_metadata",
        )
        manifest = TwoStageSourceManifest(
            name="metadata-only",
            version="1",
            repository="test/synthetic-two-stage",
            revision="b" * 40,
            license_reference="test-only",
            files=(metadata,),
        )
        eligible = {
            (item.task_variant, item.source_participant_id)
            for item in manifest.files
            if item.purpose == "scientific_evidence"
        }
        self.assertNotIn(("spaceship", "meta_only"), eligible)

    def test_verification_uses_local_checkout_and_never_network(self):
        self.require_source()
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(Path(tmp))
            snapshot = verify_two_stage_snapshot(root, self._manifest(root, revision))
            self.assertEqual(snapshot.repository_revision, revision)
            self.assertEqual(snapshot.manifest_hash, self._manifest(root, revision).content_hash)


if __name__ == "__main__":
    unittest.main()
