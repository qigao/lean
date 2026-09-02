from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from narrative_dynamics.abm import scenario_checkpoint_store as checkpoint_store_module
from narrative_dynamics.abm.scenario_checkpoint_store import (
    LocalScenarioCheckpointStore,
)
from narrative_dynamics.abm.scenario_compiler import (
    compile_situated_scenario_package,
    initialize_compiled_scenario,
)
from narrative_dynamics.abm.scenario_coordinator_contracts import ScenarioCheckpoint
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.situated_network import simulate_situated_network_round
from narrative_dynamics.abm.situated_percept_memory import (
    hash_situated_percept_memory_store,
)
from tests.scenario_package_fixtures import write_law_firm_package


class LocalScenarioCheckpointStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package_temporary = TemporaryDirectory()
        package_root = write_law_firm_package(
            Path(cls.package_temporary.name) / "law-firm"
        )
        cls.scenario = compile_situated_scenario_package(
            load_situated_scenario_package(package_root)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.package_temporary.cleanup()

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "source-memory.sqlite3"
        initial = initialize_compiled_scenario(self.database, self.scenario)
        self.state = simulate_situated_network_round(
            self.database,
            self.scenario.runtime_model,
            initial,
        ).next_state
        self.checkpoint = ScenarioCheckpoint(
            "round-one",
            "law-firm-run",
            self.scenario.content_hash,
            1,
            self.state,
            7,
        )
        self.store_root = self.root / "checkpoint-store"
        self.store = LocalScenarioCheckpointStore(self.store_root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_create_load_and_restore_exact_typed_checkpoint(self) -> None:
        stored = self.store.create(self.checkpoint, self.database)

        self.assertIs(stored, self.checkpoint)
        self.assertIs(
            self.store.load(self.checkpoint.content_hash),
            self.checkpoint,
        )
        restored = self.root / "restored-memory.sqlite3"
        self.store.restore(self.checkpoint.content_hash, restored)
        self.assertEqual(
            hash_situated_percept_memory_store(restored),
            self.checkpoint.memory_store_hash,
        )

    def test_human_checkpoint_id_never_controls_artifact_path(self) -> None:
        path_like = replace(
            self.checkpoint,
            checkpoint_id="../../outside\\private.sqlite3",
        )

        self.store.create(path_like, self.database)

        artifacts = tuple(self.store_root.iterdir())
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].parent.resolve(), self.store_root.resolve())
        self.assertNotIn("outside", artifacts[0].name)
        self.assertNotIn("private", artifacts[0].name)
        self.assertFalse((self.root.parent / "outside").exists())

    def test_checkpoint_id_reuse_is_exactly_idempotent_or_rejected(self) -> None:
        first = self.store.create(self.checkpoint, self.database)
        self.assertIs(self.store.create(self.checkpoint, self.database), first)

        conflict = replace(self.checkpoint, next_sequence=8)
        with self.assertRaisesRegex(ValueError, "checkpoint id was reused"):
            self.store.create(conflict, self.database)
        self.assertIs(self.store.load(first.content_hash), first)

    def test_unknown_hash_load_restore_and_discard_are_path_safe(self) -> None:
        unknown = "sha256:" + "f" * 64
        with self.assertRaisesRegex(KeyError, "unknown scenario checkpoint"):
            self.store.load(unknown)
        with self.assertRaisesRegex(KeyError, "unknown scenario checkpoint"):
            self.store.restore(unknown, self.root / "never-created.sqlite3")

        self.store.discard(unknown)
        self.assertFalse((self.root / "never-created.sqlite3").exists())

    def test_restore_requires_missing_target_and_preserves_existing_bytes(self) -> None:
        self.store.create(self.checkpoint, self.database)
        target = self.root / "existing.sqlite3"
        target.write_bytes(b"existing-private-bytes")

        with self.assertRaisesRegex(ValueError, "restore target already exists"):
            self.store.restore(self.checkpoint.content_hash, target)

        self.assertEqual(target.read_bytes(), b"existing-private-bytes")

    def test_create_rejects_source_whose_logical_hash_does_not_match(self) -> None:
        stale_database = self.root / "stale-memory.sqlite3"
        initialize_compiled_scenario(stale_database, self.scenario)
        connection = sqlite3.connect(stale_database)
        try:
            connection.execute(
                "INSERT INTO percept_memory_metadata(key, value) VALUES (?, ?)",
                ("test-logical-difference", "different"),
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(ValueError, "source memory hash does not match"):
            self.store.create(self.checkpoint, stale_database)

        self.assertEqual(tuple(self.store_root.iterdir()), ())

    def test_existing_content_artifact_is_reused_without_clobbering_bytes(self) -> None:
        self.store.create(self.checkpoint, self.database)
        artifact = tuple(self.store_root.iterdir())[0]
        prior_bytes = artifact.read_bytes()
        second_process_store = LocalScenarioCheckpointStore(self.store_root)

        with patch(
            "narrative_dynamics.abm.scenario_checkpoint_store.os.link",
            side_effect=AssertionError("existing artifact must not be republished"),
        ):
            reused = second_process_store.create(self.checkpoint, self.database)

        self.assertIs(reused, self.checkpoint)
        self.assertEqual(artifact.read_bytes(), prior_bytes)
        self.assertEqual(tuple(self.store_root.glob("*.stage-*")), ())

        second_process_store.discard(self.checkpoint.content_hash)
        self.assertTrue(artifact.exists())
        self.assertEqual(artifact.read_bytes(), prior_bytes)

    def test_concurrent_content_artifact_winner_is_validated_and_reused(self) -> None:
        artifact_hash = self.checkpoint.content_hash.split(":", 1)[1]
        artifact = self.store_root / f"checkpoint-{artifact_hash}.sqlite3"
        real_link = os.link

        def publish_winner(source, destination) -> None:
            real_link(source, destination)
            raise FileExistsError("private concurrent checkpoint winner")

        with patch(
            "narrative_dynamics.abm.scenario_checkpoint_store.os.link",
            side_effect=publish_winner,
        ):
            stored = self.store.create(self.checkpoint, self.database)

        self.assertIs(stored, self.checkpoint)
        self.assertTrue(artifact.exists())
        self.assertEqual(
            hash_situated_percept_memory_store(artifact),
            self.checkpoint.memory_store_hash,
        )
        self.assertEqual(tuple(self.store_root.glob("*.stage-*")), ())

    def test_invalid_existing_artifact_is_never_replaced_or_deleted(self) -> None:
        self.store.create(self.checkpoint, self.database)
        artifact = tuple(self.store_root.iterdir())[0]
        artifact.write_bytes(b"prior-invalid-bytes")
        prior_bytes = artifact.read_bytes()
        second_process_store = LocalScenarioCheckpointStore(self.store_root)

        with self.assertRaisesRegex(ValueError, "integrity validation"):
            second_process_store.create(self.checkpoint, self.database)

        self.assertTrue(artifact.exists())
        self.assertEqual(artifact.read_bytes(), prior_bytes)

    def test_restore_concurrent_target_winner_is_preserved_byte_for_byte(self) -> None:
        self.store.create(self.checkpoint, self.database)
        target = self.root / "concurrent-target.sqlite3"
        winner_bytes = b"concurrent-winner-must-survive"

        def publish_winner(_, destination) -> None:
            Path(destination).write_bytes(winner_bytes)
            raise FileExistsError("private concurrent detail")

        with patch(
            "narrative_dynamics.abm.scenario_checkpoint_store.os.link",
            side_effect=publish_winner,
        ):
            with self.assertRaisesRegex(ValueError, "restore target already exists"):
                self.store.restore(self.checkpoint.content_hash, target)

        self.assertEqual(target.read_bytes(), winner_bytes)
        self.assertEqual(tuple(target.parent.glob(f".{target.name}.stage-*")), ())

    def test_restore_validation_failure_removes_owned_target_and_can_retry(self) -> None:
        self.store.create(self.checkpoint, self.database)
        target = self.root / "validation-retry.sqlite3"
        real_logical_hash = checkpoint_store_module._logical_hash
        call_count = 0

        def fail_published_target(path, *, message):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise ValueError("private post-publication validation failure")
            return real_logical_hash(path, message=message)

        with patch.object(
            checkpoint_store_module,
            "_logical_hash",
            side_effect=fail_published_target,
        ):
            with self.assertRaisesRegex(ValueError, "private post-publication"):
                self.store.restore(self.checkpoint.content_hash, target)

        self.assertFalse(target.exists())
        self.assertEqual(tuple(target.parent.glob(f".{target.name}.stage-*")), ())
        self.store.restore(self.checkpoint.content_hash, target)
        self.assertEqual(
            hash_situated_percept_memory_store(target),
            self.checkpoint.memory_store_hash,
        )

    def test_restore_publication_failure_removes_stage_and_can_retry(self) -> None:
        self.store.create(self.checkpoint, self.database)
        target = self.root / "publication-retry.sqlite3"

        with patch(
            "narrative_dynamics.abm.scenario_checkpoint_store.os.link",
            side_effect=OSError("private publication failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "restore publication failed"):
                self.store.restore(self.checkpoint.content_hash, target)

        self.assertFalse(target.exists())
        self.assertEqual(tuple(target.parent.glob(f".{target.name}.stage-*")), ())
        self.store.restore(self.checkpoint.content_hash, target)
        self.assertTrue(target.exists())

    def test_cleanup_failure_is_visible_and_path_redacted(self) -> None:
        self.store.create(self.checkpoint, self.database)
        target = self.root / "cleanup-private-target.sqlite3"
        real_unlink = Path.unlink

        def fail_stage_unlink(path, *args, **kwargs):
            if ".stage-" in path.name:
                raise OSError(f"private cleanup path {path}")
            return real_unlink(path, *args, **kwargs)

        with patch(
            "narrative_dynamics.abm.scenario_checkpoint_store.os.link",
            side_effect=OSError("private publication failure"),
        ), patch.object(Path, "unlink", autospec=True, side_effect=fail_stage_unlink):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed") as raised:
                self.store.restore(self.checkpoint.content_hash, target)

        self.assertNotIn(str(self.root), str(raised.exception))
        self.assertNotIn(str(target), str(raised.exception))

    def test_create_cleanup_failure_rolls_back_new_artifact_and_can_retry(self) -> None:
        real_unlink = Path.unlink
        failed_once = False

        def fail_first_stage_unlink(path, *args, **kwargs):
            nonlocal failed_once
            if ".stage-" in path.name and not failed_once:
                failed_once = True
                raise OSError(f"private cleanup path {path}")
            return real_unlink(path, *args, **kwargs)

        with patch.object(
            Path,
            "unlink",
            autospec=True,
            side_effect=fail_first_stage_unlink,
        ):
            with self.assertRaisesRegex(RuntimeError, "publication cleanup failed"):
                self.store.create(self.checkpoint, self.database)

        self.assertEqual(tuple(self.store_root.iterdir()), ())
        stored = self.store.create(self.checkpoint, self.database)
        self.assertIs(stored, self.checkpoint)

    def test_symlink_artifact_is_rejected_with_redacted_error(self) -> None:
        self.store.create(self.checkpoint, self.database)
        artifact = tuple(self.store_root.iterdir())[0]
        prior_bytes = artifact.read_bytes()
        backing = self.root / "private-outside.sqlite3"
        backing.write_bytes(prior_bytes)
        artifact.unlink()
        used_real_symlink = False
        try:
            artifact.symlink_to(backing)
            used_real_symlink = True
        except OSError:
            artifact.write_bytes(prior_bytes)

        if used_real_symlink:
            context = self.assertRaisesRegex(ValueError, "symlink")
            with context:
                self.store.load(self.checkpoint.content_hash)
            raised = context.exception
        else:
            real_is_symlink = Path.is_symlink

            def identify_artifact(path):
                if path == artifact:
                    return True
                return real_is_symlink(path)

            with patch.object(
                Path,
                "is_symlink",
                autospec=True,
                side_effect=identify_artifact,
            ):
                context = self.assertRaisesRegex(ValueError, "symlink")
                with context:
                    self.store.load(self.checkpoint.content_hash)
                raised = context.exception

        self.assertNotIn(str(self.store_root), str(raised))
        self.assertNotIn(str(backing), str(raised))

    def test_integrity_and_io_errors_never_expose_local_paths(self) -> None:
        self.store.create(self.checkpoint, self.database)
        artifact = tuple(self.store_root.iterdir())[0]
        with artifact.open("r+b") as stream:
            stream.seek(0)
            stream.write(b"not-a-sqlite-header")
            stream.flush()
            os.fsync(stream.fileno())

        with self.assertRaises(Exception) as raised:
            self.store.restore(
                self.checkpoint.content_hash,
                self.root / "private-target.sqlite3",
            )

        rendered = str(raised.exception)
        self.assertNotIn(str(self.store_root), rendered)
        self.assertNotIn(str(self.database), rendered)
        self.assertNotIn(str(self.root / "private-target.sqlite3"), rendered)

    def test_discard_is_exact_and_idempotent(self) -> None:
        self.store.create(self.checkpoint, self.database)
        artifact = tuple(self.store_root.iterdir())[0]

        self.store.discard(self.checkpoint.content_hash)
        self.store.discard(self.checkpoint.content_hash)

        self.assertFalse(artifact.exists())
        with self.assertRaisesRegex(KeyError, "unknown scenario checkpoint"):
            self.store.load(self.checkpoint.content_hash)


if __name__ == "__main__":
    unittest.main()
