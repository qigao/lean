from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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

    def test_replace_failure_preserves_prior_snapshot_and_removes_stage(self) -> None:
        self.store.create(self.checkpoint, self.database)
        artifact = tuple(self.store_root.iterdir())[0]
        prior_bytes = artifact.read_bytes()
        second_process_store = LocalScenarioCheckpointStore(self.store_root)

        with patch(
            "narrative_dynamics.abm.scenario_checkpoint_store.os.replace",
            side_effect=OSError("private replace detail"),
        ):
            with self.assertRaisesRegex(RuntimeError, "checkpoint publication failed"):
                second_process_store.create(self.checkpoint, self.database)

        self.assertEqual(artifact.read_bytes(), prior_bytes)
        self.assertEqual(tuple(self.store_root.glob("*.stage-*")), ())

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
