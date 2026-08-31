from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.cross_dataset_source import (
    CanonicalTransferTrial,
    DatasetSourceFile,
    DatasetSourceManifest,
    SemanticInvarianceEvidence,
    TransformReceipt,
    VerifiedSourceSnapshot,
    evaluate_end_to_end_semantic_invariance,
    validate_canonical_trials,
    verify_source_snapshot,
)
from tests.cross_dataset_transfer_fixtures import (
    digest,
    five_coherent_relabelings,
    recording_semantic_pipeline,
    same_prechoice_different_postchoice_rows,
    source_file,
    source_manifest,
    synthetic_rows,
    synthetic_source_tree,
)


class CrossDatasetSourceTests(unittest.TestCase):
    def test_snapshot_verification_is_local_exact_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = synthetic_source_tree(Path(directory))

            verified = verify_source_snapshot(root, manifest)

        self.assertEqual(verified.source_manifest_hash, manifest.content_hash)
        self.assertEqual(
            verified.selected_catalog_entry_hash,
            manifest.selected_catalog_entry_hash,
        )
        self.assertTrue(all(not Path(path).is_absolute() for path, _, _ in verified.files))
        self.assertTrue(verified.source_snapshot_hash.startswith("sha256:"))

    def test_snapshot_rejects_missing_size_or_hash_drift(self) -> None:
        mutations = (b"", b"changed bytes")
        for payload in mutations:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as directory:
                    root, manifest = synthetic_source_tree(Path(directory))
                    (root / "behavior.csv").write_bytes(payload)
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        verify_source_snapshot(root, manifest)
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = synthetic_source_tree(Path(directory))
            (root / "README.txt").unlink()
            with self.assertRaisesRegex(ValueError, "missing source file"):
                verify_source_snapshot(root, manifest)

    def test_source_file_rejects_path_traversal_non_https_and_duplicate_paths(self) -> None:
        for path in ("../secret.csv", "/absolute.csv", "folder\\windows.csv", "./row.csv"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    source_file(path, b"row", "behavioral_rows")
        with self.assertRaises(ValueError):
            DatasetSourceFile(
                path="rows.csv",
                locator="http://example.invalid/rows.csv",
                byte_size=3,
                sha256=digest("row"),
                purpose="behavioral_rows",
            )
        duplicate = source_file("rows.csv", b"row", "behavioral_rows")
        with self.assertRaises(ValueError):
            source_manifest(files=(duplicate, duplicate))

    def test_canonical_trials_reject_malformed_duplicate_or_ambiguous_order(self) -> None:
        valid = CanonicalTransferTrial(
            participant_key="private-p1",
            trial_id=1,
            source_stratum="condition-a",
            first_stage_action="action_0",
            transition_common=True,
            final_state="state_0",
            second_stage_action="second_1",
            reward=1,
            row_commitment=digest("row-1"),
        )
        self.assertEqual(validate_canonical_trials((valid,)), (valid,))
        for field, value in (
            ("first_stage_action", "left"),
            ("final_state", "unknown"),
            ("second_stage_action", "unknown"),
            ("reward", 2),
            ("transition_common", 1),
        ):
            with self.subTest(field=field):
                with self.assertRaises((TypeError, ValueError)):
                    replace(valid, **{field: value})
        with self.assertRaisesRegex(ValueError, "duplicate trial"):
            validate_canonical_trials((valid, valid))
        later = replace(valid, trial_id=2, row_commitment=digest("row-2"))
        with self.assertRaisesRegex(ValueError, "chronological"):
            validate_canonical_trials((later, valid))

    def test_semantic_invariance_runs_the_whole_pipeline(self) -> None:
        pipeline = recording_semantic_pipeline()

        evidence = evaluate_end_to_end_semantic_invariance(
            pipeline=pipeline,
            source_rows=synthetic_rows(),
            relabelings=five_coherent_relabelings(),
        )

        self.assertIsInstance(evidence, SemanticInvarianceEvidence)
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.relabeling_count, 5)
        self.assertEqual(
            pipeline.calls,
            ("parse", "transform", "scenario", "predict", "score", "report") * 6,
        )

    def test_semantic_invariance_rejects_missing_pipeline_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage sequence"):
            evaluate_end_to_end_semantic_invariance(
                pipeline=recording_semantic_pipeline(omit_report=True),
                source_rows=synthetic_rows(),
                relabelings=five_coherent_relabelings(),
            )

    def test_semantic_target_drift_produces_failed_evidence(self) -> None:
        drifted = five_coherent_relabelings()[2]
        evidence = evaluate_end_to_end_semantic_invariance(
            pipeline=recording_semantic_pipeline(drift_on=drifted.name),
            source_rows=synthetic_rows(),
            relabelings=five_coherent_relabelings(),
        )
        self.assertFalse(evidence.passed)
        self.assertEqual(evidence.failed_relabelings, (drifted.name,))

    def test_postchoice_mutation_does_not_change_model_visible_scenario(self) -> None:
        original, mutated = same_prechoice_different_postchoice_rows()
        pipeline = recording_semantic_pipeline()
        self.assertEqual(
            pipeline.scenario_hash(original),
            pipeline.scenario_hash(mutated),
        )

    def test_durable_source_records_are_strict_and_participant_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = synthetic_source_tree(Path(directory))
            verified = verify_source_snapshot(root, manifest)
        transform = TransformReceipt(
            source_manifest_hash=manifest.content_hash,
            source_snapshot_hash=verified.source_snapshot_hash,
            adapter_identity_hash=digest("adapter"),
            schema_version="cross-dataset-canonical-two-stage-v1",
            endpoint="observed_first_stage_binary_choice",
            participant_count=3,
            raw_trial_count=12,
            retained_trial_count=10,
            excluded_trial_count=2,
        )
        records = (
            (manifest, DatasetSourceManifest.from_payload),
            (verified, VerifiedSourceSnapshot.from_payload),
            (transform, TransformReceipt.from_payload),
        )
        for record, decoder in records:
            payload = record.to_payload()
            self.assertEqual(decoder(payload), record)
            serialized = json.dumps(payload, sort_keys=True).lower()
            self.assertNotIn("participant_id", serialized)
            self.assertNotIn("private-p1", serialized)
            with self.assertRaises(ValueError):
                decoder({**payload, "unknown": "field"})
            first = next(iter(payload))
            with self.assertRaises(ValueError):
                decoder({key: value for key, value in payload.items() if key != first})


if __name__ == "__main__":
    unittest.main()
