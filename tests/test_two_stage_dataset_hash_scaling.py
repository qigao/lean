from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from narrative_dynamics.studies.two_stage_source import (
    TwoStageSourceManifest,
    verify_two_stage_snapshot,
)
from narrative_dynamics.studies.two_stage_transform import (
    TwoStageParticipantSplitPlan,
    TwoStageTransformReport,
    assign_two_stage_participants,
    build_two_stage_observation_dataset,
    transform_two_stage_snapshot,
)
from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout


class TwoStageDatasetHashScalingTests(unittest.TestCase):
    def test_dataset_builder_computes_transform_report_hash_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(
                Path(tmp),
                magic_n=3,
                spaceship_n=3,
            )
            manifest = TwoStageSourceManifest(
                name="synthetic-two-stage-source",
                version="1",
                repository="test/synthetic-two-stage",
                revision=revision,
                license_reference="test-only",
                files=_files(root),
            )
            snapshot = verify_two_stage_snapshot(root, manifest)
            report = transform_two_stage_snapshot(snapshot, manifest)
            assignment = assign_two_stage_participants(
                report,
                TwoStageParticipantSplitPlan(),
            )

            original_getter = TwoStageTransformReport.content_hash.fget
            assert original_getter is not None
            calls = 0

            def counted_content_hash(instance: TwoStageTransformReport) -> str:
                nonlocal calls
                calls += 1
                return original_getter(instance)

            with patch.object(
                TwoStageTransformReport,
                "content_hash",
                new=property(counted_content_hash),
            ):
                dataset = build_two_stage_observation_dataset(report, assignment)

            record_count = sum(len(partition.records) for partition in dataset.partitions)
            self.assertGreater(record_count, 1)
            self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
