from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.studies.two_stage_source import (
    TwoStageSourceManifest,
    verify_two_stage_snapshot,
)
from narrative_dynamics.studies.two_stage_transform import transform_two_stage_snapshot
from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout, run_git


class TwoStageRealSlowSentinelTests(unittest.TestCase):
    def test_slow_source_sentinels_are_excluded_before_scorable_field_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = build_synthetic_two_stage_checkout(Path(tmp))
            path = root / "results/magic_carpet/choices/m000_game.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

            header = rows[0]
            slow_index = header.index("slow")
            slow_row = next(row for row in rows[1:] if row[slow_index] == "1")
            slow_trial_id = int(slow_row[header.index("trial")])
            for field in (
                "rt1",
                "choice1",
                "final_state",
                "fsymbol_lft",
                "fsymbol_rgt",
                "rt2",
                "choice2",
            ):
                slow_row[header.index(field)] = "-1"

            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            run_git(root, "add", path.relative_to(root).as_posix())
            run_git(root, "commit", "-m", "source slow sentinel row")
            revision = run_git(root, "rev-parse", "HEAD")

            manifest = TwoStageSourceManifest(
                name="synthetic-source-slow-sentinel",
                version="1",
                repository="test/synthetic-two-stage",
                revision=revision,
                license_reference="test-only",
                files=_files(root),
            )
            snapshot = verify_two_stage_snapshot(root, manifest)
            report = transform_two_stage_snapshot(snapshot, manifest)

            self.assertGreater(report.slow_trial_count, 0)
            retained_ids = {
                trial.pre_choice.trial_id
                for trial in report.trials
                if trial.pre_choice.task_variant == "magic_carpet"
                and trial.pre_choice.source_participant_id == "m000"
            }
            self.assertNotIn(slow_trial_id, retained_ids)


if __name__ == "__main__":
    unittest.main()
