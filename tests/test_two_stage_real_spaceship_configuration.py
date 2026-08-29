from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.adapters.narrative_two_stage import (
    create_narrative_two_stage_intentional_source,
    create_narrative_two_stage_planning_source,
    create_narrative_two_stage_reactive_source,
)
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.studies.two_stage_source import (
    TwoStageSourceManifest,
    verify_two_stage_snapshot,
)
from narrative_dynamics.studies.two_stage_transform import transform_two_stage_snapshot
from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout, run_git


class TwoStageRealSpaceshipConfigurationTests(unittest.TestCase):
    def test_equal_spaceship_and_planet_indices_are_valid_and_preserve_target_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = build_synthetic_two_stage_checkout(
                Path(tmp), magic_n=1, spaceship_n=1
            )
            path = root / "results/spaceship/choices/s000.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

            header = rows[0]
            retained = next(row for row in rows[1:] if row[header.index("slow")] == "0")
            retained[header.index("symbol1")] = retained[header.index("symbol0")]
            common = int(retained[header.index("common")])
            final_state = int(retained[header.index("final_state")])
            relative = final_state + 1 if common else 2 - final_state
            expected_action = "action_0" if relative == 1 else "action_1"
            trial_id = int(retained[header.index("trial")])

            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            run_git(root, "add", path.relative_to(root).as_posix())
            run_git(root, "commit", "-m", "real spaceship equal configuration")
            revision = run_git(root, "rev-parse", "HEAD")

            manifest = TwoStageSourceManifest(
                name="synthetic-real-spaceship-configuration",
                version="1",
                repository="test/synthetic-two-stage",
                revision=revision,
                license_reference="test-only",
                files=_files(root),
            )
            snapshot = verify_two_stage_snapshot(root, manifest)
            report = transform_two_stage_snapshot(snapshot, manifest)
            trial = next(
                item
                for item in report.trials
                if item.pre_choice.task_variant == "spaceship"
                and item.pre_choice.source_participant_id == "s000"
                and item.pre_choice.trial_id == trial_id
            )

            configuration = dict(trial.pre_choice.first_stage_configuration)
            self.assertEqual(configuration["symbol0"], configuration["symbol1"])
            self.assertEqual(trial.outcome.first_stage_action, expected_action)

            scenario = Scenario(
                id="real-spaceship-equal-configuration",
                payload={
                    "task_variant": "spaceship",
                    "first_stage_configuration": trial.pre_choice.first_stage_configuration,
                    "history": (),
                },
            )
            runner = SimulationRunner()
            cases = (
                (create_narrative_two_stage_reactive_source(), {"beta": 1.0}),
                (
                    create_narrative_two_stage_intentional_source(),
                    {"beta": 1.0, "memory_decay": 1.0},
                ),
                (
                    create_narrative_two_stage_planning_source(),
                    {"beta": 1.0, "memory_decay": 1.0},
                ),
            )
            for source, parameters in cases:
                trace = runner.run_once(source, scenario, parameters, seed=1)
                self.assertEqual(
                    set(trace.outcome["first_stage_policy"]),
                    {"action_0", "action_1"},
                )


if __name__ == "__main__":
    unittest.main()
