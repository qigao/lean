from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.observations import ObservationPartitionRole
from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout, run_git

_TRANSFORM_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.studies.two_stage_source import (
        TwoStageSourceManifest,
        verify_two_stage_snapshot,
    )
    from narrative_dynamics.studies.two_stage_transform import (
        TwoStageParticipantSplitPlan,
        assign_two_stage_participants,
        build_two_stage_observation_dataset,
        transform_two_stage_snapshot,
    )
except Exception as error:
    _TRANSFORM_IMPORT_ERROR = error


class TwoStageTransformTests(unittest.TestCase):
    def require_transform(self) -> None:
        self.assertIsNone(
            _TRANSFORM_IMPORT_ERROR,
            f"two-stage transform boundary is missing: {_TRANSFORM_IMPORT_ERROR}",
        )

    def _prepared(self, *, magic_n=6, spaceship_n=6):
        self.require_transform()
        temp = tempfile.TemporaryDirectory()
        root, revision = build_synthetic_two_stage_checkout(
            Path(temp.name),
            magic_n=magic_n,
            spaceship_n=spaceship_n,
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
        return temp, root, manifest, snapshot, report

    def test_magic_carpet_and_spaceship_rows_normalize_to_one_canonical_trial_schema(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        self.assertEqual({trial.pre_choice.task_variant for trial in report.trials}, {"magic_carpet", "spaceship"})
        for trial in report.trials:
            self.assertIn(trial.outcome.first_stage_action, ("action_0", "action_1"))
            self.assertIn(trial.outcome.final_state, ("state_0", "state_1"))
            self.assertIn(trial.outcome.second_stage_action, ("action_0", "action_1"))

    def test_slow_trials_are_removed_before_history_construction(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        self.assertGreater(report.slow_trial_count, 0)
        self.assertTrue(all(trial.pre_choice.trial_id != 1 for trial in report.trials))
        dataset = build_two_stage_observation_dataset(
            report,
            assign_two_stage_participants(report, TwoStageParticipantSplitPlan()),
        )
        for partition in dataset.partitions:
            for record in partition.records:
                history = tuple(record.scenario.payload["history"])
                self.assertTrue(all(int(item["trial_id"]) != 1 for item in history))

    def test_malformed_included_row_fails_entire_transform(self):
        self.require_transform()
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = build_synthetic_two_stage_checkout(Path(tmp))
            path = root / "results/magic_carpet/choices/m000_game.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            reward_index = rows[0].index("reward")
            slow_index = rows[0].index("slow")
            self.assertEqual(rows[1][slow_index], "0")
            rows[1][reward_index] = "9"
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            run_git(root, "add", path.relative_to(root).as_posix())
            run_git(root, "commit", "-m", "malformed retained row")
            revision = run_git(root, "rev-parse", "HEAD")
            manifest = TwoStageSourceManifest(
                name="malformed",
                version="1",
                repository="test/synthetic-two-stage",
                revision=revision,
                license_reference="test-only",
                files=_files(root),
            )
            snapshot = verify_two_stage_snapshot(root, manifest)
            with self.assertRaises(ValueError):
                transform_two_stage_snapshot(snapshot, manifest)

    def test_spaceship_target_decoding_can_use_current_transition_only_outside_model_input(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        ship = next(trial for trial in report.trials if trial.pre_choice.task_variant == "spaceship")
        self.assertIn(ship.outcome.first_stage_action, ("action_0", "action_1"))
        self.assertNotIn("transition_common", dict(ship.pre_choice.first_stage_configuration))

    def test_same_trial_postchoice_mutations_do_not_change_scenario_hash(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        trial = next(trial for trial in report.trials if trial.pre_choice.trial_id == 0)
        original = stable_content_hash(trial.pre_choice)
        changed = replace(
            trial,
            outcome=replace(
                trial.outcome,
                transition_common=not trial.outcome.transition_common,
                reward=1 - trial.outcome.reward,
            ),
        )
        self.assertEqual(original, stable_content_hash(changed.pre_choice))

    def test_current_latent_reward_probability_mutation_does_not_change_scenario_hash(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        trial = report.trials[0]
        changed = replace(
            trial,
            audit_latent_reward_probabilities=tuple(
                min(1.0, value + 0.01)
                for value in trial.audit_latent_reward_probabilities
            ),
        )
        self.assertEqual(stable_content_hash(trial.pre_choice), stable_content_hash(changed.pre_choice))

    def test_prior_retained_reward_mutation_changes_history_identity(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        assignment = assign_two_stage_participants(report, TwoStageParticipantSplitPlan())
        dataset = build_two_stage_observation_dataset(report, assignment)
        records = [
            record
            for partition in dataset.partitions
            for record in partition.records
            if record.metadata["source_participant_id"] == "m000"
        ]
        records.sort(key=lambda record: int(record.metadata["source_trial_id"]))
        self.assertGreaterEqual(len(records), 2)
        self.assertNotEqual(
            records[0].metadata["causal_history_hash"],
            records[1].metadata["causal_history_hash"],
        )

    def test_participant_split_is_task_stratified_deterministic_and_disjoint(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        plan = TwoStageParticipantSplitPlan()
        first = assign_two_stage_participants(report, plan)
        second = assign_two_stage_participants(report, plan)
        self.assertEqual(first, second)
        identities = [(task, participant) for task, participant, _ in first.assignments]
        self.assertEqual(len(identities), len(set(identities)))

    def test_pinned_24_21_inventory_apportions_to_14_5_5_and_13_4_4(self):
        temp, _, _, _, report = self._prepared(magic_n=24, spaceship_n=21)
        self.addCleanup(temp.cleanup)
        assignment = assign_two_stage_participants(report, TwoStageParticipantSplitPlan())
        counts = {}
        for task, _, role in assignment.assignments:
            counts[(task, role)] = counts.get((task, role), 0) + 1
        self.assertEqual(
            (counts[("magic_carpet", "train")], counts[("magic_carpet", "selection_validation")], counts[("magic_carpet", "final_test")]),
            (14, 5, 5),
        )
        self.assertEqual(
            (counts[("spaceship", "train")], counts[("spaceship", "selection_validation")], counts[("spaceship", "final_test")]),
            (13, 4, 4),
        )

    def test_participant_assignment_hash_changes_on_any_role_change(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        assignment = assign_two_stage_participants(report, TwoStageParticipantSplitPlan())
        task, participant, role = assignment.assignments[0]
        replacement_role = "final_test" if role != "final_test" else "train"
        changed = replace(
            assignment,
            assignments=((task, participant, replacement_role),) + assignment.assignments[1:],
        )
        self.assertNotEqual(assignment.content_hash, changed.content_hash)

    def test_observation_dataset_has_positive_external_origin_and_record_level_partitions(self):
        temp, _, _, _, report = self._prepared()
        self.addCleanup(temp.cleanup)
        assignment = assign_two_stage_participants(report, TwoStageParticipantSplitPlan())
        dataset = build_two_stage_observation_dataset(report, assignment)
        self.assertEqual(dataset.source["kind"], "external_observational")
        self.assertIs(dataset.provenance["external_observational"], True)
        self.assertEqual(
            {partition.role for partition in dataset.partitions},
            {
                ObservationPartitionRole.TRAIN,
                ObservationPartitionRole.SELECTION_VALIDATION,
                ObservationPartitionRole.FINAL_TEST,
            },
        )


if __name__ == "__main__":
    unittest.main()
