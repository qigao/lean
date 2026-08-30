from __future__ import annotations

from dataclasses import replace
import math
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.measurement_validity import (
    MeasurementAggregateLoss,
    MeasurementAggregation,
    MeasurementCaseLoss,
    MeasurementPairwiseDelta,
    MeasurementRobustnessProfile,
    MeasurementScore,
    MeasurementValidityStatus,
    ParticipantInfluenceRange,
    aggregate_measurement_losses,
    build_measurement_robustness_profile,
    participant_equal_mean,
    trial_equal_mean,
)
from narrative_dynamics.observations import ObservationPartitionRole

from tests.measurement_validity_fixtures import measurement_protocol


_ROLES = (
    ObservationPartitionRole.TRAIN,
    ObservationPartitionRole.SELECTION_VALIDATION,
)
_TASKS = ("magic_carpet", "spaceship")
_MODELS = ("reactive", "intentional", "planning")


def _fixture_hash(*parts: object) -> str:
    return stable_content_hash(("measurement-robustness-fixture",) + parts)


def _reactive_delta(
    role: ObservationPartitionRole,
    task: str,
    score: MeasurementScore,
    participant_index: int,
) -> float:
    if role is ObservationPartitionRole.TRAIN and task == "magic_carpet":
        if score is MeasurementScore.BRIER:
            return (-0.02, 0.03)[participant_index]
        return 0.01
    if role is ObservationPartitionRole.TRAIN:
        return 0.01
    if task == "magic_carpet":
        return -0.006 if score is MeasurementScore.BRIER else 0.004
    return -0.006 if score is MeasurementScore.BRIER else -0.008


def _case_losses(
    *,
    one_participant_stratum: tuple[ObservationPartitionRole, str] | None = None,
) -> tuple[MeasurementCaseLoss, ...]:
    rows: list[MeasurementCaseLoss] = []
    for role in _ROLES:
        for task in _TASKS:
            participant_counts = (4, 1)
            if one_participant_stratum == (role, task):
                participant_counts = (4,)
            for participant_index, count in enumerate(participant_counts):
                participant_hash = _fixture_hash(
                    role.value,
                    task,
                    "participant",
                    participant_index,
                )
                for case_index in range(count):
                    case_hash = _fixture_hash(
                        role.value,
                        task,
                        participant_index,
                        case_index,
                    )
                    for score in MeasurementScore:
                        reactive_delta = _reactive_delta(
                            role,
                            task,
                            score,
                            participant_index,
                        )
                        model_losses = {
                            "reactive": 0.5 + reactive_delta,
                            "intentional": 0.502,
                            "planning": 0.5,
                        }
                        for model_name in _MODELS:
                            rows.append(
                                MeasurementCaseLoss(
                                    case_hash=case_hash,
                                    role=role,
                                    task_variant=task,
                                    participant_group_hash=participant_hash,
                                    model_name=model_name,
                                    score=score,
                                    value=model_losses[model_name],
                                )
                            )
    return tuple(rows)


def _finding(rows, **expected):
    matches = tuple(
        row
        for row in rows
        if all(getattr(row, key) == value for key, value in expected.items())
    )
    if len(matches) != 1:
        raise AssertionError(f"expected one finding, got {len(matches)}: {expected}")
    return matches[0]


class MeasurementRobustnessTests(unittest.TestCase):
    def test_trial_and_participant_equal_formulas_are_frozen(self) -> None:
        participant_losses = {
            "sha256:" + "1" * 64: (0.1, 0.1, 0.1, 0.1),
            "sha256:" + "2" * 64: (0.9,),
        }
        self.assertEqual(trial_equal_mean(participant_losses), 0.26)
        self.assertEqual(participant_equal_mean(participant_losses), 0.5)

        rows = _case_losses()
        trial_rows = aggregate_measurement_losses(
            rows,
            MeasurementAggregation.TRIAL_EQUAL,
        )
        participant_rows = aggregate_measurement_losses(
            tuple(reversed(rows)),
            MeasurementAggregation.PARTICIPANT_EQUAL,
        )
        trial = _finding(
            trial_rows,
            role=ObservationPartitionRole.TRAIN,
            task_variant="magic_carpet",
            score=MeasurementScore.BRIER,
            model_name="reactive",
        )
        participant = _finding(
            participant_rows,
            role=ObservationPartitionRole.TRAIN,
            task_variant="magic_carpet",
            score=MeasurementScore.BRIER,
            model_name="reactive",
        )
        self.assertIsInstance(trial, MeasurementAggregateLoss)
        self.assertAlmostEqual(trial.mean_loss, 0.49)
        self.assertAlmostEqual(participant.mean_loss, 0.505)
        self.assertEqual((trial.case_count, trial.participant_count), (5, 2))
        self.assertEqual(
            (participant.case_count, participant.participant_count),
            (5, 2),
        )

    def test_profile_covers_tasks_roles_scores_aggregations_and_pairs(self) -> None:
        profile = build_measurement_robustness_profile(
            _case_losses(),
            measurement_protocol(),
        )
        self.assertIsInstance(profile, MeasurementRobustnessProfile)
        self.assertEqual(len(profile.aggregate_losses), 48)
        self.assertEqual(len(profile.pairwise_deltas), 48)
        self.assertTrue(
            all(
                isinstance(row, MeasurementPairwiseDelta)
                for row in profile.pairwise_deltas
            )
        )
        self.assertEqual(
            {
                (row.first_model, row.second_model)
                for row in profile.pairwise_deltas
            },
            {
                ("reactive", "intentional"),
                ("reactive", "planning"),
                ("intentional", "planning"),
            },
        )
        self.assertEqual(
            {row.role for row in profile.aggregate_losses},
            set(_ROLES),
        )
        self.assertEqual(
            {row.task_variant for row in profile.aggregate_losses},
            set(_TASKS),
        )
        self.assertEqual(
            {row.score for row in profile.aggregate_losses},
            set(MeasurementScore),
        )
        self.assertEqual(
            {row.aggregation for row in profile.aggregate_losses},
            set(MeasurementAggregation),
        )

    def test_profile_classifies_task_aggregation_score_and_stability(self) -> None:
        profile = build_measurement_robustness_profile(
            _case_losses(),
            measurement_protocol(),
        )
        model_pair = ("reactive", "planning")
        material_task = _finding(
            profile.task_findings,
            score=MeasurementScore.BRIER,
            aggregation=MeasurementAggregation.TRIAL_EQUAL,
            task="train:magic_carpet_vs_spaceship",
            model_pair=model_pair,
        )
        self.assertAlmostEqual(material_task.deltas[0], -0.01)
        self.assertAlmostEqual(material_task.deltas[1], 0.01)
        self.assertIs(
            material_task.status,
            MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT,
        )

        material_aggregation = _finding(
            profile.aggregation_findings,
            score=MeasurementScore.BRIER,
            aggregation=MeasurementAggregation.TRIAL_EQUAL,
            task="train:magic_carpet",
            model_pair=model_pair,
        )
        self.assertAlmostEqual(material_aggregation.deltas[0], -0.01)
        self.assertAlmostEqual(material_aggregation.deltas[1], 0.005)
        self.assertIs(
            material_aggregation.status,
            MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT,
        )

        material_score = _finding(
            profile.score_findings,
            score=MeasurementScore.BRIER,
            aggregation=MeasurementAggregation.TRIAL_EQUAL,
            task="train:magic_carpet",
            model_pair=model_pair,
        )
        self.assertAlmostEqual(material_score.deltas[0], -0.01)
        self.assertAlmostEqual(material_score.deltas[1], 0.01)
        self.assertEqual(
            material_score.references,
            (0.005, 0.006931471805599453),
        )
        self.assertIs(
            material_score.status,
            MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT,
        )

        inconclusive = _finding(
            profile.score_findings,
            score=MeasurementScore.BRIER,
            aggregation=MeasurementAggregation.TRIAL_EQUAL,
            task="selection_validation:magic_carpet",
            model_pair=model_pair,
        )
        self.assertIs(
            inconclusive.status,
            MeasurementValidityStatus.INCONCLUSIVE_SENSITIVITY,
        )
        stable = _finding(
            profile.task_findings,
            score=MeasurementScore.BRIER,
            aggregation=MeasurementAggregation.TRIAL_EQUAL,
            task="selection_validation:magic_carpet_vs_spaceship",
            model_pair=model_pair,
        )
        self.assertIs(
            stable.status,
            MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
        )

    def test_leave_one_participant_out_range_is_deterministic_and_material(self) -> None:
        profile = build_measurement_robustness_profile(
            _case_losses(),
            measurement_protocol(),
        )
        influence = _finding(
            profile.participant_influence,
            role=ObservationPartitionRole.TRAIN,
            task_variant="magic_carpet",
            score=MeasurementScore.BRIER,
            aggregation=MeasurementAggregation.TRIAL_EQUAL,
            first_model="reactive",
            second_model="planning",
        )
        self.assertIsInstance(influence, ParticipantInfluenceRange)
        self.assertAlmostEqual(influence.minimum_delta, -0.02)
        self.assertAlmostEqual(influence.maximum_delta, 0.03)
        self.assertEqual(influence.omitted_participant_count, 2)
        self.assertIs(
            influence.status,
            MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT,
        )

        reordered = build_measurement_robustness_profile(
            tuple(reversed(_case_losses())),
            measurement_protocol(),
        )
        self.assertEqual(profile.content_hash, reordered.content_hash)
        self.assertEqual(profile, reordered)

    def test_one_participant_support_is_explicitly_not_established(self) -> None:
        profile = build_measurement_robustness_profile(
            _case_losses(
                one_participant_stratum=(
                    ObservationPartitionRole.SELECTION_VALIDATION,
                    "spaceship",
                )
            ),
            measurement_protocol(),
        )
        influence = _finding(
            profile.participant_influence,
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            task_variant="spaceship",
            score=MeasurementScore.LOG,
            aggregation=MeasurementAggregation.PARTICIPANT_EQUAL,
            first_model="reactive",
            second_model="planning",
        )
        self.assertIsNone(influence.minimum_delta)
        self.assertIsNone(influence.maximum_delta)
        self.assertEqual(influence.omitted_participant_count, 1)
        self.assertIs(
            influence.status,
            MeasurementValidityStatus.NOT_ESTABLISHED,
        )

    def test_profile_fails_closed_on_incomplete_duplicate_or_nonfinite_rows(self) -> None:
        rows = _case_losses()
        invalid_sets = (
            (
                tuple(row for row in rows if row.task_variant != "spaceship"),
                "task strata",
            ),
            (
                tuple(row for row in rows if row.model_name != "planning"),
                "model",
            ),
            (rows + (rows[0],), "duplicate"),
        )
        for invalid_rows, message in invalid_sets:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_measurement_robustness_profile(
                        invalid_rows,
                        measurement_protocol(),
                    )

        with self.assertRaises((TypeError, ValueError)):
            replace(rows[0], value=math.nan)
        with self.assertRaises((TypeError, ValueError)):
            trial_equal_mean({_fixture_hash("participant"): ()})
        with self.assertRaises((TypeError, ValueError)):
            participant_equal_mean(
                {_fixture_hash("participant"): (0.1, math.inf)}
            )


if __name__ == "__main__":
    unittest.main()
