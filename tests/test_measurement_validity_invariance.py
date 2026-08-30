from __future__ import annotations

from dataclasses import replace
import math
import unittest

from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_log_loss,
)
from narrative_dynamics.losses import evaluate_metric_loss
from narrative_dynamics.measurement_validity import (
    ExactInvarianceFinding,
    MeasurementAggregation,
    MeasurementDependenceFinding,
    MeasurementScore,
    MeasurementValidityStatus,
    classify_material_reversal,
    evaluate_categorical_coordinate_invariance,
    permute_binary_metric_map,
)

from tests.measurement_validity_fixtures import digest


class MeasurementInvarianceTests(unittest.TestCase):
    def test_simultaneous_binary_coordinate_swap_preserves_brier_and_log(self) -> None:
        target = {
            "first_stage.action_0": 1.0,
            "first_stage.action_1": 0.0,
        }
        prediction = {
            "first_stage.action_0": 0.8,
            "first_stage.action_1": 0.2,
        }
        permuted_target = permute_binary_metric_map(target)
        permuted_prediction = permute_binary_metric_map(prediction)

        original_brier = evaluate_metric_loss(
            two_stage_brier_loss(), prediction, target
        )
        permuted_brier = evaluate_metric_loss(
            two_stage_brier_loss(), permuted_prediction, permuted_target
        )
        original_log = evaluate_metric_loss(
            two_stage_log_loss(), prediction, target
        )
        permuted_log = evaluate_metric_loss(
            two_stage_log_loss(), permuted_prediction, permuted_target
        )

        self.assertEqual(original_brier, 0.07999999999999999)
        self.assertEqual(permuted_brier, original_brier)
        self.assertEqual(original_log, -math.log(0.8))
        self.assertEqual(permuted_log, original_log)
        self.assertEqual(
            permute_binary_metric_map(permuted_target),
            target,
        )

        finding = evaluate_categorical_coordinate_invariance(target, prediction)
        self.assertIs(
            finding.status,
            MeasurementValidityStatus.EXACT_INVARIANCE_MET,
        )
        self.assertEqual(finding.original_hash, finding.transformed_hash)

    def test_coordinate_permutation_rejects_invalid_probability_maps(self) -> None:
        invalid_maps = (
            {"first_stage.action_0": 1.0},
            {
                "first_stage.action_0": 0.5,
                "first_stage.action_1": 0.4,
            },
            {
                "first_stage.action_0": math.nan,
                "first_stage.action_1": 0.0,
            },
            {"action_0": 0.5, "action_1": 0.5},
            {
                "first_stage.action_0": 0.3,
                "first_stage.action_1": 0.3,
                "first_stage.action_2": 0.4,
            },
        )
        valid = {
            "first_stage.action_0": 0.5,
            "first_stage.action_1": 0.5,
        }
        for values in invalid_maps:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    permute_binary_metric_map(values)
                with self.assertRaises((TypeError, ValueError)):
                    evaluate_categorical_coordinate_invariance(values, valid)

    def test_material_reversal_classification_uses_both_direction_margins(self) -> None:
        rows = (
            (
                (-0.006, -0.010),
                (0.005, 0.005),
                MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
            ),
            (
                (-0.006, 0.006),
                (0.005, 0.005),
                MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT,
            ),
            (
                (-0.006, 0.004),
                (0.005, 0.005),
                MeasurementValidityStatus.INCONCLUSIVE_SENSITIVITY,
            ),
            (
                (0.0, 0.0),
                (0.005, 0.005),
                MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
            ),
        )
        for deltas, references, expected in rows:
            with self.subTest(deltas=deltas):
                self.assertIs(
                    classify_material_reversal(deltas, references),
                    expected,
                )

    def test_material_reversal_rejects_ambiguous_numeric_inputs(self) -> None:
        invalid = (
            ((), ()),
            ((-0.1,), (0.1, 0.1)),
            ((math.nan, 0.1), (0.1, 0.1)),
            ((-0.1, 0.1), (0.0, 0.1)),
            ((-0.1, 0.1), (-0.1, 0.1)),
            ((-0.1, 0.1), (math.inf, 0.1)),
        )
        for deltas, references in invalid:
            with self.subTest(deltas=deltas, references=references):
                with self.assertRaises((TypeError, ValueError)):
                    classify_material_reversal(deltas, references)

    def test_findings_are_typed_hashed_and_reject_wrong_status_family(self) -> None:
        exact = ExactInvarianceFinding(
            check_name="binary_coordinate_swap",
            status=MeasurementValidityStatus.EXACT_INVARIANCE_MET,
            original_hash=digest("1"),
            transformed_hash=digest("1"),
            details_hash=digest("2"),
        )
        self.assertEqual(exact.content_hash, exact.content_hash)
        with self.assertRaisesRegex(ValueError, "exact-invariance status"):
            replace(
                exact,
                status=MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
            )

        dependence = MeasurementDependenceFinding(
            dimension="task",
            score=MeasurementScore.BRIER,
            aggregation=MeasurementAggregation.TRIAL_EQUAL,
            task="magic_carpet_vs_spaceship",
            model_pair=("reactive", "planning"),
            deltas=(-0.006, 0.006),
            references=(0.005, 0.005),
            status=MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT,
        )
        self.assertNotEqual(dependence.content_hash, exact.content_hash)
        with self.assertRaisesRegex(ValueError, "dependence status"):
            replace(
                dependence,
                status=MeasurementValidityStatus.EXACT_INVARIANCE_MET,
            )


if __name__ == "__main__":
    unittest.main()
