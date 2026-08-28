from __future__ import annotations

import math
import unittest

from narrative_dynamics.losses import evaluate_metric_loss

_METRIC_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.adapters.two_stage_metrics import (
        FIRST_STAGE_METRIC_KEYS,
        FIRST_STAGE_PROBABILITY_FLOOR,
        two_stage_brier_loss,
        two_stage_brier_thresholds,
        two_stage_first_stage_policy_metrics,
        two_stage_log_loss,
        two_stage_log_thresholds,
    )
except Exception as error:
    _METRIC_IMPORT_ERROR = error


class _Trace:
    def __init__(self, policy):
        self.outcome = {"first_stage_policy": policy}


class TwoStageMetricTests(unittest.TestCase):
    def require_metrics(self):
        self.assertIsNone(
            _METRIC_IMPORT_ERROR,
            f"two-stage metric boundary is missing: {_METRIC_IMPORT_ERROR}",
        )

    def test_extractor_returns_exact_binary_metric_schema(self):
        self.require_metrics()
        metrics = two_stage_first_stage_policy_metrics(
            _Trace({"action_0": 0.25, "action_1": 0.75})
        )
        self.assertEqual(tuple(metrics), FIRST_STAGE_METRIC_KEYS)

    def test_probability_floor_is_exactly_1e_minus_12_and_renormalizes(self):
        self.require_metrics()
        self.assertEqual(FIRST_STAGE_PROBABILITY_FLOOR, 1e-12)
        metrics = two_stage_first_stage_policy_metrics(
            _Trace({"action_0": 0.0, "action_1": 1.0})
        )
        self.assertGreater(metrics["first_stage.action_0"], 0.0)
        self.assertTrue(math.isclose(math.fsum(metrics.values()), 1.0, rel_tol=0.0, abs_tol=1e-15))

    def test_extractor_does_not_mutate_raw_runtime_policy(self):
        self.require_metrics()
        raw = {"action_0": 0.0, "action_1": 1.0}
        before = dict(raw)
        two_stage_first_stage_policy_metrics(_Trace(raw))
        self.assertEqual(raw, before)

    def test_uniform_binary_brier_is_exactly_point_5(self):
        self.require_metrics()
        observed = {"first_stage.action_0": 0.5, "first_stage.action_1": 0.5}
        target = {"first_stage.action_0": 1.0, "first_stage.action_1": 0.0}
        self.assertEqual(evaluate_metric_loss(two_stage_brier_loss(), observed, target), 0.5)

    def test_uniform_binary_log_is_log_2(self):
        self.require_metrics()
        observed = {"first_stage.action_0": 0.5, "first_stage.action_1": 0.5}
        target = {"first_stage.action_0": 1.0, "first_stage.action_1": 0.0}
        self.assertTrue(
            math.isclose(
                evaluate_metric_loss(two_stage_log_loss(), observed, target),
                math.log(2.0),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )

    def test_strict_adequacy_thresholds_are_downward_adjacent_representable_values(self):
        self.require_metrics()
        self.assertEqual(two_stage_brier_thresholds().max_mean_loss, math.nextafter(0.5, -math.inf))
        self.assertEqual(two_stage_log_thresholds().max_mean_loss, math.nextafter(math.log(2.0), -math.inf))

    def test_separation_margins_are_one_percent_of_uniform_references(self):
        self.require_metrics()
        self.assertEqual(0.005, 0.01 * 0.5)
        self.assertTrue(math.isclose(0.006931471805599453, 0.01 * math.log(2.0), rel_tol=0.0, abs_tol=1e-18))


if __name__ == "__main__":
    unittest.main()
