from __future__ import annotations

import unittest

import narrative_dynamics


class ModelAdequacyPublicAPITests(unittest.TestCase):
    def test_metric_loss_api_is_exported_from_package_root(self):
        expected = (
            "DEFAULT_METRIC_LOSS",
            "CategoricalBrierLoss",
            "CategoricalLogLoss",
            "CategoricalMetricGroup",
            "MetricLoss",
            "WeightedSquaredErrorLoss",
            "evaluate_metric_loss",
            "metric_loss_identity",
        )
        missing = tuple(
            name for name in expected if getattr(narrative_dynamics, name, None) is None
        )
        self.assertEqual(missing, ())
        self.assertTrue(set(expected).issubset(set(narrative_dynamics.__all__)))


if __name__ == "__main__":
    unittest.main()
