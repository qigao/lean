from dataclasses import replace
import unittest

from narrative_dynamics.abm.adaptive_metrics import (
    AdaptiveTrustMetrics,
    measure_adaptive_trust,
)
from narrative_dynamics.abm.learning import simulate_adaptive_round
from tests.adaptive_trust_fixtures import adaptive_case, truth_feedback


class AdaptiveTrustMetricTests(unittest.TestCase):
    def test_feedback_outcome_has_hand_computed_trust_metrics(self):
        model, initial = adaptive_case()
        learned = simulate_adaptive_round(
            model,
            initial,
            feedback=truth_feedback(),
        ).next_state

        metrics = measure_adaptive_trust(model, learned)

        self.assertIsInstance(metrics, AdaptiveTrustMetrics)
        self.assertEqual(metrics.mean_trust, 0.5)
        self.assertEqual(metrics.min_trust, 0.25)
        self.assertEqual(metrics.max_trust, 0.75)
        self.assertEqual(metrics.learned_edge_rate, 1.0)
        self.assertEqual(
            metrics.to_dict(),
            {
                "mean_trust": 0.5,
                "min_trust": 0.25,
                "max_trust": 0.75,
                "learned_edge_rate": 1.0,
            },
        )

    def test_initial_trust_is_uniform_and_unlearned(self):
        model, initial = adaptive_case(initial_trust=0.4)

        metrics = measure_adaptive_trust(model, initial)

        self.assertEqual(metrics.mean_trust, 0.4)
        self.assertEqual(metrics.min_trust, 0.4)
        self.assertEqual(metrics.max_trust, 0.4)
        self.assertEqual(metrics.learned_edge_rate, 0.0)

    def test_trust_metrics_reject_model_state_mismatch(self):
        model, initial = adaptive_case()
        with self.assertRaisesRegex(ValueError, "model identity"):
            measure_adaptive_trust(replace(model, model_id="other"), initial)


if __name__ == "__main__":
    unittest.main()
