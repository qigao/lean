from dataclasses import replace
import unittest

from narrative_dynamics.abm import initialize_population, simulate_population
from narrative_dynamics.abm.metrics import EmergenceMetrics, measure_emergence
from tests.network_abm_fixtures import line_case


class NetworkABMMetricsTests(unittest.TestCase):
    def test_split_population_metrics_are_hand_computable(self):
        model, _ = line_case()
        state = initialize_population(model, beliefs={"a": 1.0, "b": 1.0})

        metrics = measure_emergence(model, state)

        self.assertIsInstance(metrics, EmergenceMetrics)
        self.assertEqual(metrics.mean_belief, 0.5)
        self.assertEqual(metrics.adoption_rate, 0.5)
        self.assertEqual(metrics.broadcasting_rate, 0.5)
        self.assertEqual(metrics.informed_rate, 0.5)
        self.assertEqual(metrics.consensus, 0.0)
        self.assertEqual(metrics.polarization, 1.0)
        self.assertEqual(
            metrics.to_dict(),
            {
                "mean_belief": 0.5,
                "adoption_rate": 0.5,
                "broadcasting_rate": 0.5,
                "informed_rate": 0.5,
                "consensus": 0.0,
                "polarization": 1.0,
            },
        )

    def test_uniform_population_has_consensus_without_polarization(self):
        model, _ = line_case()
        state = initialize_population(
            model,
            beliefs={agent_id: 0.5 for agent_id in model.network.agent_ids},
        )

        metrics = measure_emergence(model, state)

        self.assertEqual(metrics.mean_belief, 0.5)
        self.assertEqual(metrics.adoption_rate, 1.0)
        self.assertEqual(metrics.broadcasting_rate, 1.0)
        self.assertEqual(metrics.informed_rate, 1.0)
        self.assertEqual(metrics.consensus, 1.0)
        self.assertEqual(metrics.polarization, 0.0)

    def test_line_diffusion_produces_expected_population_change(self):
        model, initial = line_case()
        final = simulate_population(model, initial, rounds=2).final_state

        metrics = measure_emergence(model, final)

        self.assertEqual(metrics.mean_belief, 0.75)
        self.assertEqual(metrics.adoption_rate, 0.75)
        self.assertEqual(metrics.broadcasting_rate, 0.75)
        self.assertEqual(metrics.informed_rate, 0.75)
        self.assertEqual(metrics.consensus, 0.0)
        self.assertEqual(metrics.polarization, 0.75)

    def test_metrics_reject_model_state_mismatch(self):
        model, initial = line_case()
        with self.assertRaisesRegex(ValueError, "model identity"):
            measure_emergence(replace(model, model_id="other"), initial)


if __name__ == "__main__":
    unittest.main()
