from dataclasses import replace
import unittest

from narrative_dynamics.abm import NetworkABMModel, SocialNetwork
from narrative_dynamics.abm.rewiring import simulate_rewiring_round
from narrative_dynamics.abm.rewiring_contracts import (
    EndogenousRewiringModel,
    initialize_rewiring_population,
)
from narrative_dynamics.abm.rewiring_metrics import (
    NetworkStructureMetrics,
    measure_network_structure,
)
from tests.rewiring_fixtures import rewiring_base_model


def rewiring_model(base=None):
    return EndogenousRewiringModel(
        "endogenous-rewiring",
        "1",
        rewiring_base_model() if base is None else base,
        0.2,
        0.8,
    )


class NetworkStructureMetricTests(unittest.TestCase):
    def test_fragmented_topology_metrics_match_by_hand(self):
        model = rewiring_model()
        state = initialize_rewiring_population(
            model,
            beliefs={"a": 1.0, "b": 0.0, "c": 0.0},
        )

        metrics = measure_network_structure(model, state)

        self.assertEqual(metrics.catalog_edge_count, 2)
        self.assertEqual(metrics.active_edge_count, 1)
        self.assertEqual(metrics.active_edge_rate, 0.5)
        self.assertEqual(metrics.mean_active_similarity, 0.0)
        self.assertEqual(metrics.rewired_edge_rate, 0.0)
        self.assertEqual(metrics.cumulative_rewirings, 0)
        self.assertEqual(metrics.weak_component_count, 2)
        self.assertEqual(metrics.largest_component_share, 2 / 3)

    def test_formed_edge_updates_density_similarity_churn_and_connectivity(self):
        model = rewiring_model()
        initial = initialize_rewiring_population(
            model,
            beliefs={"a": 1.0, "b": 0.0, "c": 1.0},
        )
        state = simulate_rewiring_round(model, initial).next_state

        metrics = measure_network_structure(model, state)

        self.assertEqual(metrics.active_edge_count, 2)
        self.assertEqual(metrics.active_edge_rate, 1.0)
        self.assertEqual(metrics.mean_active_similarity, 1.0)
        self.assertEqual(metrics.rewired_edge_rate, 0.5)
        self.assertEqual(metrics.cumulative_rewirings, 1)
        self.assertEqual(metrics.weak_component_count, 1)
        self.assertEqual(metrics.largest_component_share, 1.0)

    def test_no_active_edges_count_every_agent_as_an_isolated_component(self):
        base = rewiring_base_model()
        inactive_edges = tuple(replace(item, active=False) for item in base.network.edges)
        inactive_base = NetworkABMModel(
            base.model_id,
            base.version,
            base.agents,
            SocialNetwork(base.network.agent_ids, inactive_edges),
        )
        model = rewiring_model(inactive_base)
        state = initialize_rewiring_population(
            model,
            beliefs={"a": 1.0, "b": 0.0, "c": 0.5},
        )

        metrics = measure_network_structure(model, state)

        self.assertEqual(metrics.active_edge_count, 0)
        self.assertEqual(metrics.mean_active_similarity, 0.0)
        self.assertEqual(metrics.weak_component_count, 3)
        self.assertEqual(metrics.largest_component_share, 1 / 3)

    def test_metric_contract_rejects_inconsistent_counts_and_rates(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            NetworkStructureMetrics(2, 3, 1.0, 0.5, 0.0, 0, 1, 1.0)
        with self.assertRaisesRegex(ValueError, "active edge rate"):
            NetworkStructureMetrics(2, 1, 0.75, 0.5, 0.0, 0, 2, 2 / 3)

    def test_measurement_requires_exact_model_state_binding(self):
        first = rewiring_model()
        second = EndogenousRewiringModel(
            "different",
            "1",
            rewiring_base_model(),
            0.2,
            0.8,
        )
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            measure_network_structure(
                second,
                initialize_rewiring_population(first),
            )


if __name__ == "__main__":
    unittest.main()
