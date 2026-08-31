import unittest

from narrative_dynamics.abm.lifecycle import simulate_lifecycle_round
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    PopulationLifecycleEvent,
    PopulationLifecycleModel,
    initialize_lifecycle_population,
)
from narrative_dynamics.abm.lifecycle_metrics import (
    PopulationLifecycleMetrics,
    measure_population_lifecycle,
)
from tests.lifecycle_fixtures import lifecycle_base_model


class PopulationLifecycleMetricTests(unittest.TestCase):
    def test_metrics_match_hand_computed_catalog_state(self):
        model = PopulationLifecycleModel(
            "population-lifecycle",
            "1",
            lifecycle_base_model(),
            ("a",),
        )
        initial = initialize_lifecycle_population(model, beliefs={"a": 1.0})
        result = simulate_lifecycle_round(
            model,
            initial,
            events=(
                PopulationLifecycleEvent("a", LifecycleEventKind.EXIT),
                PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),
                PopulationLifecycleEvent("c", LifecycleEventKind.DEATH),
            ),
        )

        metrics = measure_population_lifecycle(model, result.next_state)

        self.assertEqual(metrics.catalog_population, 3)
        self.assertEqual(metrics.active_population, 1)
        self.assertEqual(metrics.inactive_population, 1)
        self.assertEqual(metrics.dead_population, 1)
        self.assertEqual(metrics.active_share, 1 / 3)
        self.assertEqual(metrics.dead_share, 1 / 3)
        self.assertEqual(metrics.cumulative_entries, 2)
        self.assertEqual(metrics.cumulative_exits, 1)
        self.assertEqual(
            metrics.to_dict(),
            {
                "catalog_population": 3,
                "active_population": 1,
                "inactive_population": 1,
                "dead_population": 1,
                "active_share": 1 / 3,
                "dead_share": 1 / 3,
                "cumulative_entries": 2,
                "cumulative_exits": 1,
            },
        )

    def test_metric_contract_rejects_inconsistent_counts(self):
        with self.assertRaisesRegex(ValueError, "sum to catalog"):
            PopulationLifecycleMetrics(3, 1, 1, 0, 1 / 3, 0.0, 1, 0)
        with self.assertRaisesRegex(ValueError, "active share"):
            PopulationLifecycleMetrics(3, 1, 1, 1, 0.5, 1 / 3, 1, 0)
        with self.assertRaisesRegex(ValueError, "dead share"):
            PopulationLifecycleMetrics(3, 1, 1, 1, 1 / 3, 0.5, 1, 0)

    def test_measurement_requires_exact_model_state_binding(self):
        first = PopulationLifecycleModel(
            "first",
            "1",
            lifecycle_base_model(),
            ("a",),
        )
        second = PopulationLifecycleModel(
            "second",
            "1",
            lifecycle_base_model(),
            ("a",),
        )
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            measure_population_lifecycle(
                second,
                initialize_lifecycle_population(first),
            )


if __name__ == "__main__":
    unittest.main()
