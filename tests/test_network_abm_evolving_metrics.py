import unittest

from narrative_dynamics.abm import EdgeSelector
from narrative_dynamics.abm.adaptive_contracts import TruthFeedback
from narrative_dynamics.abm.evolving import simulate_evolving_round
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel,
    initialize_evolving_population,
)
from narrative_dynamics.abm.evolving_metrics import (
    EvolvingSystemMetrics,
    measure_evolving_system,
)
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    PopulationLifecycleEvent,
)
from tests.evolving_fixtures import evolving_base_model


def model_for(model_id="unified-evolution"):
    return EvolvingNetworkModel(
        model_id,
        "1",
        evolving_base_model(),
        ("a", "c"),
        0.5,
        0.5,
        0.2,
        0.8,
    )


def unified_state():
    model = model_for()
    initial = initialize_evolving_population(
        model,
        beliefs={"a": 1.0, "c": 0.5},
    )
    state = simulate_evolving_round(
        model,
        initial,
        events=(
            PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),
        ),
        feedback=(
            TruthFeedback(EdgeSelector("a", "b", "peer"), 1.0),
        ),
    ).next_state
    return model, state


class EvolvingSystemMetricTests(unittest.TestCase):
    def test_metrics_cover_membership_cognition_trust_and_structure(self):
        model, state = unified_state()

        metrics = measure_evolving_system(model, state)

        self.assertEqual(metrics.catalog_population, 3)
        self.assertEqual(metrics.active_population, 3)
        self.assertEqual(metrics.inactive_population, 0)
        self.assertEqual(metrics.dead_population, 0)
        self.assertEqual(metrics.active_share, 1.0)
        self.assertEqual(metrics.dead_share, 0.0)
        self.assertEqual(metrics.mean_active_belief, 2 / 3)
        self.assertEqual(metrics.active_adoption_rate, 1.0)
        self.assertEqual(metrics.mean_trust, 0.625)
        self.assertEqual(metrics.learned_edge_rate, 0.5)
        self.assertEqual(metrics.catalog_edge_count, 2)
        self.assertEqual(metrics.effective_active_edge_count, 2)
        self.assertEqual(metrics.effective_active_edge_rate, 1.0)
        self.assertEqual(metrics.cumulative_entries, 3)
        self.assertEqual(metrics.cumulative_exits, 0)
        self.assertEqual(metrics.cumulative_rewirings, 1)

    def test_zero_active_population_uses_none_for_active_belief_metrics(self):
        model = model_for()
        initial = initialize_evolving_population(model)
        state = simulate_evolving_round(
            model,
            initial,
            events=(
                PopulationLifecycleEvent("a", LifecycleEventKind.EXIT),
                PopulationLifecycleEvent("c", LifecycleEventKind.DEATH),
            ),
        ).next_state

        metrics = measure_evolving_system(model, state)

        self.assertEqual(metrics.active_population, 0)
        self.assertIsNone(metrics.mean_active_belief)
        self.assertIsNone(metrics.active_adoption_rate)
        self.assertEqual(metrics.effective_active_edge_count, 0)
        self.assertEqual(metrics.effective_active_edge_rate, 0.0)

    def test_metric_contract_rejects_inconsistent_counts_rates_and_nullability(self):
        values = dict(
            catalog_population=3,
            active_population=3,
            inactive_population=0,
            dead_population=0,
            active_share=1.0,
            dead_share=0.0,
            mean_active_belief=0.5,
            active_adoption_rate=1.0,
            mean_trust=0.5,
            learned_edge_rate=0.0,
            catalog_edge_count=2,
            effective_active_edge_count=1,
            effective_active_edge_rate=0.5,
            cumulative_entries=3,
            cumulative_exits=0,
            cumulative_rewirings=0,
        )
        with self.assertRaisesRegex(ValueError, "sum to catalog"):
            EvolvingSystemMetrics(**{**values, "inactive_population": 1})
        with self.assertRaisesRegex(ValueError, "effective edge rate"):
            EvolvingSystemMetrics(**{**values, "effective_active_edge_rate": 1.0})
        with self.assertRaisesRegex(ValueError, "must be None"):
            EvolvingSystemMetrics(
                **{
                    **values,
                    "active_population": 0,
                    "inactive_population": 3,
                    "active_share": 0.0,
                }
            )

    def test_measurement_requires_exact_model_state_binding(self):
        model, state = unified_state()
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            measure_evolving_system(model_for("different"), state)


if __name__ == "__main__":
    unittest.main()
