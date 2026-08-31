from dataclasses import replace
import math
import unittest

from narrative_dynamics.abm.autonomy_contracts import TruthObservation
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    PopulationLifecycleEvent,
)
from narrative_dynamics.abm.role_contracts import (
    DynamicRoleModel,
    initialize_dynamic_role_population,
)
from narrative_dynamics.abm.role_metrics import (
    RoleDynamicsMetrics,
    RolePopulationCount,
    measure_role_dynamics,
)
from narrative_dynamics.abm.roles import simulate_dynamic_role_round
from tests.dynamic_role_fixtures import dynamic_role_model


def role_count(metrics, role: str):
    return next(item for item in metrics.active_role_counts if item.role == role)


def transitioned_state():
    model = dynamic_role_model()
    initial = initialize_dynamic_role_population(
        model,
        beliefs={"a": 1.0, "c": 0.5},
    )
    state = simulate_dynamic_role_round(
        model,
        initial,
        environment_events=(
            PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),
        ),
        truth_observations=(
            TruthObservation(EdgeSelector("a", "b", "peer"), 1.0),
        ),
    ).next_state
    return model, state


class RoleDynamicsMetricTests(unittest.TestCase):
    def test_metrics_cover_composition_diversity_transitions_and_tenure(self):
        model, state = transitioned_state()

        metrics = measure_role_dynamics(model, state)

        self.assertEqual(metrics.active_population, 3)
        self.assertEqual(role_count(metrics, "source").count, 2)
        self.assertEqual(role_count(metrics, "source").share, 2 / 3)
        self.assertEqual(role_count(metrics, "relay").count, 0)
        self.assertEqual(role_count(metrics, "recipient").count, 1)
        expected_entropy = -(
            (2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)
        ) / math.log(3)
        self.assertAlmostEqual(metrics.role_entropy, expected_entropy)
        self.assertEqual(metrics.decision_count, 3)
        self.assertEqual(metrics.transition_count, 1)
        self.assertEqual(metrics.transition_rate, 1 / 3)
        self.assertEqual(metrics.active_rounds_in_role, 2)
        self.assertEqual(metrics.mean_active_rounds_in_role, 2 / 3)

    def test_initial_metrics_include_zero_count_roles_and_zero_history(self):
        model = dynamic_role_model()
        state = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        metrics = measure_role_dynamics(model, state)

        self.assertEqual(metrics.active_population, 2)
        self.assertEqual(role_count(metrics, "source").share, 0.5)
        self.assertEqual(role_count(metrics, "relay").share, 0.0)
        self.assertEqual(role_count(metrics, "recipient").share, 0.5)
        self.assertAlmostEqual(metrics.role_entropy, math.log(2) / math.log(3))
        self.assertEqual(metrics.transition_rate, 0.0)
        self.assertEqual(metrics.mean_active_rounds_in_role, 0.0)

    def test_empty_population_has_zero_composition_rates_entropy_and_tenure(self):
        model = dynamic_role_model()
        initial = initialize_dynamic_role_population(model)
        state = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(
                PopulationLifecycleEvent("a", LifecycleEventKind.DEATH),
                PopulationLifecycleEvent("c", LifecycleEventKind.DEATH),
            ),
        ).next_state

        metrics = measure_role_dynamics(model, state)

        self.assertEqual(metrics.active_population, 0)
        self.assertTrue(all(item.count == 0 for item in metrics.active_role_counts))
        self.assertTrue(all(item.share == 0.0 for item in metrics.active_role_counts))
        self.assertEqual(metrics.role_entropy, 0.0)
        self.assertEqual(metrics.transition_rate, 0.0)
        self.assertEqual(metrics.mean_active_rounds_in_role, 0.0)

    def test_metric_contract_rejects_inconsistent_aggregates(self):
        model, state = transitioned_state()
        metrics = measure_role_dynamics(model, state)
        with self.assertRaisesRegex(ValueError, "role counts must sum"):
            RoleDynamicsMetrics(
                4,
                metrics.active_role_counts,
                metrics.role_entropy,
                metrics.decision_count,
                metrics.transition_count,
                metrics.transition_rate,
                metrics.active_rounds_in_role,
                metrics.mean_active_rounds_in_role,
            )
        with self.assertRaisesRegex(ValueError, "transition rate"):
            replace(metrics, transition_rate=0.5)
        with self.assertRaisesRegex(ValueError, "role shares must match"):
            replace(
                metrics,
                active_role_counts=(
                    RolePopulationCount("recipient", 1, 0.5),
                    RolePopulationCount("relay", 0, 0.0),
                    RolePopulationCount("source", 2, 0.5),
                ),
            )

    def test_measurement_requires_exact_model_state_binding(self):
        model, state = transitioned_state()
        different = DynamicRoleModel(
            "different",
            model.version,
            model.autonomy_model,
            model.transition_rules,
        )
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            measure_role_dynamics(different, state)


if __name__ == "__main__":
    unittest.main()
