from dataclasses import replace
import unittest

from narrative_dynamics.abm import EdgeSelector
from narrative_dynamics.abm.autonomy import simulate_autonomous_round
from narrative_dynamics.abm.autonomy_contracts import (
    AutonomousNetworkModel,
    TruthObservation,
    initialize_autonomous_population,
)
from narrative_dynamics.abm.autonomy_metrics import (
    AgentAutonomyMetrics,
    measure_agent_autonomy,
)
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    PopulationLifecycleEvent,
)
from tests.autonomy_fixtures import autonomous_model, role_policies


def autonomous_state(*, exit_threshold=None):
    base_model = autonomous_model()
    if exit_threshold is None:
        model = base_model
    else:
        policies = tuple(
            replace(item, exit_belief_threshold=exit_threshold)
            if item.role == "relay"
            else item
            for item in role_policies()
        )
        model = AutonomousNetworkModel(
            base_model.model_id,
            base_model.version,
            base_model.evolving_model,
            policies,
        )
    initial = initialize_autonomous_population(
        model,
        beliefs={"a": 1.0, "c": 0.5},
    )
    state = simulate_autonomous_round(
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


class AgentAutonomyMetricTests(unittest.TestCase):
    def test_behavior_metrics_cover_decisions_resources_and_current_sharing(self):
        model, state = autonomous_state()

        metrics = measure_agent_autonomy(model, state)

        self.assertEqual(metrics.decision_count, 3)
        self.assertEqual(metrics.verification_count, 1)
        self.assertEqual(metrics.verification_rate, 1 / 3)
        self.assertEqual(metrics.silent_decision_count, 1)
        self.assertEqual(metrics.silence_rate, 1 / 3)
        self.assertEqual(metrics.autonomous_exit_count, 0)
        self.assertEqual(metrics.exit_rate, 0.0)
        self.assertEqual(metrics.initial_verification_budget, 4)
        self.assertEqual(metrics.remaining_verification_budget, 3)
        self.assertEqual(metrics.budget_utilization, 0.25)
        self.assertEqual(metrics.active_population, 3)
        self.assertEqual(metrics.active_sharing_count, 2)
        self.assertEqual(metrics.active_sharing_rate, 2 / 3)

    def test_autonomous_exit_is_reflected_in_cumulative_behavior(self):
        model, state = autonomous_state(exit_threshold=0.6)

        metrics = measure_agent_autonomy(model, state)

        self.assertEqual(metrics.autonomous_exit_count, 1)
        self.assertEqual(metrics.exit_rate, 1 / 3)
        self.assertEqual(metrics.silent_decision_count, 2)
        self.assertEqual(metrics.active_population, 2)
        self.assertEqual(metrics.active_sharing_count, 1)
        self.assertEqual(metrics.active_sharing_rate, 0.5)

    def test_initial_state_has_zero_cumulative_rates_and_full_budget(self):
        model = autonomous_model()
        state = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        metrics = measure_agent_autonomy(model, state)

        self.assertEqual(metrics.decision_count, 0)
        self.assertEqual(metrics.verification_rate, 0.0)
        self.assertEqual(metrics.silence_rate, 0.0)
        self.assertEqual(metrics.exit_rate, 0.0)
        self.assertEqual(metrics.budget_utilization, 0.0)
        self.assertEqual(metrics.active_sharing_rate, 0.5)

    def test_metric_contract_rejects_inconsistent_counts_and_rates(self):
        values = dict(
            decision_count=3,
            verification_count=1,
            verification_rate=1 / 3,
            silent_decision_count=1,
            silence_rate=1 / 3,
            autonomous_exit_count=0,
            exit_rate=0.0,
            initial_verification_budget=4,
            remaining_verification_budget=3,
            budget_utilization=0.25,
            active_population=3,
            active_sharing_count=2,
            active_sharing_rate=2 / 3,
        )
        with self.assertRaisesRegex(ValueError, "verification count cannot exceed"):
            AgentAutonomyMetrics(**{**values, "verification_count": 4})
        with self.assertRaisesRegex(ValueError, "verification rate"):
            AgentAutonomyMetrics(**{**values, "verification_rate": 0.5})
        with self.assertRaisesRegex(ValueError, "sharing count cannot exceed"):
            AgentAutonomyMetrics(**{**values, "active_sharing_count": 4})

    def test_measurement_requires_exact_model_state_binding(self):
        model, state = autonomous_state()
        different = AutonomousNetworkModel(
            "different",
            model.version,
            model.evolving_model,
            model.role_policies,
        )
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            measure_agent_autonomy(different, state)


if __name__ == "__main__":
    unittest.main()
