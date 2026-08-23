from __future__ import annotations

import math
import unittest

from narrative_dynamics.attestation import RepositoryIdentity, measure_implementation
from narrative_dynamics.contracts import Scenario, SimulationTrace
from narrative_dynamics.execution_policy import TrustedExecutionPolicy, TrustedModelPin
from narrative_dynamics.process_execution import ProcessLimits
from narrative_dynamics.simulation import SimulationRunner


class PrisonInitialActionMetricTests(unittest.TestCase):
    def _trace(self, policy):
        return SimulationTrace(
            model_name="fixture-model",
            scenario_id="fixture-scenario",
            parameters=(("beta", 1.0),),
            seed=1,
            events=(),
            outcome={"initial_policy": policy},
        )

    def test_shared_extractor_returns_exact_three_initial_action_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import (
            prison_initial_action_metrics,
        )

        metrics = prison_initial_action_metrics(
            self._trace({"scout": 0.25, "escape": 0.5, "submit": 0.25})
        )
        self.assertEqual(
            metrics,
            {
                "initial.scout": 0.25,
                "initial.escape": 0.5,
                "initial.submit": 0.25,
            },
        )
        self.assertEqual(prison_initial_action_metrics.version, "1.0.0")

    def test_shared_extractor_rejects_missing_or_non_finite_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import (
            prison_initial_action_metrics,
        )

        with self.assertRaises(ValueError):
            prison_initial_action_metrics(
                self._trace({"scout": 0.5, "escape": 0.5})
            )
        with self.assertRaises(ValueError):
            prison_initial_action_metrics(
                self._trace({"scout": math.nan, "escape": 0.5, "submit": 0.5})
            )


def prison_scenario(
    scenario_id: str,
    *,
    prior_weak: float = 0.5,
    signal_accuracy: float = 0.75,
    guard_persistence: float = 0.9,
    horizon: int = 2,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        payload={
            "prior_weak": prior_weak,
            "signal_accuracy": signal_accuracy,
            "guard_persistence": guard_persistence,
            "escape_reward": 8.0,
            "capture_cost": 10.0,
            "submit_reward": 1.0,
            "scout_cost": 0.25,
            "discount": 0.95,
            "horizon": horizon,
        },
    )


class PrisonReactiveModelTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        )

    def test_reactive_values_follow_declared_cue_rule_and_replay_seed(self):
        from narrative_dynamics.adapters.prison_reactive import (
            create_prison_reactive_model,
        )

        model = create_prison_reactive_model()
        scenario = prison_scenario("reactive-equations")
        first = self.runner.run_once(model, scenario, {"beta": 2.0}, seed=17)
        replay = self.runner.run_once(model, scenario, {"beta": 2.0}, seed=17)

        self.assertEqual(first, replay)
        self.assertAlmostEqual(first.outcome["initial_values"]["escape"], -1.0)
        self.assertAlmostEqual(
            first.outcome["cue_values"]["clear"]["escape"], 3.5
        )
        self.assertAlmostEqual(
            first.outcome["cue_values"]["alarm"]["escape"], -5.5
        )
        self.assertAlmostEqual(sum(first.outcome["initial_policy"].values()), 1.0)
        self.assertNotIn("posterior_weak", first.outcome)

    def test_guard_persistence_does_not_change_reactive_policy(self):
        from narrative_dynamics.adapters.prison_reactive import (
            create_prison_reactive_model,
        )

        model = create_prison_reactive_model()
        low = self.runner.run_once(
            model,
            prison_scenario("low-persistence", guard_persistence=0.1),
            {"beta": 2.0},
            seed=11,
        )
        high = self.runner.run_once(
            model,
            prison_scenario("high-persistence", guard_persistence=0.9),
            {"beta": 2.0},
            seed=11,
        )
        self.assertEqual(
            low.outcome["initial_policy"], high.outcome["initial_policy"]
        )
        self.assertEqual(
            low.outcome["initial_values"], high.outcome["initial_values"]
        )

    def test_reactive_module_does_not_import_prison_pomdp(self):
        import inspect
        import narrative_dynamics.adapters.prison_reactive as reactive

        source = inspect.getsource(reactive)
        self.assertNotIn("adapters.prison_pomdp", source)
        self.assertNotIn("_posterior_weak", source)
        self.assertNotIn("_future_weak_probability", source)

    def test_reactive_production_source_is_pinned_isolated_and_schema_attested(self):
        from narrative_dynamics.adapters.prison_reactive import (
            prison_reactive_contract,
            prison_reactive_source,
        )

        source = prison_reactive_source(
            limits=ProcessLimits(
                timeout_seconds=3.0,
                max_output_bytes=64 * 1024,
                max_trace_bytes=128 * 1024,
            )
        )
        contract = prison_reactive_contract()
        policy = TrustedExecutionPolicy(
            name="reactive-test-policy",
            version="1",
            pins=(
                TrustedModelPin(
                    model_name=source.name,
                    declared_contract_hash=contract.content_hash,
                    expected_implementation_hash=(
                        measure_implementation(source).content_hash
                    ),
                ),
            ),
        )
        bound = policy.bind(source, contract=contract)
        trace = self.runner.run_once(
            bound,
            prison_scenario("reactive-production"),
            {"beta": 2.0},
            seed=23,
        )

        self.assertIsNotNone(trace.execution)
        self.assertTrue(trace.execution.isolated)
        for boundary in ("parameters", "scenario", "events", "outcome"):
            self.assertEqual(
                trace.manifest.inputs["schema_validation"][boundary]["status"],
                "validated",
            )
        self.assertEqual(
            trace.manifest.inputs["model"]["implementation_attestation"][
                "verification"
            ],
            "matched",
        )
        self.assertIn("result_artifact", trace.manifest.inputs)


if __name__ == "__main__":
    unittest.main()
