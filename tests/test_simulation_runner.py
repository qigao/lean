import math
import unittest

from narrative_dynamics.contracts import ModelRun, Scenario, TraceEvent
from narrative_dynamics.simulation import SimulationRunner


class CoinModel:
    name = "coin"

    def simulate(self, scenario, parameters, rng):
        draw = rng.random() + parameters.get("offset", 0.0)
        return ModelRun(
            events=(
                TraceEvent(
                    tick=0,
                    kind="draw",
                    data={"value": draw},
                ),
            ),
            outcome={"value": draw},
        )


class SimulationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = CoinModel()
        self.scenario = Scenario(id="coin-scenario", payload={"label": "demo"})

    def test_same_inputs_and_seed_replay_exactly(self):
        first = self.runner.run_once(
            self.model,
            self.scenario,
            {"offset": 0.25},
            seed=17,
        )
        second = self.runner.run_once(
            self.model,
            self.scenario,
            {"offset": 0.25},
            seed=17,
        )

        self.assertEqual(first, second)

    def test_different_seeds_change_stochastic_outcome(self):
        first = self.runner.run_once(
            self.model,
            self.scenario,
            {"offset": 0.0},
            seed=1,
        )
        second = self.runner.run_once(
            self.model,
            self.scenario,
            {"offset": 0.0},
            seed=2,
        )

        self.assertNotEqual(first.outcome["value"], second.outcome["value"])

    def test_trace_metadata_is_canonical(self):
        trace = self.runner.run_once(
            self.model,
            self.scenario,
            {"zeta": 2, "alpha": 1},
            seed=9,
        )

        self.assertEqual(trace.model_name, "coin")
        self.assertEqual(trace.scenario_id, "coin-scenario")
        self.assertEqual(trace.seed, 9)
        self.assertEqual(trace.parameters, (("alpha", 1.0), ("zeta", 2.0)))
        self.assertEqual(trace.events[0].kind, "draw")

    def test_batch_preserves_input_seed_order(self):
        traces = self.runner.run_batch(
            self.model,
            self.scenario,
            {"offset": 0.0},
            seeds=(3, 1, 2),
        )

        self.assertEqual(tuple(trace.seed for trace in traces), (3, 1, 2))

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.run_batch(
                self.model,
                self.scenario,
                {"offset": 0.0},
                seeds=(),
            )

    def test_non_finite_parameters_are_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.runner.run_once(
                        self.model,
                        self.scenario,
                        {"offset": invalid},
                        seed=1,
                    )


if __name__ == "__main__":
    unittest.main()
