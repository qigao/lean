import math
import unittest

from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.metrics import aggregate_metrics, weighted_squared_error
from narrative_dynamics.simulation import SimulationRunner


class LevelModel:
    name = "level"

    def simulate(self, scenario, parameters, rng):
        noise_scale = float(scenario.payload.get("noise_scale", 0.0))
        value = parameters["level"] + noise_scale * (rng.random() - 0.5)
        return ModelRun(events=(), outcome={"value": value})


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = LevelModel()
        self.scenario = Scenario(id="level", payload={"noise_scale": 0.0})

    def test_metrics_are_averaged_over_nonempty_trace_batch(self):
        traces = self.runner.run_batch(
            self.model,
            self.scenario,
            {"level": 2.5},
            seeds=(1, 2, 3),
        )

        self.assertEqual(aggregate_metrics(traces, value_metrics), {"value": 2.5})

    def test_metric_key_drift_is_rejected(self):
        traces = self.runner.run_batch(
            self.model,
            self.scenario,
            {"level": 1.0},
            seeds=(1, 2),
        )

        def drifting_metrics(trace):
            return {"odd": 1.0} if trace.seed == 1 else {"even": 1.0}

        with self.assertRaises(ValueError):
            aggregate_metrics(traces, drifting_metrics)

    def test_non_finite_metric_is_rejected(self):
        trace = self.runner.run_once(
            self.model,
            self.scenario,
            {"level": 1.0},
            seed=1,
        )

        with self.assertRaises(ValueError):
            aggregate_metrics((trace,), lambda _: {"value": math.nan})

    def test_weighted_squared_error_validates_metric_schema_and_weights(self):
        self.assertEqual(
            weighted_squared_error(
                {"value": 2.0},
                {"value": 1.0},
                weights={"value": 2.0},
            ),
            2.0,
        )
        with self.assertRaises(ValueError):
            weighted_squared_error(
                {"value": 2.0},
                {"other": 1.0},
            )
        with self.assertRaises(ValueError):
            weighted_squared_error(
                {"value": 2.0},
                {"value": 1.0},
                weights={"value": -1.0},
            )

    def test_grid_calibration_finds_best_candidate_and_breaks_ties_lexically(self):
        result = calibrate_grid(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (3.0, 1.0, 2.0)},
            seeds=(0, 1, 2),
            extractor=value_metrics,
            target={"value": 2.0},
        )

        self.assertEqual(result.best.parameters, (("level", 2.0),))
        self.assertEqual(result.best.metrics, (("value", 2.0),))
        self.assertEqual(result.best.loss, 0.0)
        self.assertEqual(
            tuple(candidate.parameters for candidate in result.ranking),
            (
                (("level", 2.0),),
                (("level", 1.0),),
                (("level", 3.0),),
            ),
        )

    def test_empty_grid_dimension_and_seed_set_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_grid(
                runner=self.runner,
                model=self.model,
                scenario=self.scenario,
                parameter_grid={"level": ()},
                seeds=(1,),
                extractor=value_metrics,
                target={"value": 1.0},
            )
        with self.assertRaises(ValueError):
            calibrate_grid(
                runner=self.runner,
                model=self.model,
                scenario=self.scenario,
                parameter_grid={"level": (1.0,)},
                seeds=(),
                extractor=value_metrics,
                target={"value": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
