import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.metrics import aggregate_metrics
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import (
    diagnose_identifiability,
    repeated_grid_calibration,
)


class LevelModel:
    name = "uncertainty-level"

    def simulate(self, scenario, parameters, rng):
        noise_scale = float(scenario.payload.get("noise_scale", 0.0))
        value = float(parameters["level"]) + noise_scale * (rng.random() - 0.5)
        return ModelRun(events=(), outcome={"value": value})


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class RepeatedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = LevelModel()
        self.scenario = Scenario(id="uncertainty", payload={"noise_scale": 0.0})

    def test_repeated_blocks_report_stability_and_unique_accepted_parameter(self):
        report = repeated_grid_calibration(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seed_blocks=((1, 2), (3, 4), (5, 6)),
            extractor=value_metrics,
            target={"value": 2.0},
            acceptance_loss_delta=0.0,
            min_acceptance_fraction=1.0,
        )

        self.assertEqual(len(report.blocks), 3)
        self.assertEqual(
            tuple(block.best.parameters for block in report.blocks),
            ((("level", 2.0),),) * 3,
        )
        self.assertEqual(report.accepted_parameters, ((('level', 2.0),),))

        stability = {entry.parameter_map["level"]: entry for entry in report.stability}
        self.assertEqual(stability[2.0].wins, 3)
        self.assertEqual(stability[2.0].accepted_blocks, 3)
        self.assertEqual(stability[2.0].acceptance_fraction, 1.0)
        self.assertEqual(stability[2.0].loss_stddev, 0.0)
        self.assertEqual(stability[1.0].wins, 0)

    def test_loss_delta_produces_an_explicit_accepted_parameter_set(self):
        report = repeated_grid_calibration(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seed_blocks=((1,), (2,)),
            extractor=value_metrics,
            target={"value": 2.0},
            acceptance_loss_delta=1.0,
            min_acceptance_fraction=1.0,
        )

        self.assertEqual(
            report.accepted_parameters,
            (
                (("level", 1.0),),
                (("level", 2.0),),
                (("level", 3.0),),
            ),
        )

    def test_identifiability_is_reported_per_parameter_over_accepted_set(self):
        report = diagnose_identifiability(
            (
                (("beta", 2.0), ("risk", 0.5)),
                (("beta", 4.0), ("risk", 0.5)),
            )
        )

        diagnostics = {entry.name: entry for entry in report.parameters}
        self.assertFalse(report.fully_identified)
        self.assertEqual(diagnostics["beta"].values, (2.0, 4.0))
        self.assertFalse(diagnostics["beta"].identified)
        self.assertEqual(diagnostics["risk"].values, (0.5,))
        self.assertTrue(diagnostics["risk"].identified)

    def test_repeated_calibration_validates_blocks_and_thresholds(self):
        common = dict(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0)},
            extractor=value_metrics,
            target={"value": 1.0},
        )
        with self.assertRaises(ValueError):
            repeated_grid_calibration(**common, seed_blocks=())
        with self.assertRaises(ValueError):
            repeated_grid_calibration(**common, seed_blocks=((1,), ()))
        with self.assertRaises(ValueError):
            repeated_grid_calibration(
                **common,
                seed_blocks=((1,),),
                acceptance_loss_delta=-1.0,
            )
        with self.assertRaises(ValueError):
            repeated_grid_calibration(
                **common,
                seed_blocks=((1,),),
                min_acceptance_fraction=0.0,
            )
        with self.assertRaises(ValueError):
            diagnose_identifiability(())


if __name__ == "__main__":
    unittest.main()
