import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.validation import synthetic_recovery


class IdentifiableLevelModel:
    name = "identifiable-level"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(events=(), outcome={"value": parameters["level"]})


class ConstantOutcomeModel:
    name = "constant-outcome"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(events=(), outcome={"value": 1.0})


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class SyntheticRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.scenario = Scenario(id="recovery", payload={})

    def test_identifiable_true_parameter_is_recovered(self):
        report = synthetic_recovery(
            runner=self.runner,
            model=IdentifiableLevelModel(),
            scenario=self.scenario,
            true_parameters={"level": 2.0},
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            observation_seeds=(1, 2),
            calibration_seeds=(3, 4),
            extractor=value_metrics,
        )

        self.assertTrue(report.recovered)
        self.assertEqual(report.true_parameters, (("level", 2.0),))
        self.assertEqual(report.target_metrics, (("value", 2.0),))
        self.assertEqual(report.calibration.best.parameters, (("level", 2.0),))

    def test_non_identifiable_metric_is_reported_as_non_recovery(self):
        report = synthetic_recovery(
            runner=self.runner,
            model=ConstantOutcomeModel(),
            scenario=self.scenario,
            true_parameters={"level": 2.0},
            parameter_grid={"level": (2.0, 1.0)},
            observation_seeds=(1,),
            calibration_seeds=(2,),
            extractor=value_metrics,
        )

        self.assertFalse(report.recovered)
        self.assertEqual(report.calibration.best.parameters, (("level", 1.0),))
        self.assertEqual(
            tuple(candidate.loss for candidate in report.calibration.ranking),
            (0.0, 0.0),
        )

    def test_observation_and_calibration_seed_sets_must_be_nonempty(self):
        common = dict(
            runner=self.runner,
            model=IdentifiableLevelModel(),
            scenario=self.scenario,
            true_parameters={"level": 2.0},
            parameter_grid={"level": (1.0, 2.0)},
            extractor=value_metrics,
        )
        with self.assertRaises(ValueError):
            synthetic_recovery(
                **common,
                observation_seeds=(),
                calibration_seeds=(1,),
            )
        with self.assertRaises(ValueError):
            synthetic_recovery(
                **common,
                observation_seeds=(1,),
                calibration_seeds=(),
            )


if __name__ == "__main__":
    unittest.main()
