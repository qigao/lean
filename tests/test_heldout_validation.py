import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.metrics import aggregate_metrics
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.validation import HeldOutCase, validate_held_out


class ScaledLevelModel:
    name = "scaled-level"

    def simulate(self, scenario, parameters, rng):
        scale = float(scenario.payload["scale"])
        value = float(parameters["level"]) * scale
        return ModelRun(events=(), outcome={"value": value})


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class HeldOutValidationTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = ScaledLevelModel()
        self.cases = (
            HeldOutCase(
                scenario=Scenario(id="scale-2", payload={"scale": 2.0}),
                seeds=(11, 12),
                target={"value": 4.0},
            ),
            HeldOutCase(
                scenario=Scenario(id="scale-3", payload={"scale": 3.0}),
                seeds=(21, 22),
                target={"value": 6.0},
            ),
        )

    def test_correct_parameters_generalize_across_held_out_scenarios(self):
        report = validate_held_out(
            runner=self.runner,
            model=self.model,
            parameters={"level": 2.0},
            cases=self.cases,
            extractor=value_metrics,
        )

        self.assertEqual(report.parameters, (("level", 2.0),))
        self.assertEqual(
            tuple(case.scenario_id for case in report.cases),
            ("scale-2", "scale-3"),
        )
        self.assertEqual(tuple(case.loss for case in report.cases), (0.0, 0.0))
        self.assertEqual(report.mean_loss, 0.0)
        self.assertEqual(report.worst_loss, 0.0)

    def test_wrong_parameters_expose_out_of_sample_error(self):
        report = validate_held_out(
            runner=self.runner,
            model=self.model,
            parameters={"level": 1.0},
            cases=self.cases,
            extractor=value_metrics,
        )

        self.assertEqual(tuple(case.loss for case in report.cases), (4.0, 9.0))
        self.assertEqual(report.mean_loss, 6.5)
        self.assertEqual(report.worst_loss, 9.0)

    def test_held_out_validation_rejects_ambiguous_or_empty_cases(self):
        with self.assertRaises(ValueError):
            validate_held_out(
                runner=self.runner,
                model=self.model,
                parameters={"level": 2.0},
                cases=(),
                extractor=value_metrics,
            )

        duplicate = HeldOutCase(
            scenario=Scenario(id="scale-2", payload={"scale": 4.0}),
            seeds=(31,),
            target={"value": 8.0},
        )
        with self.assertRaises(ValueError):
            validate_held_out(
                runner=self.runner,
                model=self.model,
                parameters={"level": 2.0},
                cases=(self.cases[0], duplicate),
                extractor=value_metrics,
            )

        empty_seeds = HeldOutCase(
            scenario=Scenario(id="empty-seeds", payload={"scale": 1.0}),
            seeds=(),
            target={"value": 2.0},
        )
        with self.assertRaises(ValueError):
            validate_held_out(
                runner=self.runner,
                model=self.model,
                parameters={"level": 2.0},
                cases=(empty_seeds,),
                extractor=value_metrics,
            )


if __name__ == "__main__":
    unittest.main()
