import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.diagnostics import (
    HeldOutCase,
    central_difference_sensitivity,
    validate_held_out,
)
from narrative_dynamics.simulation import SimulationRunner


class AffineModel:
    name = "affine-diagnostics"

    def simulate(self, scenario, parameters, rng):
        scale = float(scenario.payload.get("scale", 1.0))
        a = float(parameters["a"])
        b = float(parameters["b"])
        return ModelRun(
            events=(),
            outcome={
                "y": a * scale + b,
                "z": 2.0 * a - b,
            },
        )


def affine_metrics(trace):
    return {
        "y": float(trace.outcome["y"]),
        "z": float(trace.outcome["z"]),
    }


class HeldOutValidationTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = AffineModel()
        self.parameters = {"a": 2.0, "b": 1.0}

    def test_parameters_are_evaluated_on_independent_named_scenarios(self):
        report = validate_held_out(
            runner=self.runner,
            model=self.model,
            parameters=self.parameters,
            cases=(
                HeldOutCase(
                    name="double",
                    scenario=Scenario(id="double", payload={"scale": 2.0}),
                    seeds=(1, 2),
                    target={"y": 5.0, "z": 3.0},
                ),
                HeldOutCase(
                    name="negative",
                    scenario=Scenario(id="negative", payload={"scale": -1.0}),
                    seeds=(3, 4),
                    target={"y": -1.0, "z": 3.0},
                ),
            ),
            extractor=affine_metrics,
        )

        self.assertEqual(report.parameters, (("a", 2.0), ("b", 1.0)))
        self.assertEqual(tuple(case.name for case in report.cases), ("double", "negative"))
        self.assertEqual(report.mean_loss, 0.0)
        self.assertEqual(report.max_loss, 0.0)
        self.assertEqual(report.cases[0].metric_map, {"y": 5.0, "z": 3.0})

    def test_wrong_parameters_fail_held_out_scenarios(self):
        report = validate_held_out(
            runner=self.runner,
            model=self.model,
            parameters={"a": 1.0, "b": 1.0},
            cases=(
                HeldOutCase(
                    name="double",
                    scenario=Scenario(id="double", payload={"scale": 2.0}),
                    seeds=(1,),
                    target={"y": 5.0, "z": 3.0},
                ),
            ),
            extractor=affine_metrics,
        )

        self.assertGreater(report.mean_loss, 0.0)
        self.assertEqual(report.mean_loss, report.max_loss)

    def test_held_out_validation_rejects_empty_or_ambiguous_cases(self):
        with self.assertRaises(ValueError):
            validate_held_out(
                runner=self.runner,
                model=self.model,
                parameters=self.parameters,
                cases=(),
                extractor=affine_metrics,
            )
        with self.assertRaises(ValueError):
            validate_held_out(
                runner=self.runner,
                model=self.model,
                parameters=self.parameters,
                cases=(
                    HeldOutCase(
                        name="same",
                        scenario=Scenario(id="one", payload={}),
                        seeds=(1,),
                        target={"y": 3.0, "z": 3.0},
                    ),
                    HeldOutCase(
                        name="same",
                        scenario=Scenario(id="two", payload={}),
                        seeds=(2,),
                        target={"y": 3.0, "z": 3.0},
                    ),
                ),
                extractor=affine_metrics,
            )
        with self.assertRaises(ValueError):
            validate_held_out(
                runner=self.runner,
                model=self.model,
                parameters=self.parameters,
                cases=(
                    HeldOutCase(
                        name="empty-seeds",
                        scenario=Scenario(id="empty-seeds", payload={}),
                        seeds=(),
                        target={"y": 3.0, "z": 3.0},
                    ),
                ),
                extractor=affine_metrics,
            )


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = AffineModel()
        self.scenario = Scenario(id="sensitivity", payload={"scale": 3.0})

    def test_central_difference_recovers_local_metric_derivatives(self):
        report = central_difference_sensitivity(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            baseline_parameters={"a": 2.0, "b": 1.0, "ignored": 7.0},
            steps={"a": 0.5, "b": 0.25, "ignored": 1.0},
            seeds=(1, 2),
            extractor=affine_metrics,
        )

        self.assertEqual(report.baseline_metrics, (("y", 7.0), ("z", 3.0)))
        entries = {entry.parameter: entry for entry in report.entries}
        self.assertAlmostEqual(entries["a"].derivative_map["y"], 3.0)
        self.assertAlmostEqual(entries["a"].derivative_map["z"], 2.0)
        self.assertAlmostEqual(entries["b"].derivative_map["y"], 1.0)
        self.assertAlmostEqual(entries["b"].derivative_map["z"], -1.0)
        self.assertEqual(entries["ignored"].derivative_map, {"y": 0.0, "z": 0.0})

    def test_sensitivity_validates_seed_step_and_parameter_schemas(self):
        common = dict(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            baseline_parameters={"a": 2.0, "b": 1.0},
            extractor=affine_metrics,
        )
        with self.assertRaises(ValueError):
            central_difference_sensitivity(**common, steps={"a": 0.1}, seeds=())
        with self.assertRaises(ValueError):
            central_difference_sensitivity(**common, steps={}, seeds=(1,))
        with self.assertRaises(ValueError):
            central_difference_sensitivity(
                **common,
                steps={"missing": 0.1},
                seeds=(1,),
            )
        with self.assertRaises(ValueError):
            central_difference_sensitivity(
                **common,
                steps={"a": 0.0},
                seeds=(1,),
            )


if __name__ == "__main__":
    unittest.main()
