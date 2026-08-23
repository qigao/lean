import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import (
    ParameterAcceptanceSet,
    calibrate_seed_block_variants,
    repeated_grid_calibration,
)
from narrative_dynamics.validation import (
    HeldOutCase,
    local_sensitivity_report,
    validate_acceptance_set_held_out,
)


class ScaledLevelModel:
    name = "scaled-level"

    def simulate(self, scenario, parameters, rng):
        scale = float(scenario.payload.get("scale", 1.0))
        return ModelRun(
            events=(),
            outcome={"value": float(parameters["level"]) * scale},
        )


class SeedSignModel:
    name = "seed-sign"

    def simulate(self, scenario, parameters, rng):
        noise = 1.0 if rng.random() < 0.5 else -1.0
        return ModelRun(
            events=(),
            outcome={"value": float(parameters["level"]) + noise},
        )


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class UncertaintyExternalValidationTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = ScaledLevelModel()
        self.calibration_scenario = Scenario(id="calibration", payload={"scale": 1.0})
        self.held_out_cases = (
            HeldOutCase(
                scenario=Scenario(id="scale-2", payload={"scale": 2.0}),
                seeds=(101, 102),
                target={"value": 4.0},
            ),
            HeldOutCase(
                scenario=Scenario(id="scale-3", payload={"scale": 3.0}),
                seeds=(201, 202),
                target={"value": 6.0},
            ),
        )

    def test_parameter_acceptance_set_is_first_class_and_canonical(self):
        accepted = ParameterAcceptanceSet.from_parameters(
            (
                (("level", 3),),
                (("level", 1.0),),
                (("level", 3.0),),
                (("level", 2),),
            )
        )

        self.assertEqual(
            accepted.parameters,
            (
                (("level", 1.0),),
                (("level", 2.0),),
                (("level", 3.0),),
            ),
        )
        self.assertEqual(accepted.count, 3)
        self.assertEqual(
            accepted.parameter_maps,
            ({"level": 1.0}, {"level": 2.0}, {"level": 3.0}),
        )

        repeated = repeated_grid_calibration(
            runner=self.runner,
            model=self.model,
            scenario=self.calibration_scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seed_blocks=((1, 2), (3, 4)),
            extractor=value_metrics,
            target={"value": 2.0},
        )
        self.assertEqual(
            repeated.acceptance_set.parameters,
            ((("level", 2.0),),),
        )

    def test_seed_block_variants_shift_whole_blocks_and_report_robust_set(self):
        report = calibrate_seed_block_variants(
            runner=self.runner,
            model=self.model,
            scenario=self.calibration_scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seed_blocks=((10, 11), (20, 21)),
            seed_offsets=(0, 100, 1000),
            extractor=value_metrics,
            target={"value": 2.0},
        )

        self.assertEqual(tuple(v.offset for v in report.variants), (0, 100, 1000))
        self.assertEqual(report.variants[1].seed_blocks, ((110, 111), (120, 121)))
        self.assertEqual(
            report.accepted_union.parameters,
            ((("level", 2.0),),),
        )
        self.assertEqual(report.accepted_intersection, report.accepted_union)
        self.assertEqual(
            tuple(v.calibration.acceptance_set for v in report.variants),
            (report.accepted_union,) * 3,
        )

    def test_seed_block_variants_expose_seed_dependent_acceptance_drift(self):
        report = calibrate_seed_block_variants(
            runner=self.runner,
            model=SeedSignModel(),
            scenario=self.calibration_scenario,
            parameter_grid={"level": (-1.0, 0.0, 1.0)},
            seed_blocks=((1,),),
            seed_offsets=(0, 1),
            extractor=value_metrics,
            target={"value": 0.0},
        )

        self.assertEqual(
            report.variants[0].calibration.acceptance_set.parameters,
            ((("level", -1.0),),),
        )
        self.assertEqual(
            report.variants[1].calibration.acceptance_set.parameters,
            ((("level", 1.0),),),
        )
        self.assertEqual(
            report.accepted_union.parameters,
            ((("level", -1.0),), (("level", 1.0),)),
        )
        self.assertEqual(report.accepted_intersection.parameters, ())

    def test_acceptance_set_is_filtered_on_held_out_scenarios(self):
        accepted = ParameterAcceptanceSet.from_parameters(
            (
                (("level", 1.0),),
                (("level", 2.0),),
                (("level", 3.0),),
            )
        )

        report = validate_acceptance_set_held_out(
            runner=self.runner,
            model=self.model,
            accepted_parameters=accepted,
            cases=self.held_out_cases,
            extractor=value_metrics,
            max_mean_loss=0.0,
            max_worst_loss=0.0,
        )

        self.assertEqual(report.best.parameters, (("level", 2.0),))
        self.assertEqual(report.best.validation.mean_loss, 0.0)
        self.assertEqual(
            report.retained_parameters.parameters,
            ((("level", 2.0),),),
        )
        by_level = {
            evaluation.parameter_map["level"]: evaluation.validation.mean_loss
            for evaluation in report.evaluations
        }
        self.assertEqual(by_level, {1.0: 6.5, 2.0: 0.0, 3.0: 6.5})

    def test_local_sensitivity_reports_slope_curvature_and_perturbed_losses(self):
        report = local_sensitivity_report(
            runner=self.runner,
            model=self.model,
            parameters={"level": 2.0},
            cases=self.held_out_cases,
            extractor=value_metrics,
            step_sizes={"level": 0.5},
        )

        self.assertEqual(report.baseline.mean_loss, 0.0)
        self.assertEqual(len(report.parameters), 1)
        level = report.parameters[0]
        self.assertEqual(level.name, "level")
        self.assertEqual(level.lower.parameters, (("level", 1.5),))
        self.assertEqual(level.upper.parameters, (("level", 2.5),))
        self.assertAlmostEqual(level.lower.mean_loss, 1.625)
        self.assertAlmostEqual(level.upper.mean_loss, 1.625)
        self.assertAlmostEqual(level.mean_loss_slope, 0.0)
        self.assertAlmostEqual(level.mean_loss_curvature, 13.0)
        self.assertAlmostEqual(level.worst_loss_curvature, 18.0)
        self.assertAlmostEqual(level.max_absolute_mean_loss_change, 1.625)

    def test_new_uncertainty_inputs_reject_ambiguous_configuration(self):
        with self.assertRaises(ValueError):
            ParameterAcceptanceSet.from_parameters(
                (
                    (("level", 1.0),),
                    (("other", 2.0),),
                )
            )
        with self.assertRaises(ValueError):
            calibrate_seed_block_variants(
                runner=self.runner,
                model=self.model,
                scenario=self.calibration_scenario,
                parameter_grid={"level": (1.0, 2.0)},
                seed_blocks=((1,),),
                seed_offsets=(),
                extractor=value_metrics,
                target={"value": 2.0},
            )
        with self.assertRaises(ValueError):
            local_sensitivity_report(
                runner=self.runner,
                model=self.model,
                parameters={"level": 2.0},
                cases=self.held_out_cases,
                extractor=value_metrics,
                step_sizes={"level": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
