import importlib
import unittest

from narrative_dynamics.contracts import ModelRun, Scenario, TraceEvent
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import repeated_grid_calibration
from narrative_dynamics.validation import HeldOutCase


class ExternalReferenceModel:
    name = "external-reference"

    def __init__(self, event_data, outcome):
        self.event_data = event_data
        self.outcome = outcome

    def simulate(self, scenario, parameters, rng):
        return ModelRun(
            events=(TraceEvent(tick=0, kind="snapshot", data=self.event_data),),
            outcome=self.outcome,
        )


class LevelModel:
    name = "level"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(events=(), outcome={"value": float(parameters["level"])})


class StatefulLevelModel:
    name = "stateful-level"

    def __init__(self):
        self.call_count = 0

    def simulate(self, scenario, parameters, rng):
        value = float(parameters["level"]) + float(self.call_count)
        self.call_count += 1
        return ModelRun(events=(), outcome={"value": value})


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class CanonicalTraceBoundaryTests(unittest.TestCase):
    def test_scenario_events_and_outcomes_are_recursively_frozen_and_detached(self):
        payload = {"nested": {"labels": ["a"]}}
        event_data = {"nested": {"count": 1}}
        outcome = {"nested": {"value": 2}}
        scenario = Scenario(id="immutable", payload=payload)

        trace = SimulationRunner().run_once(
            ExternalReferenceModel(event_data, outcome),
            scenario,
            {},
            seed=7,
        )

        payload["nested"]["labels"].append("mutated")
        event_data["nested"]["count"] = 99
        outcome["nested"]["value"] = 100

        self.assertEqual(scenario.payload["nested"]["labels"], ("a",))
        self.assertEqual(trace.events[0].data["nested"]["count"], 1)
        self.assertEqual(trace.outcome["nested"]["value"], 2)
        with self.assertRaises(TypeError):
            scenario.payload["new"] = "forbidden"
        with self.assertRaises(TypeError):
            trace.events[0].data["nested"]["count"] = 3
        with self.assertRaises(TypeError):
            trace.outcome["nested"]["value"] = 4


class FreshModelLifecycleTests(unittest.TestCase):
    def test_factory_model_is_fresh_for_each_calibration_candidate(self):
        simulation = importlib.import_module("narrative_dynamics.simulation")
        model_factory_type = getattr(simulation, "ModelFactory", None)
        self.assertIsNotNone(
            model_factory_type,
            "simulation must expose a factory-backed model source",
        )

        instances = []

        def create_model():
            model = StatefulLevelModel()
            instances.append(model)
            return model

        model = model_factory_type(name="stateful-level", create=create_model)
        calibration = importlib.import_module("narrative_dynamics.calibration")
        report = calibration.calibrate_grid(
            runner=SimulationRunner(),
            model=model,
            scenario=Scenario(id="stateful", payload={}),
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seeds=(11, 12),
            extractor=value_metrics,
            target={"value": 2.5},
        )

        self.assertEqual(report.best.parameters, (("level", 2.0),))
        self.assertEqual(len(instances), 3)
        self.assertTrue(all(instance.call_count == 2 for instance in instances))


class ReplayableGridTests(unittest.TestCase):
    def test_repeated_calibration_materializes_generator_dimensions_once(self):
        try:
            report = repeated_grid_calibration(
                runner=SimulationRunner(),
                model=LevelModel(),
                scenario=Scenario(id="generator-grid", payload={}),
                parameter_grid={"level": iter((1.0, 2.0, 3.0))},
                seed_blocks=((1,), (2,)),
                extractor=value_metrics,
                target={"value": 2.0},
            )
        except Exception as error:  # pragma: no cover - RED documents current bug
            self.fail(f"generator-backed parameter grids must be replayable: {error}")

        self.assertEqual(tuple(len(block.ranking) for block in report.blocks), (3, 3))
        self.assertEqual(
            tuple(block.best.parameters for block in report.blocks),
            ((("level", 2.0),), (("level", 2.0),)),
        )


class EvaluationDataRoleTests(unittest.TestCase):
    def test_selection_validation_and_final_test_have_distinct_apis(self):
        validation = importlib.import_module("narrative_dynamics.validation")
        role_type = getattr(validation, "EvaluationRole", None)
        suite_type = getattr(validation, "HeldOutSuite", None)
        select = getattr(validation, "select_on_validation_suite", None)
        evaluate_final = getattr(validation, "evaluate_on_final_test_suite", None)
        self.assertIsNotNone(role_type, "validation must expose explicit dataset roles")
        self.assertIsNotNone(suite_type, "validation must expose role-tagged suites")
        self.assertIsNotNone(select, "candidate selection must use a validation-only API")
        self.assertIsNotNone(
            evaluate_final,
            "final-test evaluation must accept one fixed parameter mapping",
        )

        cases = (
            HeldOutCase(
                scenario=Scenario(id="scale-2", payload={"scale": 2.0}),
                seeds=(1,),
                target={"value": 4.0},
            ),
        )

        class ScaledLevelModel:
            name = "scaled-level"

            def simulate(self, scenario, parameters, rng):
                return ModelRun(
                    events=(),
                    outcome={
                        "value": float(parameters["level"])
                        * float(scenario.payload["scale"])
                    },
                )

        selection_suite = suite_type(
            name="candidate-selection",
            role=role_type.SELECTION_VALIDATION,
            cases=cases,
        )
        selection = select(
            runner=SimulationRunner(),
            model=ScaledLevelModel(),
            accepted_parameters=(
                (("level", 1.0),),
                (("level", 2.0),),
                (("level", 3.0),),
            ),
            suite=selection_suite,
            extractor=value_metrics,
        )
        self.assertEqual(selection.selected_parameters, (("level", 2.0),))

        final_suite = suite_type(
            name="untouched-final",
            role=role_type.FINAL_TEST,
            cases=cases,
        )
        final = evaluate_final(
            runner=SimulationRunner(),
            model=ScaledLevelModel(),
            parameters=dict(selection.selected_parameters),
            suite=final_suite,
            extractor=value_metrics,
        )
        self.assertEqual(final.role, role_type.FINAL_TEST)
        self.assertEqual(final.validation.mean_loss, 0.0)
        self.assertFalse(hasattr(final, "best"))
        self.assertFalse(hasattr(final, "retained_parameters"))

        with self.assertRaises(ValueError):
            select(
                runner=SimulationRunner(),
                model=ScaledLevelModel(),
                accepted_parameters=((("level", 2.0),),),
                suite=final_suite,
                extractor=value_metrics,
            )

    def test_selection_returns_best_candidate_that_survives_thresholds(self):
        validation = importlib.import_module("narrative_dynamics.validation")

        class ScaledLevelModel:
            name = "threshold-scaled-level"

            def simulate(self, scenario, parameters, rng):
                return ModelRun(
                    events=(),
                    outcome={
                        "value": float(parameters["level"])
                        * float(scenario.payload["scale"])
                    },
                )

        suite = validation.HeldOutSuite(
            name="threshold-selection",
            role=validation.EvaluationRole.SELECTION_VALIDATION,
            cases=(
                HeldOutCase(
                    scenario=Scenario(id="high-scale", payload={"scale": 3.0}),
                    seeds=(1,),
                    target={"value": 0.0},
                ),
                HeldOutCase(
                    scenario=Scenario(id="low-scale", payload={"scale": 1.0}),
                    seeds=(2,),
                    target={"value": 4.0},
                ),
            ),
        )
        selection = validation.select_on_validation_suite(
            runner=SimulationRunner(),
            model=ScaledLevelModel(),
            accepted_parameters=(
                (("level", 0.0),),
                (("level", 1.0),),
            ),
            suite=suite,
            extractor=value_metrics,
            max_worst_loss=10.0,
        )

        self.assertEqual(
            selection.candidate_report.best.parameters,
            (("level", 0.0),),
        )
        self.assertEqual(
            selection.candidate_report.retained_parameters.parameters,
            ((("level", 1.0),),),
        )
        self.assertEqual(selection.selected_parameters, (("level", 1.0),))


if __name__ == "__main__":
    unittest.main()
