import importlib
import math
import unittest

from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import repeated_grid_calibration
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    evaluate_on_final_test_suite,
    select_on_validation_suite,
)


class VersionedLevelModel:
    name = "versioned-level"
    version = "2.1.0"

    def simulate(self, scenario, parameters, rng):
        scale = float(scenario.payload.get("scale", 1.0))
        return ModelRun(
            events=(),
            outcome={"value": float(parameters["level"]) * scale},
        )


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


value_metrics.version = "1.0.0"


def require_manifest_module(test_case):
    try:
        return importlib.import_module("narrative_dynamics.manifest")
    except ModuleNotFoundError as error:  # RED: module is intentionally absent first.
        test_case.fail(f"experiment manifest module is missing: {error}")


class StableContentHashTests(unittest.TestCase):
    def test_hash_is_order_independent_typed_and_finite(self):
        manifest = require_manifest_module(self)

        first = manifest.stable_content_hash(
            {"b": [1, 2.0], "a": {"flag": True, "label": "x"}}
        )
        second = manifest.stable_content_hash(
            {"a": {"label": "x", "flag": True}, "b": (1, 2.0)}
        )

        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(first, second)
        self.assertNotEqual(
            manifest.stable_content_hash({"value": 1}),
            manifest.stable_content_hash({"value": 1.0}),
        )
        with self.assertRaises(ValueError):
            manifest.stable_content_hash({"value": math.nan})

    def test_manifest_is_versioned_detached_and_parent_hashes_are_canonical(self):
        manifest = require_manifest_module(self)
        parent_a = manifest.stable_content_hash({"parent": "a"})
        parent_b = manifest.stable_content_hash({"parent": "b"})
        source = {"nested": {"values": [1, 2]}}

        report_manifest = manifest.ExperimentManifest(
            stage=manifest.ExperimentStage.GRID_CALIBRATION,
            inputs=source,
            parent_hashes=(parent_b, parent_a, parent_a),
        )
        source["nested"]["values"].append(3)

        self.assertEqual(report_manifest.schema_version, 1)
        self.assertEqual(
            report_manifest.inputs["nested"]["values"],
            (1, 2),
        )
        self.assertEqual(
            report_manifest.parent_hashes,
            tuple(sorted((parent_a, parent_b))),
        )
        with self.assertRaises(TypeError):
            report_manifest.inputs["new"] = "forbidden"

        same = manifest.ExperimentManifest(
            stage="grid_calibration",
            inputs={"nested": {"values": (1, 2)}},
            parent_hashes=(parent_a, parent_b),
        )
        newer_schema = manifest.ExperimentManifest(
            schema_version=2,
            stage=manifest.ExperimentStage.GRID_CALIBRATION,
            inputs={"nested": {"values": (1, 2)}},
            parent_hashes=(parent_a, parent_b),
        )
        self.assertEqual(report_manifest.content_hash, same.content_hash)
        self.assertNotEqual(report_manifest.content_hash, newer_schema.content_hash)


class RuntimeManifestTests(unittest.TestCase):
    def test_runner_attaches_stable_versioned_manifest_to_each_trace(self):
        manifest = require_manifest_module(self)
        runner = SimulationRunner()
        scenario = Scenario(id="run", payload={"scale": 2.0, "labels": ["a"]})

        first = runner.run_once(
            VersionedLevelModel(),
            scenario,
            {"z": 9.0, "level": 2.0},
            seed=7,
        )
        second = runner.run_once(
            VersionedLevelModel(),
            scenario,
            {"level": 2.0, "z": 9.0},
            seed=7,
        )
        changed_seed = runner.run_once(
            VersionedLevelModel(),
            scenario,
            {"level": 2.0, "z": 9.0},
            seed=8,
        )

        self.assertEqual(first.manifest.stage, manifest.ExperimentStage.SIMULATION_RUN)
        self.assertEqual(first.manifest.inputs["model"]["name"], "versioned-level")
        self.assertEqual(first.manifest.inputs["model"]["version"], "2.1.0")
        self.assertEqual(first.manifest.content_hash, second.manifest.content_hash)
        self.assertNotEqual(first.manifest.content_hash, changed_seed.manifest.content_hash)
        self.assertEqual(first.manifest_hash, first.manifest.content_hash)


class ReportManifestTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = VersionedLevelModel()
        self.scenario = Scenario(id="calibration", payload={"scale": 1.0})

    def test_calibration_and_repeated_calibration_form_parent_hash_chains(self):
        manifest = require_manifest_module(self)
        calibration = calibrate_grid(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seeds=(10, 11),
            extractor=value_metrics,
            target={"value": 2.0},
        )
        same = calibrate_grid(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seeds=(10, 11),
            extractor=value_metrics,
            target={"value": 2.0},
        )
        changed_target = calibrate_grid(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seeds=(10, 11),
            extractor=value_metrics,
            target={"value": 2.5},
        )

        self.assertEqual(
            calibration.manifest.stage,
            manifest.ExperimentStage.GRID_CALIBRATION,
        )
        self.assertEqual(calibration.manifest.content_hash, same.manifest.content_hash)
        self.assertNotEqual(
            calibration.manifest.content_hash,
            changed_target.manifest.content_hash,
        )
        self.assertEqual(len(calibration.manifest.parent_hashes), 6)

        repeated = repeated_grid_calibration(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seed_blocks=((1, 2), (3, 4)),
            extractor=value_metrics,
            target={"value": 2.0},
        )
        self.assertEqual(
            repeated.manifest.stage,
            manifest.ExperimentStage.REPEATED_CALIBRATION,
        )
        self.assertEqual(
            repeated.manifest.parent_hashes,
            tuple(sorted(block.manifest.content_hash for block in repeated.blocks)),
        )

    def test_selection_and_final_test_reports_keep_distinct_manifest_stages(self):
        manifest = require_manifest_module(self)
        cases = (
            HeldOutCase(
                scenario=Scenario(id="scale-2", payload={"scale": 2.0}),
                seeds=(101, 102),
                target={"value": 4.0},
            ),
        )
        selection = select_on_validation_suite(
            runner=self.runner,
            model=self.model,
            accepted_parameters=(
                (("level", 1.0),),
                (("level", 2.0),),
                (("level", 3.0),),
            ),
            suite=HeldOutSuite(
                name="selection",
                role=EvaluationRole.SELECTION_VALIDATION,
                cases=cases,
            ),
            extractor=value_metrics,
        )
        final = evaluate_on_final_test_suite(
            runner=self.runner,
            model=self.model,
            parameters=dict(selection.selected_parameters),
            suite=HeldOutSuite(
                name="final",
                role=EvaluationRole.FINAL_TEST,
                cases=cases,
            ),
            extractor=value_metrics,
        )

        self.assertEqual(
            selection.manifest.stage,
            manifest.ExperimentStage.SELECTION_VALIDATION,
        )
        self.assertEqual(
            selection.candidate_report.manifest.stage,
            manifest.ExperimentStage.ACCEPTANCE_VALIDATION,
        )
        self.assertEqual(final.manifest.stage, manifest.ExperimentStage.FINAL_TEST)
        self.assertEqual(
            final.validation.manifest.stage,
            manifest.ExperimentStage.HELD_OUT_VALIDATION,
        )
        self.assertIn(
            final.validation.manifest.content_hash,
            final.manifest.parent_hashes,
        )


if __name__ == "__main__":
    unittest.main()
