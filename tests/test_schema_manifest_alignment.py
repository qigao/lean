import unittest

from narrative_dynamics.manifest import scenario_identity, stable_content_hash
from narrative_dynamics.registry import ModelRegistry
from narrative_dynamics.schema_validation import ModelSchemaViolation
from narrative_dynamics.simulation import SimulationRunner
from tests.test_runtime_schema_enforcement import (
    CountingContractModel,
    in_process_contract,
)


class ManifestOnlyScenario:
    id = "manifest-only"

    def manifest_payload(self):
        return {"scale": 1.0}


class PayloadOnlyScenario:
    id = "payload-only"
    payload = {"scale": 1.0}


class OneShotManifestScenario:
    id = "one-shot-manifest"

    def __init__(self):
        self.calls = 0

    def manifest_payload(self):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("manifest_payload() must be snapshotted exactly once")
        return {"scale": 1.0}


class SchemaManifestAlignmentTests(unittest.TestCase):
    def test_custom_scenario_validates_the_same_payload_recorded_by_manifest(self):
        scenario = ManifestOnlyScenario()
        model = CountingContractModel()
        descriptor = ModelRegistry().register(
            model,
            contract=in_process_contract(),
        )

        trace = SimulationRunner().run_once(
            descriptor,
            scenario,
            {"level": 5.0},
            seed=7,
        )

        self.assertEqual(trace.outcome["value"], 5.0)
        self.assertEqual(model.calls, 1)
        self.assertEqual(
            trace.manifest.inputs["scenario"]["content_hash"],
            scenario_identity(scenario)["content_hash"],
        )
        self.assertEqual(
            trace.manifest.inputs["schema_validation"]["scenario"]["status"],
            "validated",
        )

    def test_contracted_custom_scenario_cannot_validate_unrecorded_payload(self):
        model = CountingContractModel()
        descriptor = ModelRegistry().register(
            model,
            contract=in_process_contract(),
        )

        with self.assertRaises(ModelSchemaViolation) as raised:
            SimulationRunner().run_once(
                descriptor,
                PayloadOnlyScenario(),
                {"level": 5.0},
                seed=8,
            )

        self.assertEqual(raised.exception.boundary, "scenario")
        self.assertEqual(raised.exception.path, "$")
        self.assertEqual(model.calls, 0)

    def test_custom_manifest_payload_is_snapshotted_once_for_validation_and_identity(self):
        scenario = OneShotManifestScenario()
        descriptor = ModelRegistry().register(
            CountingContractModel(),
            contract=in_process_contract(),
        )

        trace = SimulationRunner().run_once(
            descriptor,
            scenario,
            {"level": 5.0},
            seed=9,
        )

        scenario_type = (
            f"{scenario.__class__.__module__}.{scenario.__class__.__qualname__}"
        )
        expected_hash = stable_content_hash(
            {
                "id": scenario.id,
                "type": scenario_type,
                "payload": {"scale": 1.0},
            }
        )
        self.assertEqual(scenario.calls, 1)
        self.assertEqual(
            trace.manifest.inputs["scenario"]["content_hash"],
            expected_hash,
        )


if __name__ == "__main__":
    unittest.main()
