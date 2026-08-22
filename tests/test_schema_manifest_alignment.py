import unittest

from narrative_dynamics.manifest import scenario_identity
from narrative_dynamics.registry import ModelRegistry
from narrative_dynamics.simulation import SimulationRunner
from tests.test_runtime_schema_enforcement import (
    CountingContractModel,
    in_process_contract,
)


class ManifestOnlyScenario:
    id = "manifest-only"

    def manifest_payload(self):
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


if __name__ == "__main__":
    unittest.main()
