from __future__ import annotations

from unittest.mock import patch
import unittest

import narrative_dynamics.attestation as attestation
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.model_contract import ModelContract, ModelSchema
from narrative_dynamics.process_execution import ProcessLimits, SubprocessModel
from narrative_dynamics.registry import ModelRegistry
from narrative_dynamics.simulation import SimulationRunner


def _contract(source: SubprocessModel) -> ModelContract:
    return ModelContract(
        version=source.version,
        implementation_revision=source.implementation_revision,
        expected_implementation_hash=measure_implementation(source).content_hash,
        parameter_schema=ModelSchema(
            name="echo-parameters",
            version="1.0.0",
            definition={"type": "object"},
        ),
        scenario_schema=ModelSchema(
            name="echo-scenario",
            version="1.0.0",
            definition={"type": "object"},
        ),
        event_schema=ModelSchema(
            name="echo-events",
            version="1.0.0",
            definition={"event_kinds": ("echo",)},
        ),
        outcome_schema=ModelSchema(
            name="echo-outcome",
            version="1.0.0",
            definition={"type": "object"},
        ),
    )


class FreshProcessPinTests(unittest.TestCase):
    def test_batch_remeasures_pin_before_each_fresh_worker_launch(self):
        source = SubprocessModel(
            name="process-echo",
            factory="tests.subprocess_fixtures:create_echo_model",
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            limits=ProcessLimits(timeout_seconds=2.0),
        )
        descriptor = ModelRegistry().register(
            source,
            kind="pomdp",
            contract=_contract(source),
        )

        with patch(
            "narrative_dynamics.simulation.implementation_attestation_identity",
            wraps=attestation.implementation_attestation_identity,
        ) as measured:
            traces = SimulationRunner().run_batch(
                descriptor,
                Scenario(id="remeasure", payload={}),
                {"value": 2.0},
                seeds=(41, 42),
            )

        self.assertEqual(len(traces), 2)
        self.assertEqual(measured.call_count, 2)


if __name__ == "__main__":
    unittest.main()
