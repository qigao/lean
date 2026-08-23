from __future__ import annotations

import unittest

from narrative_dynamics.attestation import RepositoryIdentity, measure_implementation
from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.execution_policy import (
    TrustedExecutionPolicy,
    TrustedModelPin,
)
from narrative_dynamics.model_contract import ModelContract
from narrative_dynamics.process_execution import ProcessLimits, SubprocessModel
from narrative_dynamics.simulation import ModelFactory, SimulationRunner


class FactoryPolicyModel:
    name = "factory-policy"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(events=(), outcome={"value": float(parameters["value"])})


def create_factory_policy_model():
    return FactoryPolicyModel()


class ExecutionPolicyLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        )

    @staticmethod
    def _policy(name, contract, expected_hash):
        return TrustedExecutionPolicy(
            name="lifecycle-policy",
            version="1",
            pins=(
                TrustedModelPin(
                    model_name=name,
                    declared_contract_hash=contract.content_hash,
                    expected_implementation_hash=expected_hash,
                ),
            ),
        )

    def test_policy_bound_factory_preserves_fresh_batch_construction(self):
        source = ModelFactory(
            name="factory-policy",
            create=create_factory_policy_model,
            version="1.0.0",
            implementation_revision="test:factory-policy-v1",
        )
        contract = ModelContract(
            version="1.0.0",
            implementation_revision="test:factory-policy-v1",
        )
        policy = self._policy(
            source.name,
            contract,
            measure_implementation(source).content_hash,
        )
        bound = policy.bind(source, contract=contract)

        trace = self.runner.run_once(
            bound,
            Scenario(id="factory-policy", payload={}),
            {"value": 5.0},
            seed=1,
        )

        self.assertEqual(trace.outcome["value"], 5.0)
        self.assertEqual(
            trace.manifest.inputs["model"]["trusted_execution_policy"]["content_hash"],
            policy.content_hash,
        )

    def test_policy_bound_subprocess_preserves_fresh_process_execution(self):
        source = SubprocessModel(
            name="process-echo",
            factory="tests.subprocess_fixtures:create_echo_model",
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            limits=ProcessLimits(timeout_seconds=2.0),
        )
        contract = ModelContract(
            version="1.0.0",
            implementation_revision="git:test-process-v1",
        )
        policy = self._policy(
            source.name,
            contract,
            measure_implementation(source).content_hash,
        )
        bound = policy.bind(source, contract=contract)

        trace = self.runner.run_once(
            bound,
            Scenario(id="process-policy", payload={}),
            {"value": 6.0},
            seed=3,
        )

        self.assertEqual(trace.outcome["value"], 6.0)
        self.assertTrue(trace.execution.isolated)
        self.assertEqual(
            trace.manifest.inputs["model"]["trusted_execution_policy"]["content_hash"],
            policy.content_hash,
        )


if __name__ == "__main__":
    unittest.main()
