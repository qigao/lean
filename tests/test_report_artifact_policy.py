from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.attestation import (
    ImplementationAttestationMismatch,
    RepositoryIdentity,
    measure_implementation,
)
from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.execution_policy import (
    TrustedExecutionPolicy,
    TrustedExecutionPolicyViolation,
    TrustedModelPin,
)
from narrative_dynamics.model_contract import ModelContract
from narrative_dynamics.report_artifact import (
    AggregateReportArtifact,
    attest_report,
)
from narrative_dynamics.simulation import SimulationRunner


class LevelModel:
    name = "report-level"
    version = "1.0.0"
    implementation_revision = "test:report-level-v1"

    def simulate(self, scenario, parameters, rng):
        scale = float(scenario.payload.get("scale", 1.0))
        return ModelRun(
            events=(),
            outcome={"value": float(parameters["level"]) * scale},
        )


class CountingPolicyModel:
    name = "policy-model"
    version = "1.0.0"
    implementation_revision = "test:policy-model-v1"

    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        return ModelRun(events=(), outcome={"value": float(parameters["value"])})


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


class AggregateReportArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        )
        self.model = LevelModel()
        self.scenario = Scenario(id="report", payload={"scale": 1.0})

    def _calibrate(self, target: float):
        return calibrate_grid(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seeds=(1, 2),
            extractor=value_metrics,
            target={"value": target},
        )

    def test_report_artifact_binds_payload_to_existing_manifest(self):
        report = self._calibrate(2.0)
        attested = attest_report(report)

        self.assertIs(attested.report, report)
        self.assertIsInstance(attested.artifact, AggregateReportArtifact)
        self.assertEqual(attested.artifact.manifest_hash, report.manifest.content_hash)
        self.assertTrue(attested.artifact.matches(report))
        self.assertIs(attested.require_integrity(), report)

    def test_report_payload_or_manifest_change_changes_artifact_identity(self):
        first = self._calibrate(2.0)
        second = self._calibrate(3.0)
        first_artifact = AggregateReportArtifact.from_report(first)
        second_artifact = AggregateReportArtifact.from_report(second)

        self.assertNotEqual(first_artifact.payload_hash, second_artifact.payload_hash)
        self.assertNotEqual(first_artifact.content_hash, second_artifact.content_hash)

        relinked = replace(first, manifest=second.manifest)
        relinked_artifact = AggregateReportArtifact.from_report(relinked)
        self.assertEqual(first_artifact.payload_hash, relinked_artifact.payload_hash)
        self.assertNotEqual(first_artifact.manifest_hash, relinked_artifact.manifest_hash)
        self.assertNotEqual(first_artifact.content_hash, relinked_artifact.content_hash)

    def test_non_report_values_are_rejected(self):
        with self.assertRaises(TypeError):
            AggregateReportArtifact.from_report({"loss": 1.0})


class TrustedExecutionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        )
        self.contract = ModelContract(
            version="1.0.0",
            implementation_revision="test:policy-model-v1",
        )

    def _policy(self, expected_hash: str) -> TrustedExecutionPolicy:
        return TrustedExecutionPolicy(
            name="production",
            version="2026-08-23",
            pins=(
                TrustedModelPin(
                    model_name="policy-model",
                    declared_contract_hash=self.contract.content_hash,
                    expected_implementation_hash=expected_hash,
                ),
            ),
        )

    def test_external_policy_pin_is_injected_and_attested(self):
        model = CountingPolicyModel()
        measured = measure_implementation(model)
        policy = self._policy(measured.content_hash)
        bound = policy.bind(model, contract=self.contract)

        trace = self.runner.run_once(
            bound,
            Scenario(id="policy", payload={}),
            {"value": 4.0},
            seed=7,
        )

        self.assertEqual(model.calls, 1)
        policy_identity = trace.manifest.inputs["model"]["trusted_execution_policy"]
        self.assertEqual(policy_identity["content_hash"], policy.content_hash)
        self.assertEqual(policy_identity["verification"], "matched")
        attestation = trace.manifest.inputs["model"]["implementation_attestation"]
        self.assertEqual(
            attestation["expected_content_hash"],
            measured.content_hash,
        )
        self.assertEqual(attestation["verification"], "matched")

    def test_policy_mismatch_rejects_before_model_execution(self):
        model = CountingPolicyModel()
        bound = self._policy("sha256:" + "0" * 64).bind(
            model,
            contract=self.contract,
        )

        with self.assertRaises(ImplementationAttestationMismatch):
            self.runner.run_once(
                bound,
                Scenario(id="policy", payload={}),
                {"value": 4.0},
                seed=7,
            )
        self.assertEqual(model.calls, 0)

    def test_policy_requires_exact_model_and_declared_contract(self):
        model = CountingPolicyModel()
        measured = measure_implementation(model)
        policy = self._policy(measured.content_hash)

        with self.assertRaises(TrustedExecutionPolicyViolation):
            policy.bind(
                model,
                contract=replace(
                    self.contract,
                    implementation_revision="test:different",
                ),
            )

        with self.assertRaises(TrustedExecutionPolicyViolation):
            TrustedExecutionPolicy(
                name="production",
                version="2026-08-23",
                pins=(),
            ).bind(model, contract=self.contract)


if __name__ == "__main__":
    unittest.main()
