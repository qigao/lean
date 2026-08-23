from __future__ import annotations

import importlib
import unittest

from narrative_dynamics.contracts import ModelRun, Scenario, stable_content_hash
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
    PreregisteredEvaluationProtocol,
)


class ReleaseProbabilityModel:
    name = "release-probability-model"
    version = "1.0.0"
    implementation_revision = "release-probability-v1"

    def simulate(self, scenario, parameters, rng):
        probability = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={
                "policy": {"a": probability, "b": 1.0 - probability}
            },
        )


class ReleaseAlternativeModel:
    name = "release-alternative-model"
    version = "1.0.0"
    implementation_revision = "release-alternative-v1"

    def simulate(self, scenario, parameters, rng):
        probability = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={
                "policy": {"a": probability, "b": 1.0 - probability}
            },
        )


def policy_metrics(trace):
    policy = trace.outcome["policy"]
    return {
        "choice.a": float(policy["a"]),
        "choice.b": float(policy["b"]),
    }


policy_metrics.version = "1"


def observation_dataset() -> ObservationDataset:
    def record(record_id: str, scenario_id: str, a: int, b: int) -> ObservationRecord:
        return ObservationRecord(
            id=record_id,
            scenario=Scenario(id=scenario_id, payload={"condition": scenario_id}),
            counts={"a": a, "b": b},
        )

    return ObservationDataset(
        name="release-observations",
        version="1.0.0",
        source={"kind": "synthetic_test"},
        provenance={"purpose": "release-boundary"},
        partitions=(
            ObservationPartition(
                name="train",
                role=ObservationPartitionRole.TRAIN,
                records=(record("train-1", "train-scenario", 6, 4),),
            ),
            ObservationPartition(
                name="selection",
                role=ObservationPartitionRole.SELECTION_VALIDATION,
                records=(record("selection-1", "selection-scenario", 7, 3),),
            ),
            ObservationPartition(
                name="final",
                role=ObservationPartitionRole.FINAL_TEST,
                records=(record("final-1", "final-scenario", 8, 2),),
            ),
        ),
    )


def protocol_fixture(*, name: str = "release-protocol") -> PreregisteredEvaluationProtocol:
    dataset = observation_dataset()
    spec = CategoricalTargetSpec(
        name="release-choice-target",
        version="1",
        categories=("a", "b"),
        metric_prefix="choice",
    )
    loss = CategoricalBrierLoss(
        (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),)
    )
    selection_hash = stable_content_hash({"selection": "frozen"})
    baseline = FrozenModelCandidate.freeze(
        name="baseline",
        model=ReleaseProbabilityModel(),
        parameters={"p": 0.8},
        selection_manifest_hash=selection_hash,
    )
    alternative = FrozenModelCandidate.freeze(
        name="alternative",
        model=ReleaseAlternativeModel(),
        parameters={"p": 0.2},
        selection_manifest_hash=selection_hash,
    )
    return PreregisteredEvaluationProtocol.create(
        name=name,
        version="1",
        dataset=dataset,
        target_spec=spec,
        extractor=policy_metrics,
        loss=loss,
        simulation_seeds=(11, 12),
        baseline_name="baseline",
        candidates=(baseline, alternative),
        thresholds=AdequacyThresholds(1.0, 1.0),
    )


def release_api(test_case):
    try:
        return importlib.import_module("narrative_dynamics.observations.release")
    except ModuleNotFoundError as error:
        test_case.fail(f"protocol release API is missing: {error}")


class FixtureWitnessVerifier:
    name = "fixture-witness-verifier"
    version = "1"

    def __init__(self, *, accept: bool = True):
        self.accept = accept
        self.calls = 0

    def verify(self, release, receipt):
        self.calls += 1
        return (
            self.accept
            and receipt.provider == "test-fixture"
            and dict(receipt.proof) == {"nonce": "release-v1"}
        )


class ProtocolReleaseIdentityTests(unittest.TestCase):
    def test_release_and_receipt_roundtrip_recompute_declared_hashes(self):
        api = release_api(self)
        protocol = protocol_fixture()
        release = api.ProtocolRelease.create(
            name="prison-comparison-release",
            version="1",
            protocol=protocol,
            source_revision={
                "kind": "declared_revision",
                "repository": "qigao/lean",
                "revision": "prison-alternative-release-v1",
            },
        )
        self.assertEqual(api.ProtocolRelease.from_payload(release.to_payload()), release)

        receipt = api.WitnessReceipt.create(
            provider="test-fixture",
            authority="unit-test",
            subject_hash=release.content_hash,
            reference="fixture-receipt-1",
            claimed_at="2026-08-23T00:00:00Z",
            proof={"nonce": "release-v1"},
        )
        self.assertEqual(api.WitnessReceipt.from_payload(receipt.to_payload()), receipt)

        forged = dict(release.to_payload())
        forged["version"] = "2"
        with self.assertRaises(ValueError):
            api.ProtocolRelease.from_payload(forged)

    def test_release_binds_protocol_dataset_target_and_candidate_identities(self):
        api = release_api(self)
        protocol = protocol_fixture()
        release = api.ProtocolRelease.create(
            name="prison-comparison-release",
            version="1",
            protocol=protocol,
            source_revision={"repository": "qigao/lean", "revision": "fixture"},
        )
        self.assertEqual(release.protocol_hash, protocol.content_hash)
        self.assertEqual(release.dataset_hash, protocol.dataset_hash)
        self.assertEqual(release.target_spec_hash, protocol.target_spec_hash)
        self.assertEqual(
            release.candidate_hashes,
            tuple(candidate.content_hash for candidate in protocol.candidates),
        )


class ProtocolReleaseVerificationTests(unittest.TestCase):
    def _fixture(self):
        api = release_api(self)
        protocol = protocol_fixture()
        release = api.ProtocolRelease.create(
            name="prison-comparison-release",
            version="1",
            protocol=protocol,
            source_revision={"repository": "qigao/lean", "revision": "fixture"},
        )
        receipt = api.WitnessReceipt.create(
            provider="test-fixture",
            authority="unit-test",
            subject_hash=release.content_hash,
            reference="fixture-receipt-1",
            claimed_at="2026-08-23T00:00:00Z",
            proof={"nonce": "release-v1"},
        )
        return api, protocol, release, receipt

    def test_verification_requires_at_least_one_external_receipt_acceptance(self):
        api, protocol, release, receipt = self._fixture()
        verifier = FixtureWitnessVerifier(accept=False)
        with self.assertRaises(api.ProtocolReleaseVerificationError):
            api.verify_protocol_release(
                release,
                protocol=protocol,
                receipts=(receipt,),
                verifier=verifier,
            )
        self.assertEqual(verifier.calls, 1)

    def test_subject_duplicate_and_protocol_drift_fail_before_verifier_call(self):
        api, protocol, release, receipt = self._fixture()
        verifier = FixtureWitnessVerifier()
        wrong_subject = api.WitnessReceipt.create(
            provider="test-fixture",
            authority="unit-test",
            subject_hash=stable_content_hash({"other": True}),
            reference="wrong-subject",
            claimed_at="2026-08-23T00:00:00Z",
            proof={"nonce": "release-v1"},
        )
        with self.assertRaises(api.ProtocolReleaseVerificationError):
            api.verify_protocol_release(
                release,
                protocol=protocol,
                receipts=(wrong_subject,),
                verifier=verifier,
            )
        self.assertEqual(verifier.calls, 0)

        with self.assertRaises(api.ProtocolReleaseVerificationError):
            api.verify_protocol_release(
                release,
                protocol=protocol,
                receipts=(receipt, receipt),
                verifier=verifier,
            )
        self.assertEqual(verifier.calls, 0)

        with self.assertRaises(api.ProtocolReleaseVerificationError):
            api.verify_protocol_release(
                release,
                protocol=protocol_fixture(name="different-protocol"),
                receipts=(receipt,),
                verifier=verifier,
            )
        self.assertEqual(verifier.calls, 0)

    def test_verified_release_records_verifier_and_receipt_identity(self):
        api, protocol, release, receipt = self._fixture()
        verifier = FixtureWitnessVerifier()
        verified = api.verify_protocol_release(
            release,
            protocol=protocol,
            receipts=(receipt,),
            verifier=verifier,
        )
        self.assertEqual(verified.status, "verified")
        self.assertEqual(verified.release_hash, release.content_hash)
        self.assertEqual(verified.protocol_hash, protocol.content_hash)
        self.assertEqual(verified.verified_receipt_hashes, (receipt.content_hash,))
        self.assertEqual(verified.verifier_identity["name"], verifier.name)
        self.assertTrue(verified.content_hash.startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
