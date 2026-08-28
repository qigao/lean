from __future__ import annotations

import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.observations import ProtocolRelease, VerifiedProtocolRelease
from narrative_dynamics.simulation import SimulationRunner

from tests.external_validation_fixtures import (
    brier_loss,
    build_external_declaration,
    build_preregistration,
    final_targets,
    log_loss,
    make_release,
    policy_metrics,
    runtime_models,
    sibling_protocols,
    verify_release,
)

try:
    from narrative_dynamics.external_validation import (
        ExternalValidationError,
        preflight_external_releases,
    )
except ImportError as error:
    ExternalValidationError = None
    preflight_external_releases = None
    PREFLIGHT_IMPORT_ERROR = error
else:
    PREFLIGHT_IMPORT_ERROR = None

try:
    from narrative_dynamics.external_validation import evaluate_external_final
except ImportError as error:
    evaluate_external_final = None
    FINAL_IMPORT_ERROR = error
else:
    FINAL_IMPORT_ERROR = None

HASH0 = "sha256:" + "0" * 64


class ExternalValidationReleaseGateTests(unittest.TestCase):
    def _fixture(self):
        if preflight_external_releases is None:
            self.fail(f"external release gate API is missing: {PREFLIGHT_IMPORT_ERROR}")
        data, brier, log = sibling_protocols()
        declaration = build_external_declaration(data)
        prereg = build_preregistration(
            dataset=data,
            brier_protocol=brier,
            log_protocol=log,
        )
        brier_release = make_release(
            brier,
            evidence_hash=declaration.content_hash,
            preregistration_hash=prereg.content_hash,
            score_role="brier",
        )
        log_release = make_release(
            log,
            evidence_hash=declaration.content_hash,
            preregistration_hash=prereg.content_hash,
            score_role="log",
        )
        return {
            "dataset": data,
            "declaration": declaration,
            "prereg": prereg,
            "brier": brier,
            "log": log,
            "brier_release": brier_release,
            "log_release": log_release,
            "brier_verified": verify_release(brier_release, brier),
            "log_verified": verify_release(log_release, log),
        }

    def _preflight(self, fixture, **overrides):
        values = dict(
            preregistration=fixture["prereg"],
            evidence=fixture["declaration"],
            brier_protocol=fixture["brier"],
            log_protocol=fixture["log"],
            brier_release=fixture["brier_release"],
            log_release=fixture["log_release"],
            brier_verified=fixture["brier_verified"],
            log_verified=fixture["log_verified"],
        )
        values.update(overrides)
        return preflight_external_releases(**values)

    def test_release_source_revision_must_bind_evidence_preregistration_repository_and_score_role(self):
        fixture = self._fixture()
        incomplete = ProtocolRelease.create(
            name="incomplete-log",
            version="1",
            protocol=fixture["log"],
            source_revision={
                "repository_revision": "test-repository-revision",
                "score_role": "log",
            },
        )
        incomplete_verified = verify_release(incomplete, fixture["log"])
        with self.assertRaises(ExternalValidationError):
            self._preflight(
                fixture,
                log_release=incomplete,
                log_verified=incomplete_verified,
            )

    def test_brier_release_cannot_be_substituted_for_log_release(self):
        fixture = self._fixture()
        with self.assertRaises(ExternalValidationError):
            self._preflight(fixture, log_release=fixture["brier_release"])

    def test_verified_release_hash_must_match_the_actual_release_hash(self):
        fixture = self._fixture()
        forged = VerifiedProtocolRelease(
            release_hash=HASH0,
            protocol_hash=fixture["log"].content_hash,
            verifier_identity={"name": "forged", "version": "1"},
            verified_receipt_hashes=(stable_content_hash({"receipt": "forged"}),),
        )
        with self.assertRaises(ExternalValidationError):
            self._preflight(fixture, log_verified=forged)

    def test_one_unverified_or_drifted_sibling_prevents_both_final_executions(self):
        if evaluate_external_final is None:
            self.fail(f"external final API is missing: {FINAL_IMPORT_ERROR}")
        fixture = self._fixture()
        invalid_log = ProtocolRelease.create(
            name="drifted-log",
            version="1",
            protocol=fixture["log"],
            source_revision={
                "repository_revision": "test-repository-revision",
                "external_evidence_declaration_hash": fixture["declaration"].content_hash,
                "external_validation_preregistration_hash": HASH0,
                "score_role": "log",
            },
        )
        invalid_verified = verify_release(invalid_log, fixture["log"])
        brier_models, brier_sources = runtime_models(fixture["brier"].candidates)
        log_models, log_sources = runtime_models(fixture["log"].candidates)
        with self.assertRaises(ExternalValidationError):
            evaluate_external_final(
                runner=SimulationRunner(),
                preregistration=fixture["prereg"],
                evidence=fixture["declaration"],
                brier_protocol=fixture["brier"],
                brier_release=fixture["brier_release"],
                brier_verified=fixture["brier_verified"],
                brier_models=brier_models,
                brier_loss=brier_loss(),
                log_protocol=fixture["log"],
                log_release=invalid_log,
                log_verified=invalid_verified,
                log_models=log_models,
                log_loss=log_loss(),
                final_targets=final_targets(fixture["dataset"]),
                extractor=policy_metrics,
            )
        self.assertEqual(sum(model.calls for model in brier_sources + log_sources), 0)

    def test_both_verified_siblings_produce_stable_preflight_identity(self):
        fixture = self._fixture()
        first = self._preflight(fixture)
        second = self._preflight(fixture)
        self.assertEqual(first, second)
        self.assertTrue(first.content_hash.startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
