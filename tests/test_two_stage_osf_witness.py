from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.observations.release import WitnessReceipt, verify_protocol_release
from tests.external_validation_fixtures import (
    build_external_declaration,
    build_preregistration,
    make_release,
    sibling_protocols,
)

_OSF_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.studies.two_stage_osf import (
        OSFRegistrationProof,
        OSFRegistrationVerifier,
        TwoStageOSFBundle,
    )
except Exception as error:
    _OSF_IMPORT_ERROR = error


def _bundle_and_releases():
    data, brier_protocol, log_protocol = sibling_protocols()
    evidence = build_external_declaration(data)
    preregistration = build_preregistration(
        dataset=data,
        brier_protocol=brier_protocol,
        log_protocol=log_protocol,
    )
    brier_release = make_release(
        brier_protocol,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role="brier",
    )
    log_release = make_release(
        log_protocol,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role="log",
    )
    bundle = TwoStageOSFBundle(
        source_manifest_hash=stable_content_hash({"source": "manifest"}),
        source_snapshot_hash=stable_content_hash({"source": "snapshot"}),
        transform_hash=stable_content_hash({"transform": "v1"}),
        participant_assignment_hash=stable_content_hash({"split": "v1"}),
        dataset_hash=data.content_hash,
        final_target_hash=brier_protocol.final_target_hash,
        frozen_candidate_hashes=tuple(candidate.content_hash for candidate in brier_protocol.candidates),
        brier_protocol_hash=brier_protocol.content_hash,
        log_protocol_hash=log_protocol.content_hash,
        external_preregistration_hash=preregistration.content_hash,
        brier_release_hash=brier_release.content_hash,
        log_release_hash=log_release.content_hash,
        repository_revision="test-repository-revision",
        claim_scope="external_observational_predictive_only",
        scientific_contract={
            "probability_floor": 1e-12,
            "separation": (0.005, 0.006931471805599453),
            "constraint_plans": (),
        },
    )
    return bundle, brier_protocol, log_protocol, brier_release, log_release


class TwoStageOSFWitnessTests(unittest.TestCase):
    def require_osf(self):
        self.assertIsNone(
            _OSF_IMPORT_ERROR,
            f"OSF witness boundary is missing: {_OSF_IMPORT_ERROR}",
        )

    def test_bundle_identity_is_immutable_and_content_hashed(self):
        self.require_osf()
        bundle, *_ = _bundle_and_releases()
        changed = replace(bundle, repository_revision="changed")
        self.assertNotEqual(bundle.content_hash, changed.content_hash)

    def test_two_receipts_can_bind_one_registration_and_same_bundle_digest(self):
        self.require_osf()
        bundle, brier_protocol, log_protocol, brier_release, log_release = _bundle_and_releases()
        proof = OSFRegistrationProof(
            registration_reference="https://osf.io/registrations/test",
            registered_at="2026-08-29T00:00:00Z",
            bundle_hash=bundle.content_hash,
        )
        receipts = tuple(
            WitnessReceipt.create(
                provider="osf-registration",
                authority="public-registration",
                subject_hash=release.content_hash,
                reference=proof.registration_reference,
                claimed_at=proof.registered_at,
                proof={"bundle_hash": proof.bundle_hash},
            )
            for release in (brier_release, log_release)
        )
        self.assertEqual(receipts[0].reference, receipts[1].reference)
        self.assertEqual(dict(receipts[0].proof), dict(receipts[1].proof))
        self.assertNotEqual(receipts[0].subject_hash, receipts[1].subject_hash)
        verifier = OSFRegistrationVerifier(proof)
        self.assertEqual(
            verify_protocol_release(
                brier_release,
                protocol=brier_protocol,
                receipts=(receipts[0],),
                verifier=verifier,
            ).release_hash,
            brier_release.content_hash,
        )
        self.assertEqual(
            verify_protocol_release(
                log_release,
                protocol=log_protocol,
                receipts=(receipts[1],),
                verifier=verifier,
            ).release_hash,
            log_release.content_hash,
        )

    def test_bundle_digest_drift_is_rejected(self):
        self.require_osf()
        bundle, brier_protocol, _, brier_release, _ = _bundle_and_releases()
        proof = OSFRegistrationProof(
            registration_reference="https://osf.io/registrations/test",
            registered_at="2026-08-29T00:00:00Z",
            bundle_hash=bundle.content_hash,
        )
        receipt = WitnessReceipt.create(
            provider="osf-registration",
            authority="public-registration",
            subject_hash=brier_release.content_hash,
            reference=proof.registration_reference,
            claimed_at=proof.registered_at,
            proof={"bundle_hash": stable_content_hash({"wrong": True})},
        )
        with self.assertRaises(ValueError):
            verify_protocol_release(
                brier_release,
                protocol=brier_protocol,
                receipts=(receipt,),
                verifier=OSFRegistrationVerifier(proof),
            )

    def test_witness_receipt_schema_is_not_changed(self):
        self.require_osf()
        self.assertEqual(
            set(WitnessReceipt.__dataclass_fields__),
            {
                "provider",
                "authority",
                "subject_hash",
                "reference",
                "claimed_at",
                "proof",
                "declared_content_hash",
                "schema_version",
            },
        )


if __name__ == "__main__":
    unittest.main()
