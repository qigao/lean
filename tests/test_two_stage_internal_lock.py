from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.observations.preregistration import FrozenModelSpec
from narrative_dynamics.observations.release import WitnessReceipt, verify_protocol_release
from narrative_dynamics.external_validation import preflight_external_releases
from narrative_dynamics.studies.two_stage_osf import (
    OSFRegistrationProof,
    OSFRegistrationVerifier,
)
from narrative_dynamics.studies import feher_hare_two_stage_v1 as study
from tests.external_validation_fixtures import (
    build_external_declaration,
    build_preregistration,
    runtime_models,
    sibling_protocols,
)

_INTERNAL_LOCK_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.studies.two_stage_internal_lock import (
        INTERNAL_LOCK_AUTHORITY,
        INTERNAL_LOCK_GOVERNANCE_MODE,
        INTERNAL_LOCK_PROVIDER,
        InternalRepositoryLockProof,
        InternalRepositoryLockVerifier,
        TwoStageInternalLockBundle,
        create_internal_locked_protocol_release,
        create_internal_repository_lock_receipt,
    )
except Exception as error:
    _INTERNAL_LOCK_IMPORT_ERROR = error


FROZEN_SCIENTIFIC_REVISION = "r3-scientific-revision"
LOCK_COMMIT = "1" * 40
LOCKED_AT = "2026-08-30T00:00:00Z"
FREEZE_DIGEST = stable_content_hash({"freeze": "r2-real-train-selection"})


def _fixture():
    data, brier_protocol, log_protocol = sibling_protocols()
    evidence = build_external_declaration(data)
    preregistration = build_preregistration(
        dataset=data,
        brier_protocol=brier_protocol,
        log_protocol=log_protocol,
    )
    brier_release = create_internal_locked_protocol_release(
        name="external-validation:brier:r3-release",
        version="3",
        protocol=brier_protocol,
        repository_revision=FROZEN_SCIENTIFIC_REVISION,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role="brier",
        train_selection_freeze_digest=FREEZE_DIGEST,
    )
    log_release = create_internal_locked_protocol_release(
        name="external-validation:log:r3-release",
        version="3",
        protocol=log_protocol,
        repository_revision=FROZEN_SCIENTIFIC_REVISION,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role="log",
        train_selection_freeze_digest=FREEZE_DIGEST,
    )
    bundle = TwoStageInternalLockBundle(
        source_manifest_hash=stable_content_hash({"source": "manifest"}),
        source_snapshot_hash=stable_content_hash({"source": "snapshot"}),
        transform_hash=stable_content_hash({"transform": "v1"}),
        participant_assignment_hash=stable_content_hash({"split": "v1"}),
        dataset_hash=data.content_hash,
        final_target_hash=brier_protocol.final_target_hash,
        frozen_candidate_hashes=tuple(
            candidate.content_hash for candidate in brier_protocol.candidates
        ),
        brier_protocol_hash=brier_protocol.content_hash,
        log_protocol_hash=log_protocol.content_hash,
        evaluation_preregistration_hash=preregistration.content_hash,
        brier_release_hash=brier_release.content_hash,
        log_release_hash=log_release.content_hash,
        scientific_repository_revision=FROZEN_SCIENTIFIC_REVISION,
        train_selection_freeze_digest=FREEZE_DIGEST,
        claim_scope="external_observational_predictive_only",
        scientific_contract={
            "probability_floor": 1e-12,
            "final_seeds": (301, 302),
            "constraint_plans": (),
        },
    )
    proof = InternalRepositoryLockProof(
        repository="qigao/lean",
        lock_commit=LOCK_COMMIT,
        locked_at=LOCKED_AT,
        lock_payload_hash=bundle.content_hash,
        scientific_repository_revision=FROZEN_SCIENTIFIC_REVISION,
        release_hashes=(brier_release.content_hash, log_release.content_hash),
    )
    return {
        "data": data,
        "evidence": evidence,
        "preregistration": preregistration,
        "brier_protocol": brier_protocol,
        "log_protocol": log_protocol,
        "brier_release": brier_release,
        "log_release": log_release,
        "bundle": bundle,
        "proof": proof,
    }


class TwoStageInternalLockTests(unittest.TestCase):
    def require_internal_lock(self):
        self.assertIsNone(
            _INTERNAL_LOCK_IMPORT_ERROR,
            f"R3 internal lock boundary is missing: {_INTERNAL_LOCK_IMPORT_ERROR}",
        )

    def test_internal_lock_bundle_is_content_hashed_and_immutable(self):
        self.require_internal_lock()
        fixture = _fixture()
        bundle = fixture["bundle"]
        changed = replace(
            bundle,
            scientific_repository_revision="r3-other-revision",
        )
        self.assertTrue(bundle.content_hash.startswith("sha256:"))
        self.assertNotEqual(bundle.content_hash, changed.content_hash)

    def test_internal_lock_bundle_requires_internal_governance_and_no_external_registration(self):
        self.require_internal_lock()
        bundle = _fixture()["bundle"]
        self.assertEqual(bundle.governance_mode, "internal_locked_final")
        self.assertFalse(bundle.external_registration)
        with self.assertRaises(ValueError):
            replace(bundle, governance_mode="external_witness")
        with self.assertRaises(ValueError):
            replace(bundle, external_registration=True)

    def test_internal_lock_bundle_requires_two_distinct_release_hashes(self):
        self.require_internal_lock()
        bundle = _fixture()["bundle"]
        with self.assertRaises(ValueError):
            replace(bundle, log_release_hash=bundle.brier_release_hash)

    def test_internal_lock_bundle_requires_final_model_execution_false(self):
        self.require_internal_lock()
        bundle = _fixture()["bundle"]
        self.assertFalse(bundle.final_model_execution)
        with self.assertRaises(ValueError):
            replace(bundle, final_model_execution=True)

    def test_internal_locked_release_binds_r3_governance_metadata(self):
        self.require_internal_lock()
        fixture = _fixture()
        brier = fixture["brier_release"]
        log = fixture["log_release"]
        for role, release in (("brier", brier), ("log", log)):
            revision = release.source_revision
            self.assertEqual(
                revision["repository_revision"],
                FROZEN_SCIENTIFIC_REVISION,
            )
            self.assertEqual(revision["score_role"], role)
            self.assertEqual(
                revision["governance_mode"],
                "internal_locked_final",
            )
            self.assertIs(revision["external_registration"], False)
            self.assertEqual(
                revision["train_selection_freeze_digest"],
                FREEZE_DIGEST,
            )
        self.assertNotEqual(brier.content_hash, log.content_hash)

    def test_internal_locked_release_hash_changes_with_repository_revision(self):
        self.require_internal_lock()
        fixture = _fixture()
        changed = create_internal_locked_protocol_release(
            name="external-validation:brier:r3-release",
            version="3",
            protocol=fixture["brier_protocol"],
            repository_revision="r3-other-revision",
            evidence_hash=fixture["evidence"].content_hash,
            preregistration_hash=fixture["preregistration"].content_hash,
            score_role="brier",
            train_selection_freeze_digest=FREEZE_DIGEST,
        )
        self.assertNotEqual(changed.content_hash, fixture["brier_release"].content_hash)

    def test_internal_lock_proof_binds_repository_commit_payload_revision_and_releases(self):
        self.require_internal_lock()
        proof = _fixture()["proof"]
        self.assertEqual(proof.repository, "qigao/lean")
        self.assertEqual(proof.lock_commit, LOCK_COMMIT)
        self.assertEqual(proof.locked_at, LOCKED_AT)
        self.assertEqual(
            proof.reference,
            f"https://github.com/qigao/lean/commit/{LOCK_COMMIT}",
        )
        self.assertTrue(proof.content_hash.startswith("sha256:"))
        self.assertEqual(len(proof.release_hashes), 2)

    def test_two_internal_receipts_verify_two_sibling_releases(self):
        self.require_internal_lock()
        fixture = _fixture()
        verifier = InternalRepositoryLockVerifier(fixture["proof"])
        verified = []
        for release, protocol in (
            (fixture["brier_release"], fixture["brier_protocol"]),
            (fixture["log_release"], fixture["log_protocol"]),
        ):
            receipt = create_internal_repository_lock_receipt(
                release,
                fixture["proof"],
            )
            self.assertEqual(receipt.provider, INTERNAL_LOCK_PROVIDER)
            self.assertEqual(receipt.authority, INTERNAL_LOCK_AUTHORITY)
            verified.append(
                verify_protocol_release(
                    release,
                    protocol=protocol,
                    receipts=(receipt,),
                    verifier=verifier,
                )
            )
        self.assertNotEqual(verified[0].release_hash, verified[1].release_hash)

    def test_receipt_for_other_release_is_rejected(self):
        self.require_internal_lock()
        fixture = _fixture()
        verifier = InternalRepositoryLockVerifier(fixture["proof"])
        receipt = create_internal_repository_lock_receipt(
            fixture["brier_release"],
            fixture["proof"],
        )
        self.assertFalse(verifier.verify(fixture["log_release"], receipt))

    def test_wrong_lock_commit_reference_or_time_is_rejected(self):
        self.require_internal_lock()
        fixture = _fixture()
        verifier = InternalRepositoryLockVerifier(fixture["proof"])
        release = fixture["brier_release"]
        receipt = create_internal_repository_lock_receipt(release, fixture["proof"])
        wrong_reference = WitnessReceipt.create(
            provider=receipt.provider,
            authority=receipt.authority,
            subject_hash=receipt.subject_hash,
            reference="https://github.com/qigao/lean/commit/" + "2" * 40,
            claimed_at=receipt.claimed_at,
            proof=receipt.proof,
        )
        wrong_time = WitnessReceipt.create(
            provider=receipt.provider,
            authority=receipt.authority,
            subject_hash=receipt.subject_hash,
            reference=receipt.reference,
            claimed_at="2026-08-30T00:00:01Z",
            proof=receipt.proof,
        )
        self.assertFalse(verifier.verify(release, wrong_reference))
        self.assertFalse(verifier.verify(release, wrong_time))

    def test_wrong_lock_payload_or_scientific_revision_is_rejected(self):
        self.require_internal_lock()
        fixture = _fixture()
        verifier = InternalRepositoryLockVerifier(fixture["proof"])
        release = fixture["brier_release"]
        receipt = create_internal_repository_lock_receipt(release, fixture["proof"])
        wrong_payload = dict(receipt.proof)
        wrong_payload["lock_payload_hash"] = stable_content_hash({"wrong": "payload"})
        wrong_revision = dict(receipt.proof)
        wrong_revision["scientific_repository_revision"] = "wrong-r3-revision"
        self.assertFalse(
            verifier.verify(
                release,
                WitnessReceipt.create(
                    provider=receipt.provider,
                    authority=receipt.authority,
                    subject_hash=receipt.subject_hash,
                    reference=receipt.reference,
                    claimed_at=receipt.claimed_at,
                    proof=wrong_payload,
                ),
            )
        )
        self.assertFalse(
            verifier.verify(
                release,
                WitnessReceipt.create(
                    provider=receipt.provider,
                    authority=receipt.authority,
                    subject_hash=receipt.subject_hash,
                    reference=receipt.reference,
                    claimed_at=receipt.claimed_at,
                    proof=wrong_revision,
                ),
            )
        )

    def test_osf_receipt_does_not_verify_as_internal_lock(self):
        self.require_internal_lock()
        fixture = _fixture()
        release = fixture["brier_release"]
        osf_receipt = WitnessReceipt.create(
            provider="osf-registration",
            authority="public-registration",
            subject_hash=release.content_hash,
            reference="https://osf.io/abc12",
            claimed_at=LOCKED_AT,
            proof={"bundle_hash": fixture["bundle"].content_hash},
        )
        self.assertFalse(
            InternalRepositoryLockVerifier(fixture["proof"]).verify(
                release,
                osf_receipt,
            )
        )

    def test_internal_receipt_does_not_verify_with_osf_verifier(self):
        self.require_internal_lock()
        fixture = _fixture()
        release = fixture["brier_release"]
        receipt = create_internal_repository_lock_receipt(release, fixture["proof"])
        osf_proof = OSFRegistrationProof(
            registration_reference=receipt.reference,
            registered_at=receipt.claimed_at,
            bundle_hash=fixture["bundle"].content_hash,
        )
        self.assertFalse(OSFRegistrationVerifier(osf_proof).verify(release, receipt))

    def test_internal_verified_siblings_complete_generic_preflight_without_model_calls(self):
        self.require_internal_lock()
        fixture = _fixture()
        verifier = InternalRepositoryLockVerifier(fixture["proof"])
        brier_receipt = create_internal_repository_lock_receipt(
            fixture["brier_release"],
            fixture["proof"],
        )
        log_receipt = create_internal_repository_lock_receipt(
            fixture["log_release"],
            fixture["proof"],
        )
        brier_verified = verify_protocol_release(
            fixture["brier_release"],
            protocol=fixture["brier_protocol"],
            receipts=(brier_receipt,),
            verifier=verifier,
        )
        log_verified = verify_protocol_release(
            fixture["log_release"],
            protocol=fixture["log_protocol"],
            receipts=(log_receipt,),
            verifier=verifier,
        )
        _models, sources = runtime_models(fixture["brier_protocol"].candidates)
        preflight = preflight_external_releases(
            preregistration=fixture["preregistration"],
            evidence=fixture["evidence"],
            brier_protocol=fixture["brier_protocol"],
            brier_release=fixture["brier_release"],
            brier_verified=brier_verified,
            log_protocol=fixture["log_protocol"],
            log_release=fixture["log_release"],
            log_verified=log_verified,
        )
        self.assertTrue(preflight.content_hash.startswith("sha256:"))
        self.assertEqual(sum(source.calls for source in sources), 0)

    def test_r2_frozen_candidates_reconstruct_exactly_without_train_or_selection(self):
        self.require_internal_lock()
        sources = {
            family: source
            for family, source, _grid in study._family_sources()
        }
        frozen = {
            "reactive": (
                (("beta", 0.5),),
                "sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e",
                "sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456",
            ),
            "intentional": (
                (("beta", 2.0), ("memory_decay", 0.5)),
                "sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d",
                "sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839",
            ),
            "planning": (
                (("beta", 4.0), ("memory_decay", 0.75)),
                "sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74",
                "sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c",
            ),
        }
        for family, (parameters, selection_hash, expected_hash) in frozen.items():
            candidate = FrozenModelSpec.freeze(
                name=family,
                model=sources[family],
                parameters=parameters,
                selection_manifest_hash=selection_hash,
            )
            self.assertEqual(candidate.content_hash, expected_hash)


if __name__ == "__main__":
    unittest.main()
