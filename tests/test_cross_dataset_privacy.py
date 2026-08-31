from __future__ import annotations

import json
import pickle
import unittest

from narrative_dynamics.cross_dataset_privacy import (
    ParticipantRole,
    PrivateParticipant,
    PublicSplitManifest,
    RestrictedStudySecret,
    assign_participant_roles,
)
from tests.cross_dataset_transfer_fixtures import (
    assigned_split,
    digest,
    synthetic_stratified_participants,
)


class CrossDatasetPrivacyTests(unittest.TestCase):
    def test_split_is_stratified_deterministic_disjoint_and_exact(self) -> None:
        secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
        private_index, public_manifest = assign_participant_roles(
            namespace="cross-dataset-transfer-v1",
            source_snapshot_hash=digest("snapshot"),
            participants=synthetic_stratified_participants((10, 15)),
            secret=secret,
            transform_attestation_hash=digest("transform-attestation"),
        )

        self.assertEqual(
            private_index.role_counts,
            (("s0", 6, 2, 2), ("s1", 9, 3, 3)),
        )
        self.assertFalse(private_index.has_overlap)
        self.assertEqual(public_manifest.role_counts, private_index.role_counts)
        self.assertEqual(
            private_index.role_total(ParticipantRole.TRAIN),
            15,
        )

    def test_same_secret_is_input_order_invariant(self) -> None:
        participants = synthetic_stratified_participants((10, 15))
        left = assign_participant_roles(
            namespace="cross-dataset-transfer-v1",
            source_snapshot_hash=digest("snapshot"),
            participants=participants,
            secret=RestrictedStudySecret.from_bytes(bytes(range(32))),
            transform_attestation_hash=digest("transform-attestation"),
        )
        right = assign_participant_roles(
            namespace="cross-dataset-transfer-v1",
            source_snapshot_hash=digest("snapshot"),
            participants=tuple(reversed(participants)),
            secret=RestrictedStudySecret.from_bytes(bytes(range(32))),
            transform_attestation_hash=digest("transform-attestation"),
        )
        self.assertEqual(left[0].assignment_hash, right[0].assignment_hash)
        self.assertEqual(left[1], right[1])

    def test_public_manifest_is_aggregate_only_and_non_enumerable(self) -> None:
        private_index, manifest = assigned_split()
        payload = manifest.to_payload()
        serialized = json.dumps(payload, sort_keys=True).lower()
        for forbidden in (
            "participant_id",
            "participant_token",
            "private-s0",
            "hmac_output",
            "commitment_rows",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(payload["hmac_algorithm"], "HMAC-SHA256")
        self.assertTrue(manifest.merkle_root.startswith("sha256:"))
        self.assertTrue(manifest.key_commitment.startswith("sha256:"))
        self.assertFalse(hasattr(manifest, "participant_commitments"))
        self.assertFalse(hasattr(private_index, "to_payload"))

    def test_secret_close_zeroizes_and_blocks_reuse(self) -> None:
        secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
        before = secret.commit("p-1", "s0")
        self.assertEqual(len(before), 32)
        secret.close()
        self.assertTrue(secret.closed)
        with self.assertRaisesRegex(RuntimeError, "secret is closed"):
            secret.commit("p-1", "s0")

    def test_secret_is_exact_length_redacted_and_nonserializable(self) -> None:
        for payload in (b"", bytes(range(31)), bytes(range(32)) + b"x"):
            with self.subTest(length=len(payload)):
                with self.assertRaises(ValueError):
                    RestrictedStudySecret.from_bytes(payload)
        secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
        self.assertNotIn(bytes(range(32)).hex(), repr(secret))
        self.assertFalse(hasattr(secret, "raw_key"))
        self.assertFalse(hasattr(secret, "nonce"))
        with self.assertRaises(TypeError):
            pickle.dumps(secret)

    def test_duplicate_cross_stratum_and_empty_role_inputs_fail_closed(self) -> None:
        duplicate = PrivateParticipant("private-duplicate", "s0")
        cases = (
            (duplicate, duplicate),
            (
                PrivateParticipant("private-cross", "s0"),
                PrivateParticipant("private-cross", "s1"),
            ),
            synthetic_stratified_participants((4,)),
        )
        for participants in cases:
            with self.subTest(count=len(participants)):
                with self.assertRaises(ValueError):
                    assign_participant_roles(
                        namespace="cross-dataset-transfer-v1",
                        source_snapshot_hash=digest("snapshot"),
                        participants=participants,
                        secret=RestrictedStudySecret.from_bytes(bytes(range(32))),
                        transform_attestation_hash=digest("transform-attestation"),
                    )

    def test_private_participant_rejects_bool_empty_or_surrounding_space(self) -> None:
        for participant_id, stratum in (
            (True, "s0"),
            ("", "s0"),
            ("private-p1 ", "s0"),
            ("private-p1", ""),
        ):
            with self.subTest(participant_id=participant_id, stratum=stratum):
                with self.assertRaises((TypeError, ValueError)):
                    PrivateParticipant(participant_id, stratum)

    def test_public_manifest_round_trip_is_strict(self) -> None:
        _, manifest = assigned_split()
        payload = manifest.to_payload()
        self.assertEqual(PublicSplitManifest.from_payload(payload), manifest)
        with self.assertRaises(ValueError):
            PublicSplitManifest.from_payload({**payload, "unknown": "field"})
        first = next(iter(payload))
        with self.assertRaises(ValueError):
            PublicSplitManifest.from_payload(
                {key: value for key, value in payload.items() if key != first}
            )


if __name__ == "__main__":
    unittest.main()
