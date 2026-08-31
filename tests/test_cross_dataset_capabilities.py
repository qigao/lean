from __future__ import annotations

from dataclasses import replace
import json
import pickle
import unittest

from narrative_dynamics.cross_dataset_capabilities import (
    FinalProjectionCommitment,
    FinalUnlockGrant,
    FinalVaultHandle,
    PreparedCrossDatasetTransferV1,
    SelectionProjection,
    TrainProjection,
)
from tests.cross_dataset_transfer_fixtures import (
    digest,
    final_unlock_grant,
    prepared_transfer,
)


class CrossDatasetCapabilityTests(unittest.TestCase):
    def test_prepared_surface_has_no_general_dataset_or_final_rows(self) -> None:
        prepared, _backend = prepared_transfer()
        self.assertEqual(
            set(prepared.__dataclass_fields__),
            {
                "source_identity_hash",
                "split_manifest",
                "train",
                "selection_validation",
                "final_commitment",
                "final_vault_handle",
            },
        )
        payload = json.dumps(prepared.identity_payload(), sort_keys=True).lower()
        for forbidden in (
            "participant_id",
            "participant_key",
            "final_target",
            "final_scenario",
            "vault_path",
        ):
            self.assertNotIn(forbidden, payload)
        self.assertIsInstance(prepared.train, TrainProjection)
        self.assertIsInstance(prepared.selection_validation, SelectionProjection)
        self.assertIsInstance(prepared.final_commitment, FinalProjectionCommitment)

    def test_role_projections_have_only_their_exact_private_rows(self) -> None:
        prepared, _backend = prepared_transfer()
        train_rows = prepared.train.consume_for_refit()
        selection_rows = prepared.selection_validation.consume_for_refit()
        self.assertEqual(len(train_rows), 15)
        self.assertEqual(len(selection_rows), 5)
        self.assertTrue(
            set(row.participant_key for row in train_rows).isdisjoint(
                row.participant_key for row in selection_rows
            )
        )

    def test_final_handle_is_noniterable_nonserializable_and_path_free(self) -> None:
        prepared, _backend = prepared_transfer()
        handle = prepared.final_vault_handle
        self.assertIsInstance(handle, FinalVaultHandle)
        with self.assertRaises(TypeError):
            iter(handle)
        with self.assertRaises(TypeError):
            len(handle)
        with self.assertRaises(TypeError):
            pickle.dumps(handle)
        self.assertNotIn("path", repr(handle).lower())
        self.assertFalse(hasattr(handle, "open"))
        self.assertFalse(hasattr(handle, "rows"))

    def test_vault_unlock_requires_exact_single_use_grant(self) -> None:
        prepared, backend = prepared_transfer()
        projection = backend.unlock(
            prepared.final_vault_handle,
            final_unlock_grant(),
        )
        self.assertEqual(
            projection.commitment_hash,
            prepared.final_commitment.content_hash,
        )
        self.assertTrue(backend.final_projection_opened)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            backend.unlock(prepared.final_vault_handle, final_unlock_grant())

    def test_every_unlock_identity_drift_fails_before_open(self) -> None:
        mutations = {
            "scientific_revision": "c" * 40,
            "ledger_head_hash": "e" * 40,
            "preflight_hash": digest("wrong-preflight"),
            "authorization_receipt_hash": digest("wrong-authorization"),
            "brier_release_hash": digest("wrong-brier"),
            "log_release_hash": digest("wrong-log"),
            "lock_commit": "d" * 40,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                prepared, backend = prepared_transfer()
                forged = replace(final_unlock_grant(), **{field: value})
                with self.assertRaisesRegex(ValueError, "unlock grant mismatch"):
                    backend.unlock(prepared.final_vault_handle, forged)
                self.assertFalse(backend.final_projection_opened)

    def test_forged_handle_and_policy_rebinding_fail_closed(self) -> None:
        prepared, backend = prepared_transfer()
        forged = FinalVaultHandle._for_backend(
            capability_id="f" * 64,
            commitment_hash=prepared.final_commitment.content_hash,
        )
        with self.assertRaisesRegex(ValueError, "unknown FINAL vault handle"):
            backend.unlock(forged, final_unlock_grant())
        with self.assertRaisesRegex(RuntimeError, "policy already bound"):
            backend.bind_unlock_policy(
                prepared.final_vault_handle,
                final_unlock_grant(),
            )

    def test_backend_is_not_reachable_from_prepared_or_refit_capabilities(self) -> None:
        prepared, backend = prepared_transfer()
        self.assertNotIn(backend, prepared.__dict__.values())
        for capability in (prepared.train, prepared.selection_validation):
            names = set(capability.__dict__)
            self.assertTrue(
                names.isdisjoint(
                    {"backend", "vault", "final", "final_rows", "final_handle"}
                )
            )
            self.assertFalse(hasattr(capability, "unlock"))

    def test_unlock_grant_and_final_commitment_round_trip_strictly(self) -> None:
        prepared, _backend = prepared_transfer()
        records = (
            (final_unlock_grant(), FinalUnlockGrant.from_payload),
            (prepared.final_commitment, FinalProjectionCommitment.from_payload),
        )
        for record, decoder in records:
            payload = record.to_payload()
            self.assertEqual(decoder(payload), record)
            with self.assertRaises(ValueError):
                decoder({**payload, "unknown": "field"})
            first = next(iter(payload))
            with self.assertRaises(ValueError):
                decoder({key: value for key, value in payload.items() if key != first})


if __name__ == "__main__":
    unittest.main()
