from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.cross_dataset_ledger import (
    TransferAttemptEventType,
)
from narrative_dynamics.cross_dataset_release import (
    CARRY_FORWARD_REQUIREMENT_IDS,
    LIMITATIONS,
    TOP_LEVEL_TERMINALS,
    TransferScore,
    TransferScoreRelease,
    preflight_transfer_releases,
)
from tests.cross_dataset_transfer_fixtures import (
    attempt_event,
    digest,
    final_started_event,
    local_git_attempt_store,
    sibling_releases,
)


class CrossDatasetReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_sibling_releases_differ_only_by_score_identity(self) -> None:
        brier, log = sibling_releases()
        self.assertIs(brier.score, TransferScore.BRIER)
        self.assertIs(log.score, TransferScore.LOG)
        self.assertEqual(brier.shared_identity_payload(), log.shared_identity_payload())
        self.assertEqual(
            brier.prediction_artifact_identity,
            log.prediction_artifact_identity,
        )
        self.assertNotEqual(brier.content_hash, log.content_hash)

    def test_protocol_freezes_all_governance_vocabularies(self) -> None:
        protocol = sibling_releases()[0].protocol
        self.assertEqual(protocol.train_seeds, (101, 102))
        self.assertEqual(protocol.selection_seeds, (201, 202))
        self.assertEqual(protocol.final_seeds, (301, 302))
        self.assertEqual(protocol.aggregations, ("PARTICIPANT_EQUAL", "TRIAL_EQUAL"))
        self.assertEqual(protocol.bootstrap_seed, 43001)
        self.assertEqual(protocol.bootstrap_replicates, 10000)
        self.assertEqual(protocol.relative_improvement_threshold, 0.01)
        self.assertEqual(protocol.family_vocabulary, ("reactive", "intentional", "planning"))
        self.assertEqual(protocol.top_level_terminals, TOP_LEVEL_TERMINALS)
        self.assertEqual(protocol.limitations, LIMITATIONS)
        self.assertEqual(protocol.carry_forward_requirement_ids, CARRY_FORWARD_REQUIREMENT_IDS)
        self.assertFalse(protocol.external_registration)

    def test_preflight_binds_current_authoritative_head_and_zero_final_history(self) -> None:
        store = local_git_attempt_store(self.root)
        brier, log = sibling_releases()
        preflight = preflight_transfer_releases(
            brier,
            log,
            store=store,
            completed_at_utc="2026-08-30T13:00:00Z",
        )
        self.assertEqual(preflight.ledger_head_hash, store.head())
        self.assertEqual(preflight.final_started_count, 0)
        self.assertEqual(preflight.final_projection_openings, 0)
        self.assertEqual(preflight.completed_model_runs, 0)
        self.assertEqual(preflight.prediction_artifact_hashes, ())

    def test_shared_identity_drift_is_rejected(self) -> None:
        brier, log = sibling_releases()
        fields = (
            "source_identity_hash",
            "transform_identity_hash",
            "split_manifest_hash",
            "baseline_hash",
            "final_commitment_hash",
            "prediction_artifact_identity",
        )
        for field in fields:
            with self.subTest(field=field):
                drifted_protocol = replace(
                    brier.protocol,
                    **{field: digest(f"wrong-{field}")},
                )
                drifted = TransferScoreRelease.create(
                    drifted_protocol,
                    release_receipt_hash=digest(f"wrong-{field}-receipt"),
                )
                with self.assertRaisesRegex(ValueError, "sibling"):
                    preflight_transfer_releases(
                        drifted,
                        log,
                        store=local_git_attempt_store(self.root / field),
                        completed_at_utc="2026-08-30T13:00:00Z",
                    )

        candidates = list(brier.protocol.candidate_hashes)
        candidates[-1] = digest("wrong-sixth-candidate")
        drifted = TransferScoreRelease.create(
            replace(brier.protocol, candidate_hashes=tuple(candidates)),
            release_receipt_hash=digest("wrong-candidates-receipt"),
        )
        with self.assertRaisesRegex(ValueError, "sibling"):
            preflight_transfer_releases(
                drifted,
                log,
                store=local_git_attempt_store(self.root / "candidates"),
                completed_at_utc="2026-08-30T13:00:00Z",
            )

    def test_exact_protocol_fields_reject_drift_at_construction(self) -> None:
        protocol = sibling_releases()[0].protocol
        mutations = (
            ("train_seeds", (101,)),
            ("selection_seeds", (201,)),
            ("final_seeds", (301,)),
            ("aggregations", ("PARTICIPANT_EQUAL",)),
            ("bootstrap_seed", 1),
            ("bootstrap_replicates", 9999),
            ("relative_improvement_threshold", 0.02),
            ("family_vocabulary", ("reactive", "planning")),
            ("top_level_terminals", ("GREEN",)),
            ("limitations", LIMITATIONS[:-1]),
            ("external_registration", True),
            ("carry_forward_requirement_ids", CARRY_FORWARD_REQUIREMENT_IDS[:-1]),
            ("scientific_revision", "b" * 40),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(protocol, **{field: value})

    def test_preflight_rejects_any_prior_final_activity(self) -> None:
        event_sequences = (
            (),
            ((TransferAttemptEventType.FINAL_VAULT_OPENED, {"final_commitment_hash": digest("final")}),),
            ((TransferAttemptEventType.RUN_PROGRESS, {"completed_model_runs": 1}),),
            ((TransferAttemptEventType.PREDICTION_SEALED, {"prediction_artifact_hash": digest("prediction")}),),
        )
        for index, tail in enumerate(event_sequences):
            with self.subTest(index=index):
                store = local_git_attempt_store(self.root / f"activity-{index}")
                head = store.compare_and_append(
                    store.head(),
                    final_started_event(parent=store.head()),
                )
                for event_type, details in tail:
                    head = store.compare_and_append(
                        head,
                        attempt_event(event_type, parent=head, **details),
                    )
                with self.assertRaisesRegex(ValueError, "zero-FINAL"):
                    preflight_transfer_releases(
                        *sibling_releases(),
                        store=store,
                        completed_at_utc="2026-08-30T13:00:00Z",
                    )

    def test_in_memory_or_stale_caller_ledger_is_rejected(self) -> None:
        class MemoryStore:
            def head(self):
                return "a" * 40

            def history(self):
                raise AssertionError("must reject before trusting caller history")

        with self.assertRaisesRegex(TypeError, "GitTransferAttemptStore"):
            preflight_transfer_releases(
                *sibling_releases(),
                store=MemoryStore(),
                completed_at_utc="2026-08-30T13:00:00Z",
            )

        real = local_git_attempt_store(self.root / "stale")

        class StaleStore(type(real)):
            def head(self):
                return "f" * 40

        stale = object.__new__(StaleStore)
        stale.__dict__.update(real.__dict__)
        with self.assertRaisesRegex(ValueError, "current head"):
            preflight_transfer_releases(
                *sibling_releases(),
                store=stale,
                completed_at_utc="2026-08-30T13:00:00Z",
            )

    def test_release_payload_round_trip_is_strict(self) -> None:
        release = sibling_releases()[0]
        payload = release.to_payload()
        self.assertEqual(TransferScoreRelease.from_payload(payload), release)
        with self.assertRaises(ValueError):
            TransferScoreRelease.from_payload({**payload, "unknown": True})


if __name__ == "__main__":
    unittest.main()
