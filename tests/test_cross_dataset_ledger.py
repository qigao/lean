from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from narrative_dynamics.cross_dataset_ledger import (
    ConcurrentAttemptError,
    GitTransferAttemptStore,
    TransferAttemptEvent,
    TransferAttemptEventType,
)
from tests.cross_dataset_transfer_fixtures import (
    attempt_event,
    digest,
    failure_event,
    final_started_event,
    local_git_attempt_store,
    two_git_store_clients,
)


class CrossDatasetLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_branch_name_is_exact_and_empty_history_is_zero(self) -> None:
        store = local_git_attempt_store(self.root)
        self.assertEqual(
            store.branch_name,
            "ledger/narrative-cross-dataset-transfer-v1",
        )
        history = store.history()
        self.assertEqual(history.final_started_count, 0)
        self.assertEqual(history.final_projection_openings, 0)
        self.assertEqual(history.completed_model_runs, 0)
        self.assertEqual(history.prediction_artifact_hashes, ())
        self.assertEqual(history.score_artifact_hashes, ())

    def test_compare_and_append_rejects_stale_replacement(self) -> None:
        store = local_git_attempt_store(self.root)
        genesis = store.genesis_hash
        first = store.compare_and_append(
            store.head(),
            final_started_event(parent=store.head()),
        )
        with self.assertRaises(ConcurrentAttemptError):
            store.compare_and_append(
                genesis,
                failure_event(parent=genesis),
            )
        self.assertEqual(store.head(), first)

    def test_two_writers_cannot_both_start_final(self) -> None:
        left, right = two_git_store_clients(self.root)
        expected = left.head()
        winner = left.compare_and_append(
            expected,
            final_started_event(parent=expected),
        )
        with self.assertRaises(ConcurrentAttemptError):
            right.compare_and_append(
                expected,
                final_started_event(parent=expected),
            )
        self.assertEqual(right.head(), winner)

    def test_event_payload_is_canonical_attested_and_strict(self) -> None:
        store = local_git_attempt_store(self.root)
        event = final_started_event(parent=store.head())
        payload = event.to_payload()
        self.assertEqual(TransferAttemptEvent.from_payload(payload), event)
        self.assertEqual(
            event.event_payload_hash,
            event.recompute_event_payload_hash(),
        )
        self.assertEqual(event.attestation_hash, event.recompute_attestation_hash())
        with self.assertRaises(ValueError):
            TransferAttemptEvent.from_payload({**payload, "unknown": True})
        with self.assertRaises(ValueError):
            TransferAttemptEvent.from_payload(
                {**payload, "attestation_hash": digest("forged")}
            )

    def test_started_and_progress_are_visible_from_fresh_clone(self) -> None:
        store = local_git_attempt_store(self.root)
        started_head = store.compare_and_append(
            store.head(),
            final_started_event(parent=store.head()),
        )
        progress = attempt_event(
            TransferAttemptEventType.RUN_PROGRESS,
            parent=started_head,
            completed_model_runs=3,
        )
        progress_head = store.compare_and_append(started_head, progress)
        fresh = local_git_attempt_store(self.root)
        self.assertEqual(fresh.head(), progress_head)
        history = fresh.history()
        self.assertEqual(history.final_started_count, 1)
        self.assertEqual(history.completed_model_runs, 3)

    def test_vault_prediction_score_and_terminal_evidence_persist(self) -> None:
        store = local_git_attempt_store(self.root)
        head = store.compare_and_append(
            store.head(),
            final_started_event(parent=store.head()),
        )
        sequence = (
            (
                TransferAttemptEventType.FINAL_VAULT_OPENED,
                {"final_commitment_hash": digest("final-commitment")},
            ),
            (
                TransferAttemptEventType.PREDICTION_SEALED,
                {"prediction_artifact_hash": digest("prediction")},
            ),
            (
                TransferAttemptEventType.SCORE_SEALED,
                {"score_artifact_hash": digest("score")},
            ),
            (
                TransferAttemptEventType.FINAL_COMPLETED,
                {"report_hash": digest("report")},
            ),
        )
        for event_type, details in sequence:
            event = attempt_event(event_type, parent=head, **details)
            head = store.compare_and_append(head, event)
        history = local_git_attempt_store(self.root).history()
        self.assertEqual(history.final_projection_openings, 1)
        self.assertEqual(
            history.prediction_artifact_hashes,
            (digest("prediction"),),
        )
        self.assertEqual(history.score_artifact_hashes, (digest("score"),))
        self.assertEqual(history.terminal_count, 1)
        self.assertEqual(
            history.last_event.event_type,
            TransferAttemptEventType.FINAL_COMPLETED,
        )

    def test_duplicate_started_and_second_terminal_fail_closed(self) -> None:
        store = local_git_attempt_store(self.root)
        head = store.compare_and_append(
            store.head(),
            final_started_event(parent=store.head()),
        )
        with self.assertRaisesRegex(ValueError, "duplicate FINAL_STARTED"):
            store.compare_and_append(head, final_started_event(parent=head))
        terminal = failure_event(parent=head)
        terminal_head = store.compare_and_append(head, terminal)
        with self.assertRaisesRegex(ValueError, "terminal"):
            store.compare_and_append(
                terminal_head,
                attempt_event(
                    TransferAttemptEventType.EXACT_REPLAY_ALLOWED,
                    parent=terminal_head,
                    failure_class="INFRASTRUCTURE",
                ),
            )

    def test_corrupt_remote_branch_history_is_rejected(self) -> None:
        store = local_git_attempt_store(self.root)
        workspace = store._workspace
        corrupt = workspace / "events" / "corrupt.json"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_text(json.dumps({"invalid": True}), encoding="utf-8")
        subprocess.run(
            ("git", "add", "events/corrupt.json"),
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ("git", "commit", "-m", "corrupt ledger event"),
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            env=store._git_environment(),
        )
        subprocess.run(
            (
                "git",
                "push",
                "origin",
                f"HEAD:refs/heads/{store.branch_name}",
            ),
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        fresh = GitTransferAttemptStore(
            remote=self.root / "attempt-ledger.git",
            workspace=self.root / "corrupt-reader",
        )
        with self.assertRaisesRegex(ValueError, "corrupt ledger"):
            fresh.history()

    def test_parent_chain_mismatch_is_rejected_before_write(self) -> None:
        store = local_git_attempt_store(self.root)
        with self.assertRaisesRegex(ValueError, "parent"):
            store.compare_and_append(
                store.head(),
                final_started_event(parent="f" * 40),
            )


if __name__ == "__main__":
    unittest.main()
