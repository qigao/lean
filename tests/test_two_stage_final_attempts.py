from __future__ import annotations

import unittest

from narrative_dynamics.contracts import stable_content_hash

_ATTEMPT_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.studies.feher_hare_two_stage_v1 import (
        TwoStageFinalAttemptLedger,
        TwoStageFinalAttemptStatus,
        complete_final_attempt,
        mark_infrastructure_failed,
        mark_revision_required,
        require_exact_retry,
        start_final_attempt,
    )
except Exception as error:
    _ATTEMPT_IMPORT_ERROR = error


def identities():
    return {
        "preregistration_hash": stable_content_hash({"prereg": 1}),
        "preflight_hash": stable_content_hash({"preflight": 1}),
        "repository_revision": "repo-sha",
        "dataset_hash": stable_content_hash({"dataset": 1}),
        "final_target_hash": stable_content_hash({"target": 1}),
        "frozen_candidate_hashes": (
            stable_content_hash({"candidate": "reactive"}),
            stable_content_hash({"candidate": "intentional"}),
            stable_content_hash({"candidate": "planning"}),
        ),
        "brier_release_hash": stable_content_hash({"release": "brier"}),
        "log_release_hash": stable_content_hash({"release": "log"}),
    }


class TwoStageFinalAttemptTests(unittest.TestCase):
    def require_attempts(self):
        self.assertIsNone(
            _ATTEMPT_IMPORT_ERROR,
            f"FINAL attempt lineage is missing: {_ATTEMPT_IMPORT_ERROR}",
        )

    def test_preflight_failure_creates_no_started_attempt(self):
        self.require_attempts()
        ledger = TwoStageFinalAttemptLedger(())
        self.assertEqual(ledger.attempts, ())

    def test_first_model_execution_starts_attempt(self):
        self.require_attempts()
        ledger = start_final_attempt(
            TwoStageFinalAttemptLedger(()),
            attempt_id="attempt-1",
            started_at="2026-08-29T00:00:00Z",
            **identities(),
        )
        self.assertEqual(ledger.attempts[-1].status, TwoStageFinalAttemptStatus.STARTED)

    def test_infrastructure_failure_is_append_only_and_exact_retry_requires_all_scientific_identities(self):
        self.require_attempts()
        ledger = start_final_attempt(
            TwoStageFinalAttemptLedger(()),
            attempt_id="attempt-1",
            started_at="2026-08-29T00:00:00Z",
            **identities(),
        )
        failed = mark_infrastructure_failed(
            ledger,
            attempt_id="attempt-1",
            failure_class="executor_cancelled",
        )
        self.assertEqual(len(failed.attempts), 2)
        self.assertEqual(failed.attempts[-1].status, TwoStageFinalAttemptStatus.INFRASTRUCTURE_FAILED)
        require_exact_retry(failed, **identities())
        changed = dict(identities())
        changed["dataset_hash"] = stable_content_hash({"dataset": 2})
        with self.assertRaises(ValueError):
            require_exact_retry(failed, **changed)

    def test_scientific_failure_marks_revision_required(self):
        self.require_attempts()
        ledger = start_final_attempt(
            TwoStageFinalAttemptLedger(()),
            attempt_id="attempt-1",
            started_at="2026-08-29T00:00:00Z",
            **identities(),
        )
        revised = mark_revision_required(
            ledger,
            attempt_id="attempt-1",
            failure_class="transform_identity_drift",
        )
        self.assertEqual(revised.attempts[-1].status, TwoStageFinalAttemptStatus.REVISION_REQUIRED)

    def test_completed_attempt_cannot_be_overwritten_or_reopened(self):
        self.require_attempts()
        ledger = start_final_attempt(
            TwoStageFinalAttemptLedger(()),
            attempt_id="attempt-1",
            started_at="2026-08-29T00:00:00Z",
            **identities(),
        )
        completed = complete_final_attempt(
            ledger,
            attempt_id="attempt-1",
            completed_run_manifest_hashes=(stable_content_hash({"run": 1}),),
            result_hash=stable_content_hash({"result": 1}),
        )
        with self.assertRaises(ValueError):
            complete_final_attempt(
                completed,
                attempt_id="attempt-1",
                completed_run_manifest_hashes=(stable_content_hash({"run": 2}),),
                result_hash=stable_content_hash({"result": 2}),
            )

    def test_disappointing_result_is_completed_not_revision_required(self):
        self.require_attempts()
        ledger = start_final_attempt(
            TwoStageFinalAttemptLedger(()),
            attempt_id="attempt-1",
            started_at="2026-08-29T00:00:00Z",
            **identities(),
        )
        completed = complete_final_attempt(
            ledger,
            attempt_id="attempt-1",
            completed_run_manifest_hashes=(stable_content_hash({"run": "poor-fit"}),),
            result_hash=stable_content_hash({"predictive_adequacy": "not_met"}),
        )
        self.assertEqual(completed.attempts[-1].status, TwoStageFinalAttemptStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
