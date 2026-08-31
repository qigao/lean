from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.studies.cross_dataset_transfer_locked_final import (
    TransferFinalInfrastructureError,
    run_locked_cross_dataset_transfer_final,
)
from tests.cross_dataset_transfer_fixtures import (
    final_started_event,
    locked_final_inputs,
)


class CrossDatasetLockedFinalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_started_is_persisted_before_vault_unlock(self) -> None:
        order: list[str] = []
        result = run_locked_cross_dataset_transfer_final(
            **locked_final_inputs(self.root / "success", order=order)
        )
        self.assertLess(
            order.index("ledger:FINAL_STARTED"),
            order.index("vault:open"),
        )
        self.assertEqual(result.terminal_event, "FINAL_COMPLETED")
        self.assertEqual(result.report.terminal.value, "GREEN")

    def test_crash_after_vault_open_requires_revision(self) -> None:
        inputs = locked_final_inputs(
            self.root / "after-open",
            fail_at="after_vault_open",
        )
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        history = inputs["store"].history()
        self.assertEqual(history.last_event.event_type.value, "REVISION_REQUIRED")
        self.assertEqual(history.final_projection_openings, 1)

    def test_preunlock_infrastructure_failure_can_only_record_replay_allowed(self) -> None:
        inputs = locked_final_inputs(
            self.root / "before-open",
            fail_at="before_vault_open",
        )
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        history = inputs["store"].history()
        self.assertEqual(history.last_event.event_type.value, "EXACT_REPLAY_ALLOWED")
        self.assertEqual(history.final_projection_openings, 0)
        self.assertEqual(history.completed_model_runs, 0)
        self.assertEqual(history.prediction_artifact_hashes, ())

    def test_preunlock_schema_failure_requires_revision(self) -> None:
        inputs = locked_final_inputs(
            self.root / "schema",
            fail_at="before_vault_open",
            infrastructure_failure=False,
        )
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        self.assertEqual(
            inputs["store"].history().last_event.event_type.value,
            "REVISION_REQUIRED",
        )

    def test_failure_before_started_creates_no_attempt(self) -> None:
        inputs = locked_final_inputs(
            self.root / "before-started",
            fail_at="before_started",
        )
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        self.assertEqual(inputs["store"].history().events, ())

    def test_every_post_unlock_failure_is_revision_required(self) -> None:
        stages = (
            "during_unlock",
            "after_first_model_run",
            "during_prediction_seal",
            "during_brier_scoring",
            "during_log_scoring",
            "during_report_assembly",
            "during_final_completed_append",
        )
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                inputs = locked_final_inputs(
                    self.root / f"post-{index}",
                    fail_at=stage,
                )
                with self.assertRaises(TransferFinalInfrastructureError):
                    run_locked_cross_dataset_transfer_final(**inputs)
                history = inputs["store"].history()
                self.assertEqual(
                    history.last_event.event_type.value,
                    "REVISION_REQUIRED",
                )
                self.assertTrue(inputs["evidence_sink"].snapshots)

    def test_always_run_evidence_failure_prevents_success_and_requires_revision(self) -> None:
        inputs = locked_final_inputs(
            self.root / "evidence-failure",
            evidence_failure=True,
        )
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        history = inputs["store"].history()
        self.assertEqual(history.last_event.event_type.value, "REVISION_REQUIRED")

    def test_authorization_cannot_be_reused(self) -> None:
        inputs = locked_final_inputs(self.root / "reuse")
        run_locked_cross_dataset_transfer_final(**inputs)
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        self.assertEqual(inputs["store"].history().final_started_count, 1)

    def test_stale_preflight_or_concurrent_started_fails_before_vault(self) -> None:
        inputs = locked_final_inputs(self.root / "stale")
        store = inputs["store"]
        store.compare_and_append(
            store.head(),
            final_started_event(parent=store.head()),
        )
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        self.assertFalse(inputs["vault_backend"].final_projection_opened)

    def test_replay_allowed_does_not_trigger_automatic_retry(self) -> None:
        inputs = locked_final_inputs(
            self.root / "no-auto-retry",
            fail_at="before_vault_open",
        )
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        first_count = inputs["store"].history().final_started_count
        with self.assertRaises(TransferFinalInfrastructureError):
            run_locked_cross_dataset_transfer_final(**inputs)
        self.assertEqual(inputs["store"].history().final_started_count, first_count)


if __name__ == "__main__":
    unittest.main()
