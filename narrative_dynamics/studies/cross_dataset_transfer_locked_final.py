"""Locked exactly-once FINAL state machine for cross-dataset transfer V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_authorization import (
    TransferAuthorizationReceipt,
)
from narrative_dynamics.cross_dataset_capabilities import (
    FinalUnlockGrant,
    FinalVaultHandle,
)
from narrative_dynamics.cross_dataset_candidates import (
    RefitCandidateFreeze,
    ZeroShotCandidateFreeze,
)
from narrative_dynamics.cross_dataset_inference import FrozenBaseRateComparator
from narrative_dynamics.cross_dataset_ledger import (
    GitTransferAttemptStore,
    TransferAttemptEvent,
    TransferAttemptEventType,
)
from narrative_dynamics.cross_dataset_prediction import (
    FINAL_SEEDS,
    TransferEvaluator,
    TransferRunProgress,
    execute_transfer_final_predictions,
)
from narrative_dynamics.cross_dataset_release import (
    DualTransferPreflight,
    TransferScore,
    TransferScoreRelease,
)
from narrative_dynamics.cross_dataset_reporting import (
    CarryForwardSensitivityStatus,
    CrossDatasetTransferReport,
    score_and_report_transfer,
)


class TransferFinalInfrastructureError(RuntimeError):
    """Raised when locked FINAL cannot return a valid completed result."""


class _EvidenceSink(Protocol):
    def persist_aggregate_state(self, history: object) -> None:
        ...


@dataclass(frozen=True)
class LockedCrossDatasetTransferResult:
    report: CrossDatasetTransferReport
    terminal_event: str
    ledger_head_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.report, CrossDatasetTransferReport):
            raise TypeError("locked FINAL result requires CrossDatasetTransferReport")
        if self.terminal_event != "FINAL_COMPLETED":
            raise ValueError("successful locked FINAL result must be FINAL_COMPLETED")


def _event(
    *,
    event_type: TransferAttemptEventType,
    parent: str,
    preflight: DualTransferPreflight,
    authorization: TransferAuthorizationReceipt,
    details: dict[str, object],
) -> TransferAttemptEvent:
    attempt_id = f"authorization-{authorization.comment_id}"
    return TransferAttemptEvent.create(
        event_type=event_type,
        parent_ledger_head=parent,
        scientific_revision=preflight.scientific_revision,
        attempt_id=attempt_id,
        timestamp_utc="2026-08-30T15:00:00Z",
        repository_receipt_hash=stable_content_hash(
            {
                "scientific_revision": preflight.scientific_revision,
                "preflight_hash": preflight.content_hash,
            }
        ),
        run_receipt_hash=stable_content_hash(
            {
                "authorization_receipt_hash": authorization.content_hash,
                "attempt_id": attempt_id,
            }
        ),
        job_receipt_hash=stable_content_hash(
            {
                "ledger_parent": parent,
                "event_type": event_type.value,
            }
        ),
        details=details,
    )


def _verify_locked_inputs(
    *,
    store: object,
    preflight: object,
    authorization: object,
    vault_handle: object,
    zero_shot: object,
    refit: object,
    baseline: object,
    brier_release: object,
    log_release: object,
) -> None:
    if not isinstance(store, GitTransferAttemptStore):
        raise TypeError("locked FINAL requires GitTransferAttemptStore")
    if not isinstance(preflight, DualTransferPreflight):
        raise TypeError("locked FINAL requires DualTransferPreflight")
    if not isinstance(authorization, TransferAuthorizationReceipt):
        raise TypeError("locked FINAL requires TransferAuthorizationReceipt")
    if not isinstance(vault_handle, FinalVaultHandle):
        raise TypeError("locked FINAL requires FinalVaultHandle")
    if not isinstance(zero_shot, ZeroShotCandidateFreeze) or not isinstance(
        refit,
        RefitCandidateFreeze,
    ):
        raise TypeError("locked FINAL requires both candidate freezes")
    if not isinstance(baseline, FrozenBaseRateComparator):
        raise TypeError("locked FINAL requires frozen TRAIN baseline")
    if (
        not isinstance(brier_release, TransferScoreRelease)
        or not isinstance(log_release, TransferScoreRelease)
        or brier_release.score is not TransferScore.BRIER
        or log_release.score is not TransferScore.LOG
        or brier_release.shared_identity_payload()
        != log_release.shared_identity_payload()
    ):
        raise ValueError("locked FINAL requires exact Brier/Log sibling releases")
    if (
        preflight.brier_release_hash != brier_release.content_hash
        or preflight.log_release_hash != log_release.content_hash
        or preflight.scientific_revision
        != brier_release.protocol.scientific_revision
        or authorization.preflight_hash != preflight.content_hash
        or authorization.brier_release_hash != preflight.brier_release_hash
        or authorization.log_release_hash != preflight.log_release_hash
        or authorization.scientific_revision != preflight.scientific_revision
        or authorization.pre_authorization_ledger_head
        != preflight.ledger_head_hash
    ):
        raise ValueError("locked FINAL release/preflight/authorization identity drift")
    if (
        brier_release.protocol.baseline_hash != baseline.content_hash
        or brier_release.protocol.final_commitment_hash
        != vault_handle.commitment_hash
    ):
        raise ValueError("locked FINAL baseline or FINAL commitment drift")
    candidate_hashes = tuple(
        row.content_hash
        for row in zero_shot.candidates + refit.selected_candidates
    )
    if brier_release.protocol.candidate_hashes != candidate_hashes:
        raise ValueError("locked FINAL candidate freeze drift")
    head = store.head()
    history = store.history()
    if (
        head != preflight.ledger_head_hash
        or history.head_hash != head
        or history.events
    ):
        raise ValueError("locked FINAL preflight is stale or authorization was consumed")


def _append_failure(
    *,
    store: GitTransferAttemptStore,
    preflight: DualTransferPreflight,
    authorization: TransferAuthorizationReceipt,
    cause: BaseException,
    unlock_attempted: bool,
) -> None:
    history = store.history()
    if not history.events:
        return
    already_terminal = history.terminal_count > 0
    if already_terminal:
        return
    independently_safe = (
        not unlock_attempted
        and isinstance(cause, TransferFinalInfrastructureError)
        and history.final_projection_openings == 0
        and history.completed_model_runs == 0
        and not history.prediction_artifact_hashes
        and not history.score_artifact_hashes
    )
    event_type = (
        TransferAttemptEventType.EXACT_REPLAY_ALLOWED
        if independently_safe
        else TransferAttemptEventType.REVISION_REQUIRED
    )
    failure_class = "INFRASTRUCTURE" if independently_safe else "LOCKED_FINAL_FAILURE"
    current = store.head()
    store.compare_and_append(
        current,
        _event(
            event_type=event_type,
            parent=current,
            preflight=preflight,
            authorization=authorization,
            details={"failure_class": failure_class},
        ),
    )


def run_locked_cross_dataset_transfer_final(
    *,
    store: GitTransferAttemptStore,
    preflight: DualTransferPreflight,
    authorization: TransferAuthorizationReceipt,
    vault_backend: object,
    vault_handle: FinalVaultHandle,
    zero_shot: ZeroShotCandidateFreeze,
    refit: RefitCandidateFreeze,
    baseline: FrozenBaseRateComparator,
    brier_release: TransferScoreRelease,
    log_release: TransferScoreRelease,
    evaluator: TransferEvaluator,
    semantic_invariance_receipt_hash: str,
    carry_forward_statuses: tuple[
        tuple[str, CarryForwardSensitivityStatus], ...
    ],
    task_condition_strata: tuple[str, ...],
    participant_influence_hash: str,
    evidence_sink: _EvidenceSink,
    stage_hook: Callable[[str], None] = lambda _stage: None,
) -> LockedCrossDatasetTransferResult:
    started = False
    unlock_attempted = False
    evidence_persisted = False
    result: LockedCrossDatasetTransferResult | None = None
    active_cause: BaseException | None = None
    try:
        stage_hook("before_started")
        _verify_locked_inputs(
            store=store,
            preflight=preflight,
            authorization=authorization,
            vault_handle=vault_handle,
            zero_shot=zero_shot,
            refit=refit,
            baseline=baseline,
            brier_release=brier_release,
            log_release=log_release,
        )
        head = store.compare_and_append(
            preflight.ledger_head_hash,
            _event(
                event_type=TransferAttemptEventType.FINAL_STARTED,
                parent=preflight.ledger_head_hash,
                preflight=preflight,
                authorization=authorization,
                details={
                    "authorization_receipt_hash": authorization.content_hash,
                },
            ),
        )
        started = True
        stage_hook("ledger:FINAL_STARTED")
        stage_hook("before_vault_open")
        unlock_attempted = True
        stage_hook("during_unlock")
        stage_hook("vault:open")
        projection = vault_backend.unlock(
            vault_handle,
            FinalUnlockGrant(
                scientific_revision=preflight.scientific_revision,
                ledger_head_hash=preflight.ledger_head_hash,
                preflight_hash=preflight.content_hash,
                authorization_receipt_hash=authorization.content_hash,
                brier_release_hash=preflight.brier_release_hash,
                log_release_hash=preflight.log_release_hash,
                lock_commit=authorization.lock_commit,
            ),
        )
        head = store.compare_and_append(
            head,
            _event(
                event_type=TransferAttemptEventType.FINAL_VAULT_OPENED,
                parent=head,
                preflight=preflight,
                authorization=authorization,
                details={
                    "final_commitment_hash": vault_handle.commitment_hash,
                },
            ),
        )
        stage_hook("after_vault_open")

        def persist_progress(row: TransferRunProgress) -> None:
            nonlocal head
            head = store.compare_and_append(
                head,
                _event(
                    event_type=TransferAttemptEventType.RUN_PROGRESS,
                    parent=head,
                    preflight=preflight,
                    authorization=authorization,
                    details={
                        "completed_model_runs": row.completed_model_runs,
                    },
                ),
            )
            if row.completed_model_runs == 1:
                stage_hook("after_first_model_run")

        artifact = execute_transfer_final_predictions(
            projection=projection,
            zero_shot=zero_shot,
            refit=refit,
            seeds=FINAL_SEEDS,
            evaluator=evaluator,
            progress=persist_progress,
            expected_final_commitment_hash=vault_handle.commitment_hash,
            prediction_artifact_identity=(
                brier_release.prediction_artifact_identity
            ),
        )
        stage_hook("during_prediction_seal")
        head = store.compare_and_append(
            head,
            _event(
                event_type=TransferAttemptEventType.PREDICTION_SEALED,
                parent=head,
                preflight=preflight,
                authorization=authorization,
                details={
                    "prediction_artifact_hash": artifact.sealed_artifact_hash,
                },
            ),
        )
        stage_hook("during_brier_scoring")
        stage_hook("during_log_scoring")
        stage_hook("during_report_assembly")
        report = score_and_report_transfer(
            artifact=artifact,
            baseline=baseline,
            brier_release=brier_release,
            log_release=log_release,
            semantic_invariance_receipt_hash=(
                semantic_invariance_receipt_hash
            ),
            semantic_invariance_pass=True,
            carry_forward_statuses=carry_forward_statuses,
            task_condition_strata=task_condition_strata,
            participant_influence_hash=participant_influence_hash,
        )
        head = store.compare_and_append(
            head,
            _event(
                event_type=TransferAttemptEventType.SCORE_SEALED,
                parent=head,
                preflight=preflight,
                authorization=authorization,
                details={"score_artifact_hash": report.content_hash},
            ),
        )
        evidence_sink.persist_aggregate_state(store.history())
        evidence_persisted = True
        stage_hook("during_final_completed_append")
        head = store.compare_and_append(
            head,
            _event(
                event_type=TransferAttemptEventType.FINAL_COMPLETED,
                parent=head,
                preflight=preflight,
                authorization=authorization,
                details={"report_hash": report.content_hash},
            ),
        )
        result = LockedCrossDatasetTransferResult(
            report=report,
            terminal_event="FINAL_COMPLETED",
            ledger_head_hash=head,
        )
    except BaseException as exc:
        active_cause = exc
        if started:
            try:
                _append_failure(
                    store=store,
                    preflight=preflight,
                    authorization=authorization,
                    cause=exc,
                    unlock_attempted=unlock_attempted,
                )
            except BaseException as persistence_error:
                active_cause = persistence_error
    finally:
        if started and not evidence_persisted:
            try:
                evidence_sink.persist_aggregate_state(store.history())
                evidence_persisted = True
            except BaseException as evidence_error:
                if active_cause is None:
                    active_cause = evidence_error
                    try:
                        _append_failure(
                            store=store,
                            preflight=preflight,
                            authorization=authorization,
                            cause=evidence_error,
                            unlock_attempted=unlock_attempted,
                        )
                    except BaseException as persistence_error:
                        active_cause = persistence_error
    if active_cause is not None or result is None:
        raise TransferFinalInfrastructureError(
            "locked FINAL did not complete"
        ) from active_cause
    return result
