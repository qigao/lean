from __future__ import annotations

from dataclasses import dataclass

from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_first_stage_policy_metrics,
    two_stage_log_loss,
)
from narrative_dynamics.external_prediction import (
    ExternalFinalPredictionArtifact,
    _require_sibling_protocols,
    _resolve_models,
    predict_external_final_once,
    score_external_prediction_artifact,
)
from narrative_dynamics.external_validation import (
    ExternalFinalEvaluation,
    ExternalReleasePreflight,
    ExternalValidationReport,
    assemble_external_final_from_reports,
    build_external_validation_report,
    preflight_external_releases,
)
from narrative_dynamics.model_comparison import ComparisonModel
from narrative_dynamics.observations.release import (
    ProtocolRelease,
    ReleasedModelComparisonReport,
    VerifiedProtocolRelease,
)
from narrative_dynamics.process_execution import (
    ModelCancelled,
    ModelOutputLimitExceeded,
    ModelResourceLimitExceeded,
    ModelTimeout,
    ModelTraceLimitExceeded,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner

from .feher_hare_two_stage_v1 import (
    FeherHareProtocolBundle,
    PreparedFeherHareTwoStageV1,
)
from .two_stage_final import (
    TwoStageFinalAttemptLedger,
    TwoStageFinalAttemptStatus,
    TwoStageStaySwitchDiagnostic,
    build_two_stage_stay_switch_diagnostic,
    complete_final_attempt,
    mark_infrastructure_failed,
    mark_revision_required,
    require_exact_retry,
    start_final_attempt,
)


_INFRASTRUCTURE_FAILURES = (
    ModelCancelled,
    ModelTimeout,
    ModelOutputLimitExceeded,
    ModelTraceLimitExceeded,
    ModelResourceLimitExceeded,
)


class FinalInfrastructureError(RuntimeError):
    """Environment-only failure after a locked FINAL attempt has started."""

    def __init__(self, *, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.attempt_ledger: TwoStageFinalAttemptLedger | None = None


@dataclass(frozen=True)
class LockedFeherHareFinalResult:
    attempt_ledger: TwoStageFinalAttemptLedger
    prediction_artifact: ExternalFinalPredictionArtifact
    brier_report: ReleasedModelComparisonReport
    log_report: ReleasedModelComparisonReport
    external_evaluation: ExternalFinalEvaluation
    diagnostic: TwoStageStaySwitchDiagnostic
    report: ExternalValidationReport


def _repository_revision(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("locked two-stage FINAL repository revision must be a trimmed string")
    return value


def _preflight_locked_final(
    *,
    runner: SimulationRunner,
    prepared: PreparedFeherHareTwoStageV1,
    protocols: FeherHareProtocolBundle,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
    runtime_models: tuple[ComparisonModel, ...],
    repository_revision: str,
    attempt_ledger: TwoStageFinalAttemptLedger,
) -> tuple[ExternalReleasePreflight, tuple[ComparisonModel, ...], dict[str, object]]:
    if not isinstance(runner, SimulationRunner):
        raise TypeError("locked two-stage FINAL requires SimulationRunner")
    if not isinstance(prepared, PreparedFeherHareTwoStageV1):
        raise TypeError("locked two-stage FINAL requires prepared study")
    if not isinstance(protocols, FeherHareProtocolBundle):
        raise TypeError("locked two-stage FINAL requires protocol bundle")
    if not isinstance(brier_release, ProtocolRelease):
        raise TypeError("locked two-stage FINAL requires Brier ProtocolRelease")
    if not isinstance(log_release, ProtocolRelease):
        raise TypeError("locked two-stage FINAL requires Log ProtocolRelease")
    if not isinstance(brier_verified, VerifiedProtocolRelease):
        raise TypeError("locked two-stage FINAL requires verified Brier release")
    if not isinstance(log_verified, VerifiedProtocolRelease):
        raise TypeError("locked two-stage FINAL requires verified Log release")
    if not isinstance(attempt_ledger, TwoStageFinalAttemptLedger):
        raise TypeError("locked two-stage FINAL requires attempt ledger")

    revision = _repository_revision(repository_revision)
    models = tuple(runtime_models)
    preflight = preflight_external_releases(
        preregistration=protocols.preregistration,
        evidence=prepared.evidence,
        brier_protocol=protocols.brier_protocol,
        brier_release=brier_release,
        brier_verified=brier_verified,
        log_protocol=protocols.log_protocol,
        log_release=log_release,
        log_verified=log_verified,
    )
    for role, release in (("Brier", brier_release), ("Log", log_release)):
        if release.source_revision.get("repository_revision") != revision:
            raise ValueError(
                f"locked two-stage FINAL {role} release repository revision changed"
            )

    _require_sibling_protocols(
        protocols.preregistration,
        preflight,
        protocols.brier_protocol,
        protocols.log_protocol,
        prepared.final_targets,
        two_stage_first_stage_policy_metrics,
    )
    resolved_models = _resolve_models(models, protocols.brier_protocol)

    identity: dict[str, object] = {
        "preregistration_hash": protocols.preregistration.content_hash,
        "preflight_hash": preflight.content_hash,
        "repository_revision": revision,
        "dataset_hash": prepared.dataset.content_hash,
        "final_target_hash": prepared.final_targets.content_hash,
        "frozen_candidate_hashes": tuple(
            candidate.content_hash for candidate in protocols.brier_protocol.candidates
        ),
        "brier_release_hash": brier_release.content_hash,
        "log_release_hash": log_release.content_hash,
    }
    if (
        attempt_ledger.attempts
        and attempt_ledger.attempts[-1].status
        is TwoStageFinalAttemptStatus.INFRASTRUCTURE_FAILED
    ):
        require_exact_retry(attempt_ledger, **identity)
    return preflight, resolved_models, identity


def _predict_locked_final(
    *,
    runner: SimulationRunner,
    prepared: PreparedFeherHareTwoStageV1,
    protocols: FeherHareProtocolBundle,
    preflight: ExternalReleasePreflight,
    runtime_models: tuple[ComparisonModel, ...],
    repository_revision: str,
) -> ExternalFinalPredictionArtifact:
    try:
        return predict_external_final_once(
            runner=runner,
            preregistration=protocols.preregistration,
            preflight=preflight,
            brier_protocol=protocols.brier_protocol,
            log_protocol=protocols.log_protocol,
            models=runtime_models,
            final_targets=prepared.final_targets,
            extractor=two_stage_first_stage_policy_metrics,
            repository_revision=repository_revision,
        )
    except _INFRASTRUCTURE_FAILURES as error:
        raise FinalInfrastructureError(
            failure_class=type(error).__name__,
            message=str(error),
        ) from error


def run_locked_feher_hare_final(
    *,
    runner: SimulationRunner,
    prepared: PreparedFeherHareTwoStageV1,
    protocols: FeherHareProtocolBundle,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
    runtime_models: tuple[ComparisonModel, ...],
    repository_revision: str,
    attempt_ledger: TwoStageFinalAttemptLedger,
    attempt_id: str,
    started_at: str,
) -> LockedFeherHareFinalResult:
    preflight, resolved_models, identity = _preflight_locked_final(
        runner=runner,
        prepared=prepared,
        protocols=protocols,
        brier_release=brier_release,
        brier_verified=brier_verified,
        log_release=log_release,
        log_verified=log_verified,
        runtime_models=tuple(runtime_models),
        repository_revision=repository_revision,
        attempt_ledger=attempt_ledger,
    )

    started_ledger = start_final_attempt(
        attempt_ledger,
        attempt_id=attempt_id,
        started_at=started_at,
        **identity,
    )
    try:
        artifact = _predict_locked_final(
            runner=runner,
            prepared=prepared,
            protocols=protocols,
            preflight=preflight,
            runtime_models=resolved_models,
            repository_revision=repository_revision,
        )
        brier_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=brier_verified,
            protocol=protocols.brier_protocol,
            target_set=prepared.final_targets,
            loss=two_stage_brier_loss(),
        )
        log_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=log_verified,
            protocol=protocols.log_protocol,
            target_set=prepared.final_targets,
            loss=two_stage_log_loss(),
        )
        external_evaluation = assemble_external_final_from_reports(
            preregistration=protocols.preregistration,
            evidence=prepared.evidence,
            brier_protocol=protocols.brier_protocol,
            brier_release=brier_release,
            brier_verified=brier_verified,
            brier_report=brier_report,
            log_protocol=protocols.log_protocol,
            log_release=log_release,
            log_verified=log_verified,
            log_report=log_report,
            final_targets=prepared.final_targets,
        )
        diagnostic = build_two_stage_stay_switch_diagnostic(
            dataset=prepared.dataset,
            artifact=artifact,
        )
        report = build_external_validation_report(
            preregistration=protocols.preregistration,
            evidence=prepared.evidence,
            final_evaluation=external_evaluation,
            constraint_findings=(),
        )
        attested = attest_report(report)
        attested.require_integrity()
        completed_ledger = complete_final_attempt(
            started_ledger,
            attempt_id=attempt_id,
            completed_run_manifest_hashes=tuple(
                prediction.run_manifest_hash
                for model in artifact.model_predictions
                for prediction in model.predictions
            ),
            result_hash=attested.content_hash,
        )
        return LockedFeherHareFinalResult(
            attempt_ledger=completed_ledger,
            prediction_artifact=artifact,
            brier_report=brier_report,
            log_report=log_report,
            external_evaluation=external_evaluation,
            diagnostic=diagnostic,
            report=report,
        )
    except FinalInfrastructureError as error:
        failed_ledger = mark_infrastructure_failed(
            started_ledger,
            attempt_id=attempt_id,
            failure_class=error.failure_class,
        )
        error.attempt_ledger = failed_ledger
        raise
    except Exception as error:
        revised_ledger = mark_revision_required(
            started_ledger,
            attempt_id=attempt_id,
            failure_class=type(error).__name__,
        )
        try:
            setattr(error, "final_attempt_ledger", revised_ledger)
        except Exception:
            pass
        raise


__all__ = [
    "FinalInfrastructureError",
    "LockedFeherHareFinalResult",
    "run_locked_feher_hare_final",
]
