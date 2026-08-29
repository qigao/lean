from __future__ import annotations

from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.external_validation import (
    ExternalFinalEvaluation,
    ExternalScoreRole,
    ExternalValidationProtocolError,
    _external_adequacy_findings,
    _external_final_target_matches,
    _external_separation_findings,
    _external_stratum_scores,
    _hash,
    preflight_external_releases,
)
from narrative_dynamics.observations.external import ExternalEvidenceDeclaration
from narrative_dynamics.observations.preregistration import (
    PreregisteredEvaluationProtocol,
)
from narrative_dynamics.observations.release import (
    ProtocolRelease,
    ReleasedModelComparisonReport,
    VerifiedProtocolRelease,
)
from narrative_dynamics.observations.targets import TargetConstructionReport


def _require_precomputed_external_report(
    *,
    report: ReleasedModelComparisonReport,
    protocol: PreregisteredEvaluationProtocol,
    release: ProtocolRelease,
    verified: VerifiedProtocolRelease,
    final_targets: TargetConstructionReport,
    score_role: ExternalScoreRole,
) -> str:
    if not isinstance(report, ReleasedModelComparisonReport):
        raise TypeError("external final assembly requires released comparison reports")
    expected_loss_name = (
        "categorical_brier"
        if score_role is ExternalScoreRole.BRIER
        else "categorical_log"
    )
    if protocol.loss_identity.get("name") != expected_loss_name:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child report has the wrong loss role"
        )
    if report.release_hash != release.content_hash:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child report release identity changed"
        )
    if report.verification_hash != verified.content_hash:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child report verification identity changed"
        )
    if report.manifest.stage is not ExperimentStage.RELEASED_MODEL_COMPARISON:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child report manifest stage changed"
        )

    released_inputs = report.manifest.inputs
    expected_released_inputs = {
        "verified_release_hash": verified.content_hash,
        "release_hash": release.content_hash,
        "protocol_hash": protocol.content_hash,
        "verifier_identity": verified.verifier_identity,
        "verified_receipt_hashes": verified.verified_receipt_hashes,
        "comparison_manifest_hash": report.comparison.manifest.content_hash,
    }
    for key, expected in expected_released_inputs.items():
        if released_inputs.get(key) != expected:
            raise ExternalValidationProtocolError(
                f"external {score_role.value} child released manifest changed: {key}"
            )
    prediction_artifact_hash = _hash(
        released_inputs.get("prediction_artifact_hash"),
        label=f"external {score_role.value} prediction artifact hash",
    )
    if report.manifest.parent_hashes != (report.comparison.manifest.content_hash,):
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child released lineage changed"
        )

    comparison = report.comparison
    if comparison.manifest.stage is not ExperimentStage.MODEL_COMPARISON:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child comparison stage changed"
        )
    if comparison.baseline_name != protocol.baseline_name:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child baseline changed"
        )
    names = tuple(sorted(comparison.entry_map))
    if names != protocol.candidate_names:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child candidate names changed"
        )

    comparison_inputs = comparison.manifest.inputs
    expected_comparison_inputs = {
        "protocol_hash": protocol.content_hash,
        "declared_precommitment_hash": protocol.declared_precommitment_hash,
        "dataset_hash": protocol.dataset_hash,
        "final_partition_hash": protocol.final_partition_hash,
        "final_target_hash": protocol.final_target_hash,
        "final_target_manifest_hash": protocol.final_target_manifest_hash,
        "metric_identity": protocol.metric_identity,
        "loss_identity": protocol.loss_identity,
        "simulation_seeds": protocol.simulation_seeds,
        "baseline_name": protocol.baseline_name,
        "prediction_artifact_hash": prediction_artifact_hash,
    }
    for key, expected in expected_comparison_inputs.items():
        if comparison_inputs.get(key) != expected:
            raise ExternalValidationProtocolError(
                f"external {score_role.value} child comparison manifest changed: {key}"
            )

    expected_entries = tuple(
        {
            "name": entry.name,
            "parameters": entry.parameters,
            "mean_loss": entry.mean_loss,
            "worst_loss": entry.worst_loss,
            "delta_mean_from_baseline": entry.delta_mean_from_baseline,
            "delta_worst_from_baseline": entry.delta_worst_from_baseline,
            "adequate": entry.adequate,
        }
        for entry in comparison.ranking
    )
    if comparison_inputs.get("entries") != expected_entries:
        raise ExternalValidationProtocolError(
            f"external {score_role.value} child comparison entries were mutated"
        )

    targets = {
        case.name: (case.scenario.id, tuple(sorted(case.target_map.items())))
        for case in final_targets.cases
    }
    frozen_by_name = {candidate.name: candidate for candidate in protocol.candidates}
    for name in names:
        entry = comparison.entry_map[name]
        frozen = frozen_by_name[name]
        if entry.parameters != frozen.parameters:
            raise ExternalValidationProtocolError(
                f"external {score_role.value} child parameters changed for {name!r}"
            )
        if entry.mean_loss != entry.final_test.validation.mean_loss:
            raise ExternalValidationProtocolError(
                f"external {score_role.value} child mean loss changed for {name!r}"
            )
        if entry.worst_loss != entry.final_test.validation.worst_loss:
            raise ExternalValidationProtocolError(
                f"external {score_role.value} child worst loss changed for {name!r}"
            )
        expected_adequate = (
            entry.mean_loss <= protocol.thresholds.max_mean_loss
            and entry.worst_loss <= protocol.thresholds.max_worst_loss
        )
        if entry.adequate is not expected_adequate:
            raise ExternalValidationProtocolError(
                f"external {score_role.value} child adequacy changed for {name!r}"
            )
        actual_cases = {
            case.name: (case.scenario_id, tuple(sorted(case.target)))
            for case in entry.final_test.validation.cases
        }
        if actual_cases != targets:
            raise ExternalValidationProtocolError(
                f"external {score_role.value} child final targets changed for {name!r}"
            )
    return prediction_artifact_hash


def assemble_external_final_from_reports(
    *,
    preregistration,
    evidence: ExternalEvidenceDeclaration,
    brier_protocol: PreregisteredEvaluationProtocol,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    brier_report: ReleasedModelComparisonReport,
    log_protocol: PreregisteredEvaluationProtocol,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
    log_report: ReleasedModelComparisonReport,
    final_targets: TargetConstructionReport,
) -> ExternalFinalEvaluation:
    if not isinstance(final_targets, TargetConstructionReport):
        raise TypeError("external final assembly requires final targets")
    preflight = preflight_external_releases(
        preregistration=preregistration,
        evidence=evidence,
        brier_protocol=brier_protocol,
        brier_release=brier_release,
        brier_verified=brier_verified,
        log_protocol=log_protocol,
        log_release=log_release,
        log_verified=log_verified,
    )
    if not _external_final_target_matches(final_targets, brier_protocol):
        raise ExternalValidationProtocolError(
            "external final targets do not match the Brier protocol"
        )
    if not _external_final_target_matches(final_targets, log_protocol):
        raise ExternalValidationProtocolError(
            "external final targets do not match the Log protocol"
        )

    brier_artifact_hash = _require_precomputed_external_report(
        report=brier_report,
        protocol=brier_protocol,
        release=brier_release,
        verified=brier_verified,
        final_targets=final_targets,
        score_role=ExternalScoreRole.BRIER,
    )
    log_artifact_hash = _require_precomputed_external_report(
        report=log_report,
        protocol=log_protocol,
        release=log_release,
        verified=log_verified,
        final_targets=final_targets,
        score_role=ExternalScoreRole.LOG,
    )
    if brier_artifact_hash != log_artifact_hash:
        raise ExternalValidationProtocolError(
            "external sibling reports were not scored from one sealed prediction artifact"
        )

    adequacy_findings, aggregate_adequacy = _external_adequacy_findings(
        brier_protocol=brier_protocol,
        brier_report=brier_report,
        log_protocol=log_protocol,
        log_report=log_report,
    )
    separation_findings = _external_separation_findings(
        preregistration=preregistration,
        brier_report=brier_report,
        log_report=log_report,
    )
    stratum_scores = _external_stratum_scores(
        preregistration=preregistration,
        brier_report=brier_report,
        log_report=log_report,
    )
    return ExternalFinalEvaluation(
        preflight=preflight,
        brier_report=brier_report,
        log_report=log_report,
        adequacy_findings=adequacy_findings,
        aggregate_adequacy=aggregate_adequacy,
        separation_findings=separation_findings,
        stratum_scores=stratum_scores,
    )


__all__ = ["assemble_external_final_from_reports"]
