"""Aggregate-only scoring and typed reporting for transfer V1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from typing import Mapping

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_inference import (
    FrozenBaseRateComparator,
    TransferInferenceEvidence,
    _ParticipantLossBlock,
    paired_participant_bootstrap,
)
from narrative_dynamics.cross_dataset_prediction import (
    SealedTransferPredictionArtifact,
)
from narrative_dynamics.cross_dataset_release import (
    CARRY_FORWARD_REQUIREMENT_IDS,
    LIMITATIONS,
    TransferScore,
    TransferScoreRelease,
)


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_FAMILIES = ("reactive", "intentional", "planning")
_KNOWN_FORBIDDEN_KEYS = (
    "participant_id",
    "participant_token",
    "case_prediction",
    "final_target",
    "source_path",
)


class TransferAdequacyStatus(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    MIXED_OR_INCONCLUSIVE = "MIXED_OR_INCONCLUSIVE"


class CarryForwardSensitivityStatus(str, Enum):
    COMPARABLE = "COMPARABLE"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class TransferStudyTerminal(str, Enum):
    GREEN = "GREEN"
    SCIENTIFIC_RED = "SCIENTIFIC_RED"
    INFRASTRUCTURE_INCOMPLETE = "INFRASTRUCTURE_INCOMPLETE"


def _strict_fields(
    payload: object,
    *,
    expected: tuple[str, ...],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = frozenset(payload)
    required = frozenset(expected)
    if actual != required:
        missing = tuple(sorted(required - actual))
        unknown = tuple(sorted(actual - required))
        raise ValueError(
            f"{label} fields mismatch: missing={missing!r}, unknown={unknown!r}"
        )
    return payload


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return value


def _optional_hash(value: object, *, label: str) -> str | None:
    return None if value is None else _hash(value, label=label)


def classify_family_finding(
    zero_shot: bool,
    refit: bool,
    mixed: bool,
) -> str:
    if not all(isinstance(value, bool) for value in (zero_shot, refit, mixed)):
        raise TypeError("family finding inputs must be boolean")
    if mixed:
        return "MIXED_OR_INCONCLUSIVE_EVIDENCE"
    return {
        (True, True): "PARAMETER_AND_FAMILY_TRANSFER_EVIDENCE",
        (False, True): "FAMILY_TRANSFER_ONLY_RETRAINING_REQUIRED",
        (True, False): "ZERO_SHOT_TRANSFER_REFIT_NOT_ESTABLISHED",
        (False, False): "NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL",
    }[(zero_shot, refit)]


@dataclass(frozen=True)
class TransferFamilyFinding:
    family: str
    zero_shot_adequacy: TransferAdequacyStatus
    refit_adequacy: TransferAdequacyStatus
    finding: str
    refit_material_gain: bool
    refit_gain_mixed: bool

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError("unknown transfer family finding")
        for field_name in ("zero_shot_adequacy", "refit_adequacy"):
            value = getattr(self, field_name)
            if not isinstance(value, TransferAdequacyStatus):
                try:
                    value = TransferAdequacyStatus(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("unknown transfer adequacy status") from exc
                object.__setattr__(self, field_name, value)
        allowed = {
            "PARAMETER_AND_FAMILY_TRANSFER_EVIDENCE",
            "FAMILY_TRANSFER_ONLY_RETRAINING_REQUIRED",
            "ZERO_SHOT_TRANSFER_REFIT_NOT_ESTABLISHED",
            "NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL",
            "MIXED_OR_INCONCLUSIVE_EVIDENCE",
        }
        if self.finding not in allowed:
            raise ValueError("unknown family finding")
        if not isinstance(self.refit_material_gain, bool) or not isinstance(
            self.refit_gain_mixed,
            bool,
        ):
            raise TypeError("refit material-gain fields must be boolean")

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "zero_shot_adequacy": self.zero_shot_adequacy.value,
            "refit_adequacy": self.refit_adequacy.value,
            "finding": self.finding,
            "refit_material_gain": self.refit_material_gain,
            "refit_gain_mixed": self.refit_gain_mixed,
        }

    @classmethod
    def from_payload(cls, payload: object) -> TransferFamilyFinding:
        expected = (
            "family",
            "zero_shot_adequacy",
            "refit_adequacy",
            "finding",
            "refit_material_gain",
            "refit_gain_mixed",
        )
        values = _strict_fields(payload, expected=expected, label="family finding")
        return cls(**values)


@dataclass(frozen=True)
class CarryForwardSensitivityFinding:
    requirement_id: str
    status: CarryForwardSensitivityStatus

    def __post_init__(self) -> None:
        if self.requirement_id not in CARRY_FORWARD_REQUIREMENT_IDS:
            raise ValueError("unknown carry-forward requirement ID")
        if not isinstance(self.status, CarryForwardSensitivityStatus):
            try:
                object.__setattr__(
                    self,
                    "status",
                    CarryForwardSensitivityStatus(self.status),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("unknown carry-forward sensitivity status") from exc

    def to_payload(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status.value,
        }

    @classmethod
    def from_payload(cls, payload: object) -> CarryForwardSensitivityFinding:
        values = _strict_fields(
            payload,
            expected=("requirement_id", "status"),
            label="carry-forward sensitivity",
        )
        return cls(**values)


def _inference_from_payload(payload: object) -> TransferInferenceEvidence:
    expected = (
        "candidate_identity",
        "comparator_identity",
        "score",
        "aggregation",
        "candidate_loss",
        "comparator_loss",
        "relative_improvement",
        "lower_95",
        "upper_95",
        "bootstrap_seed",
        "bootstrap_replicates",
        "percentile_rule",
        "participant_count",
        "threshold",
        "pass_status",
        "source_stratum_identity",
    )
    values = _strict_fields(payload, expected=expected, label="inference evidence")
    return TransferInferenceEvidence(**values)


@dataclass(frozen=True)
class CrossDatasetTransferReport:
    terminal: TransferStudyTerminal
    prediction_artifact_hash: str | None
    semantic_invariance_receipt_hash: str
    semantic_invariance_pass: bool
    family_findings: tuple[TransferFamilyFinding, ...]
    inference_evidence: tuple[TransferInferenceEvidence, ...]
    carry_forward_sensitivities: tuple[CarryForwardSensitivityFinding, ...]
    primary_aggregation: str
    diagnostic_aggregation: str
    task_condition_strata: tuple[str, ...]
    participant_influence_hash: str
    limitations: tuple[str, ...]
    scientific_failure_hash: str | None
    infrastructure_failure_hash: str | None
    privacy_scan_passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.terminal, TransferStudyTerminal):
            try:
                object.__setattr__(self, "terminal", TransferStudyTerminal(self.terminal))
            except (TypeError, ValueError) as exc:
                raise ValueError("unknown transfer study terminal") from exc
        object.__setattr__(
            self,
            "prediction_artifact_hash",
            _optional_hash(
                self.prediction_artifact_hash,
                label="prediction_artifact_hash",
            ),
        )
        object.__setattr__(
            self,
            "semantic_invariance_receipt_hash",
            _hash(
                self.semantic_invariance_receipt_hash,
                label="semantic_invariance_receipt_hash",
            ),
        )
        if not isinstance(self.semantic_invariance_pass, bool):
            raise TypeError("semantic_invariance_pass must be boolean")
        findings = tuple(self.family_findings)
        evidence = tuple(self.inference_evidence)
        carry = tuple(self.carry_forward_sensitivities)
        if not all(isinstance(row, TransferFamilyFinding) for row in findings):
            raise TypeError("family_findings contain invalid rows")
        if not all(isinstance(row, TransferInferenceEvidence) for row in evidence):
            raise TypeError("inference_evidence contain invalid rows")
        if not all(isinstance(row, CarryForwardSensitivityFinding) for row in carry):
            raise TypeError("carry_forward_sensitivities contain invalid rows")
        if tuple(row.requirement_id for row in carry) != CARRY_FORWARD_REQUIREMENT_IDS:
            raise ValueError("carry-forward requirement set/order changed")
        object.__setattr__(self, "family_findings", findings)
        object.__setattr__(self, "inference_evidence", evidence)
        object.__setattr__(self, "carry_forward_sensitivities", carry)
        if self.primary_aggregation != "PARTICIPANT_EQUAL":
            raise ValueError("primary aggregation must be PARTICIPANT_EQUAL")
        if self.diagnostic_aggregation != "TRIAL_EQUAL":
            raise ValueError("diagnostic aggregation must be TRIAL_EQUAL")
        strata = tuple(self.task_condition_strata)
        if not strata or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in strata
        ):
            raise ValueError("task_condition_strata must be non-empty trimmed text")
        object.__setattr__(self, "task_condition_strata", strata)
        object.__setattr__(
            self,
            "participant_influence_hash",
            _hash(self.participant_influence_hash, label="participant_influence_hash"),
        )
        if tuple(self.limitations) != LIMITATIONS:
            raise ValueError("report limitations changed from frozen protocol")
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(
            self,
            "scientific_failure_hash",
            _optional_hash(self.scientific_failure_hash, label="scientific_failure_hash"),
        )
        object.__setattr__(
            self,
            "infrastructure_failure_hash",
            _optional_hash(
                self.infrastructure_failure_hash,
                label="infrastructure_failure_hash",
            ),
        )
        if self.privacy_scan_passed is not True:
            raise ValueError("report privacy scan must pass")
        if self.terminal is TransferStudyTerminal.GREEN:
            if (
                tuple(row.family for row in findings) != _FAMILIES
                or not evidence
                or self.prediction_artifact_hash is None
                or self.scientific_failure_hash is not None
                or self.infrastructure_failure_hash is not None
                or not self.semantic_invariance_pass
            ):
                raise ValueError("GREEN report is incomplete or has conflicting terminals")
        elif self.terminal is TransferStudyTerminal.SCIENTIFIC_RED:
            if (
                self.scientific_failure_hash is None
                or self.infrastructure_failure_hash is not None
                or findings
                or evidence
            ):
                raise ValueError("SCIENTIFIC_RED terminal evidence is not exclusive")
        elif (
            self.infrastructure_failure_hash is None
            or self.scientific_failure_hash is not None
            or findings
            or evidence
        ):
            raise ValueError("INFRASTRUCTURE_INCOMPLETE terminal evidence is not exclusive")

    def to_payload(self) -> dict[str, object]:
        return {
            "terminal": self.terminal.value,
            "prediction_artifact_hash": self.prediction_artifact_hash,
            "semantic_invariance_receipt_hash": self.semantic_invariance_receipt_hash,
            "semantic_invariance_pass": self.semantic_invariance_pass,
            "family_findings": [row.to_payload() for row in self.family_findings],
            "inference_evidence": [row.to_payload() for row in self.inference_evidence],
            "carry_forward_sensitivities": [
                row.to_payload() for row in self.carry_forward_sensitivities
            ],
            "primary_aggregation": self.primary_aggregation,
            "diagnostic_aggregation": self.diagnostic_aggregation,
            "task_condition_strata": list(self.task_condition_strata),
            "participant_influence_hash": self.participant_influence_hash,
            "limitations": list(self.limitations),
            "scientific_failure_hash": self.scientific_failure_hash,
            "infrastructure_failure_hash": self.infrastructure_failure_hash,
            "privacy_scan_passed": self.privacy_scan_passed,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> CrossDatasetTransferReport:
        expected = (
            "terminal",
            "prediction_artifact_hash",
            "semantic_invariance_receipt_hash",
            "semantic_invariance_pass",
            "family_findings",
            "inference_evidence",
            "carry_forward_sensitivities",
            "primary_aggregation",
            "diagnostic_aggregation",
            "task_condition_strata",
            "participant_influence_hash",
            "limitations",
            "scientific_failure_hash",
            "infrastructure_failure_hash",
            "privacy_scan_passed",
        )
        values = _strict_fields(payload, expected=expected, label="transfer report")
        return cls(
            terminal=values["terminal"],
            prediction_artifact_hash=values["prediction_artifact_hash"],
            semantic_invariance_receipt_hash=values[
                "semantic_invariance_receipt_hash"
            ],
            semantic_invariance_pass=values["semantic_invariance_pass"],
            family_findings=tuple(
                TransferFamilyFinding.from_payload(row)
                for row in values["family_findings"]
            ),
            inference_evidence=tuple(
                _inference_from_payload(row) for row in values["inference_evidence"]
            ),
            carry_forward_sensitivities=tuple(
                CarryForwardSensitivityFinding.from_payload(row)
                for row in values["carry_forward_sensitivities"]
            ),
            primary_aggregation=values["primary_aggregation"],
            diagnostic_aggregation=values["diagnostic_aggregation"],
            task_condition_strata=tuple(values["task_condition_strata"]),
            participant_influence_hash=values["participant_influence_hash"],
            limitations=tuple(values["limitations"]),
            scientific_failure_hash=values["scientific_failure_hash"],
            infrastructure_failure_hash=values["infrastructure_failure_hash"],
            privacy_scan_passed=values["privacy_scan_passed"],
        )


def _canonical_carry_forward(
    value: object,
) -> tuple[CarryForwardSensitivityFinding, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("carry_forward_statuses must be a sequence")
    rows = tuple(
        CarryForwardSensitivityFinding(requirement_id=row[0], status=row[1])
        for row in value
        if isinstance(row, (tuple, list)) and len(row) == 2
    )
    if len(rows) != len(tuple(value)):
        raise ValueError("carry_forward_statuses rows must contain two fields")
    if tuple(row.requirement_id for row in rows) != CARRY_FORWARD_REQUIREMENT_IDS:
        raise ValueError("carry-forward requirement set/order changed")
    return rows


def _case_loss(row: object, score: str) -> float:
    target_index = 1 if row.target_action == "action_1" else 0
    probabilities = (row.probability_action_0, row.probability_action_1)
    if score == "BRIER":
        return math.fsum(
            (probability - (1.0 if index == target_index else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
    return -math.log(max(probabilities[target_index], 1e-12))


def _baseline_loss(
    row: object,
    score: str,
    baseline: FrozenBaseRateComparator,
) -> float:
    p0, p1 = baseline.probabilities_for(row.source_stratum)
    target_index = 1 if row.target_action == "action_1" else 0
    if score == "BRIER":
        return math.fsum(
            (probability - (1.0 if index == target_index else 0.0)) ** 2
            for index, probability in enumerate((p0, p1))
        )
    return -math.log(max((p0, p1)[target_index], 1e-12))


def _loss_blocks(
    rows: tuple[object, ...],
    *,
    loss,
) -> tuple[_ParticipantLossBlock, ...]:
    by_token: dict[str, list[float]] = {}
    for row in rows:
        by_token.setdefault(row.participant_token, []).append(loss(row))
    return tuple(
        _ParticipantLossBlock(token=token, losses=tuple(losses))
        for token, losses in sorted(by_token.items())
    )


def _adequacy_status(left: bool, right: bool) -> TransferAdequacyStatus:
    if left != right:
        return TransferAdequacyStatus.MIXED_OR_INCONCLUSIVE
    return TransferAdequacyStatus.MET if left else TransferAdequacyStatus.NOT_MET


def _scan_archive(payload: object, forbidden_values: tuple[str, ...]) -> None:
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in _KNOWN_FORBIDDEN_KEYS:
        if forbidden in serialized:
            raise ValueError(f"archive contains forbidden key {forbidden!r}")
    for value in forbidden_values:
        if not isinstance(value, str) or not value:
            raise ValueError("forbidden archive values must be non-empty text")
        if value.lower() in serialized:
            raise ValueError("archive contains a forbidden private value")


def score_and_report_transfer(
    *,
    artifact: SealedTransferPredictionArtifact | None,
    baseline: FrozenBaseRateComparator,
    brier_release: TransferScoreRelease,
    log_release: TransferScoreRelease,
    semantic_invariance_receipt_hash: str,
    semantic_invariance_pass: bool,
    carry_forward_statuses: tuple[tuple[str, CarryForwardSensitivityStatus], ...],
    task_condition_strata: tuple[str, ...],
    participant_influence_hash: str,
    forbidden_archive_values: tuple[str, ...] = (),
    scientific_failure_hash: str | None = None,
    infrastructure_failure_hash: str | None = None,
) -> CrossDatasetTransferReport:
    if scientific_failure_hash is not None and infrastructure_failure_hash is not None:
        raise ValueError("scientific and infrastructure terminals must be exclusive")
    if not isinstance(baseline, FrozenBaseRateComparator):
        raise TypeError("baseline must be FrozenBaseRateComparator")
    if (
        not isinstance(brier_release, TransferScoreRelease)
        or not isinstance(log_release, TransferScoreRelease)
        or brier_release.score is not TransferScore.BRIER
        or log_release.score is not TransferScore.LOG
        or brier_release.shared_identity_payload()
        != log_release.shared_identity_payload()
    ):
        raise ValueError("report scoring requires exact Brier/Log siblings")
    if brier_release.protocol.baseline_hash != baseline.content_hash:
        raise ValueError("score releases do not bind the supplied TRAIN baseline")
    semantic_hash = _hash(
        semantic_invariance_receipt_hash,
        label="semantic_invariance_receipt_hash",
    )
    influence_hash = _hash(
        participant_influence_hash,
        label="participant_influence_hash",
    )
    carry = _canonical_carry_forward(carry_forward_statuses)
    strata = tuple(task_condition_strata)

    if not semantic_invariance_pass:
        report = CrossDatasetTransferReport(
            terminal=TransferStudyTerminal.SCIENTIFIC_RED,
            prediction_artifact_hash=None,
            semantic_invariance_receipt_hash=semantic_hash,
            semantic_invariance_pass=False,
            family_findings=(),
            inference_evidence=(),
            carry_forward_sensitivities=carry,
            primary_aggregation="PARTICIPANT_EQUAL",
            diagnostic_aggregation="TRIAL_EQUAL",
            task_condition_strata=strata,
            participant_influence_hash=influence_hash,
            limitations=LIMITATIONS,
            scientific_failure_hash=(
                _hash(scientific_failure_hash, label="scientific_failure_hash")
                if scientific_failure_hash is not None
                else stable_content_hash(
                    {"semantic_invariance_receipt_hash": semantic_hash}
                )
            ),
            infrastructure_failure_hash=None,
            privacy_scan_passed=True,
        )
        _scan_archive(report.to_payload(), tuple(forbidden_archive_values))
        return report

    if artifact is None:
        if infrastructure_failure_hash is None:
            raise ValueError("missing artifact requires infrastructure failure evidence")
        report = CrossDatasetTransferReport(
            terminal=TransferStudyTerminal.INFRASTRUCTURE_INCOMPLETE,
            prediction_artifact_hash=None,
            semantic_invariance_receipt_hash=semantic_hash,
            semantic_invariance_pass=True,
            family_findings=(),
            inference_evidence=(),
            carry_forward_sensitivities=carry,
            primary_aggregation="PARTICIPANT_EQUAL",
            diagnostic_aggregation="TRIAL_EQUAL",
            task_condition_strata=strata,
            participant_influence_hash=influence_hash,
            limitations=LIMITATIONS,
            scientific_failure_hash=None,
            infrastructure_failure_hash=_hash(
                infrastructure_failure_hash,
                label="infrastructure_failure_hash",
            ),
            privacy_scan_passed=True,
        )
        _scan_archive(report.to_payload(), tuple(forbidden_archive_values))
        return report

    if scientific_failure_hash is not None or infrastructure_failure_hash is not None:
        raise ValueError("successful scoring cannot include failure terminal evidence")
    if not isinstance(artifact, SealedTransferPredictionArtifact):
        raise TypeError("artifact must be SealedTransferPredictionArtifact")
    if (
        artifact.prediction_artifact_identity
        != brier_release.prediction_artifact_identity
        or artifact.final_commitment_hash
        != brier_release.protocol.final_commitment_hash
    ):
        raise ValueError("sealed prediction identity differs from score releases")
    private_rows = artifact.consume_for_scoring_once()

    by_candidate: dict[tuple[str, str], tuple[object, ...]] = {}
    for path in ("ZERO_SHOT", "REFIT"):
        for family in _FAMILIES:
            rows = tuple(
                row
                for row in private_rows
                if row.path == path and row.family.value == family
            )
            if not rows:
                raise ValueError("sealed artifact is missing one candidate path/family")
            identities = {row.candidate_hash for row in rows}
            if len(identities) != 1:
                raise ValueError("sealed artifact candidate identity drift")
            by_candidate[(path, family)] = rows

    evidence: list[TransferInferenceEvidence] = []
    base_lookup: dict[tuple[str, str, str], TransferInferenceEvidence] = {}
    gain_lookup: dict[tuple[str, str], TransferInferenceEvidence] = {}
    for score in ("BRIER", "CLIPPED_LOG"):
        for path in ("ZERO_SHOT", "REFIT"):
            for family in _FAMILIES:
                rows = by_candidate[(path, family)]
                candidate_blocks = _loss_blocks(
                    rows,
                    loss=lambda row, score=score: _case_loss(row, score),
                )
                baseline_blocks = _loss_blocks(
                    rows,
                    loss=lambda row, score=score: _baseline_loss(
                        row,
                        score,
                        baseline,
                    ),
                )
                candidate_identity = rows[0].candidate_hash
                for aggregation in ("PARTICIPANT_EQUAL", "TRIAL_EQUAL"):
                    row_evidence = paired_participant_bootstrap(
                        candidate=candidate_blocks,
                        comparator=baseline_blocks,
                        score=score,
                        aggregation=aggregation,
                        candidate_identity=candidate_identity,
                        comparator_identity=baseline.content_hash,
                        seed=43001,
                        replicates=10000,
                    )
                    evidence.append(row_evidence)
                    if aggregation == "PARTICIPANT_EQUAL":
                        base_lookup[(path, family, score)] = row_evidence

        for family in _FAMILIES:
            refit_rows = by_candidate[("REFIT", family)]
            zero_rows = by_candidate[("ZERO_SHOT", family)]
            refit_blocks = _loss_blocks(
                refit_rows,
                loss=lambda row, score=score: _case_loss(row, score),
            )
            zero_blocks = _loss_blocks(
                zero_rows,
                loss=lambda row, score=score: _case_loss(row, score),
            )
            for aggregation in ("PARTICIPANT_EQUAL", "TRIAL_EQUAL"):
                row_evidence = paired_participant_bootstrap(
                    candidate=refit_blocks,
                    comparator=zero_blocks,
                    score=score,
                    aggregation=aggregation,
                    candidate_identity=refit_rows[0].candidate_hash,
                    comparator_identity=zero_rows[0].candidate_hash,
                    seed=43001,
                    replicates=10000,
                )
                evidence.append(row_evidence)
                if aggregation == "PARTICIPANT_EQUAL":
                    gain_lookup[(family, score)] = row_evidence

    findings: list[TransferFamilyFinding] = []
    for family in _FAMILIES:
        zero_brier = base_lookup[("ZERO_SHOT", family, "BRIER")].pass_status
        zero_log = base_lookup[("ZERO_SHOT", family, "CLIPPED_LOG")].pass_status
        refit_brier = base_lookup[("REFIT", family, "BRIER")].pass_status
        refit_log = base_lookup[("REFIT", family, "CLIPPED_LOG")].pass_status
        gain_brier = gain_lookup[(family, "BRIER")].pass_status
        gain_log = gain_lookup[(family, "CLIPPED_LOG")].pass_status
        mixed = zero_brier != zero_log or refit_brier != refit_log
        zero_met = zero_brier and zero_log
        refit_met = refit_brier and refit_log
        findings.append(
            TransferFamilyFinding(
                family=family,
                zero_shot_adequacy=_adequacy_status(zero_brier, zero_log),
                refit_adequacy=_adequacy_status(refit_brier, refit_log),
                finding=classify_family_finding(zero_met, refit_met, mixed),
                refit_material_gain=gain_brier and gain_log,
                refit_gain_mixed=gain_brier != gain_log,
            )
        )

    report = CrossDatasetTransferReport(
        terminal=TransferStudyTerminal.GREEN,
        prediction_artifact_hash=artifact.sealed_artifact_hash,
        semantic_invariance_receipt_hash=semantic_hash,
        semantic_invariance_pass=True,
        family_findings=tuple(findings),
        inference_evidence=tuple(evidence),
        carry_forward_sensitivities=carry,
        primary_aggregation="PARTICIPANT_EQUAL",
        diagnostic_aggregation="TRIAL_EQUAL",
        task_condition_strata=strata,
        participant_influence_hash=influence_hash,
        limitations=LIMITATIONS,
        scientific_failure_hash=None,
        infrastructure_failure_hash=None,
        privacy_scan_passed=True,
    )
    _scan_archive(report.to_payload(), tuple(forbidden_archive_values))
    return report
