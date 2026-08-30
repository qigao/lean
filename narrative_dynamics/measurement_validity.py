from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_log_loss,
)
from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    Scenario,
    stable_content_hash,
)
from narrative_dynamics.losses import (
    MetricLoss,
    evaluate_metric_loss,
    metric_loss_identity,
)
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.observations.preregistration import FrozenModelSpec


MEASUREMENT_CLAIM_SCOPE = "external_observational_measurement_audit_only"
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PROBABILITY_TOLERANCE = 1e-12
_ALLOWED_ROLES = (
    ObservationPartitionRole.TRAIN,
    ObservationPartitionRole.SELECTION_VALIDATION,
)
_SEEDS_BY_ROLE = {
    ObservationPartitionRole.TRAIN: (101, 102),
    ObservationPartitionRole.SELECTION_VALIDATION: (201, 202),
}
_FAMILY_ORDER = ("reactive", "intentional", "planning")
_TASK_ORDER = ("magic_carpet", "spaceship")
_TARGET_KEYS = (
    "first_stage.action_0",
    "first_stage.action_1",
)
_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {"participant_id", "source_participant_id"}
)


class MeasurementScore(str, Enum):
    BRIER = "brier"
    LOG = "log"


class MeasurementAggregation(str, Enum):
    TRIAL_EQUAL = "trial_equal"
    PARTICIPANT_EQUAL = "participant_equal"


class MeasurementValidityStatus(str, Enum):
    EXACT_INVARIANCE_MET = "exact_invariance_met"
    EXACT_INVARIANCE_FAILED = "exact_invariance_failed"
    STABLE_UNDER_FROZEN_AUDIT = "stable_under_frozen_audit"
    MATERIALLY_MEASUREMENT_DEPENDENT = "materially_measurement_dependent"
    INCONCLUSIVE_SENSITIVITY = "inconclusive_sensitivity"
    NOT_ESTABLISHED = "not_established"


class MeasurementTerminalClass(str, Enum):
    GREEN = "green"
    SCIENTIFIC_RED = "scientific_red"
    INFRASTRUCTURE_INCOMPLETE = "infrastructure_incomplete"


_EXACT_STATUSES = frozenset(
    {
        MeasurementValidityStatus.EXACT_INVARIANCE_MET,
        MeasurementValidityStatus.EXACT_INVARIANCE_FAILED,
    }
)
_DEPENDENCE_STATUSES = frozenset(
    {
        MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
        MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT,
        MeasurementValidityStatus.INCONCLUSIVE_SENSITIVITY,
        MeasurementValidityStatus.NOT_ESTABLISHED,
    }
)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _freeze_value(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        rows: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            rows[key] = _freeze_value(value[key], label=f"{label}.{key}")
        return MappingProxyType(rows)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} contains a non-canonical value")


def _freeze_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = _freeze_value(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _role(value: ObservationPartitionRole | str) -> ObservationPartitionRole:
    try:
        selected = (
            value
            if isinstance(value, ObservationPartitionRole)
            else ObservationPartitionRole(value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("measurement role must be supported") from error
    if selected not in _ALLOWED_ROLES:
        raise ValueError(
            "measurement cases may use only TRAIN or SELECTION_VALIDATION"
        )
    return selected


def _contains_direct_identity(value: object) -> bool:
    if isinstance(value, Mapping):
        if any(key in _FORBIDDEN_IDENTITY_KEYS for key in value):
            return True
        return any(_contains_direct_identity(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_direct_identity(item) for item in value)
    return False


def _target_rows(value: object) -> tuple[tuple[str, float], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    try:
        rows = tuple(items)
    except TypeError as error:
        raise TypeError("measurement target must be a mapping or pair sequence") from error
    if len(rows) != 2 or any(
        not isinstance(row, tuple) or len(row) != 2 for row in rows
    ):
        raise ValueError("measurement target must contain exactly two coordinates")
    canonical: list[tuple[str, float]] = []
    for key, raw_value in rows:
        if not isinstance(key, str):
            raise ValueError("measurement target keys must be strings")
        canonical.append(
            (key, _finite(raw_value, label=f"measurement target {key!r}"))
        )
    canonical.sort()
    if tuple(key for key, _ in canonical) != _TARGET_KEYS:
        raise ValueError("measurement target must use the first-stage metric keys")
    if any(value < 0.0 or value > 1.0 for _, value in canonical):
        raise ValueError("measurement target probabilities must be in [0, 1]")
    if not math.isclose(
        math.fsum(value for _, value in canonical),
        1.0,
        rel_tol=_PROBABILITY_TOLERANCE,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("measurement target probabilities must sum to one")
    return tuple(canonical)


def _validity_status(
    value: MeasurementValidityStatus | str,
    *,
    allowed: frozenset[MeasurementValidityStatus],
    label: str,
) -> MeasurementValidityStatus:
    try:
        status = (
            value
            if isinstance(value, MeasurementValidityStatus)
            else MeasurementValidityStatus(value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is unsupported") from error
    if status not in allowed:
        raise ValueError(f"{label} is unsupported")
    return status


@dataclass(frozen=True)
class ExactInvarianceFinding:
    check_name: str
    status: MeasurementValidityStatus | str
    original_hash: str
    transformed_hash: str
    details_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "check_name",
            _text(self.check_name, label="measurement invariance check name"),
        )
        status = _validity_status(
            self.status,
            allowed=_EXACT_STATUSES,
            label="measurement exact-invariance status",
        )
        object.__setattr__(self, "status", status)
        original = _hash(
            self.original_hash,
            label="measurement invariance original hash",
        )
        transformed = _hash(
            self.transformed_hash,
            label="measurement invariance transformed hash",
        )
        if (
            status is MeasurementValidityStatus.EXACT_INVARIANCE_MET
            and original != transformed
        ):
            raise ValueError("exact invariance cannot be met when hashes differ")
        if (
            status is MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
            and original == transformed
        ):
            raise ValueError("exact invariance cannot fail when hashes match")
        object.__setattr__(self, "original_hash", original)
        object.__setattr__(self, "transformed_hash", transformed)
        object.__setattr__(
            self,
            "details_hash",
            _hash(self.details_hash, label="measurement invariance details hash"),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "original_hash": self.original_hash,
            "transformed_hash": self.transformed_hash,
            "details_hash": self.details_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def classify_material_reversal(
    deltas: object,
    references: object,
) -> MeasurementValidityStatus:
    try:
        raw_deltas = tuple(deltas)
        raw_references = tuple(references)
    except TypeError as error:
        raise TypeError("measurement deltas and references must be sequences") from error
    if not raw_deltas or len(raw_deltas) != len(raw_references):
        raise ValueError(
            "measurement deltas and references must be non-empty aligned sequences"
        )
    canonical_deltas = tuple(
        _finite(value, label="measurement comparison delta")
        for value in raw_deltas
    )
    canonical_references = tuple(
        _finite(value, label="measurement comparison reference")
        for value in raw_references
    )
    if any(reference <= 0.0 for reference in canonical_references):
        raise ValueError("measurement comparison references must be positive")

    negative_material = any(
        delta <= -reference
        for delta, reference in zip(
            canonical_deltas,
            canonical_references,
            strict=True,
        )
    )
    positive_material = any(
        delta >= reference
        for delta, reference in zip(
            canonical_deltas,
            canonical_references,
            strict=True,
        )
    )
    if negative_material and positive_material:
        return MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT
    if any(delta < 0.0 for delta in canonical_deltas) and any(
        delta > 0.0 for delta in canonical_deltas
    ):
        return MeasurementValidityStatus.INCONCLUSIVE_SENSITIVITY
    return MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT


@dataclass(frozen=True)
class MeasurementDependenceFinding:
    dimension: str
    score: MeasurementScore | str
    aggregation: MeasurementAggregation | str
    task: str
    model_pair: tuple[str, str]
    deltas: tuple[float, ...]
    references: tuple[float, ...]
    status: MeasurementValidityStatus | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dimension",
            _text(self.dimension, label="measurement dependence dimension"),
        )
        try:
            score = (
                self.score
                if isinstance(self.score, MeasurementScore)
                else MeasurementScore(self.score)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("measurement dependence score is unsupported") from error
        object.__setattr__(self, "score", score)
        try:
            aggregation = (
                self.aggregation
                if isinstance(self.aggregation, MeasurementAggregation)
                else MeasurementAggregation(self.aggregation)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "measurement dependence aggregation is unsupported"
            ) from error
        object.__setattr__(self, "aggregation", aggregation)
        object.__setattr__(
            self,
            "task",
            _text(self.task, label="measurement dependence task"),
        )

        try:
            pair = tuple(self.model_pair)
        except TypeError as error:
            raise TypeError("measurement dependence model pair must be a sequence") from error
        if len(pair) != 2 or any(name not in _FAMILY_ORDER for name in pair):
            raise ValueError("measurement dependence model pair is unsupported")
        if pair[0] == pair[1]:
            raise ValueError("measurement dependence model pair must be distinct")
        canonical_pair = tuple(sorted(pair, key=_FAMILY_ORDER.index))
        if pair != canonical_pair:
            raise ValueError("measurement dependence model pair must be canonical")
        object.__setattr__(self, "model_pair", pair)

        try:
            raw_deltas = tuple(self.deltas)
            raw_references = tuple(self.references)
        except TypeError as error:
            raise TypeError(
                "measurement dependence values must be sequences"
            ) from error
        canonical_deltas = tuple(
            _finite(value, label="measurement dependence delta")
            for value in raw_deltas
        )
        canonical_references = tuple(
            _finite(value, label="measurement dependence reference")
            for value in raw_references
        )
        if not canonical_deltas or len(canonical_deltas) != len(
            canonical_references
        ):
            raise ValueError(
                "measurement dependence values must be non-empty and aligned"
            )
        if any(reference <= 0.0 for reference in canonical_references):
            raise ValueError("measurement dependence references must be positive")
        object.__setattr__(self, "deltas", canonical_deltas)
        object.__setattr__(self, "references", canonical_references)

        status = _validity_status(
            self.status,
            allowed=_DEPENDENCE_STATUSES,
            label="measurement dependence status",
        )
        if status is not MeasurementValidityStatus.NOT_ESTABLISHED:
            expected = classify_material_reversal(
                canonical_deltas,
                canonical_references,
            )
            if status is not expected:
                raise ValueError(
                    "measurement dependence status disagrees with frozen rule"
                )
        object.__setattr__(self, "status", status)

    def identity_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "score": self.score.value,
            "aggregation": self.aggregation.value,
            "task": self.task,
            "model_pair": self.model_pair,
            "deltas": self.deltas,
            "references": self.references,
            "status": self.status.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def permute_binary_metric_map(values: object) -> dict[str, float]:
    canonical = dict(_target_rows(values))
    return {
        _TARGET_KEYS[0]: canonical[_TARGET_KEYS[1]],
        _TARGET_KEYS[1]: canonical[_TARGET_KEYS[0]],
    }


def evaluate_categorical_coordinate_invariance(
    target: object,
    prediction: object,
) -> ExactInvarianceFinding:
    canonical_target = dict(_target_rows(target))
    canonical_prediction = dict(_target_rows(prediction))
    transformed_target = permute_binary_metric_map(canonical_target)
    transformed_prediction = permute_binary_metric_map(canonical_prediction)

    original_losses = {
        MeasurementScore.BRIER.value: evaluate_metric_loss(
            two_stage_brier_loss(),
            canonical_prediction,
            canonical_target,
        ),
        MeasurementScore.LOG.value: evaluate_metric_loss(
            two_stage_log_loss(),
            canonical_prediction,
            canonical_target,
        ),
    }
    transformed_losses = {
        MeasurementScore.BRIER.value: evaluate_metric_loss(
            two_stage_brier_loss(),
            transformed_prediction,
            transformed_target,
        ),
        MeasurementScore.LOG.value: evaluate_metric_loss(
            two_stage_log_loss(),
            transformed_prediction,
            transformed_target,
        ),
    }
    original_payload = {
        "target": _target_rows(canonical_target),
        "prediction": _target_rows(canonical_prediction),
        "losses": original_losses,
    }
    transformed_payload = {
        "target": _target_rows(permute_binary_metric_map(transformed_target)),
        "prediction": _target_rows(
            permute_binary_metric_map(transformed_prediction)
        ),
        "losses": transformed_losses,
    }
    original_hash = stable_content_hash(original_payload)
    transformed_hash = stable_content_hash(transformed_payload)
    status = (
        MeasurementValidityStatus.EXACT_INVARIANCE_MET
        if original_hash == transformed_hash
        else MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
    )
    return ExactInvarianceFinding(
        check_name="binary_coordinate_swap",
        status=status,
        original_hash=original_hash,
        transformed_hash=transformed_hash,
        details_hash=stable_content_hash(
            {
                "permuted_target": _target_rows(transformed_target),
                "permuted_prediction": _target_rows(transformed_prediction),
                "original_losses": original_losses,
                "transformed_losses": transformed_losses,
            }
        ),
    )


@dataclass(frozen=True)
class MeasurementAuditCase:
    role: ObservationPartitionRole | str
    scenario: Scenario
    target: tuple[tuple[str, float], ...]
    task_variant: str
    participant_group_hash: str
    trial_index: int
    record_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _role(self.role))
        if not isinstance(self.scenario, Scenario):
            raise TypeError("measurement audit case requires a Scenario")
        if _contains_direct_identity(self.scenario.payload):
            raise ValueError(
                "measurement scenario cannot expose a direct participant identity"
            )
        object.__setattr__(self, "target", _target_rows(self.target))
        task = _text(self.task_variant, label="measurement task variant")
        if task not in _TASK_ORDER:
            raise ValueError("measurement task variant is unsupported")
        if self.scenario.payload.get("task_variant") != task:
            raise ValueError("measurement scenario task variant changed")
        object.__setattr__(self, "task_variant", task)
        object.__setattr__(
            self,
            "participant_group_hash",
            _hash(
                self.participant_group_hash,
                label="measurement opaque participant group hash",
            ),
        )
        if (
            not isinstance(self.trial_index, int)
            or isinstance(self.trial_index, bool)
            or self.trial_index < 0
        ):
            raise ValueError("measurement trial index must be a non-negative integer")
        object.__setattr__(
            self,
            "record_hash",
            _hash(self.record_hash, label="measurement record hash"),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "scenario": {
                "id": self.scenario.id,
                "content_hash": self.scenario.content_hash,
            },
            "target": self.target,
            "task_variant": self.task_variant,
            "participant_group_hash": self.participant_group_hash,
            "trial_index": self.trial_index,
            "record_hash": self.record_hash,
        }

    @property
    def case_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _role_hash_pairs(
    value: object,
    *,
    label: str,
) -> tuple[tuple[ObservationPartitionRole, str], ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence") from error
    canonical: dict[ObservationPartitionRole, str] = {}
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(f"{label} rows must be role/hash pairs")
        selected_role = _role(row[0])
        if selected_role in canonical:
            raise ValueError(f"{label} roles must be unique")
        canonical[selected_role] = _hash(row[1], label=f"{label} hash")
    if set(canonical) != set(_ALLOWED_ROLES):
        raise ValueError(f"{label} must contain TRAIN and SELECTION_VALIDATION")
    return tuple((role, canonical[role]) for role in _ALLOWED_ROLES)


@dataclass(frozen=True)
class MeasurementAuditInput:
    claim_scope: str
    source_manifest_hash: str
    source_snapshot_hash: str
    transform_hash: str
    participant_assignment_hash: str
    dataset_hash: str
    target_spec_hash: str
    allowed_partition_hashes: tuple[tuple[ObservationPartitionRole, str], ...]
    allowed_target_report_hashes: tuple[tuple[ObservationPartitionRole, str], ...]
    allowed_case_commitments: tuple[tuple[ObservationPartitionRole, str], ...]
    frozen_candidates: tuple[FrozenModelSpec, ...]
    excluded_final_partition_hash: str
    excluded_final_target_hash: str
    cases: tuple[MeasurementAuditCase, ...]

    def __post_init__(self) -> None:
        if self.claim_scope != MEASUREMENT_CLAIM_SCOPE:
            raise ValueError("measurement audit claim scope changed")
        for attribute, label in (
            ("source_manifest_hash", "source manifest hash"),
            ("source_snapshot_hash", "source snapshot hash"),
            ("transform_hash", "transform hash"),
            ("participant_assignment_hash", "participant assignment hash"),
            ("dataset_hash", "dataset hash"),
            ("target_spec_hash", "target specification hash"),
        ):
            object.__setattr__(
                self,
                attribute,
                _hash(getattr(self, attribute), label=f"measurement {label}"),
            )
        object.__setattr__(
            self,
            "allowed_partition_hashes",
            _role_hash_pairs(
                self.allowed_partition_hashes,
                label="measurement allowed partition hashes",
            ),
        )
        object.__setattr__(
            self,
            "allowed_target_report_hashes",
            _role_hash_pairs(
                self.allowed_target_report_hashes,
                label="measurement allowed target report hashes",
            ),
        )
        object.__setattr__(
            self,
            "allowed_case_commitments",
            _role_hash_pairs(
                self.allowed_case_commitments,
                label="measurement allowed case commitments",
            ),
        )

        candidates = tuple(self.frozen_candidates)
        if len(candidates) != 3:
            raise ValueError("measurement audit requires exactly three frozen candidates")
        if any(not isinstance(candidate, FrozenModelSpec) for candidate in candidates):
            raise TypeError("measurement candidates must be FrozenModelSpec values")
        by_name = {candidate.name: candidate for candidate in candidates}
        if set(by_name) != set(_FAMILY_ORDER) or len(by_name) != len(candidates):
            raise ValueError(
                "measurement candidates must be reactive, intentional, and planning"
            )
        candidates = tuple(by_name[name] for name in _FAMILY_ORDER)
        if len({candidate.content_hash for candidate in candidates}) != 3:
            raise ValueError("measurement candidate identities must be unique")
        object.__setattr__(self, "frozen_candidates", candidates)

        cases = tuple(self.cases)
        if not cases or any(not isinstance(case, MeasurementAuditCase) for case in cases):
            raise TypeError("measurement audit cases must contain MeasurementAuditCase values")
        cases = tuple(sorted(cases, key=lambda item: (item.role.value, item.case_hash)))
        if {case.role for case in cases} != set(_ALLOWED_ROLES):
            raise ValueError(
                "measurement audit cases must contain both TRAIN and SELECTION_VALIDATION"
            )
        uniqueness_groups = (
            ("case", tuple(case.case_hash for case in cases)),
            ("record", tuple(case.record_hash for case in cases)),
            ("scenario", tuple(case.scenario.id for case in cases)),
            (
                "participant/trial",
                tuple(
                    (
                        case.task_variant,
                        case.participant_group_hash,
                        case.trial_index,
                    )
                    for case in cases
                ),
            ),
        )
        for label, values in uniqueness_groups:
            if len(set(values)) != len(values):
                raise ValueError(f"measurement {label} identities must be unique")
        object.__setattr__(self, "cases", cases)

        declared_commitments = dict(self.allowed_case_commitments)
        for role in _ALLOWED_ROLES:
            role_cases = tuple(
                sorted(
                    (case for case in cases if case.role is role),
                    key=lambda item: item.case_hash,
                )
            )
            actual = stable_content_hash(
                tuple(case.record_hash for case in role_cases)
            )
            if declared_commitments[role] != actual:
                raise ValueError(
                    f"measurement {role.value} case commitment changed"
                )

        final_partition = _hash(
            self.excluded_final_partition_hash,
            label="measurement excluded FINAL partition hash",
        )
        final_target = _hash(
            self.excluded_final_target_hash,
            label="measurement excluded FINAL target hash",
        )
        if final_partition == final_target:
            raise ValueError("measurement excluded FINAL identities must be distinct")
        object.__setattr__(self, "excluded_final_partition_hash", final_partition)
        object.__setattr__(self, "excluded_final_target_hash", final_target)

    @property
    def empirical_anchor_hash(self) -> str:
        return stable_content_hash(
            {
                "claim_scope": self.claim_scope,
                "source_manifest_hash": self.source_manifest_hash,
                "source_snapshot_hash": self.source_snapshot_hash,
                "transform_hash": self.transform_hash,
                "participant_assignment_hash": self.participant_assignment_hash,
                "dataset_hash": self.dataset_hash,
                "target_spec_hash": self.target_spec_hash,
                "allowed_partition_hashes": tuple(
                    (role.value, value)
                    for role, value in self.allowed_partition_hashes
                ),
                "allowed_target_report_hashes": tuple(
                    (role.value, value)
                    for role, value in self.allowed_target_report_hashes
                ),
                "frozen_candidate_hashes": tuple(
                    candidate.content_hash for candidate in self.frozen_candidates
                ),
                "excluded_final_partition_hash": self.excluded_final_partition_hash,
                "excluded_final_target_hash": self.excluded_final_target_hash,
            }
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "claim_scope": self.claim_scope,
            "empirical_anchor_hash": self.empirical_anchor_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "transform_hash": self.transform_hash,
            "participant_assignment_hash": self.participant_assignment_hash,
            "dataset_hash": self.dataset_hash,
            "target_spec_hash": self.target_spec_hash,
            "allowed_partition_hashes": tuple(
                (role.value, value) for role, value in self.allowed_partition_hashes
            ),
            "allowed_target_report_hashes": tuple(
                (role.value, value)
                for role, value in self.allowed_target_report_hashes
            ),
            "allowed_case_commitments": tuple(
                (role.value, value) for role, value in self.allowed_case_commitments
            ),
            "frozen_candidates": tuple(
                candidate.identity_payload() for candidate in self.frozen_candidates
            ),
            "excluded_final_partition_hash": self.excluded_final_partition_hash,
            "excluded_final_target_hash": self.excluded_final_target_hash,
            "cases": tuple(case.identity_payload() for case in self.cases),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _enum_rows(
    value: object,
    *,
    enum_type: type[Enum],
    expected: tuple[Enum, ...],
    label: str,
) -> tuple[Enum, ...]:
    try:
        raw_rows = tuple(value)
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence") from error
    try:
        rows = tuple(
            item if isinstance(item, enum_type) else enum_type(item)
            for item in raw_rows
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} contains an unsupported value") from error
    if len(set(rows)) != len(rows):
        raise ValueError(f"{label} values must be unique")
    if set(rows) != set(expected):
        raise ValueError(f"{label} must contain the complete frozen set")
    return expected


@dataclass(frozen=True)
class MeasurementValidityProtocol:
    name: str
    version: str
    claim_scope: str
    empirical_anchor_hash: str
    allowed_roles: tuple[ObservationPartitionRole, ...]
    candidate_hashes: tuple[str, ...]
    seeds_by_role: tuple[tuple[ObservationPartitionRole, tuple[int, ...]], ...]
    semantic_fixture_hashes: tuple[str, ...]
    semantic_permutations: tuple[str, ...]
    task_strata: tuple[str, ...]
    aggregation_rules: tuple[MeasurementAggregation, ...]
    scores: tuple[MeasurementScore, ...]
    material_reversal_references: tuple[tuple[MeasurementScore, float], ...]
    participant_influence_rule: str
    stay_switch_definition: str
    excluded_final_partition_hash: str
    excluded_final_target_hash: str
    implementation_identities: tuple[tuple[str, Mapping[str, object]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="measurement protocol name"))
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="measurement protocol version"),
        )
        if self.claim_scope != MEASUREMENT_CLAIM_SCOPE:
            raise ValueError("measurement protocol claim scope changed")
        object.__setattr__(
            self,
            "empirical_anchor_hash",
            _hash(self.empirical_anchor_hash, label="measurement empirical anchor hash"),
        )
        roles = _enum_rows(
            self.allowed_roles,
            enum_type=ObservationPartitionRole,
            expected=_ALLOWED_ROLES,
            label="measurement protocol roles",
        )
        object.__setattr__(self, "allowed_roles", roles)

        candidate_hashes = tuple(
            _hash(value, label="measurement protocol candidate hash")
            for value in self.candidate_hashes
        )
        if len(candidate_hashes) != 3 or len(set(candidate_hashes)) != 3:
            raise ValueError(
                "measurement protocol requires exactly three unique candidate hashes"
            )
        object.__setattr__(self, "candidate_hashes", candidate_hashes)

        seeds_by_role = _seed_rows(self.seeds_by_role)
        object.__setattr__(self, "seeds_by_role", seeds_by_role)

        fixture_hashes = tuple(
            sorted(
                _hash(value, label="measurement semantic fixture hash")
                for value in self.semantic_fixture_hashes
            )
        )
        if not fixture_hashes or len(set(fixture_hashes)) != len(fixture_hashes):
            raise ValueError("measurement semantic fixture hashes must be non-empty and unique")
        object.__setattr__(self, "semantic_fixture_hashes", fixture_hashes)

        semantic_permutations = tuple(
            sorted(
                _text(value, label="measurement semantic permutation")
                for value in self.semantic_permutations
            )
        )
        if not semantic_permutations or len(set(semantic_permutations)) != len(
            semantic_permutations
        ):
            raise ValueError("measurement semantic permutations must be non-empty and unique")
        object.__setattr__(self, "semantic_permutations", semantic_permutations)

        task_strata = tuple(
            _text(value, label="measurement task stratum")
            for value in self.task_strata
        )
        if len(set(task_strata)) != len(task_strata):
            raise ValueError("measurement task strata must be unique")
        if set(task_strata) != set(_TASK_ORDER):
            raise ValueError("measurement task strata must be magic_carpet and spaceship")
        object.__setattr__(self, "task_strata", _TASK_ORDER)

        aggregations = _enum_rows(
            self.aggregation_rules,
            enum_type=MeasurementAggregation,
            expected=tuple(MeasurementAggregation),
            label="measurement aggregation rules",
        )
        object.__setattr__(self, "aggregation_rules", aggregations)
        scores = _enum_rows(
            self.scores,
            enum_type=MeasurementScore,
            expected=tuple(MeasurementScore),
            label="measurement scores",
        )
        object.__setattr__(self, "scores", scores)

        references = _reference_rows(self.material_reversal_references)
        object.__setattr__(self, "material_reversal_references", references)
        object.__setattr__(
            self,
            "participant_influence_rule",
            _text(
                self.participant_influence_rule,
                label="measurement participant influence rule",
            ),
        )
        object.__setattr__(
            self,
            "stay_switch_definition",
            _text(
                self.stay_switch_definition,
                label="measurement stay/switch definition",
            ),
        )
        final_partition = _hash(
            self.excluded_final_partition_hash,
            label="measurement protocol excluded FINAL partition hash",
        )
        final_target = _hash(
            self.excluded_final_target_hash,
            label="measurement protocol excluded FINAL target hash",
        )
        if final_partition == final_target:
            raise ValueError("measurement protocol excluded FINAL identities must differ")
        object.__setattr__(self, "excluded_final_partition_hash", final_partition)
        object.__setattr__(self, "excluded_final_target_hash", final_target)

        try:
            raw_implementations = tuple(self.implementation_identities)
        except TypeError as error:
            raise TypeError("measurement implementation identities must be a sequence") from error
        implementations: list[tuple[str, Mapping[str, object]]] = []
        names: set[str] = set()
        for row in raw_implementations:
            if not isinstance(row, tuple) or len(row) != 2:
                raise ValueError(
                    "measurement implementation identities must be name/mapping pairs"
                )
            name = _text(row[0], label="measurement implementation identity name")
            if name in names:
                raise ValueError("measurement implementation identity names must be unique")
            names.add(name)
            implementations.append(
                (
                    name,
                    _freeze_mapping(
                        row[1],
                        label=f"measurement implementation identity {name!r}",
                    ),
                )
            )
        if not implementations:
            raise ValueError("measurement protocol requires implementation identities")
        object.__setattr__(
            self,
            "implementation_identities",
            tuple(sorted(implementations, key=lambda item: item[0])),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "claim_scope": self.claim_scope,
            "empirical_anchor_hash": self.empirical_anchor_hash,
            "allowed_roles": tuple(role.value for role in self.allowed_roles),
            "candidate_hashes": self.candidate_hashes,
            "seeds_by_role": tuple(
                (role.value, seeds) for role, seeds in self.seeds_by_role
            ),
            "semantic_fixture_hashes": self.semantic_fixture_hashes,
            "semantic_permutations": self.semantic_permutations,
            "task_strata": self.task_strata,
            "aggregation_rules": tuple(
                aggregation.value for aggregation in self.aggregation_rules
            ),
            "scores": tuple(score.value for score in self.scores),
            "material_reversal_references": tuple(
                (score.value, reference)
                for score, reference in self.material_reversal_references
            ),
            "participant_influence_rule": self.participant_influence_rule,
            "stay_switch_definition": self.stay_switch_definition,
            "excluded_final_partition_hash": self.excluded_final_partition_hash,
            "excluded_final_target_hash": self.excluded_final_target_hash,
            "implementation_identities": self.implementation_identities,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    def to_payload(self) -> dict[str, object]:
        payload = _thaw(self.identity_payload())
        assert isinstance(payload, dict)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MeasurementValidityProtocol":
        if not isinstance(payload, Mapping):
            raise TypeError("measurement protocol payload must be a mapping")
        expected = {
            "name",
            "version",
            "claim_scope",
            "empirical_anchor_hash",
            "allowed_roles",
            "candidate_hashes",
            "seeds_by_role",
            "semantic_fixture_hashes",
            "semantic_permutations",
            "task_strata",
            "aggregation_rules",
            "scores",
            "material_reversal_references",
            "participant_influence_rule",
            "stay_switch_definition",
            "excluded_final_partition_hash",
            "excluded_final_target_hash",
            "implementation_identities",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown:
            raise ValueError(
                f"measurement protocol payload has unknown fields: {tuple(sorted(unknown))}"
            )
        if missing:
            raise ValueError(
                f"measurement protocol payload has missing fields: {tuple(sorted(missing))}"
            )
        return cls(
            name=payload["name"],
            version=payload["version"],
            claim_scope=payload["claim_scope"],
            empirical_anchor_hash=payload["empirical_anchor_hash"],
            allowed_roles=tuple(payload["allowed_roles"]),
            candidate_hashes=tuple(payload["candidate_hashes"]),
            seeds_by_role=tuple(
                (row[0], tuple(row[1])) for row in payload["seeds_by_role"]
            ),
            semantic_fixture_hashes=tuple(payload["semantic_fixture_hashes"]),
            semantic_permutations=tuple(payload["semantic_permutations"]),
            task_strata=tuple(payload["task_strata"]),
            aggregation_rules=tuple(payload["aggregation_rules"]),
            scores=tuple(payload["scores"]),
            material_reversal_references=tuple(
                (row[0], row[1])
                for row in payload["material_reversal_references"]
            ),
            participant_influence_rule=payload["participant_influence_rule"],
            stay_switch_definition=payload["stay_switch_definition"],
            excluded_final_partition_hash=payload[
                "excluded_final_partition_hash"
            ],
            excluded_final_target_hash=payload["excluded_final_target_hash"],
            implementation_identities=tuple(
                (row[0], row[1]) for row in payload["implementation_identities"]
            ),
        )

    def build_manifest(self, audit_input: MeasurementAuditInput) -> ExperimentManifest:
        if not isinstance(audit_input, MeasurementAuditInput):
            raise TypeError("measurement protocol manifest requires MeasurementAuditInput")
        if audit_input.empirical_anchor_hash != self.empirical_anchor_hash:
            raise ValueError("measurement protocol empirical anchor changed")
        if tuple(
            candidate.content_hash for candidate in audit_input.frozen_candidates
        ) != self.candidate_hashes:
            raise ValueError("measurement protocol frozen candidates changed")
        if (
            audit_input.excluded_final_partition_hash
            != self.excluded_final_partition_hash
            or audit_input.excluded_final_target_hash
            != self.excluded_final_target_hash
        ):
            raise ValueError("measurement protocol excluded FINAL lineage changed")
        return ExperimentManifest(
            stage=ExperimentStage.MEASUREMENT_AUDIT,
            inputs=self.identity_payload(),
            parent_hashes=(audit_input.content_hash,),
        )


def _canonical_numeric_zero(value: float) -> float:
    return (
        0.0
        if math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-15)
        else float(value)
    )


def average_seed_metrics(seed_metrics: object) -> tuple[tuple[str, float], ...]:
    try:
        raw_rows = tuple(seed_metrics)
    except TypeError as error:
        raise TypeError("measurement seed metrics must be a sequence") from error
    if not raw_rows:
        raise ValueError("measurement seed metrics must be non-empty")
    canonical = tuple(_target_rows(row) for row in raw_rows)
    averaged = tuple(
        (
            key,
            _canonical_numeric_zero(
                math.fsum(dict(row)[key] for row in canonical) / len(canonical)
            ),
        )
        for key in _TARGET_KEYS
    )
    return _target_rows(averaged)


@dataclass(frozen=True)
class MeasurementSeedPrediction:
    case_hash: str
    scenario_hash: str
    role: ObservationPartitionRole | str
    model_name: str
    seed: int
    metrics: tuple[tuple[str, float], ...]
    run_manifest_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_hash",
            _hash(self.case_hash, label="measurement prediction case hash"),
        )
        object.__setattr__(
            self,
            "scenario_hash",
            _hash(self.scenario_hash, label="measurement prediction scenario hash"),
        )
        object.__setattr__(self, "role", _role(self.role))
        object.__setattr__(
            self,
            "model_name",
            _text(self.model_name, label="measurement prediction model name"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("measurement prediction seed must be an integer")
        object.__setattr__(self, "metrics", _target_rows(self.metrics))
        object.__setattr__(
            self,
            "run_manifest_hash",
            _hash(
                self.run_manifest_hash,
                label="measurement prediction run manifest hash",
            ),
        )

    @property
    def metric_map(self) -> dict[str, float]:
        return dict(self.metrics)

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_hash": self.case_hash,
            "scenario_hash": self.scenario_hash,
            "role": self.role.value,
            "model_name": self.model_name,
            "seed": self.seed,
            "metrics": self.metrics,
            "run_manifest_hash": self.run_manifest_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _prediction_row_key(
    row: MeasurementSeedPrediction,
) -> tuple[int, str, int, str]:
    return (
        _ALLOWED_ROLES.index(row.role),
        row.case_hash,
        row.seed,
        row.scenario_hash,
    )


@dataclass(frozen=True)
class MeasurementModelPrediction:
    model_name: str
    candidate_hash: str
    rows: tuple[MeasurementSeedPrediction, ...]

    def __post_init__(self) -> None:
        name = _text(self.model_name, label="measurement prediction model name")
        object.__setattr__(self, "model_name", name)
        object.__setattr__(
            self,
            "candidate_hash",
            _hash(
                self.candidate_hash,
                label="measurement prediction candidate hash",
            ),
        )
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("measurement model prediction requires rows")
        if any(not isinstance(row, MeasurementSeedPrediction) for row in rows):
            raise TypeError(
                "measurement model prediction rows must be MeasurementSeedPrediction values"
            )
        if any(row.model_name != name for row in rows):
            raise ValueError("measurement prediction row model name changed")
        keys = tuple((row.case_hash, row.seed) for row in rows)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "measurement model prediction case/seed pairs must be unique"
            )
        object.__setattr__(self, "rows", tuple(sorted(rows, key=_prediction_row_key)))

    @property
    def row_map(self) -> dict[tuple[str, int], MeasurementSeedPrediction]:
        return {(row.case_hash, row.seed): row for row in self.rows}

    def identity_payload(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "candidate_hash": self.candidate_hash,
            "rows": tuple(row.identity_payload() for row in self.rows),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _model_order(model: MeasurementModelPrediction) -> tuple[int, str]:
    if model.model_name in _FAMILY_ORDER:
        return (_FAMILY_ORDER.index(model.model_name), model.model_name)
    return (len(_FAMILY_ORDER), model.model_name)


def _prediction_manifest(
    *,
    protocol_hash: str,
    audit_input_hash: str,
    models: tuple[MeasurementModelPrediction, ...],
    execution_manifest_hashes: tuple[str, ...],
) -> ExperimentManifest:
    return ExperimentManifest(
        stage=ExperimentStage.MEASUREMENT_AUDIT,
        inputs={
            "artifact_type": "measurement_prediction",
            "protocol_hash": protocol_hash,
            "audit_input_hash": audit_input_hash,
            "model_prediction_hashes": tuple(
                model.content_hash for model in models
            ),
            "execution_manifest_hashes": execution_manifest_hashes,
        },
        parent_hashes=execution_manifest_hashes,
    )


@dataclass(frozen=True)
class MeasurementPredictionArtifact:
    protocol_hash: str
    audit_input_hash: str
    models: tuple[MeasurementModelPrediction, ...]
    execution_manifest_hashes: tuple[str, ...]
    manifest: ExperimentManifest | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        protocol_hash = _hash(
            self.protocol_hash,
            label="measurement prediction protocol hash",
        )
        audit_input_hash = _hash(
            self.audit_input_hash,
            label="measurement prediction audit input hash",
        )
        object.__setattr__(self, "protocol_hash", protocol_hash)
        object.__setattr__(self, "audit_input_hash", audit_input_hash)

        models = tuple(self.models)
        if not models:
            raise ValueError("measurement prediction artifact requires models")
        if any(not isinstance(model, MeasurementModelPrediction) for model in models):
            raise TypeError(
                "measurement prediction artifact models must be MeasurementModelPrediction values"
            )
        models = tuple(sorted(models, key=_model_order))
        names = tuple(model.model_name for model in models)
        if len(set(names)) != len(names):
            raise ValueError("measurement prediction model names must be unique")
        coverage = tuple(
            (row.case_hash, row.scenario_hash, row.role.value, row.seed)
            for row in models[0].rows
        )
        if any(
            tuple(
                (row.case_hash, row.scenario_hash, row.role.value, row.seed)
                for row in model.rows
            )
            != coverage
            for model in models[1:]
        ):
            raise ValueError(
                "measurement prediction models must share exact coverage"
            )
        object.__setattr__(self, "models", models)

        row_manifests = tuple(
            row.run_manifest_hash for model in models for row in model.rows
        )
        if len(set(row_manifests)) != len(row_manifests):
            raise ValueError(
                "measurement prediction run manifest hashes must be globally unique"
            )
        try:
            raw_execution_manifests = tuple(self.execution_manifest_hashes)
        except TypeError as error:
            raise TypeError(
                "measurement execution manifest hashes must be a sequence"
            ) from error
        execution_manifests = tuple(
            _hash(value, label="measurement execution manifest hash")
            for value in raw_execution_manifests
        )
        if len(set(execution_manifests)) != len(execution_manifests):
            raise ValueError(
                "measurement execution manifest hashes must be unique"
            )
        execution_manifests = tuple(sorted(execution_manifests))
        if set(execution_manifests) != set(row_manifests):
            raise ValueError(
                "measurement execution manifest index does not match prediction rows"
            )
        object.__setattr__(
            self,
            "execution_manifest_hashes",
            execution_manifests,
        )

        expected_manifest = _prediction_manifest(
            protocol_hash=protocol_hash,
            audit_input_hash=audit_input_hash,
            models=models,
            execution_manifest_hashes=execution_manifests,
        )
        if self.manifest is None:
            object.__setattr__(self, "manifest", expected_manifest)
        else:
            if not isinstance(self.manifest, ExperimentManifest):
                raise TypeError(
                    "measurement prediction manifest must be ExperimentManifest"
                )
            if self.manifest.stage is not ExperimentStage.MEASUREMENT_AUDIT:
                raise ValueError(
                    "measurement prediction manifest stage must be measurement_audit"
                )
            if self.manifest.inputs.get("protocol_hash") != protocol_hash:
                raise ValueError("measurement prediction protocol hash mismatch")
            if self.manifest.inputs.get("audit_input_hash") != audit_input_hash:
                raise ValueError("measurement prediction audit input hash mismatch")
            if self.manifest.content_hash != expected_manifest.content_hash:
                raise ValueError("measurement prediction manifest binding changed")

    @property
    def model_map(self) -> dict[str, MeasurementModelPrediction]:
        return {model.model_name: model for model in self.models}

    def identity_payload(self) -> dict[str, object]:
        assert isinstance(self.manifest, ExperimentManifest)
        return {
            "protocol_hash": self.protocol_hash,
            "audit_input_hash": self.audit_input_hash,
            "models": tuple(model.identity_payload() for model in self.models),
            "execution_manifest_hashes": self.execution_manifest_hashes,
            "manifest_hash": self.manifest.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class MeasurementCaseLoss:
    case_hash: str
    role: ObservationPartitionRole | str
    task_variant: str
    participant_group_hash: str
    model_name: str
    score: MeasurementScore | str
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_hash",
            _hash(self.case_hash, label="measurement case loss case hash"),
        )
        object.__setattr__(self, "role", _role(self.role))
        task = _text(self.task_variant, label="measurement case loss task variant")
        if task not in _TASK_ORDER:
            raise ValueError("measurement case loss task variant is unsupported")
        object.__setattr__(self, "task_variant", task)
        object.__setattr__(
            self,
            "participant_group_hash",
            _hash(
                self.participant_group_hash,
                label="measurement case loss participant group hash",
            ),
        )
        model_name = _text(
            self.model_name,
            label="measurement case loss model name",
        )
        if model_name not in _FAMILY_ORDER:
            raise ValueError("measurement case loss model is unsupported")
        object.__setattr__(self, "model_name", model_name)
        try:
            score = (
                self.score
                if isinstance(self.score, MeasurementScore)
                else MeasurementScore(self.score)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("measurement case loss score is unsupported") from error
        object.__setattr__(self, "score", score)
        value = _finite(self.value, label="measurement case loss")
        if value < 0.0:
            raise ValueError("measurement case loss must be non-negative")
        object.__setattr__(self, "value", _canonical_numeric_zero(value))

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_hash": self.case_hash,
            "role": self.role.value,
            "task_variant": self.task_variant,
            "participant_group_hash": self.participant_group_hash,
            "model_name": self.model_name,
            "score": self.score.value,
            "value": self.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _measurement_losses(
    losses: object,
) -> tuple[tuple[MeasurementScore, MetricLoss], ...]:
    try:
        raw_losses = tuple(losses)
    except TypeError as error:
        raise TypeError("measurement scoring losses must be a sequence") from error
    expected = {
        MeasurementScore.BRIER: metric_loss_identity(two_stage_brier_loss()),
        MeasurementScore.LOG: metric_loss_identity(two_stage_log_loss()),
    }
    resolved: dict[MeasurementScore, MetricLoss] = {}
    for loss in raw_losses:
        identity = metric_loss_identity(loss)
        matching = tuple(
            score
            for score, expected_identity in expected.items()
            if identity == expected_identity
        )
        if len(matching) != 1:
            raise ValueError("measurement scoring loss changed")
        score = matching[0]
        if score in resolved:
            raise ValueError("measurement scoring losses must be unique")
        resolved[score] = loss
    if set(resolved) != set(MeasurementScore):
        raise ValueError("measurement scoring requires Brier and Log losses")
    return tuple((score, resolved[score]) for score in MeasurementScore)


def score_measurement_predictions(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
    losses: object,
) -> tuple[MeasurementCaseLoss, ...]:
    if not isinstance(audit_input, MeasurementAuditInput):
        raise TypeError("measurement scoring requires MeasurementAuditInput")
    if not isinstance(prediction_artifact, MeasurementPredictionArtifact):
        raise TypeError(
            "measurement scoring requires MeasurementPredictionArtifact"
        )
    if prediction_artifact.audit_input_hash != audit_input.content_hash:
        raise ValueError("measurement prediction audit input changed")
    scoring_losses = _measurement_losses(losses)

    candidate_map = {
        candidate.name: candidate for candidate in audit_input.frozen_candidates
    }
    model_map = prediction_artifact.model_map
    if tuple(model_map) != _FAMILY_ORDER:
        raise ValueError("measurement prediction candidate set changed")
    for model_name in _FAMILY_ORDER:
        if model_map[model_name].candidate_hash != candidate_map[model_name].content_hash:
            raise ValueError(
                f"measurement prediction candidate changed for {model_name!r}"
            )

    expected_coverage = {
        (
            case.case_hash,
            case.scenario.content_hash,
            case.role,
            seed,
        )
        for case in audit_input.cases
        for seed in _SEEDS_BY_ROLE[case.role]
    }
    for model_name in _FAMILY_ORDER:
        actual_coverage = {
            (row.case_hash, row.scenario_hash, row.role, row.seed)
            for row in model_map[model_name].rows
        }
        if actual_coverage != expected_coverage or len(
            model_map[model_name].rows
        ) != len(expected_coverage):
            raise ValueError(
                f"measurement prediction coverage changed for {model_name!r}"
            )

    rows: list[MeasurementCaseLoss] = []
    for case in audit_input.cases:
        seeds = _SEEDS_BY_ROLE[case.role]
        for model_name in _FAMILY_ORDER:
            prediction_map = model_map[model_name].row_map
            averaged = dict(
                average_seed_metrics(
                    tuple(
                        prediction_map[(case.case_hash, seed)].metric_map
                        for seed in seeds
                    )
                )
            )
            for score, loss in scoring_losses:
                rows.append(
                    MeasurementCaseLoss(
                        case_hash=case.case_hash,
                        role=case.role,
                        task_variant=case.task_variant,
                        participant_group_hash=case.participant_group_hash,
                        model_name=model_name,
                        score=score,
                        value=_canonical_numeric_zero(
                            evaluate_metric_loss(
                                loss,
                                averaged,
                                dict(case.target),
                            )
                        ),
                    )
                )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                _ALLOWED_ROLES.index(row.role),
                _TASK_ORDER.index(row.task_variant),
                row.case_hash,
                _FAMILY_ORDER.index(row.model_name),
                tuple(MeasurementScore).index(row.score),
            ),
        )
    )


def _aggregation(
    value: MeasurementAggregation | str,
    *,
    label: str,
) -> MeasurementAggregation:
    try:
        return (
            value
            if isinstance(value, MeasurementAggregation)
            else MeasurementAggregation(value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is unsupported") from error


def _score(value: MeasurementScore | str, *, label: str) -> MeasurementScore:
    try:
        return value if isinstance(value, MeasurementScore) else MeasurementScore(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is unsupported") from error


def _participant_loss_rows(
    participant_losses: object,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not isinstance(participant_losses, Mapping):
        raise TypeError("measurement participant losses must be a mapping")
    if not participant_losses:
        raise ValueError("measurement participant losses must be non-empty")
    rows: list[tuple[str, tuple[float, ...]]] = []
    for participant_hash, raw_losses in participant_losses.items():
        canonical_hash = _hash(
            participant_hash,
            label="measurement participant loss group hash",
        )
        try:
            losses = tuple(raw_losses)
        except TypeError as error:
            raise TypeError(
                "measurement participant loss values must be sequences"
            ) from error
        if not losses:
            raise ValueError(
                "measurement participant loss values must be non-empty"
            )
        canonical_losses = tuple(
            _finite(value, label="measurement participant loss")
            for value in losses
        )
        if any(value < 0.0 for value in canonical_losses):
            raise ValueError("measurement participant losses must be non-negative")
        rows.append((canonical_hash, canonical_losses))
    return tuple(sorted(rows))


def trial_equal_mean(participant_losses: object) -> float:
    rows = _participant_loss_rows(participant_losses)
    losses = tuple(value for _, values in rows for value in values)
    return _canonical_numeric_zero(math.fsum(losses) / len(losses))


def participant_equal_mean(participant_losses: object) -> float:
    rows = _participant_loss_rows(participant_losses)
    participant_means = tuple(
        math.fsum(values) / len(values) for _, values in rows
    )
    return _canonical_numeric_zero(
        math.fsum(participant_means) / len(participant_means)
    )


@dataclass(frozen=True)
class MeasurementAggregateLoss:
    model_name: str
    score: MeasurementScore | str
    aggregation: MeasurementAggregation | str
    task_variant: str
    role: ObservationPartitionRole | str
    mean_loss: float
    case_count: int
    participant_count: int

    def __post_init__(self) -> None:
        model_name = _text(
            self.model_name,
            label="measurement aggregate model name",
        )
        if model_name not in _FAMILY_ORDER:
            raise ValueError("measurement aggregate model is unsupported")
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(
            self,
            "score",
            _score(self.score, label="measurement aggregate score"),
        )
        object.__setattr__(
            self,
            "aggregation",
            _aggregation(
                self.aggregation,
                label="measurement aggregate aggregation",
            ),
        )
        task = _text(
            self.task_variant,
            label="measurement aggregate task variant",
        )
        if task not in _TASK_ORDER:
            raise ValueError("measurement aggregate task variant is unsupported")
        object.__setattr__(self, "task_variant", task)
        object.__setattr__(self, "role", _role(self.role))
        mean_loss = _finite(
            self.mean_loss,
            label="measurement aggregate mean loss",
        )
        if mean_loss < 0.0:
            raise ValueError("measurement aggregate mean loss must be non-negative")
        object.__setattr__(
            self,
            "mean_loss",
            _canonical_numeric_zero(mean_loss),
        )
        for attribute in ("case_count", "participant_count"):
            value = getattr(self, attribute)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"measurement aggregate {attribute} must be a positive integer"
                )
        if self.participant_count > self.case_count:
            raise ValueError(
                "measurement aggregate participant count cannot exceed case count"
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "score": self.score.value,
            "aggregation": self.aggregation.value,
            "task_variant": self.task_variant,
            "role": self.role.value,
            "mean_loss": self.mean_loss,
            "case_count": self.case_count,
            "participant_count": self.participant_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class MeasurementPairwiseDelta:
    first_model: str
    second_model: str
    score: MeasurementScore | str
    aggregation: MeasurementAggregation | str
    task_variant: str
    role: ObservationPartitionRole | str
    delta: float

    def __post_init__(self) -> None:
        pair = (self.first_model, self.second_model)
        if any(name not in _FAMILY_ORDER for name in pair) or pair[0] == pair[1]:
            raise ValueError("measurement pairwise models are unsupported")
        if pair != tuple(sorted(pair, key=_FAMILY_ORDER.index)):
            raise ValueError("measurement pairwise models must be canonical")
        object.__setattr__(
            self,
            "score",
            _score(self.score, label="measurement pairwise score"),
        )
        object.__setattr__(
            self,
            "aggregation",
            _aggregation(
                self.aggregation,
                label="measurement pairwise aggregation",
            ),
        )
        task = _text(
            self.task_variant,
            label="measurement pairwise task variant",
        )
        if task not in _TASK_ORDER:
            raise ValueError("measurement pairwise task variant is unsupported")
        object.__setattr__(self, "task_variant", task)
        object.__setattr__(self, "role", _role(self.role))
        object.__setattr__(
            self,
            "delta",
            _canonical_numeric_zero(
                _finite(self.delta, label="measurement pairwise delta")
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "first_model": self.first_model,
            "second_model": self.second_model,
            "score": self.score.value,
            "aggregation": self.aggregation.value,
            "task_variant": self.task_variant,
            "role": self.role.value,
            "delta": self.delta,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ParticipantInfluenceRange:
    role: ObservationPartitionRole | str
    task_variant: str
    score: MeasurementScore | str
    aggregation: MeasurementAggregation | str
    first_model: str
    second_model: str
    minimum_delta: float | None
    maximum_delta: float | None
    omitted_participant_count: int
    status: MeasurementValidityStatus | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _role(self.role))
        task = _text(
            self.task_variant,
            label="measurement influence task variant",
        )
        if task not in _TASK_ORDER:
            raise ValueError("measurement influence task variant is unsupported")
        object.__setattr__(self, "task_variant", task)
        object.__setattr__(
            self,
            "score",
            _score(self.score, label="measurement influence score"),
        )
        object.__setattr__(
            self,
            "aggregation",
            _aggregation(
                self.aggregation,
                label="measurement influence aggregation",
            ),
        )
        pair = (self.first_model, self.second_model)
        if any(name not in _FAMILY_ORDER for name in pair) or pair[0] == pair[1]:
            raise ValueError("measurement influence model pair is unsupported")
        if pair != tuple(sorted(pair, key=_FAMILY_ORDER.index)):
            raise ValueError("measurement influence model pair must be canonical")
        if (
            isinstance(self.omitted_participant_count, bool)
            or not isinstance(self.omitted_participant_count, int)
            or self.omitted_participant_count <= 0
        ):
            raise ValueError(
                "measurement omitted participant count must be a positive integer"
            )
        status = _validity_status(
            self.status,
            allowed=_DEPENDENCE_STATUSES,
            label="measurement influence status",
        )
        object.__setattr__(self, "status", status)
        if status is MeasurementValidityStatus.NOT_ESTABLISHED:
            if self.minimum_delta is not None or self.maximum_delta is not None:
                raise ValueError(
                    "not-established measurement influence cannot contain a range"
                )
            return
        if self.minimum_delta is None or self.maximum_delta is None:
            raise ValueError(
                "established measurement influence requires a complete range"
            )
        minimum = _canonical_numeric_zero(
            _finite(self.minimum_delta, label="measurement influence minimum")
        )
        maximum = _canonical_numeric_zero(
            _finite(self.maximum_delta, label="measurement influence maximum")
        )
        if minimum > maximum:
            raise ValueError(
                "measurement influence minimum cannot exceed maximum"
            )
        object.__setattr__(self, "minimum_delta", minimum)
        object.__setattr__(self, "maximum_delta", maximum)

    def identity_payload(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "task_variant": self.task_variant,
            "score": self.score.value,
            "aggregation": self.aggregation.value,
            "first_model": self.first_model,
            "second_model": self.second_model,
            "minimum_delta": self.minimum_delta,
            "maximum_delta": self.maximum_delta,
            "omitted_participant_count": self.omitted_participant_count,
            "status": self.status.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _aggregate_sort_key(
    row: MeasurementAggregateLoss,
) -> tuple[int, int, int, int, int]:
    return (
        _ALLOWED_ROLES.index(row.role),
        _TASK_ORDER.index(row.task_variant),
        tuple(MeasurementScore).index(row.score),
        tuple(MeasurementAggregation).index(row.aggregation),
        _FAMILY_ORDER.index(row.model_name),
    )


def aggregate_measurement_losses(
    rows: object,
    aggregation: MeasurementAggregation | str,
) -> tuple[MeasurementAggregateLoss, ...]:
    selected_aggregation = _aggregation(
        aggregation,
        label="measurement aggregation",
    )
    try:
        canonical_rows = tuple(rows)
    except TypeError as error:
        raise TypeError("measurement case losses must be a sequence") from error
    if not canonical_rows:
        raise ValueError("measurement case losses must be non-empty")
    if any(not isinstance(row, MeasurementCaseLoss) for row in canonical_rows):
        raise TypeError(
            "measurement aggregation rows must be MeasurementCaseLoss values"
        )
    row_keys = tuple(
        (row.case_hash, row.model_name, row.score) for row in canonical_rows
    )
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("measurement case loss rows contain a duplicate")

    groups: dict[
        tuple[
            ObservationPartitionRole,
            str,
            MeasurementScore,
            str,
        ],
        list[MeasurementCaseLoss],
    ] = {}
    for row in canonical_rows:
        key = (row.role, row.task_variant, row.score, row.model_name)
        groups.setdefault(key, []).append(row)

    aggregates: list[MeasurementAggregateLoss] = []
    for (role, task, score, model_name), group_rows in groups.items():
        participant_losses: dict[str, list[float]] = {}
        for row in group_rows:
            participant_losses.setdefault(row.participant_group_hash, []).append(
                row.value
            )
        mean_loss = (
            trial_equal_mean(participant_losses)
            if selected_aggregation is MeasurementAggregation.TRIAL_EQUAL
            else participant_equal_mean(participant_losses)
        )
        aggregates.append(
            MeasurementAggregateLoss(
                model_name=model_name,
                score=score,
                aggregation=selected_aggregation,
                task_variant=task,
                role=role,
                mean_loss=mean_loss,
                case_count=len(group_rows),
                participant_count=len(participant_losses),
            )
        )
    return tuple(sorted(aggregates, key=_aggregate_sort_key))


def _pairwise_sort_key(
    row: MeasurementPairwiseDelta,
) -> tuple[int, int, int, int, int, int]:
    return (
        _ALLOWED_ROLES.index(row.role),
        _TASK_ORDER.index(row.task_variant),
        tuple(MeasurementScore).index(row.score),
        tuple(MeasurementAggregation).index(row.aggregation),
        _FAMILY_ORDER.index(row.first_model),
        _FAMILY_ORDER.index(row.second_model),
    )


def _dependence_sort_key(
    row: MeasurementDependenceFinding,
) -> tuple[str, int, int, str, int, int]:
    return (
        row.dimension,
        tuple(MeasurementScore).index(row.score),
        tuple(MeasurementAggregation).index(row.aggregation),
        row.task,
        _FAMILY_ORDER.index(row.model_pair[0]),
        _FAMILY_ORDER.index(row.model_pair[1]),
    )


def _influence_sort_key(
    row: ParticipantInfluenceRange,
) -> tuple[int, int, int, int, int, int]:
    return (
        _ALLOWED_ROLES.index(row.role),
        _TASK_ORDER.index(row.task_variant),
        tuple(MeasurementScore).index(row.score),
        tuple(MeasurementAggregation).index(row.aggregation),
        _FAMILY_ORDER.index(row.first_model),
        _FAMILY_ORDER.index(row.second_model),
    )


@dataclass(frozen=True)
class MeasurementRobustnessProfile:
    aggregate_losses: tuple[MeasurementAggregateLoss, ...]
    pairwise_deltas: tuple[MeasurementPairwiseDelta, ...]
    task_findings: tuple[MeasurementDependenceFinding, ...]
    aggregation_findings: tuple[MeasurementDependenceFinding, ...]
    score_findings: tuple[MeasurementDependenceFinding, ...]
    participant_influence: tuple[ParticipantInfluenceRange, ...]

    def __post_init__(self) -> None:
        specifications = (
            (
                "aggregate_losses",
                MeasurementAggregateLoss,
                _aggregate_sort_key,
            ),
            (
                "pairwise_deltas",
                MeasurementPairwiseDelta,
                _pairwise_sort_key,
            ),
            ("task_findings", MeasurementDependenceFinding, _dependence_sort_key),
            (
                "aggregation_findings",
                MeasurementDependenceFinding,
                _dependence_sort_key,
            ),
            ("score_findings", MeasurementDependenceFinding, _dependence_sort_key),
            (
                "participant_influence",
                ParticipantInfluenceRange,
                _influence_sort_key,
            ),
        )
        for attribute, row_type, sort_key in specifications:
            rows = tuple(getattr(self, attribute))
            if not rows:
                raise ValueError(f"measurement robustness {attribute} must be non-empty")
            if any(not isinstance(row, row_type) for row in rows):
                raise TypeError(
                    f"measurement robustness {attribute} contains the wrong row type"
                )
            rows = tuple(sorted(rows, key=sort_key))
            hashes = tuple(row.content_hash for row in rows)
            if len(set(hashes)) != len(hashes):
                raise ValueError(
                    f"measurement robustness {attribute} contains duplicate rows"
                )
            object.__setattr__(self, attribute, rows)

    def identity_payload(self) -> dict[str, object]:
        return {
            "aggregate_losses": tuple(
                row.identity_payload() for row in self.aggregate_losses
            ),
            "pairwise_deltas": tuple(
                row.identity_payload() for row in self.pairwise_deltas
            ),
            "task_findings": tuple(
                row.identity_payload() for row in self.task_findings
            ),
            "aggregation_findings": tuple(
                row.identity_payload() for row in self.aggregation_findings
            ),
            "score_findings": tuple(
                row.identity_payload() for row in self.score_findings
            ),
            "participant_influence": tuple(
                row.identity_payload() for row in self.participant_influence
            ),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


_MODEL_PAIRS = tuple(
    (_FAMILY_ORDER[first_index], _FAMILY_ORDER[second_index])
    for first_index in range(len(_FAMILY_ORDER))
    for second_index in range(first_index + 1, len(_FAMILY_ORDER))
)


def _validated_profile_rows(rows: object) -> tuple[MeasurementCaseLoss, ...]:
    try:
        canonical_rows = tuple(rows)
    except TypeError as error:
        raise TypeError("measurement robustness rows must be a sequence") from error
    if not canonical_rows:
        raise ValueError("measurement robustness rows must be non-empty")
    if any(not isinstance(row, MeasurementCaseLoss) for row in canonical_rows):
        raise TypeError(
            "measurement robustness rows must be MeasurementCaseLoss values"
        )
    keys = tuple(
        (row.case_hash, row.model_name, row.score) for row in canonical_rows
    )
    if len(set(keys)) != len(keys):
        raise ValueError("measurement robustness rows contain a duplicate")
    if {row.role for row in canonical_rows} != set(_ALLOWED_ROLES):
        raise ValueError("measurement robustness requires both partition roles")
    if {row.task_variant for row in canonical_rows} != set(_TASK_ORDER):
        raise ValueError("measurement robustness requires both task strata")
    if {row.score for row in canonical_rows} != set(MeasurementScore):
        raise ValueError("measurement robustness requires both scores")
    if {row.model_name for row in canonical_rows} != set(_FAMILY_ORDER):
        raise ValueError("measurement robustness requires every model")

    cases: dict[str, list[MeasurementCaseLoss]] = {}
    for row in canonical_rows:
        cases.setdefault(row.case_hash, []).append(row)
    expected_coordinates = {
        (model_name, score)
        for model_name in _FAMILY_ORDER
        for score in MeasurementScore
    }
    for case_hash, case_rows in cases.items():
        metadata = {
            (
                row.role,
                row.task_variant,
                row.participant_group_hash,
            )
            for row in case_rows
        }
        if len(metadata) != 1:
            raise ValueError(
                f"measurement robustness case metadata changed for {case_hash}"
            )
        coordinates = {(row.model_name, row.score) for row in case_rows}
        if coordinates != expected_coordinates:
            raise ValueError(
                f"measurement robustness model/score coverage changed for {case_hash}"
            )
    for role in _ALLOWED_ROLES:
        for task in _TASK_ORDER:
            if not any(
                row.role is role and row.task_variant == task
                for row in canonical_rows
            ):
                raise ValueError(
                    "measurement robustness role/task stratum is missing"
                )
    return tuple(
        sorted(
            canonical_rows,
            key=lambda row: (
                _ALLOWED_ROLES.index(row.role),
                _TASK_ORDER.index(row.task_variant),
                row.case_hash,
                tuple(MeasurementScore).index(row.score),
                _FAMILY_ORDER.index(row.model_name),
            ),
        )
    )


def _group_mean(
    rows: tuple[MeasurementCaseLoss, ...],
    *,
    role: ObservationPartitionRole,
    task: str,
    score: MeasurementScore,
    aggregation: MeasurementAggregation,
    model_name: str,
    omitted_participant: str | None = None,
) -> float:
    participant_losses: dict[str, list[float]] = {}
    for row in rows:
        if (
            row.role is role
            and row.task_variant == task
            and row.score is score
            and row.model_name == model_name
            and row.participant_group_hash != omitted_participant
        ):
            participant_losses.setdefault(row.participant_group_hash, []).append(
                row.value
            )
    if not participant_losses:
        raise ValueError("measurement robustness omission removed all support")
    return (
        trial_equal_mean(participant_losses)
        if aggregation is MeasurementAggregation.TRIAL_EQUAL
        else participant_equal_mean(participant_losses)
    )


def build_measurement_robustness_profile(
    rows: object,
    protocol: MeasurementValidityProtocol,
) -> MeasurementRobustnessProfile:
    if not isinstance(protocol, MeasurementValidityProtocol):
        raise TypeError(
            "measurement robustness profile requires MeasurementValidityProtocol"
        )
    canonical_rows = _validated_profile_rows(rows)
    references = dict(protocol.material_reversal_references)

    aggregates = tuple(
        aggregate
        for aggregation in MeasurementAggregation
        for aggregate in aggregate_measurement_losses(
            canonical_rows,
            aggregation,
        )
    )
    aggregate_map = {
        (
            row.role,
            row.task_variant,
            row.score,
            row.aggregation,
            row.model_name,
        ): row
        for row in aggregates
    }
    expected_aggregate_count = (
        len(_ALLOWED_ROLES)
        * len(_TASK_ORDER)
        * len(MeasurementScore)
        * len(MeasurementAggregation)
        * len(_FAMILY_ORDER)
    )
    if len(aggregate_map) != expected_aggregate_count:
        raise ValueError("measurement robustness aggregate coverage changed")

    pairwise: list[MeasurementPairwiseDelta] = []
    pair_map: dict[
        tuple[
            ObservationPartitionRole,
            str,
            MeasurementScore,
            MeasurementAggregation,
            str,
            str,
        ],
        MeasurementPairwiseDelta,
    ] = {}
    for role in _ALLOWED_ROLES:
        for task in _TASK_ORDER:
            for score in MeasurementScore:
                for aggregation in MeasurementAggregation:
                    for first_model, second_model in _MODEL_PAIRS:
                        first = aggregate_map[
                            (role, task, score, aggregation, first_model)
                        ]
                        second = aggregate_map[
                            (role, task, score, aggregation, second_model)
                        ]
                        pair = MeasurementPairwiseDelta(
                            first_model=first_model,
                            second_model=second_model,
                            score=score,
                            aggregation=aggregation,
                            task_variant=task,
                            role=role,
                            delta=_canonical_numeric_zero(
                                first.mean_loss - second.mean_loss
                            ),
                        )
                        key = (
                            role,
                            task,
                            score,
                            aggregation,
                            first_model,
                            second_model,
                        )
                        pair_map[key] = pair
                        pairwise.append(pair)

    task_findings: list[MeasurementDependenceFinding] = []
    for role in _ALLOWED_ROLES:
        for score in MeasurementScore:
            reference = references[score]
            for aggregation in MeasurementAggregation:
                for model_pair in _MODEL_PAIRS:
                    deltas = tuple(
                        pair_map[
                            (
                                role,
                                task,
                                score,
                                aggregation,
                                model_pair[0],
                                model_pair[1],
                            )
                        ].delta
                        for task in _TASK_ORDER
                    )
                    margin = (reference,) * len(deltas)
                    task_findings.append(
                        MeasurementDependenceFinding(
                            dimension="task",
                            score=score,
                            aggregation=aggregation,
                            task=(
                                f"{role.value}:"
                                "magic_carpet_vs_spaceship"
                            ),
                            model_pair=model_pair,
                            deltas=deltas,
                            references=margin,
                            status=classify_material_reversal(deltas, margin),
                        )
                    )

    aggregation_findings: list[MeasurementDependenceFinding] = []
    for role in _ALLOWED_ROLES:
        for task in _TASK_ORDER:
            for score in MeasurementScore:
                reference = references[score]
                for model_pair in _MODEL_PAIRS:
                    deltas = tuple(
                        pair_map[
                            (
                                role,
                                task,
                                score,
                                aggregation,
                                model_pair[0],
                                model_pair[1],
                            )
                        ].delta
                        for aggregation in MeasurementAggregation
                    )
                    margin = (reference,) * len(deltas)
                    aggregation_findings.append(
                        MeasurementDependenceFinding(
                            dimension="aggregation",
                            score=score,
                            aggregation=MeasurementAggregation.TRIAL_EQUAL,
                            task=f"{role.value}:{task}",
                            model_pair=model_pair,
                            deltas=deltas,
                            references=margin,
                            status=classify_material_reversal(deltas, margin),
                        )
                    )

    score_findings: list[MeasurementDependenceFinding] = []
    score_order = tuple(MeasurementScore)
    for role in _ALLOWED_ROLES:
        for task in _TASK_ORDER:
            for aggregation in MeasurementAggregation:
                for model_pair in _MODEL_PAIRS:
                    deltas = tuple(
                        pair_map[
                            (
                                role,
                                task,
                                score,
                                aggregation,
                                model_pair[0],
                                model_pair[1],
                            )
                        ].delta
                        for score in score_order
                    )
                    margins = tuple(references[score] for score in score_order)
                    score_findings.append(
                        MeasurementDependenceFinding(
                            dimension="score",
                            score=MeasurementScore.BRIER,
                            aggregation=aggregation,
                            task=f"{role.value}:{task}",
                            model_pair=model_pair,
                            deltas=deltas,
                            references=margins,
                            status=classify_material_reversal(deltas, margins),
                        )
                    )

    participant_influence: list[ParticipantInfluenceRange] = []
    for role in _ALLOWED_ROLES:
        for task in _TASK_ORDER:
            participants = tuple(
                sorted(
                    {
                        row.participant_group_hash
                        for row in canonical_rows
                        if row.role is role and row.task_variant == task
                    }
                )
            )
            for score in MeasurementScore:
                reference = references[score]
                for aggregation in MeasurementAggregation:
                    for first_model, second_model in _MODEL_PAIRS:
                        if len(participants) == 1:
                            participant_influence.append(
                                ParticipantInfluenceRange(
                                    role=role,
                                    task_variant=task,
                                    score=score,
                                    aggregation=aggregation,
                                    first_model=first_model,
                                    second_model=second_model,
                                    minimum_delta=None,
                                    maximum_delta=None,
                                    omitted_participant_count=1,
                                    status=MeasurementValidityStatus.NOT_ESTABLISHED,
                                )
                            )
                            continue
                        omitted_deltas = tuple(
                            _canonical_numeric_zero(
                                _group_mean(
                                    canonical_rows,
                                    role=role,
                                    task=task,
                                    score=score,
                                    aggregation=aggregation,
                                    model_name=first_model,
                                    omitted_participant=participant,
                                )
                                - _group_mean(
                                    canonical_rows,
                                    role=role,
                                    task=task,
                                    score=score,
                                    aggregation=aggregation,
                                    model_name=second_model,
                                    omitted_participant=participant,
                                )
                            )
                            for participant in participants
                        )
                        baseline = pair_map[
                            (
                                role,
                                task,
                                score,
                                aggregation,
                                first_model,
                                second_model,
                            )
                        ].delta
                        classification_values = (baseline,) + omitted_deltas
                        participant_influence.append(
                            ParticipantInfluenceRange(
                                role=role,
                                task_variant=task,
                                score=score,
                                aggregation=aggregation,
                                first_model=first_model,
                                second_model=second_model,
                                minimum_delta=min(omitted_deltas),
                                maximum_delta=max(omitted_deltas),
                                omitted_participant_count=len(participants),
                                status=classify_material_reversal(
                                    classification_values,
                                    (reference,) * len(classification_values),
                                ),
                            )
                        )

    return MeasurementRobustnessProfile(
        aggregate_losses=aggregates,
        pairwise_deltas=tuple(pairwise),
        task_findings=tuple(task_findings),
        aggregation_findings=tuple(aggregation_findings),
        score_findings=tuple(score_findings),
        participant_influence=tuple(participant_influence),
    )


def _terminal_class(
    value: MeasurementTerminalClass | str,
) -> MeasurementTerminalClass:
    try:
        return (
            value
            if isinstance(value, MeasurementTerminalClass)
            else MeasurementTerminalClass(value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("measurement terminal class is unsupported") from error


def _report_role_hash_pairs(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (role.value, content_hash)
        for role, content_hash in _role_hash_pairs(value, label=label)
    )


def _report_hashes(
    value: object,
    *,
    label: str,
    expected_count: int | None = None,
    allow_empty: bool = False,
    sort_values: bool = False,
) -> tuple[str, ...]:
    try:
        raw_rows = tuple(value)
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence") from error
    rows = tuple(_hash(row, label=f"{label} hash") for row in raw_rows)
    if not allow_empty and not rows:
        raise ValueError(f"{label} must be non-empty")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} hashes")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{label} hashes must be unique")
    return tuple(sorted(rows)) if sort_values else rows


def _report_exact_findings(value: object) -> tuple[ExactInvarianceFinding, ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TypeError(
            "measurement report exact-invariance findings must be a sequence"
        ) from error
    if not rows or any(not isinstance(row, ExactInvarianceFinding) for row in rows):
        raise TypeError(
            "measurement report exact-invariance findings must contain "
            "ExactInvarianceFinding values"
        )
    names = tuple(row.check_name for row in rows)
    if len(set(names)) != len(names):
        raise ValueError(
            "measurement report exact-invariance finding names must be unique"
        )
    return rows


_STAY_SWITCH_PAYLOAD_FIELDS = {
    "task_variant",
    "model_name",
    "cells",
    "observed_interaction",
    "model_interaction",
    "status",
}
_STAY_SWITCH_CELL_PAYLOAD_FIELDS = {
    "task_variant",
    "reward",
    "transition_common",
    "observed_stay_probability",
    "model_expected_stay_probability",
    "count",
    "status",
}


def _stay_switch_identity_payload(value: object) -> Mapping[str, object]:
    identity_method = getattr(value, "identity_payload", None)
    if not callable(identity_method):
        raise TypeError(
            "measurement report stay/switch diagnostics must be aggregate diagnostics"
        )
    payload = identity_method()
    if not isinstance(payload, Mapping) or set(payload) != _STAY_SWITCH_PAYLOAD_FIELDS:
        raise TypeError(
            "measurement report stay/switch diagnostic payload changed"
        )
    try:
        cells = tuple(payload["cells"])
    except TypeError as error:
        raise TypeError(
            "measurement report stay/switch diagnostic cells must be a sequence"
        ) from error
    if len(cells) != 4 or any(
        not isinstance(cell, Mapping)
        or set(cell) != _STAY_SWITCH_CELL_PAYLOAD_FIELDS
        for cell in cells
    ):
        raise TypeError(
            "measurement report stay/switch diagnostic cell payload changed"
        )
    content_hash = getattr(value, "content_hash", None)
    if content_hash != stable_content_hash(payload):
        raise ValueError(
            "measurement report stay/switch diagnostic identity changed"
        )
    return payload


def _report_stay_switch_diagnostics(value: object) -> tuple[object, ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TypeError(
            "measurement report stay/switch diagnostics must be a sequence"
        ) from error
    coordinates: list[tuple[object, object]] = []
    for row in rows:
        payload = _stay_switch_identity_payload(row)
        coordinates.append((payload["task_variant"], payload["model_name"]))
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(
            "measurement report stay/switch diagnostic coordinates must be unique"
        )
    return rows


def _report_stratum_counts(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, str, int], ...]:
    try:
        raw_rows = tuple(value)
    except TypeError as error:
        raise TypeError(f"measurement report {label} must be a sequence") from error
    rows: list[tuple[str, str, int]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, tuple) or len(raw_row) != 3:
            raise ValueError(
                f"measurement report {label} rows must be role/task/count triples"
            )
        role = _role(raw_row[0])
        task = _text(raw_row[1], label=f"measurement report {label} task")
        if task not in _TASK_ORDER:
            raise ValueError(f"measurement report {label} task is unsupported")
        count = raw_row[2]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"measurement report {label} values must be positive integers"
            )
        rows.append((role.value, task, count))
    coordinates = tuple((role, task) for role, task, _ in rows)
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(f"measurement report {label} coordinates must be unique")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                tuple(role.value for role in _ALLOWED_ROLES).index(row[0]),
                _TASK_ORDER.index(row[1]),
            ),
        )
    )


def _report_diagnostic_cell_counts(
    value: object,
) -> tuple[tuple[str, str, int, bool, int], ...]:
    try:
        raw_rows = tuple(value)
    except TypeError as error:
        raise TypeError(
            "measurement report diagnostic cell counts must be a sequence"
        ) from error
    rows: list[tuple[str, str, int, bool, int]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, tuple) or len(raw_row) != 5:
            raise ValueError(
                "measurement report diagnostic cell count rows must be "
                "task/model/reward/transition/count tuples"
            )
        task = _text(
            raw_row[0],
            label="measurement report diagnostic cell task",
        )
        model = _text(
            raw_row[1],
            label="measurement report diagnostic cell model",
        )
        reward = raw_row[2]
        transition_common = raw_row[3]
        count = raw_row[4]
        if task not in _TASK_ORDER or model not in _FAMILY_ORDER:
            raise ValueError(
                "measurement report diagnostic cell coordinate is unsupported"
            )
        if isinstance(reward, bool) or reward not in (0, 1):
            raise ValueError("measurement report diagnostic cell reward is unsupported")
        if not isinstance(transition_common, bool):
            raise ValueError(
                "measurement report diagnostic cell transition must be boolean"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "measurement report diagnostic cell count must be non-negative"
            )
        rows.append((task, model, reward, transition_common, count))
    coordinates = tuple(row[:4] for row in rows)
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(
            "measurement report diagnostic cell count coordinates must be unique"
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                _TASK_ORDER.index(row[0]),
                _FAMILY_ORDER.index(row[1]),
                row[2],
                int(row[3]),
            ),
        )
    )


def _measurement_report_manifest(
    *,
    terminal_class: MeasurementTerminalClass,
    claim_scope: str,
    protocol_hash: str,
    audit_input_hash: str,
    empirical_anchor_hash: str,
    allowed_partition_hashes: tuple[tuple[str, str], ...],
    allowed_target_report_hashes: tuple[tuple[str, str], ...],
    excluded_final_partition_hash: str,
    excluded_final_target_hash: str,
    candidate_hashes: tuple[str, ...],
    prediction_artifact_hash: str | None,
    execution_manifest_hashes: tuple[str, ...],
    exact_invariance_findings: tuple[ExactInvarianceFinding, ...],
    robustness_profile: MeasurementRobustnessProfile | None,
    stay_switch_diagnostics: tuple[object, ...],
    record_counts: tuple[tuple[str, str, int], ...],
    participant_counts: tuple[tuple[str, str, int], ...],
    diagnostic_cell_counts: tuple[tuple[str, str, int, bool, int], ...],
) -> ExperimentManifest:
    finding_hashes = tuple(row.content_hash for row in exact_invariance_findings)
    diagnostic_hashes = tuple(
        str(getattr(row, "content_hash")) for row in stay_switch_diagnostics
    )
    robustness_hash = (
        None if robustness_profile is None else robustness_profile.content_hash
    )
    parents = (
        protocol_hash,
        audit_input_hash,
        empirical_anchor_hash,
        *(value for _, value in allowed_partition_hashes),
        *(value for _, value in allowed_target_report_hashes),
        excluded_final_partition_hash,
        excluded_final_target_hash,
        *candidate_hashes,
        *(() if prediction_artifact_hash is None else (prediction_artifact_hash,)),
        *execution_manifest_hashes,
        *finding_hashes,
        *(() if robustness_hash is None else (robustness_hash,)),
        *diagnostic_hashes,
    )
    return ExperimentManifest(
        stage=ExperimentStage.MEASUREMENT_AUDIT,
        inputs={
            "artifact_type": "measurement_validity_report",
            "terminal_class": terminal_class.value,
            "claim_scope": claim_scope,
            "protocol_hash": protocol_hash,
            "audit_input_hash": audit_input_hash,
            "empirical_anchor_hash": empirical_anchor_hash,
            "allowed_partition_hashes": allowed_partition_hashes,
            "allowed_target_report_hashes": allowed_target_report_hashes,
            "excluded_final_partition_hash": excluded_final_partition_hash,
            "excluded_final_target_hash": excluded_final_target_hash,
            "candidate_hashes": candidate_hashes,
            "prediction_artifact_hash": prediction_artifact_hash,
            "execution_manifest_hashes": execution_manifest_hashes,
            "exact_invariance_finding_hashes": finding_hashes,
            "robustness_profile_hash": robustness_hash,
            "stay_switch_diagnostic_hashes": diagnostic_hashes,
            "record_counts": record_counts,
            "participant_counts": participant_counts,
            "diagnostic_cell_counts": diagnostic_cell_counts,
            "execution_boundary": {
                "parameter_training_performed": False,
                "parameter_selection_performed": False,
                "final_test_values_exposed_to_audit": False,
                "final_test_outcomes_analyzed": False,
                "final_model_execution": False,
            },
        },
        parent_hashes=parents,
    )


@dataclass(frozen=True)
class MeasurementValidityReport:
    terminal_class: MeasurementTerminalClass | str
    claim_scope: str
    protocol_hash: str
    audit_input_hash: str
    empirical_anchor_hash: str
    allowed_partition_hashes: tuple[tuple[str, str], ...]
    allowed_target_report_hashes: tuple[tuple[str, str], ...]
    excluded_final_partition_hash: str
    excluded_final_target_hash: str
    candidate_hashes: tuple[str, ...]
    prediction_artifact_hash: str | None
    execution_manifest_hashes: tuple[str, ...]
    exact_invariance_findings: tuple[ExactInvarianceFinding, ...]
    robustness_profile: MeasurementRobustnessProfile | None
    stay_switch_diagnostics: tuple[object, ...]
    record_counts: tuple[tuple[str, str, int], ...]
    participant_counts: tuple[tuple[str, str, int], ...]
    diagnostic_cell_counts: tuple[tuple[str, str, int, bool, int], ...]
    parameter_training_performed: bool
    parameter_selection_performed: bool
    final_test_values_exposed_to_audit: bool
    final_test_outcomes_analyzed: bool
    final_model_execution: bool
    manifest: ExperimentManifest | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        terminal = _terminal_class(self.terminal_class)
        if terminal is MeasurementTerminalClass.INFRASTRUCTURE_INCOMPLETE:
            raise ValueError(
                "measurement infrastructure incomplete belongs in an attempt record"
            )
        object.__setattr__(self, "terminal_class", terminal)
        if self.claim_scope != MEASUREMENT_CLAIM_SCOPE:
            raise ValueError("measurement report claim scope changed")
        for attribute, label in (
            ("protocol_hash", "protocol hash"),
            ("audit_input_hash", "audit input hash"),
            ("empirical_anchor_hash", "empirical anchor hash"),
            ("excluded_final_partition_hash", "excluded FINAL partition hash"),
            ("excluded_final_target_hash", "excluded FINAL target hash"),
        ):
            object.__setattr__(
                self,
                attribute,
                _hash(getattr(self, attribute), label=f"measurement report {label}"),
            )
        if self.excluded_final_partition_hash == self.excluded_final_target_hash:
            raise ValueError("measurement report excluded FINAL identities must differ")
        object.__setattr__(
            self,
            "allowed_partition_hashes",
            _report_role_hash_pairs(
                self.allowed_partition_hashes,
                label="measurement report allowed partition hashes",
            ),
        )
        object.__setattr__(
            self,
            "allowed_target_report_hashes",
            _report_role_hash_pairs(
                self.allowed_target_report_hashes,
                label="measurement report allowed target report hashes",
            ),
        )
        object.__setattr__(
            self,
            "candidate_hashes",
            _report_hashes(
                self.candidate_hashes,
                label="measurement report candidate hashes",
                expected_count=3,
            ),
        )
        if self.prediction_artifact_hash is not None:
            object.__setattr__(
                self,
                "prediction_artifact_hash",
                _hash(
                    self.prediction_artifact_hash,
                    label="measurement report prediction artifact hash",
                ),
            )
        object.__setattr__(
            self,
            "execution_manifest_hashes",
            _report_hashes(
                self.execution_manifest_hashes,
                label="measurement report execution manifest hashes",
                allow_empty=True,
                sort_values=True,
            ),
        )
        findings = _report_exact_findings(self.exact_invariance_findings)
        object.__setattr__(self, "exact_invariance_findings", findings)
        if self.robustness_profile is not None and not isinstance(
            self.robustness_profile,
            MeasurementRobustnessProfile,
        ):
            raise TypeError(
                "measurement report robustness profile must be "
                "MeasurementRobustnessProfile or None"
            )
        diagnostics = _report_stay_switch_diagnostics(
            self.stay_switch_diagnostics
        )
        object.__setattr__(self, "stay_switch_diagnostics", diagnostics)
        record_counts = _report_stratum_counts(
            self.record_counts,
            label="record counts",
        )
        participant_counts = _report_stratum_counts(
            self.participant_counts,
            label="participant counts",
        )
        diagnostic_counts = _report_diagnostic_cell_counts(
            self.diagnostic_cell_counts
        )
        object.__setattr__(self, "record_counts", record_counts)
        object.__setattr__(self, "participant_counts", participant_counts)
        object.__setattr__(self, "diagnostic_cell_counts", diagnostic_counts)

        for attribute in (
            "parameter_training_performed",
            "parameter_selection_performed",
            "final_test_values_exposed_to_audit",
            "final_test_outcomes_analyzed",
            "final_model_execution",
        ):
            value = getattr(self, attribute)
            if not isinstance(value, bool) or value:
                raise ValueError(f"measurement report {attribute} must be false")

        failed = any(
            row.status is MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
            for row in findings
        )
        expected_terminal = (
            MeasurementTerminalClass.SCIENTIFIC_RED
            if failed
            else MeasurementTerminalClass.GREEN
        )
        if terminal is not expected_terminal:
            raise ValueError(
                "measurement report terminal class disagrees with exact-invariance gate"
            )
        has_prediction = self.prediction_artifact_hash is not None
        has_execution = bool(self.execution_manifest_hashes)
        if has_prediction != has_execution:
            raise ValueError(
                "measurement report prediction and execution identities must appear together"
            )
        if terminal is MeasurementTerminalClass.GREEN:
            if not has_prediction:
                raise ValueError("GREEN measurement report requires a prediction artifact")
            if self.robustness_profile is None:
                raise ValueError("GREEN measurement report requires robustness")
            if not diagnostics or not diagnostic_counts:
                raise ValueError("GREEN measurement report requires diagnostics")
        else:
            if self.robustness_profile is not None:
                raise ValueError(
                    "scientific RED measurement report cannot contain robustness"
                )
            if diagnostics or diagnostic_counts:
                raise ValueError(
                    "scientific RED measurement report cannot contain diagnostics"
                )

        expected_diagnostic_counts = tuple(
            (
                str(payload["task_variant"]),
                str(payload["model_name"]),
                int(cell["reward"]),
                bool(cell["transition_common"]),
                int(cell["count"]),
            )
            for row in diagnostics
            for payload in (_stay_switch_identity_payload(row),)
            for cell in tuple(payload["cells"])
        )
        expected_diagnostic_counts = _report_diagnostic_cell_counts(
            expected_diagnostic_counts
        )
        if diagnostic_counts != expected_diagnostic_counts:
            raise ValueError(
                "measurement report diagnostic cell counts changed"
            )

        expected_manifest = _measurement_report_manifest(
            terminal_class=terminal,
            claim_scope=self.claim_scope,
            protocol_hash=self.protocol_hash,
            audit_input_hash=self.audit_input_hash,
            empirical_anchor_hash=self.empirical_anchor_hash,
            allowed_partition_hashes=self.allowed_partition_hashes,
            allowed_target_report_hashes=self.allowed_target_report_hashes,
            excluded_final_partition_hash=self.excluded_final_partition_hash,
            excluded_final_target_hash=self.excluded_final_target_hash,
            candidate_hashes=self.candidate_hashes,
            prediction_artifact_hash=self.prediction_artifact_hash,
            execution_manifest_hashes=self.execution_manifest_hashes,
            exact_invariance_findings=findings,
            robustness_profile=self.robustness_profile,
            stay_switch_diagnostics=diagnostics,
            record_counts=record_counts,
            participant_counts=participant_counts,
            diagnostic_cell_counts=diagnostic_counts,
        )
        if self.manifest is None:
            object.__setattr__(self, "manifest", expected_manifest)
        else:
            if not isinstance(self.manifest, ExperimentManifest):
                raise TypeError(
                    "measurement report manifest must be ExperimentManifest"
                )
            if self.manifest.stage is not ExperimentStage.MEASUREMENT_AUDIT:
                raise ValueError(
                    "measurement report manifest stage must be measurement_audit"
                )
            if self.manifest.content_hash != expected_manifest.content_hash:
                raise ValueError("measurement report manifest binding changed")

    def identity_payload(self) -> dict[str, object]:
        assert isinstance(self.manifest, ExperimentManifest)
        return {
            "terminal_class": self.terminal_class.value,
            "claim_scope": self.claim_scope,
            "protocol_hash": self.protocol_hash,
            "audit_input_hash": self.audit_input_hash,
            "empirical_anchor_hash": self.empirical_anchor_hash,
            "allowed_partition_hashes": self.allowed_partition_hashes,
            "allowed_target_report_hashes": self.allowed_target_report_hashes,
            "excluded_final_partition_hash": self.excluded_final_partition_hash,
            "excluded_final_target_hash": self.excluded_final_target_hash,
            "candidate_hashes": self.candidate_hashes,
            "prediction_artifact_hash": self.prediction_artifact_hash,
            "execution_manifest_hashes": self.execution_manifest_hashes,
            "exact_invariance_findings": tuple(
                row.identity_payload() for row in self.exact_invariance_findings
            ),
            "robustness_profile": (
                None
                if self.robustness_profile is None
                else self.robustness_profile.identity_payload()
            ),
            "stay_switch_diagnostics": tuple(
                _stay_switch_identity_payload(row)
                for row in self.stay_switch_diagnostics
            ),
            "record_counts": self.record_counts,
            "participant_counts": self.participant_counts,
            "diagnostic_cell_counts": self.diagnostic_cell_counts,
            "parameter_training_performed": self.parameter_training_performed,
            "parameter_selection_performed": self.parameter_selection_performed,
            "final_test_values_exposed_to_audit": (
                self.final_test_values_exposed_to_audit
            ),
            "final_test_outcomes_analyzed": self.final_test_outcomes_analyzed,
            "final_model_execution": self.final_model_execution,
            "manifest_hash": self.manifest.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def measurement_report_payload(
    report: MeasurementValidityReport,
) -> dict[str, object]:
    if not isinstance(report, MeasurementValidityReport):
        raise TypeError(
            "measurement report serialization requires MeasurementValidityReport"
        )
    return {**report.identity_payload(), "content_hash": report.content_hash}


def _attempt_revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-character lowercase git revision")
    return value


def _attempt_started_at(value: object) -> str:
    started = _text(value, label="measurement attempt start time")
    if not started.endswith("Z"):
        raise ValueError("measurement attempt start time must be UTC with a Z suffix")
    try:
        parsed = datetime.fromisoformat(started[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("measurement attempt start time must be ISO-8601") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("measurement attempt start time must be UTC")
    return started


def _attempt_artifact_file_hashes(
    value: object,
) -> tuple[tuple[str, str], ...]:
    try:
        raw_rows = tuple(value)
    except TypeError as error:
        raise TypeError(
            "measurement attempt artifact file hashes must be a sequence"
        ) from error
    rows: list[tuple[str, str]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, tuple) or len(raw_row) != 2:
            raise ValueError(
                "measurement attempt artifact file hashes must be name/hash pairs"
            )
        rows.append(
            (
                _text(raw_row[0], label="measurement attempt artifact file name"),
                _hash(
                    raw_row[1],
                    label="measurement attempt artifact file hash",
                ),
            )
        )
    names = tuple(name for name, _ in rows)
    if len(set(names)) != len(names):
        raise ValueError("measurement attempt artifact file names must be unique")
    return tuple(sorted(rows))


@dataclass(frozen=True)
class MeasurementAuditAttempt:
    attempt_id: str
    scientific_revision: str
    orchestration_revision: str
    started_at_utc: str
    terminal_class: MeasurementTerminalClass | str
    protocol_hash: str
    report_hash: str | None
    error_type: str | None
    error_message_hash: str | None
    artifact_file_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_id",
            _text(self.attempt_id, label="measurement attempt id"),
        )
        object.__setattr__(
            self,
            "scientific_revision",
            _attempt_revision(
                self.scientific_revision,
                label="measurement attempt scientific revision",
            ),
        )
        object.__setattr__(
            self,
            "orchestration_revision",
            _attempt_revision(
                self.orchestration_revision,
                label="measurement attempt orchestration revision",
            ),
        )
        object.__setattr__(
            self,
            "started_at_utc",
            _attempt_started_at(self.started_at_utc),
        )
        terminal = _terminal_class(self.terminal_class)
        object.__setattr__(self, "terminal_class", terminal)
        object.__setattr__(
            self,
            "protocol_hash",
            _hash(
                self.protocol_hash,
                label="measurement attempt protocol hash",
            ),
        )
        if self.report_hash is not None:
            object.__setattr__(
                self,
                "report_hash",
                _hash(
                    self.report_hash,
                    label="measurement attempt report hash",
                ),
            )
        if self.error_type is not None:
            object.__setattr__(
                self,
                "error_type",
                _text(self.error_type, label="measurement attempt error type"),
            )
        if self.error_message_hash is not None:
            object.__setattr__(
                self,
                "error_message_hash",
                _hash(
                    self.error_message_hash,
                    label="measurement attempt error message hash",
                ),
            )
        object.__setattr__(
            self,
            "artifact_file_hashes",
            _attempt_artifact_file_hashes(self.artifact_file_hashes),
        )
        if terminal is MeasurementTerminalClass.INFRASTRUCTURE_INCOMPLETE:
            if self.report_hash is not None:
                raise ValueError(
                    "infrastructure-incomplete measurement attempt cannot bind a report"
                )
            if self.error_type is None or self.error_message_hash is None:
                raise ValueError(
                    "infrastructure-incomplete measurement attempt requires an error"
                )
        else:
            if self.report_hash is None:
                raise ValueError(
                    "scientific measurement attempt requires a report hash"
                )
            if self.error_type is not None or self.error_message_hash is not None:
                raise ValueError(
                    "scientific measurement attempt cannot contain an infrastructure error"
                )

    def identity_payload(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "scientific_revision": self.scientific_revision,
            "orchestration_revision": self.orchestration_revision,
            "started_at_utc": self.started_at_utc,
            "terminal_class": self.terminal_class.value,
            "protocol_hash": self.protocol_hash,
            "report_hash": self.report_hash,
            "error_type": self.error_type,
            "error_message_hash": self.error_message_hash,
            "artifact_file_hashes": self.artifact_file_hashes,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _seed_rows(
    value: object,
) -> tuple[tuple[ObservationPartitionRole, tuple[int, ...]], ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TypeError("measurement seed plan must be a sequence") from error
    canonical: dict[ObservationPartitionRole, tuple[int, ...]] = {}
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError("measurement seed rows must be role/seed pairs")
        role = _role(row[0])
        try:
            seeds = tuple(row[1])
        except TypeError as error:
            raise TypeError("measurement seeds must be a sequence") from error
        if (
            not seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("measurement seeds must be non-empty unique integers")
        if any(seed in (301, 302) for seed in seeds):
            raise ValueError("measurement protocol cannot contain FINAL seeds")
        if role in canonical:
            raise ValueError("measurement seed roles must be unique")
        canonical[role] = seeds
    if set(canonical) != set(_ALLOWED_ROLES):
        raise ValueError("measurement seeds require TRAIN and SELECTION_VALIDATION")
    if canonical != _SEEDS_BY_ROLE:
        raise ValueError("measurement seed plan changed from the frozen audit")
    return tuple((role, canonical[role]) for role in _ALLOWED_ROLES)


def _reference_rows(
    value: object,
) -> tuple[tuple[MeasurementScore, float], ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TypeError("measurement reversal references must be a sequence") from error
    canonical: dict[MeasurementScore, float] = {}
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError("measurement reversal references must be score/value pairs")
        try:
            score = row[0] if isinstance(row[0], MeasurementScore) else MeasurementScore(row[0])
        except (TypeError, ValueError) as error:
            raise ValueError("measurement reversal score is unsupported") from error
        if score in canonical:
            raise ValueError("measurement reversal scores must be unique")
        reference = _finite(row[1], label="measurement reversal reference")
        if reference <= 0.0:
            raise ValueError("measurement reversal reference must be positive")
        canonical[score] = reference
    if set(canonical) != set(MeasurementScore):
        raise ValueError("measurement reversal references require Brier and Log")
    return tuple((score, canonical[score]) for score in MeasurementScore)


__all__ = [
    "ExactInvarianceFinding",
    "MEASUREMENT_CLAIM_SCOPE",
    "MeasurementAggregation",
    "MeasurementAggregateLoss",
    "MeasurementAuditAttempt",
    "MeasurementAuditCase",
    "MeasurementAuditInput",
    "MeasurementCaseLoss",
    "MeasurementDependenceFinding",
    "MeasurementModelPrediction",
    "MeasurementPairwiseDelta",
    "MeasurementPredictionArtifact",
    "MeasurementRobustnessProfile",
    "MeasurementScore",
    "MeasurementSeedPrediction",
    "MeasurementTerminalClass",
    "MeasurementValidityReport",
    "MeasurementValidityProtocol",
    "MeasurementValidityStatus",
    "ParticipantInfluenceRange",
    "aggregate_measurement_losses",
    "classify_material_reversal",
    "build_measurement_robustness_profile",
    "evaluate_categorical_coordinate_invariance",
    "measurement_report_payload",
    "average_seed_metrics",
    "participant_equal_mean",
    "permute_binary_metric_map",
    "score_measurement_predictions",
    "trial_equal_mean",
]
