from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    Scenario,
    stable_content_hash,
)
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.observations.preregistration import FrozenModelSpec


MEASUREMENT_CLAIM_SCOPE = "external_observational_measurement_audit_only"
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-12
_ALLOWED_ROLES = (
    ObservationPartitionRole.TRAIN,
    ObservationPartitionRole.SELECTION_VALIDATION,
)
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
    expected = {
        ObservationPartitionRole.TRAIN: (101, 102),
        ObservationPartitionRole.SELECTION_VALIDATION: (201, 202),
    }
    if canonical != expected:
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
    "MEASUREMENT_CLAIM_SCOPE",
    "MeasurementAggregation",
    "MeasurementAuditCase",
    "MeasurementAuditInput",
    "MeasurementScore",
    "MeasurementTerminalClass",
    "MeasurementValidityProtocol",
    "MeasurementValidityStatus",
]
