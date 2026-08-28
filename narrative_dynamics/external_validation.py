from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.losses import metric_loss_identity
from narrative_dynamics.manifest import callable_identity, component_identity
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.observations.external import (
    EXTERNAL_CLAIM_SCOPE,
    EXTERNAL_EVIDENCE_ORIGIN,
    ExternalEvidenceDeclaration,
    ExternalValidationError,
)
from narrative_dynamics.observations.preregistration import (
    AdequacyThresholds,
    PreregisteredEvaluationProtocol,
)
from narrative_dynamics.observations.release import (
    ProtocolRelease,
    VerifiedProtocolRelease,
)
from narrative_dynamics.observations.targets import TargetConstructionReport


_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalValidationError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ExternalValidationError(f"{label} cannot contain surrounding whitespace")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ExternalValidationError(f"{label} must be a sha256 content hash")
    return value


def _number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ExternalValidationError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ExternalValidationError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ExternalValidationError(f"{label} must be finite")
    if strictly_positive and number <= 0.0:
        raise ExternalValidationError(f"{label} must be strictly positive")
    if minimum is not None and number < minimum:
        raise ExternalValidationError(f"{label} must be at least {minimum}")
    return number


def _freeze(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExternalValidationError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ExternalValidationError(f"{label} keys must be non-empty strings")
            frozen[key] = _freeze(value[key], label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise ExternalValidationError(
        f"{label} values must be canonical scalars, mappings, lists, or tuples"
    )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExternalValidationError(f"{label} must be a mapping")
    frozen = _freeze(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _names(values: object, *, label: str, sort: bool = False) -> tuple[str, ...]:
    try:
        names = tuple(_text(value, label=label) for value in values)
    except TypeError as error:
        raise ExternalValidationError(f"{label} must be iterable") from error
    if not names:
        raise ExternalValidationError(f"{label} must be non-empty")
    if len(set(names)) != len(names):
        raise ExternalValidationError(f"{label} must be unique")
    return tuple(sorted(names)) if sort else names


def _seeds(values: object) -> tuple[int, ...]:
    try:
        seeds = tuple(values)
    except TypeError as error:
        raise ExternalValidationError("external constraint seeds must be iterable") from error
    if not seeds:
        raise ExternalValidationError("external constraint seeds must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ExternalValidationError("external constraint seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ExternalValidationError("external constraint seeds must be unique")
    return seeds


def _grid(
    parameter_grid: Mapping[str, object],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not isinstance(parameter_grid, Mapping) or not parameter_grid:
        raise ExternalValidationError("external constraint parameter grid must be non-empty")
    dimensions: list[tuple[str, tuple[float, ...]]] = []
    for name in sorted(parameter_grid):
        parameter_name = _text(name, label="external constraint parameter name")
        try:
            raw_values = tuple(parameter_grid[name])
        except TypeError as error:
            raise ExternalValidationError(
                f"external constraint grid {parameter_name!r} must be iterable"
            ) from error
        if not raw_values:
            raise ExternalValidationError(
                f"external constraint grid {parameter_name!r} must be non-empty"
            )
        values = tuple(
            _number(
                value,
                label=f"external constraint grid value for {parameter_name!r}",
            )
            for value in raw_values
        )
        if len(set(values)) != len(values):
            raise ExternalValidationError(
                f"external constraint grid {parameter_name!r} values must be unique"
            )
        dimensions.append((parameter_name, tuple(sorted(values))))
    return tuple(dimensions)


def _threshold_payload(thresholds: AdequacyThresholds) -> object:
    if not isinstance(thresholds, AdequacyThresholds):
        raise ExternalValidationError(
            "external validation adequacy thresholds must be AdequacyThresholds"
        )
    return thresholds.identity_payload()


def _candidate_hashes(
    protocol: PreregisteredEvaluationProtocol,
) -> tuple[str, ...]:
    return tuple(candidate.content_hash for candidate in protocol.candidates)


class ExternalScoreRole(str, Enum):
    BRIER = "brier"
    LOG = "log"


class PredictiveAdequacyStatus(str, Enum):
    MET = "predictive_adequacy_met"
    NOT_MET = "predictive_adequacy_not_met"


class PredictiveSeparationStatus(str, Enum):
    SEPARATED = "predictively_separated_under_protocol"
    NOT_SEPARATED = "not_predictively_separated_under_protocol"


class ExternalConstraintStatus(str, Enum):
    CONSTRAINED = "constrained_under_external_protocol"
    NOT_CONSTRAINED = "not_constrained_under_external_protocol"


class ExternalValidationProtocolError(ExternalValidationError):
    pass


class ExternalValidationConstraintError(ExternalValidationError):
    pass


@dataclass(frozen=True)
class PairwiseSeparationRule:
    min_mean_loss_delta_brier: float
    min_mean_loss_delta_log: float
    require_direction_agreement: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_mean_loss_delta_brier",
            _number(
                self.min_mean_loss_delta_brier,
                label="Brier separation delta",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "min_mean_loss_delta_log",
            _number(
                self.min_mean_loss_delta_log,
                label="Log separation delta",
                strictly_positive=True,
            ),
        )
        if self.require_direction_agreement is not True:
            raise ExternalValidationProtocolError(
                "external validation V1 requires Brier/Log direction agreement"
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "min_mean_loss_delta_brier": self.min_mean_loss_delta_brier,
            "min_mean_loss_delta_log": self.min_mean_loss_delta_log,
            "require_direction_agreement": True,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ExternalStratum:
    name: str
    final_case_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="external stratum name"))
        object.__setattr__(
            self,
            "final_case_names",
            _names(
                self.final_case_names,
                label="external stratum final case name",
                sort=True,
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        return {"name": self.name, "final_case_names": self.final_case_names}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ExternalConstraintPlan:
    name: str
    model_identity: Mapping[str, object]
    parameter_grid: tuple[tuple[str, tuple[float, ...]], ...]
    target_coordinates: tuple[str, ...]
    selection_case_names: tuple[str, ...]
    brier_acceptance_loss_delta: float
    log_acceptance_loss_delta: float
    simulation_seeds: tuple[int, ...]
    metric_identity: Mapping[str, object]
    brier_loss_identity: Mapping[str, object]
    log_loss_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _text(self.name, label="external constraint plan name"),
        )
        object.__setattr__(
            self,
            "model_identity",
            _mapping(self.model_identity, label="external constraint model identity"),
        )
        canonical_grid = _grid(dict(self.parameter_grid))
        object.__setattr__(self, "parameter_grid", canonical_grid)
        coordinates = _names(
            self.target_coordinates,
            label="external constraint target coordinate",
            sort=True,
        )
        grid_names = {name for name, _ in canonical_grid}
        if not set(coordinates).issubset(grid_names):
            raise ExternalValidationConstraintError(
                "external constraint target coordinates must be grid dimensions"
            )
        object.__setattr__(self, "target_coordinates", coordinates)
        object.__setattr__(
            self,
            "selection_case_names",
            _names(
                self.selection_case_names,
                label="external constraint selection case name",
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "brier_acceptance_loss_delta",
            _number(
                self.brier_acceptance_loss_delta,
                label="Brier acceptance loss delta",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "log_acceptance_loss_delta",
            _number(
                self.log_acceptance_loss_delta,
                label="Log acceptance loss delta",
                minimum=0.0,
            ),
        )
        object.__setattr__(self, "simulation_seeds", _seeds(self.simulation_seeds))
        object.__setattr__(
            self,
            "metric_identity",
            _mapping(self.metric_identity, label="external constraint metric identity"),
        )
        brier_identity = _mapping(
            self.brier_loss_identity,
            label="external constraint Brier loss identity",
        )
        log_identity = _mapping(
            self.log_loss_identity,
            label="external constraint Log loss identity",
        )
        if brier_identity.get("name") != "categorical_brier":
            raise ExternalValidationConstraintError(
                "external constraint Brier loss must be categorical_brier"
            )
        if log_identity.get("name") != "categorical_log":
            raise ExternalValidationConstraintError(
                "external constraint Log loss must be categorical_log"
            )
        object.__setattr__(self, "brier_loss_identity", brier_identity)
        object.__setattr__(self, "log_loss_identity", log_identity)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        model: object,
        parameter_grid: Mapping[str, object],
        target_coordinates: tuple[str, ...],
        selection_target_set: TargetConstructionReport,
        selection_case_names: tuple[str, ...],
        brier_acceptance_loss_delta: float,
        log_acceptance_loss_delta: float,
        simulation_seeds: tuple[int, ...],
        extractor: object,
        brier_loss: object,
        log_loss: object,
    ) -> "ExternalConstraintPlan":
        if not isinstance(selection_target_set, TargetConstructionReport):
            raise ExternalValidationConstraintError(
                "external constraint plan requires a target-construction report"
            )
        if selection_target_set.role is not ObservationPartitionRole.SELECTION_VALIDATION:
            raise ExternalValidationConstraintError(
                "external constraint plan requires selection-validation targets"
            )
        declared_cases = _names(
            selection_case_names,
            label="external constraint selection case name",
            sort=True,
        )
        available_cases = {case.name for case in selection_target_set.cases}
        if not set(declared_cases).issubset(available_cases):
            raise ExternalValidationConstraintError(
                "external constraint cases must belong to selection-validation targets"
            )
        return cls(
            name=name,
            model_identity=component_identity(model),
            parameter_grid=_grid(parameter_grid),
            target_coordinates=tuple(target_coordinates),
            selection_case_names=declared_cases,
            brier_acceptance_loss_delta=brier_acceptance_loss_delta,
            log_acceptance_loss_delta=log_acceptance_loss_delta,
            simulation_seeds=tuple(simulation_seeds),
            metric_identity=callable_identity(extractor),
            brier_loss_identity=metric_loss_identity(brier_loss),
            log_loss_identity=metric_loss_identity(log_loss),
        )

    @property
    def parameter_grid_map(self) -> dict[str, tuple[float, ...]]:
        return dict(self.parameter_grid)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_identity": self.model_identity,
            "parameter_grid": self.parameter_grid,
            "target_coordinates": self.target_coordinates,
            "selection_case_names": self.selection_case_names,
            "brier_acceptance_loss_delta": self.brier_acceptance_loss_delta,
            "log_acceptance_loss_delta": self.log_acceptance_loss_delta,
            "simulation_seeds": self.simulation_seeds,
            "metric_identity": self.metric_identity,
            "brier_loss_identity": self.brier_loss_identity,
            "log_loss_identity": self.log_loss_identity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ExternalValidationPreregistration:
    name: str
    version: str
    evidence_declaration_hash: str
    brier_protocol_hash: str
    log_protocol_hash: str
    adequacy_thresholds_by_score: tuple[
        tuple[ExternalScoreRole, AdequacyThresholds], ...
    ]
    strata: tuple[ExternalStratum, ...]
    separation_rule: PairwiseSeparationRule
    constraint_plans: tuple[ExternalConstraintPlan, ...]
    method_validation_hashes: tuple[str, ...]
    claim_scope: str = EXTERNAL_CLAIM_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _text(self.name, label="external validation preregistration name")
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="external validation preregistration version"),
        )
        for field_name in (
            "evidence_declaration_hash",
            "brier_protocol_hash",
            "log_protocol_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name.replace("_", " ")),
            )
        threshold_pairs = tuple(self.adequacy_thresholds_by_score)
        if len(threshold_pairs) != 2:
            raise ExternalValidationProtocolError(
                "external validation requires exactly Brier and Log thresholds"
            )
        expected_roles = (ExternalScoreRole.BRIER, ExternalScoreRole.LOG)
        canonical_thresholds: list[tuple[ExternalScoreRole, AdequacyThresholds]] = []
        for index, pair in enumerate(threshold_pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ExternalValidationProtocolError(
                    "external validation threshold entries must be role/threshold pairs"
                )
            role, thresholds = pair
            if not isinstance(role, ExternalScoreRole) or role is not expected_roles[index]:
                raise ExternalValidationProtocolError(
                    "external validation score roles must be Brier then Log"
                )
            _threshold_payload(thresholds)
            canonical_thresholds.append((role, thresholds))
        object.__setattr__(
            self, "adequacy_thresholds_by_score", tuple(canonical_thresholds)
        )

        strata = tuple(sorted(self.strata, key=lambda item: item.name))
        if not strata or any(not isinstance(item, ExternalStratum) for item in strata):
            raise ExternalValidationProtocolError(
                "external validation requires external strata"
            )
        if len({item.name for item in strata}) != len(strata):
            raise ExternalValidationProtocolError(
                "external validation stratum names must be unique"
            )
        object.__setattr__(self, "strata", strata)

        if not isinstance(self.separation_rule, PairwiseSeparationRule):
            raise ExternalValidationProtocolError(
                "external validation requires a pairwise separation rule"
            )

        plans = tuple(sorted(self.constraint_plans, key=lambda item: item.name))
        if any(not isinstance(item, ExternalConstraintPlan) for item in plans):
            raise ExternalValidationProtocolError(
                "external validation constraint plans are invalid"
            )
        if len({item.name for item in plans}) != len(plans):
            raise ExternalValidationProtocolError(
                "external validation constraint plan names must be unique"
            )
        object.__setattr__(self, "constraint_plans", plans)

        method_hashes = tuple(self.method_validation_hashes)
        if len(set(method_hashes)) != len(method_hashes):
            raise ExternalValidationProtocolError(
                "external validation method hashes must be unique"
            )
        method_hashes = tuple(
            sorted(
                _hash(value, label="external validation method hash")
                for value in method_hashes
            )
        )
        object.__setattr__(self, "method_validation_hashes", method_hashes)

        if self.claim_scope != EXTERNAL_CLAIM_SCOPE:
            raise ExternalValidationProtocolError(
                "external validation claim scope is fixed"
            )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        evidence_declaration: ExternalEvidenceDeclaration,
        brier_protocol: PreregisteredEvaluationProtocol,
        log_protocol: PreregisteredEvaluationProtocol,
        final_target_set: TargetConstructionReport,
        adequacy_thresholds_by_score: tuple[
            tuple[ExternalScoreRole, AdequacyThresholds], ...
        ],
        strata: tuple[ExternalStratum, ...],
        separation_rule: PairwiseSeparationRule,
        constraint_plans: tuple[ExternalConstraintPlan, ...],
        method_validation_hashes: tuple[str, ...],
    ) -> "ExternalValidationPreregistration":
        if not isinstance(evidence_declaration, ExternalEvidenceDeclaration):
            raise ExternalValidationProtocolError(
                "external validation preregistration requires external evidence"
            )
        if not isinstance(brier_protocol, PreregisteredEvaluationProtocol):
            raise ExternalValidationProtocolError(
                "external validation Brier protocol is invalid"
            )
        if not isinstance(log_protocol, PreregisteredEvaluationProtocol):
            raise ExternalValidationProtocolError(
                "external validation Log protocol is invalid"
            )
        if not isinstance(final_target_set, TargetConstructionReport):
            raise ExternalValidationProtocolError(
                "external validation requires final target construction"
            )

        if (
            evidence_declaration.dataset_hash != brier_protocol.dataset_hash
            or evidence_declaration.dataset_hash != log_protocol.dataset_hash
        ):
            raise ExternalValidationProtocolError(
                "external evidence dataset does not match both sibling protocols"
            )

        if final_target_set.role is not ObservationPartitionRole.FINAL_TEST:
            raise ExternalValidationProtocolError(
                "external validation preregistration requires final-test targets"
            )
        for protocol, label in (
            (brier_protocol, "Brier"),
            (log_protocol, "Log"),
        ):
            if final_target_set.dataset_hash != protocol.dataset_hash:
                raise ExternalValidationProtocolError(
                    f"{label} protocol final target dataset changed"
                )
            if final_target_set.partition_hash != protocol.final_partition_hash:
                raise ExternalValidationProtocolError(
                    f"{label} protocol final partition changed"
                )
            if final_target_set.spec_hash != protocol.target_spec_hash:
                raise ExternalValidationProtocolError(
                    f"{label} protocol target spec changed"
                )
            if final_target_set.content_hash != protocol.final_target_hash:
                raise ExternalValidationProtocolError(
                    f"{label} protocol final target payload changed"
                )
            if final_target_set.manifest.content_hash != protocol.final_target_manifest_hash:
                raise ExternalValidationProtocolError(
                    f"{label} protocol final target lineage changed"
                )

        if brier_protocol.loss_identity.get("name") != "categorical_brier":
            raise ExternalValidationProtocolError(
                "external Brier sibling must use categorical_brier"
            )
        if log_protocol.loss_identity.get("name") != "categorical_log":
            raise ExternalValidationProtocolError(
                "external Log sibling must use categorical_log"
            )

        sibling_fields = (
            "dataset_hash",
            "train_partition_hash",
            "selection_partition_hash",
            "final_partition_hash",
            "target_spec_hash",
            "final_target_hash",
            "final_target_manifest_hash",
            "metric_identity",
            "simulation_seeds",
            "baseline_name",
            "version",
        )
        for field_name in sibling_fields:
            if getattr(brier_protocol, field_name) != getattr(log_protocol, field_name):
                raise ExternalValidationProtocolError(
                    f"external sibling protocols disagree on {field_name}"
                )
        if _candidate_hashes(brier_protocol) != _candidate_hashes(log_protocol):
            raise ExternalValidationProtocolError(
                "external sibling protocols disagree on frozen candidates"
            )

        threshold_pairs = tuple(adequacy_thresholds_by_score)
        if len(threshold_pairs) != 2:
            raise ExternalValidationProtocolError(
                "external validation requires one threshold set per score"
            )
        expected = (
            (ExternalScoreRole.BRIER, brier_protocol.thresholds),
            (ExternalScoreRole.LOG, log_protocol.thresholds),
        )
        canonical_thresholds: list[tuple[ExternalScoreRole, AdequacyThresholds]] = []
        for index, pair in enumerate(threshold_pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ExternalValidationProtocolError(
                    "external validation threshold entries must be pairs"
                )
            role, thresholds = pair
            expected_role, protocol_thresholds = expected[index]
            if not isinstance(role, ExternalScoreRole) or role is not expected_role:
                raise ExternalValidationProtocolError(
                    "external validation score roles must be Brier then Log"
                )
            if _threshold_payload(thresholds) != _threshold_payload(protocol_thresholds):
                raise ExternalValidationProtocolError(
                    f"{expected_role.value} adequacy thresholds drifted"
                )
            canonical_thresholds.append((role, thresholds))

        canonical_strata = tuple(sorted(tuple(strata), key=lambda item: item.name))
        if not canonical_strata or any(
            not isinstance(item, ExternalStratum) for item in canonical_strata
        ):
            raise ExternalValidationProtocolError(
                "external validation requires preregistered strata"
            )
        if len({item.name for item in canonical_strata}) != len(canonical_strata):
            raise ExternalValidationProtocolError(
                "external validation stratum names must be unique"
            )
        target_cases = {case.name for case in final_target_set.cases}
        assigned_cases: list[str] = []
        for stratum in canonical_strata:
            assigned_cases.extend(stratum.final_case_names)
        if (
            len(assigned_cases) != len(set(assigned_cases))
            or set(assigned_cases) != target_cases
        ):
            raise ExternalValidationProtocolError(
                "external validation strata must cover every final case exactly once"
            )

        if not isinstance(separation_rule, PairwiseSeparationRule):
            raise ExternalValidationProtocolError(
                "external validation separation rule is invalid"
            )

        canonical_plans = tuple(
            sorted(tuple(constraint_plans), key=lambda item: item.name)
        )
        if any(not isinstance(item, ExternalConstraintPlan) for item in canonical_plans):
            raise ExternalValidationProtocolError(
                "external validation constraint plans are invalid"
            )
        if len({item.name for item in canonical_plans}) != len(canonical_plans):
            raise ExternalValidationProtocolError(
                "external validation constraint plan names must be unique"
            )

        canonical_method_hashes = tuple(method_validation_hashes)
        if len(set(canonical_method_hashes)) != len(canonical_method_hashes):
            raise ExternalValidationProtocolError(
                "external validation method hashes must be unique"
            )
        canonical_method_hashes = tuple(
            sorted(
                _hash(value, label="external validation method hash")
                for value in canonical_method_hashes
            )
        )

        return cls(
            name=name,
            version=version,
            evidence_declaration_hash=evidence_declaration.content_hash,
            brier_protocol_hash=brier_protocol.content_hash,
            log_protocol_hash=log_protocol.content_hash,
            adequacy_thresholds_by_score=tuple(canonical_thresholds),
            strata=canonical_strata,
            separation_rule=separation_rule,
            constraint_plans=canonical_plans,
            method_validation_hashes=canonical_method_hashes,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "evidence_declaration_hash": self.evidence_declaration_hash,
            "brier_protocol_hash": self.brier_protocol_hash,
            "log_protocol_hash": self.log_protocol_hash,
            "adequacy_thresholds_by_score": tuple(
                (role.value, thresholds.identity_payload())
                for role, thresholds in self.adequacy_thresholds_by_score
            ),
            "strata": tuple(item.identity_payload() for item in self.strata),
            "separation_rule": self.separation_rule.identity_payload(),
            "constraint_plans": tuple(
                item.identity_payload() for item in self.constraint_plans
            ),
            "method_validation_hashes": self.method_validation_hashes,
            "claim_scope": self.claim_scope,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ExternalReleasePreflight:
    preregistration_hash: str
    evidence_declaration_hash: str
    brier_protocol_hash: str
    brier_release_hash: str
    brier_verification_hash: str
    log_protocol_hash: str
    log_release_hash: str
    log_verification_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "preregistration_hash",
            "evidence_declaration_hash",
            "brier_protocol_hash",
            "brier_release_hash",
            "brier_verification_hash",
            "log_protocol_hash",
            "log_release_hash",
            "log_verification_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name.replace("_", " ")),
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "preregistration_hash": self.preregistration_hash,
            "evidence_declaration_hash": self.evidence_declaration_hash,
            "brier_protocol_hash": self.brier_protocol_hash,
            "brier_release_hash": self.brier_release_hash,
            "brier_verification_hash": self.brier_verification_hash,
            "log_protocol_hash": self.log_protocol_hash,
            "log_release_hash": self.log_release_hash,
            "log_verification_hash": self.log_verification_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _require_release_protocol_identity(
    release: ProtocolRelease,
    protocol: PreregisteredEvaluationProtocol,
    *,
    label: str,
) -> None:
    if release.protocol_hash != protocol.content_hash:
        raise ExternalValidationProtocolError(f"{label} release protocol identity changed")
    if release.dataset_hash != protocol.dataset_hash:
        raise ExternalValidationProtocolError(f"{label} release dataset identity changed")
    if release.target_spec_hash != protocol.target_spec_hash:
        raise ExternalValidationProtocolError(f"{label} release target identity changed")
    if release.candidate_hashes != _candidate_hashes(protocol):
        raise ExternalValidationProtocolError(f"{label} release candidate identities changed")


def _release_binding(
    release: ProtocolRelease,
    *,
    evidence_hash: str,
    preregistration_hash: str,
    score_role: ExternalScoreRole,
) -> tuple[str, str, str]:
    revision = release.source_revision
    required = (
        "repository_revision",
        "external_evidence_declaration_hash",
        "external_validation_preregistration_hash",
        "score_role",
    )
    if any(key not in revision for key in required):
        raise ExternalValidationProtocolError(
            f"{score_role.value} release is missing P3 source-revision bindings"
        )
    repository_revision = _text(
        revision["repository_revision"],
        label=f"{score_role.value} release repository revision",
    )
    declared_evidence = _hash(
        revision["external_evidence_declaration_hash"],
        label=f"{score_role.value} release evidence hash",
    )
    declared_preregistration = _hash(
        revision["external_validation_preregistration_hash"],
        label=f"{score_role.value} release preregistration hash",
    )
    if declared_evidence != evidence_hash:
        raise ExternalValidationProtocolError(
            f"{score_role.value} release external evidence identity changed"
        )
    if declared_preregistration != preregistration_hash:
        raise ExternalValidationProtocolError(
            f"{score_role.value} release P3 preregistration identity changed"
        )
    if revision["score_role"] != score_role.value:
        raise ExternalValidationProtocolError(
            f"{score_role.value} release score role changed"
        )
    return repository_revision, declared_evidence, declared_preregistration


def preflight_external_releases(
    *,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    brier_protocol: PreregisteredEvaluationProtocol,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    log_protocol: PreregisteredEvaluationProtocol,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
) -> ExternalReleasePreflight:
    if not isinstance(preregistration, ExternalValidationPreregistration):
        raise TypeError("external release preflight requires P3 preregistration")
    if not isinstance(evidence, ExternalEvidenceDeclaration):
        raise TypeError("external release preflight requires external evidence")
    for value, expected, label in (
        (brier_protocol, PreregisteredEvaluationProtocol, "Brier protocol"),
        (log_protocol, PreregisteredEvaluationProtocol, "Log protocol"),
        (brier_release, ProtocolRelease, "Brier release"),
        (log_release, ProtocolRelease, "Log release"),
        (brier_verified, VerifiedProtocolRelease, "Brier verification"),
        (log_verified, VerifiedProtocolRelease, "Log verification"),
    ):
        if not isinstance(value, expected):
            raise TypeError(f"external release preflight {label} has invalid type")

    if evidence.content_hash != preregistration.evidence_declaration_hash:
        raise ExternalValidationProtocolError(
            "external release preflight evidence does not match preregistration"
        )
    if brier_protocol.content_hash != preregistration.brier_protocol_hash:
        raise ExternalValidationProtocolError(
            "external release preflight Brier protocol changed"
        )
    if log_protocol.content_hash != preregistration.log_protocol_hash:
        raise ExternalValidationProtocolError(
            "external release preflight Log protocol changed"
        )

    brier_verified.require_matches(brier_protocol)
    log_verified.require_matches(log_protocol)
    if brier_verified.release_hash != brier_release.content_hash:
        raise ExternalValidationProtocolError(
            "Brier verification does not bind the supplied release"
        )
    if log_verified.release_hash != log_release.content_hash:
        raise ExternalValidationProtocolError(
            "Log verification does not bind the supplied release"
        )

    _require_release_protocol_identity(
        brier_release,
        brier_protocol,
        label="Brier",
    )
    _require_release_protocol_identity(
        log_release,
        log_protocol,
        label="Log",
    )

    brier_binding = _release_binding(
        brier_release,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role=ExternalScoreRole.BRIER,
    )
    log_binding = _release_binding(
        log_release,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role=ExternalScoreRole.LOG,
    )
    if brier_binding != log_binding:
        raise ExternalValidationProtocolError(
            "external sibling releases disagree on repository/evidence/preregistration identity"
        )

    return ExternalReleasePreflight(
        preregistration_hash=preregistration.content_hash,
        evidence_declaration_hash=evidence.content_hash,
        brier_protocol_hash=brier_protocol.content_hash,
        brier_release_hash=brier_release.content_hash,
        brier_verification_hash=brier_verified.content_hash,
        log_protocol_hash=log_protocol.content_hash,
        log_release_hash=log_release.content_hash,
        log_verification_hash=log_verified.content_hash,
    )


__all__ = [
    "EXTERNAL_CLAIM_SCOPE",
    "EXTERNAL_EVIDENCE_ORIGIN",
    "ExternalConstraintPlan",
    "ExternalConstraintStatus",
    "ExternalReleasePreflight",
    "ExternalScoreRole",
    "ExternalStratum",
    "ExternalValidationConstraintError",
    "ExternalValidationError",
    "ExternalValidationPreregistration",
    "ExternalValidationProtocolError",
    "PairwiseSeparationRule",
    "PredictiveAdequacyStatus",
    "PredictiveSeparationStatus",
    "preflight_external_releases",
]


@dataclass(frozen=True)
class ExternalPredictiveAdequacyFinding:
    model_name: str
    frozen_model_hash: str
    score_role: ExternalScoreRole
    loss_identity: Mapping[str, object]
    mean_loss: float
    worst_loss: float
    threshold_hash: str
    status: PredictiveAdequacyStatus
    parent_comparison_manifest_hash: str


@dataclass(frozen=True)
class ExternalPairwiseSeparationFinding:
    left_model: str
    right_model: str
    brier_mean_loss_delta: float
    log_mean_loss_delta: float
    preferred_model: str | None
    status: PredictiveSeparationStatus


@dataclass(frozen=True)
class ExternalStratumScore:
    stratum_name: str
    score_role: ExternalScoreRole
    model_name: str
    case_names: tuple[str, ...]
    mean_loss: float
    worst_loss: float


@dataclass(frozen=True)
class ExternalFinalEvaluation:
    preflight: ExternalReleasePreflight
    brier_report: object
    log_report: object
    adequacy_findings: tuple[ExternalPredictiveAdequacyFinding, ...]
    aggregate_adequacy: tuple[tuple[str, PredictiveAdequacyStatus], ...]
    separation_findings: tuple[ExternalPairwiseSeparationFinding, ...]
    stratum_scores: tuple[ExternalStratumScore, ...]


def _external_final_target_matches(
    target_set: TargetConstructionReport,
    protocol: PreregisteredEvaluationProtocol,
) -> bool:
    return (
        target_set.role is ObservationPartitionRole.FINAL_TEST
        and target_set.dataset_hash == protocol.dataset_hash
        and target_set.partition_hash == protocol.final_partition_hash
        and target_set.spec_hash == protocol.target_spec_hash
        and target_set.content_hash == protocol.final_target_hash
        and target_set.manifest.content_hash == protocol.final_target_manifest_hash
    )


def _external_adequacy_findings(
    *,
    brier_protocol: PreregisteredEvaluationProtocol,
    brier_report: object,
    log_protocol: PreregisteredEvaluationProtocol,
    log_report: object,
) -> tuple[
    tuple[ExternalPredictiveAdequacyFinding, ...],
    tuple[tuple[str, PredictiveAdequacyStatus], ...],
]:
    findings: list[ExternalPredictiveAdequacyFinding] = []
    statuses: dict[str, dict[ExternalScoreRole, PredictiveAdequacyStatus]] = {}
    for role, protocol, report in (
        (ExternalScoreRole.BRIER, brier_protocol, brier_report),
        (ExternalScoreRole.LOG, log_protocol, log_report),
    ):
        frozen_by_name = {candidate.name: candidate for candidate in protocol.candidates}
        threshold_hash = stable_content_hash(protocol.thresholds.identity_payload())
        for model_name in sorted(protocol.candidate_names):
            entry = report.entry_map[model_name]
            status = (
                PredictiveAdequacyStatus.MET
                if entry.adequate
                else PredictiveAdequacyStatus.NOT_MET
            )
            statuses.setdefault(model_name, {})[role] = status
            findings.append(
                ExternalPredictiveAdequacyFinding(
                    model_name=model_name,
                    frozen_model_hash=frozen_by_name[model_name].content_hash,
                    score_role=role,
                    loss_identity=_mapping(
                        protocol.loss_identity,
                        label=f"{role.value} predictive loss identity",
                    ),
                    mean_loss=float(entry.mean_loss),
                    worst_loss=float(entry.worst_loss),
                    threshold_hash=threshold_hash,
                    status=status,
                    parent_comparison_manifest_hash=report.manifest.content_hash,
                )
            )
    aggregate = tuple(
        (
            model_name,
            PredictiveAdequacyStatus.MET
            if statuses[model_name].get(ExternalScoreRole.BRIER)
            is PredictiveAdequacyStatus.MET
            and statuses[model_name].get(ExternalScoreRole.LOG)
            is PredictiveAdequacyStatus.MET
            else PredictiveAdequacyStatus.NOT_MET,
        )
        for model_name in sorted(statuses)
    )
    return tuple(findings), aggregate


def _external_separation_findings(
    *,
    preregistration: ExternalValidationPreregistration,
    brier_report: object,
    log_report: object,
) -> tuple[ExternalPairwiseSeparationFinding, ...]:
    names = tuple(sorted(brier_report.entry_map))
    if names != tuple(sorted(log_report.entry_map)):
        raise ExternalValidationProtocolError(
            "external final sibling comparisons have different model sets"
        )
    findings: list[ExternalPairwiseSeparationFinding] = []
    rule = preregistration.separation_rule
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            brier_delta = float(
                brier_report.entry_map[left].mean_loss
                - brier_report.entry_map[right].mean_loss
            )
            log_delta = float(
                log_report.entry_map[left].mean_loss
                - log_report.entry_map[right].mean_loss
            )
            same_direction = (
                (brier_delta < 0.0 and log_delta < 0.0)
                or (brier_delta > 0.0 and log_delta > 0.0)
            )
            separated = (
                same_direction
                and abs(brier_delta) >= rule.min_mean_loss_delta_brier
                and abs(log_delta) >= rule.min_mean_loss_delta_log
            )
            preferred = None
            if separated:
                preferred = left if brier_delta < 0.0 else right
            findings.append(
                ExternalPairwiseSeparationFinding(
                    left_model=left,
                    right_model=right,
                    brier_mean_loss_delta=brier_delta,
                    log_mean_loss_delta=log_delta,
                    preferred_model=preferred,
                    status=(
                        PredictiveSeparationStatus.SEPARATED
                        if separated
                        else PredictiveSeparationStatus.NOT_SEPARATED
                    ),
                )
            )
    return tuple(findings)


def _external_stratum_scores(
    *,
    preregistration: ExternalValidationPreregistration,
    brier_report: object,
    log_report: object,
) -> tuple[ExternalStratumScore, ...]:
    scores: list[ExternalStratumScore] = []
    for role, report in (
        (ExternalScoreRole.BRIER, brier_report),
        (ExternalScoreRole.LOG, log_report),
    ):
        for model_name in sorted(report.entry_map):
            entry = report.entry_map[model_name]
            case_losses = {
                case.name: float(case.loss)
                for case in entry.final_test.validation.cases
            }
            for stratum in preregistration.strata:
                losses = tuple(case_losses[name] for name in stratum.final_case_names)
                scores.append(
                    ExternalStratumScore(
                        stratum_name=stratum.name,
                        score_role=role,
                        model_name=model_name,
                        case_names=stratum.final_case_names,
                        mean_loss=sum(losses) / len(losses),
                        worst_loss=max(losses),
                    )
                )
    return tuple(scores)


def evaluate_external_final(
    *,
    runner: object,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    brier_protocol: PreregisteredEvaluationProtocol,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    brier_models: tuple[object, ...],
    brier_loss: object,
    log_protocol: PreregisteredEvaluationProtocol,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
    log_models: tuple[object, ...],
    log_loss: object,
    final_targets: TargetConstructionReport,
    extractor: object,
) -> ExternalFinalEvaluation:
    from narrative_dynamics.observations.release import compare_released_models
    from narrative_dynamics.simulation import SimulationRunner

    if not isinstance(runner, SimulationRunner):
        raise TypeError("external final evaluation requires SimulationRunner")
    brier_models = tuple(brier_models)
    log_models = tuple(log_models)
    if not isinstance(final_targets, TargetConstructionReport):
        raise TypeError("external final evaluation requires final targets")

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
    if callable_identity(extractor) != dict(brier_protocol.metric_identity):
        raise ExternalValidationProtocolError(
            "external final metric extractor changed from preregistration"
        )
    if callable_identity(extractor) != dict(log_protocol.metric_identity):
        raise ExternalValidationProtocolError(
            "external final metric extractor differs across siblings"
        )
    if metric_loss_identity(brier_loss) != dict(brier_protocol.loss_identity):
        raise ExternalValidationProtocolError(
            "external final Brier loss changed from preregistration"
        )
    if metric_loss_identity(log_loss) != dict(log_protocol.loss_identity):
        raise ExternalValidationProtocolError(
            "external final Log loss changed from preregistration"
        )

    brier_report = compare_released_models(
        runner=runner,
        verified_release=brier_verified,
        protocol=brier_protocol,
        models=brier_models,
        target_set=final_targets,
        extractor=extractor,
        loss=brier_loss,
    )
    log_report = compare_released_models(
        runner=runner,
        verified_release=log_verified,
        protocol=log_protocol,
        models=log_models,
        target_set=final_targets,
        extractor=extractor,
        loss=log_loss,
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


__all__ += [
    "ExternalFinalEvaluation",
    "ExternalPairwiseSeparationFinding",
    "ExternalPredictiveAdequacyFinding",
    "ExternalStratumScore",
    "evaluate_external_final",
]


@dataclass(frozen=True)
class ExternalConstraintCandidateLoss:
    parameters: tuple[tuple[str, float], ...]
    brier_loss: float
    log_loss: float
    brier_parent_manifest_hashes: tuple[str, ...]
    log_parent_manifest_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        parameters = tuple(sorted((name, float(value)) for name, value in self.parameters))
        if not parameters or len({name for name, _ in parameters}) != len(parameters):
            raise ExternalValidationConstraintError(
                "external constraint candidate parameters must be non-empty and unique"
            )
        if any(not math.isfinite(value) for _, value in parameters):
            raise ExternalValidationConstraintError(
                "external constraint candidate parameters must be finite"
            )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "brier_loss",
            _number(self.brier_loss, label="external constraint Brier candidate loss", minimum=0.0),
        )
        object.__setattr__(
            self,
            "log_loss",
            _number(self.log_loss, label="external constraint Log candidate loss", minimum=0.0),
        )
        for field_name in (
            "brier_parent_manifest_hashes",
            "log_parent_manifest_hashes",
        ):
            values = tuple(
                sorted(
                    _hash(value, label="external constraint calibration manifest hash")
                    for value in getattr(self, field_name)
                )
            )
            if not values or len(set(values)) != len(values):
                raise ExternalValidationConstraintError(
                    "external constraint calibration manifest hashes must be non-empty and unique"
                )
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True)
class ExternalCoordinateConstraint:
    parameter_name: str
    retained_values: tuple[float, ...]
    constrained: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_name",
            _text(self.parameter_name, label="external constraint parameter name"),
        )
        values = tuple(sorted(set(float(value) for value in self.retained_values)))
        if not values or any(not math.isfinite(value) for value in values):
            raise ExternalValidationConstraintError(
                "external constraint retained values must be non-empty and finite"
            )
        object.__setattr__(self, "retained_values", values)
        if not isinstance(self.constrained, bool):
            raise ExternalValidationConstraintError(
                "external constraint coordinate status must be bool"
            )
        if self.constrained != (len(values) == 1):
            raise ExternalValidationConstraintError(
                "external constraint coordinate status does not match retained values"
            )


@dataclass(frozen=True)
class ExternalConstraintFinding:
    plan_hash: str
    model_identity: Mapping[str, object]
    candidate_losses: tuple[ExternalConstraintCandidateLoss, ...]
    brier_compatible_parameters: tuple[tuple[tuple[str, float], ...], ...]
    log_compatible_parameters: tuple[tuple[tuple[str, float], ...], ...]
    compatible_parameters: tuple[tuple[tuple[str, float], ...], ...]
    coordinates: tuple[ExternalCoordinateConstraint, ...]
    status: ExternalConstraintStatus
    parent_manifest_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_hash", _hash(self.plan_hash, label="external constraint plan hash"))
        object.__setattr__(
            self,
            "model_identity",
            _mapping(self.model_identity, label="external constraint finding model identity"),
        )
        candidate_losses = tuple(sorted(self.candidate_losses, key=lambda item: item.parameters))
        if not candidate_losses or any(
            not isinstance(item, ExternalConstraintCandidateLoss) for item in candidate_losses
        ):
            raise ExternalValidationConstraintError(
                "external constraint finding requires candidate losses"
            )
        if len({item.parameters for item in candidate_losses}) != len(candidate_losses):
            raise ExternalValidationConstraintError(
                "external constraint candidate parameters must be unique"
            )
        object.__setattr__(self, "candidate_losses", candidate_losses)
        for field_name in (
            "brier_compatible_parameters",
            "log_compatible_parameters",
            "compatible_parameters",
        ):
            values = tuple(sorted(tuple(tuple(pair) for pair in item) for item in getattr(self, field_name)))
            if len(set(values)) != len(values):
                raise ExternalValidationConstraintError(
                    "external compatible parameter tuples must be unique"
                )
            object.__setattr__(self, field_name, values)
        if not self.compatible_parameters:
            raise ExternalValidationConstraintError(
                "external constraint compatible intersection must be non-empty"
            )
        coordinates = tuple(sorted(self.coordinates, key=lambda item: item.parameter_name))
        if not coordinates or any(
            not isinstance(item, ExternalCoordinateConstraint) for item in coordinates
        ):
            raise ExternalValidationConstraintError(
                "external constraint finding requires coordinate findings"
            )
        object.__setattr__(self, "coordinates", coordinates)
        if not isinstance(self.status, ExternalConstraintStatus):
            raise ExternalValidationConstraintError(
                "external constraint finding status is invalid"
            )
        expected_status = (
            ExternalConstraintStatus.CONSTRAINED
            if all(item.constrained for item in coordinates)
            else ExternalConstraintStatus.NOT_CONSTRAINED
        )
        if self.status is not expected_status:
            raise ExternalValidationConstraintError(
                "external constraint aggregate status does not match coordinates"
            )
        parents = tuple(
            sorted(
                _hash(value, label="external constraint parent manifest hash")
                for value in self.parent_manifest_hashes
            )
        )
        if not parents or len(set(parents)) != len(parents):
            raise ExternalValidationConstraintError(
                "external constraint parent manifest hashes must be non-empty and unique"
            )
        object.__setattr__(self, "parent_manifest_hashes", parents)


def _external_constraint_score_table(
    *,
    runner: object,
    plan: ExternalConstraintPlan,
    model: object,
    selection_targets: TargetConstructionReport,
    extractor: object,
    loss: object,
    expected_loss_identity: Mapping[str, object],
) -> tuple[
    dict[tuple[tuple[str, float], ...], float],
    tuple[str, ...],
]:
    from statistics import fmean

    from narrative_dynamics.calibration import calibrate_grid

    cases = {case.name: case for case in selection_targets.cases}
    losses_by_parameters: dict[tuple[tuple[str, float], ...], list[float]] = {}
    parent_hashes: list[str] = []
    expected_parameters: set[tuple[tuple[str, float], ...]] | None = None
    for case_name in plan.selection_case_names:
        case = cases[case_name]
        result = calibrate_grid(
            runner=runner,
            model=model,
            scenario=case.scenario,
            parameter_grid=plan.parameter_grid_map,
            seeds=plan.simulation_seeds,
            extractor=extractor,
            target=case.target_map,
            loss=loss,
        )
        if result.manifest is None:
            raise ExternalValidationConstraintError(
                "external constraint calibration is missing a manifest"
            )
        parent_hashes.append(result.manifest.content_hash)
        current_parameters = {candidate.parameters for candidate in result.ranking}
        if expected_parameters is None:
            expected_parameters = current_parameters
        elif current_parameters != expected_parameters:
            raise ExternalValidationConstraintError(
                "external constraint calibration candidate set drifted across cases"
            )
        for candidate in result.ranking:
            losses_by_parameters.setdefault(candidate.parameters, []).append(float(candidate.loss))
    if expected_parameters is None or not expected_parameters:
        raise ExternalValidationConstraintError(
            "external constraint calibration produced no candidates"
        )
    if metric_loss_identity(loss) != dict(expected_loss_identity):
        raise ExternalValidationConstraintError(
            "external constraint loss identity changed during evaluation"
        )
    return (
        {
            parameters: fmean(losses_by_parameters[parameters])
            for parameters in sorted(expected_parameters)
        },
        tuple(parent_hashes),
    )


def evaluate_external_constraint(
    *,
    runner: object,
    plan: ExternalConstraintPlan,
    model: object,
    selection_targets: TargetConstructionReport,
    extractor: object,
    brier_loss: object,
    log_loss: object,
) -> ExternalConstraintFinding:
    from narrative_dynamics.simulation import SimulationRunner

    if not isinstance(runner, SimulationRunner):
        raise TypeError("external constraint evaluation requires SimulationRunner")
    if not isinstance(plan, ExternalConstraintPlan):
        raise TypeError("external constraint evaluation requires ExternalConstraintPlan")
    if not isinstance(selection_targets, TargetConstructionReport):
        raise TypeError("external constraint evaluation requires constructed targets")
    if selection_targets.role is not ObservationPartitionRole.SELECTION_VALIDATION:
        raise ExternalValidationConstraintError(
            "external constraint evaluation requires selection-validation targets"
        )
    if component_identity(model) != dict(plan.model_identity):
        raise ExternalValidationConstraintError(
            "external constraint runtime model identity changed"
        )
    if callable_identity(extractor) != dict(plan.metric_identity):
        raise ExternalValidationConstraintError(
            "external constraint metric extractor changed"
        )
    if metric_loss_identity(brier_loss) != dict(plan.brier_loss_identity):
        raise ExternalValidationConstraintError(
            "external constraint Brier loss changed"
        )
    if metric_loss_identity(log_loss) != dict(plan.log_loss_identity):
        raise ExternalValidationConstraintError(
            "external constraint Log loss changed"
        )
    cases = {case.name: case for case in selection_targets.cases}
    if any(name not in cases for name in plan.selection_case_names):
        raise ExternalValidationConstraintError(
            "external constraint selection cases changed"
        )

    brier_table, brier_parent_hashes = _external_constraint_score_table(
        runner=runner,
        plan=plan,
        model=model,
        selection_targets=selection_targets,
        extractor=extractor,
        loss=brier_loss,
        expected_loss_identity=plan.brier_loss_identity,
    )
    log_table, log_parent_hashes = _external_constraint_score_table(
        runner=runner,
        plan=plan,
        model=model,
        selection_targets=selection_targets,
        extractor=extractor,
        loss=log_loss,
        expected_loss_identity=plan.log_loss_identity,
    )
    if set(brier_table) != set(log_table):
        raise ExternalValidationConstraintError(
            "external constraint Brier/Log candidate sets disagree"
        )

    brier_best = min(brier_table.values())
    log_best = min(log_table.values())
    brier_compatible = tuple(
        sorted(
            parameters
            for parameters, value in brier_table.items()
            if value <= brier_best + plan.brier_acceptance_loss_delta
        )
    )
    log_compatible = tuple(
        sorted(
            parameters
            for parameters, value in log_table.items()
            if value <= log_best + plan.log_acceptance_loss_delta
        )
    )
    compatible = tuple(sorted(set(brier_compatible).intersection(log_compatible)))
    if not compatible:
        raise ExternalValidationConstraintError(
            "external Brier/Log compatible parameter intersection is empty"
        )

    candidate_losses = tuple(
        ExternalConstraintCandidateLoss(
            parameters=parameters,
            brier_loss=brier_table[parameters],
            log_loss=log_table[parameters],
            brier_parent_manifest_hashes=brier_parent_hashes,
            log_parent_manifest_hashes=log_parent_hashes,
        )
        for parameters in sorted(brier_table)
    )
    coordinates: list[ExternalCoordinateConstraint] = []
    for parameter_name in plan.target_coordinates:
        retained_values = tuple(
            sorted({dict(parameters)[parameter_name] for parameters in compatible})
        )
        coordinates.append(
            ExternalCoordinateConstraint(
                parameter_name=parameter_name,
                retained_values=retained_values,
                constrained=len(retained_values) == 1,
            )
        )
    status = (
        ExternalConstraintStatus.CONSTRAINED
        if all(item.constrained for item in coordinates)
        else ExternalConstraintStatus.NOT_CONSTRAINED
    )
    return ExternalConstraintFinding(
        plan_hash=plan.content_hash,
        model_identity=plan.model_identity,
        candidate_losses=candidate_losses,
        brier_compatible_parameters=brier_compatible,
        log_compatible_parameters=log_compatible,
        compatible_parameters=compatible,
        coordinates=tuple(coordinates),
        status=status,
        parent_manifest_hashes=tuple(sorted(set(brier_parent_hashes + log_parent_hashes))),
    )


__all__ += [
    "ExternalConstraintCandidateLoss",
    "ExternalConstraintFinding",
    "ExternalCoordinateConstraint",
    "evaluate_external_constraint",
]


from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage


class ExternalValidationReportError(ExternalValidationError):
    pass


@dataclass(frozen=True)
class ExternalValidationReport:
    preregistration_hash: str
    evidence_declaration_hash: str
    claim_scope: str
    brier_protocol_hash: str
    brier_release_hash: str
    brier_verification_hash: str
    log_protocol_hash: str
    log_release_hash: str
    log_verification_hash: str
    brier_comparison_manifest_hash: str
    log_comparison_manifest_hash: str
    adequacy_findings: tuple[ExternalPredictiveAdequacyFinding, ...]
    aggregate_adequacy: tuple[tuple[str, PredictiveAdequacyStatus], ...]
    separation_findings: tuple[ExternalPairwiseSeparationFinding, ...]
    stratum_scores: tuple[ExternalStratumScore, ...]
    constraint_findings: tuple[ExternalConstraintFinding, ...]
    method_validation_hashes: tuple[str, ...]
    manifest: ExperimentManifest

    def __post_init__(self) -> None:
        for field_name in (
            "preregistration_hash",
            "evidence_declaration_hash",
            "brier_protocol_hash",
            "brier_release_hash",
            "brier_verification_hash",
            "log_protocol_hash",
            "log_release_hash",
            "log_verification_hash",
            "brier_comparison_manifest_hash",
            "log_comparison_manifest_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name.replace("_", " ")),
            )
        if self.claim_scope != EXTERNAL_CLAIM_SCOPE:
            raise ExternalValidationReportError(
                "external validation report claim scope is fixed"
            )

        adequacy = tuple(
            sorted(
                tuple(self.adequacy_findings),
                key=lambda item: (item.model_name, item.score_role.value),
            )
        )
        if any(
            not isinstance(item, ExternalPredictiveAdequacyFinding)
            for item in adequacy
        ):
            raise ExternalValidationReportError(
                "external validation report adequacy findings are invalid"
            )
        for item in adequacy:
            if not isinstance(item.score_role, ExternalScoreRole):
                raise ExternalValidationReportError(
                    "external validation report adequacy score role is invalid"
                )
            if not isinstance(item.status, PredictiveAdequacyStatus):
                raise ExternalValidationReportError(
                    "external validation report adequacy status is invalid"
                )
            _hash(item.frozen_model_hash, label="external adequacy frozen model hash")
            _hash(item.threshold_hash, label="external adequacy threshold hash")
            parent_hash = _hash(
                item.parent_comparison_manifest_hash,
                label="external adequacy parent comparison manifest hash",
            )
            expected_parent = (
                self.brier_comparison_manifest_hash
                if item.score_role is ExternalScoreRole.BRIER
                else self.log_comparison_manifest_hash
            )
            if parent_hash != expected_parent:
                raise ExternalValidationReportError(
                    "external adequacy finding binds the wrong child comparison"
                )
            _number(item.mean_loss, label="external adequacy mean loss", minimum=0.0)
            _number(item.worst_loss, label="external adequacy worst loss", minimum=0.0)
            _mapping(item.loss_identity, label="external adequacy loss identity")
        object.__setattr__(self, "adequacy_findings", adequacy)

        aggregate = tuple(sorted(tuple(self.aggregate_adequacy), key=lambda item: item[0]))
        if not aggregate or len({name for name, _ in aggregate}) != len(aggregate):
            raise ExternalValidationReportError(
                "external validation aggregate adequacy must be non-empty and unique"
            )
        for model_name, status in aggregate:
            _text(model_name, label="external aggregate adequacy model")
            if not isinstance(status, PredictiveAdequacyStatus):
                raise ExternalValidationReportError(
                    "external validation aggregate adequacy status is invalid"
                )
        object.__setattr__(self, "aggregate_adequacy", aggregate)

        separation = tuple(
            sorted(
                tuple(self.separation_findings),
                key=lambda item: (item.left_model, item.right_model),
            )
        )
        if any(
            not isinstance(item, ExternalPairwiseSeparationFinding)
            for item in separation
        ):
            raise ExternalValidationReportError(
                "external validation separation findings are invalid"
            )
        for item in separation:
            if not isinstance(item.status, PredictiveSeparationStatus):
                raise ExternalValidationReportError(
                    "external validation separation status is invalid"
                )
        object.__setattr__(self, "separation_findings", separation)

        strata = tuple(
            sorted(
                tuple(self.stratum_scores),
                key=lambda item: (
                    item.stratum_name,
                    item.score_role.value,
                    item.model_name,
                ),
            )
        )
        if any(not isinstance(item, ExternalStratumScore) for item in strata):
            raise ExternalValidationReportError(
                "external validation stratum scores are invalid"
            )
        for item in strata:
            if not isinstance(item.score_role, ExternalScoreRole):
                raise ExternalValidationReportError(
                    "external validation stratum score role is invalid"
                )
        object.__setattr__(self, "stratum_scores", strata)

        constraints = tuple(
            sorted(tuple(self.constraint_findings), key=lambda item: item.plan_hash)
        )
        if any(not isinstance(item, ExternalConstraintFinding) for item in constraints):
            raise ExternalValidationReportError(
                "external validation constraint findings are invalid"
            )
        if len({item.plan_hash for item in constraints}) != len(constraints):
            raise ExternalValidationReportError(
                "external validation constraint finding plan hashes must be unique"
            )
        for item in constraints:
            if not isinstance(item.status, ExternalConstraintStatus):
                raise ExternalValidationReportError(
                    "external validation constraint status is invalid"
                )
        object.__setattr__(self, "constraint_findings", constraints)

        method_hashes = tuple(
            sorted(
                _hash(value, label="external validation method lineage hash")
                for value in self.method_validation_hashes
            )
        )
        if len(set(method_hashes)) != len(method_hashes):
            raise ExternalValidationReportError(
                "external validation method lineage hashes must be unique"
            )
        object.__setattr__(self, "method_validation_hashes", method_hashes)

        if not isinstance(self.manifest, ExperimentManifest):
            raise ExternalValidationReportError(
                "external validation report requires ExperimentManifest"
            )
        if self.manifest.stage is not ExperimentStage.EXTERNAL_VALIDATION:
            raise ExternalValidationReportError(
                "external validation report manifest stage is invalid"
            )
        expected_input_keys = {
            "preregistration_hash",
            "evidence_declaration_hash",
            "claim_scope",
            "brier_protocol_hash",
            "log_protocol_hash",
            "preflight_hash",
            "method_validation_hashes",
            "constraint_plan_hashes",
        }
        if set(self.manifest.inputs) != expected_input_keys:
            raise ExternalValidationReportError(
                "external validation report manifest inputs are incomplete"
            )
        expected_inputs = {
            "preregistration_hash": self.preregistration_hash,
            "evidence_declaration_hash": self.evidence_declaration_hash,
            "claim_scope": EXTERNAL_CLAIM_SCOPE,
            "brier_protocol_hash": self.brier_protocol_hash,
            "log_protocol_hash": self.log_protocol_hash,
            "method_validation_hashes": self.method_validation_hashes,
            "constraint_plan_hashes": tuple(
                item.plan_hash for item in self.constraint_findings
            ),
        }
        for key, expected_value in expected_inputs.items():
            if self.manifest.inputs[key] != expected_value:
                raise ExternalValidationReportError(
                    f"external validation report manifest input changed: {key}"
                )
        manifest_preflight_hash = _hash(
            self.manifest.inputs["preflight_hash"],
            label="external validation report preflight hash",
        )
        expected_preflight_hash = ExternalReleasePreflight(
            preregistration_hash=self.preregistration_hash,
            evidence_declaration_hash=self.evidence_declaration_hash,
            brier_protocol_hash=self.brier_protocol_hash,
            brier_release_hash=self.brier_release_hash,
            brier_verification_hash=self.brier_verification_hash,
            log_protocol_hash=self.log_protocol_hash,
            log_release_hash=self.log_release_hash,
            log_verification_hash=self.log_verification_hash,
        ).content_hash
        if manifest_preflight_hash != expected_preflight_hash:
            raise ExternalValidationReportError(
                "external validation report preflight binding is inconsistent"
            )

        expected_parents = {
            self.brier_comparison_manifest_hash,
            self.log_comparison_manifest_hash,
            *self.method_validation_hashes,
        }
        for finding in self.constraint_findings:
            expected_parents.update(finding.parent_manifest_hashes)
        if self.manifest.parent_hashes != tuple(sorted(expected_parents)):
            raise ExternalValidationReportError(
                "external validation report parent lineage is incomplete or forged"
            )


def _validate_external_final_statuses(
    final_evaluation: ExternalFinalEvaluation,
) -> None:
    if any(
        not isinstance(item.status, PredictiveAdequacyStatus)
        for item in final_evaluation.adequacy_findings
    ):
        raise ExternalValidationReportError(
            "external final adequacy findings contain a non-P3 status"
        )
    if any(
        not isinstance(status, PredictiveAdequacyStatus)
        for _, status in final_evaluation.aggregate_adequacy
    ):
        raise ExternalValidationReportError(
            "external final aggregate adequacy contains a non-P3 status"
        )
    if any(
        not isinstance(item.status, PredictiveSeparationStatus)
        for item in final_evaluation.separation_findings
    ):
        raise ExternalValidationReportError(
            "external final separation findings contain a non-P3 status"
        )


def build_external_validation_report(
    *,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    final_evaluation: ExternalFinalEvaluation,
    constraint_findings: tuple[ExternalConstraintFinding, ...] = (),
) -> ExternalValidationReport:
    if not isinstance(preregistration, ExternalValidationPreregistration):
        raise TypeError(
            "external validation report requires ExternalValidationPreregistration"
        )
    if not isinstance(evidence, ExternalEvidenceDeclaration):
        raise TypeError(
            "external validation report requires ExternalEvidenceDeclaration"
        )
    if not isinstance(final_evaluation, ExternalFinalEvaluation):
        raise TypeError(
            "external validation report requires ExternalFinalEvaluation"
        )
    if evidence.content_hash != preregistration.evidence_declaration_hash:
        raise ExternalValidationReportError(
            "external validation report evidence does not match preregistration"
        )
    if preregistration.claim_scope != EXTERNAL_CLAIM_SCOPE:
        raise ExternalValidationReportError(
            "external validation preregistration claim scope changed"
        )

    preflight = final_evaluation.preflight
    if preflight.preregistration_hash != preregistration.content_hash:
        raise ExternalValidationReportError(
            "external final preflight does not match preregistration"
        )
    if preflight.evidence_declaration_hash != evidence.content_hash:
        raise ExternalValidationReportError(
            "external final preflight does not match evidence declaration"
        )
    if preflight.brier_protocol_hash != preregistration.brier_protocol_hash:
        raise ExternalValidationReportError(
            "external final Brier protocol does not match preregistration"
        )
    if preflight.log_protocol_hash != preregistration.log_protocol_hash:
        raise ExternalValidationReportError(
            "external final Log protocol does not match preregistration"
        )

    findings = tuple(sorted(tuple(constraint_findings), key=lambda item: item.plan_hash))
    if any(not isinstance(item, ExternalConstraintFinding) for item in findings):
        raise ExternalValidationReportError(
            "external validation report constraint finding type is invalid"
        )
    expected_plan_hashes = tuple(
        sorted(plan.content_hash for plan in preregistration.constraint_plans)
    )
    actual_plan_hashes = tuple(item.plan_hash for item in findings)
    if actual_plan_hashes != expected_plan_hashes:
        raise ExternalValidationReportError(
            "external validation report requires exactly the preregistered constraint findings"
        )
    if len(set(actual_plan_hashes)) != len(actual_plan_hashes):
        raise ExternalValidationReportError(
            "external validation report constraint findings must be unique"
        )

    _validate_external_final_statuses(final_evaluation)
    if any(
        not isinstance(item.status, ExternalConstraintStatus)
        for item in findings
    ):
        raise ExternalValidationReportError(
            "external validation constraint findings contain a non-P3 status"
        )

    brier_comparison_manifest_hash = _hash(
        final_evaluation.brier_report.manifest.content_hash,
        label="external Brier comparison manifest hash",
    )
    log_comparison_manifest_hash = _hash(
        final_evaluation.log_report.manifest.content_hash,
        label="external Log comparison manifest hash",
    )
    canonical_parent_hashes = {
        brier_comparison_manifest_hash,
        log_comparison_manifest_hash,
        *preregistration.method_validation_hashes,
    }
    for finding in findings:
        canonical_parent_hashes.update(finding.parent_manifest_hashes)

    manifest = ExperimentManifest(
        stage=ExperimentStage.EXTERNAL_VALIDATION,
        inputs={
            "preregistration_hash": preregistration.content_hash,
            "evidence_declaration_hash": evidence.content_hash,
            "claim_scope": EXTERNAL_CLAIM_SCOPE,
            "brier_protocol_hash": preflight.brier_protocol_hash,
            "log_protocol_hash": preflight.log_protocol_hash,
            "preflight_hash": preflight.content_hash,
            "method_validation_hashes": preregistration.method_validation_hashes,
            "constraint_plan_hashes": expected_plan_hashes,
        },
        parent_hashes=tuple(sorted(canonical_parent_hashes)),
    )
    return ExternalValidationReport(
        preregistration_hash=preregistration.content_hash,
        evidence_declaration_hash=evidence.content_hash,
        claim_scope=EXTERNAL_CLAIM_SCOPE,
        brier_protocol_hash=preflight.brier_protocol_hash,
        brier_release_hash=preflight.brier_release_hash,
        brier_verification_hash=preflight.brier_verification_hash,
        log_protocol_hash=preflight.log_protocol_hash,
        log_release_hash=preflight.log_release_hash,
        log_verification_hash=preflight.log_verification_hash,
        brier_comparison_manifest_hash=brier_comparison_manifest_hash,
        log_comparison_manifest_hash=log_comparison_manifest_hash,
        adequacy_findings=final_evaluation.adequacy_findings,
        aggregate_adequacy=final_evaluation.aggregate_adequacy,
        separation_findings=final_evaluation.separation_findings,
        stratum_scores=final_evaluation.stratum_scores,
        constraint_findings=findings,
        method_validation_hashes=preregistration.method_validation_hashes,
        manifest=manifest,
    )


__all__ += [
    "ExternalValidationReport",
    "ExternalValidationReportError",
    "build_external_validation_report",
]
