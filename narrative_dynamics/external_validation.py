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


__all__ = [
    "EXTERNAL_CLAIM_SCOPE",
    "EXTERNAL_EVIDENCE_ORIGIN",
    "ExternalConstraintPlan",
    "ExternalConstraintStatus",
    "ExternalScoreRole",
    "ExternalStratum",
    "ExternalValidationConstraintError",
    "ExternalValidationError",
    "ExternalValidationPreregistration",
    "ExternalValidationProtocolError",
    "PairwiseSeparationRule",
    "PredictiveAdequacyStatus",
    "PredictiveSeparationStatus",
]
