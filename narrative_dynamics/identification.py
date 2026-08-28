from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
import math
import re
from statistics import fmean
from types import MappingProxyType

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.losses import MetricLoss, metric_loss_identity
from narrative_dynamics.manifest import callable_identity, component_identity
from narrative_dynamics.observations.dataset import (
    ObservationDataset,
    ObservationPartitionRole,
)
from narrative_dynamics.observations.preregistration import (
    PreregisteredEvaluationProtocol,
)
from narrative_dynamics.observations.targets import CategoricalTargetSpec
from narrative_dynamics.uncertainty import (
    IdentifiabilityReport,
    ParameterAcceptanceSet,
    diagnose_identifiability,
)


ParameterTuple = tuple[tuple[str, float], ...]
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_CLAIM_SCOPE = "synthetic_protocol_only"
_ALLOWED_RELATIONSHIPS = frozenset(
    {"intentional_vs_reactive", "planning_vs_intentional"}
)
_EQUIVALENCE_TOLERANCE = 1e-12


class SyntheticIdentificationError(ValueError):
    """Base error for the synthetic identification research boundary."""


class IdentificationProtocolError(SyntheticIdentificationError):
    """A frozen synthetic-identification protocol is invalid or drifted."""


class IdentificationRecoveryError(SyntheticIdentificationError):
    """A synthetic generator could not support an identification conclusion."""


class InterventionCertificationError(SyntheticIdentificationError):
    """An information intervention changed undeclared semantics."""


class IdentificationComparisonError(SyntheticIdentificationError):
    """Frozen final model-comparison inputs drifted."""


class IdentificationReportError(SyntheticIdentificationError):
    """An aggregate identification report or lineage is inconsistent."""


class IdentificationStatus(str, Enum):
    IDENTIFIED_UNDER_PROTOCOL = "identified_under_protocol"
    NOT_IDENTIFIED_UNDER_PROTOCOL = "not_identified_under_protocol"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise IdentificationProtocolError(f"{label} must be a non-empty trimmed string")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IdentificationProtocolError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise IdentificationProtocolError(f"{label} must be finite")
    return numeric


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise IdentificationProtocolError(f"{label} must be a sha256 content hash")
    return value


def _freeze(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentificationProtocolError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise IdentificationProtocolError(
                    f"{label} mapping keys must be non-empty strings"
                )
            result[key] = _freeze(value[key], label=f"{label}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise IdentificationProtocolError(
        f"{label} values must be canonical scalars, mappings, lists, or tuples"
    )


def _freeze_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IdentificationProtocolError(f"{label} must be a mapping")
    frozen = _freeze(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _parameter_tuple(
    value: Mapping[str, float] | Iterable[tuple[str, float]],
    *,
    label: str,
) -> ParameterTuple:
    items = value.items() if isinstance(value, Mapping) else value
    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_name, raw_value in items:
        name = _text(raw_name, label=f"{label} parameter name")
        if name in seen:
            raise IdentificationProtocolError(f"{label} parameter names must be unique")
        seen.add(name)
        canonical.append((name, _finite(raw_value, label=f"{label}.{name}")))
    if not canonical:
        raise IdentificationProtocolError(f"{label} must contain at least one parameter")
    return tuple(sorted(canonical))


def _parameter_grid(
    value: Mapping[str, Iterable[float]],
) -> Mapping[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise IdentificationProtocolError("parameter grid must be a non-empty mapping")
    canonical: dict[str, tuple[float, ...]] = {}
    for raw_name in sorted(value):
        name = _text(raw_name, label="parameter grid name")
        raw_values = tuple(value[raw_name])
        if not raw_values:
            raise IdentificationProtocolError("parameter grid dimensions must be non-empty")
        values = tuple(
            _finite(item, label=f"parameter grid {name!r}") for item in raw_values
        )
        if len(set(values)) != len(values):
            raise IdentificationProtocolError("parameter grid values must be unique")
        canonical[name] = tuple(sorted(values))
    return MappingProxyType(canonical)


def _seed_plan(value: Iterable[int], *, label: str) -> tuple[int, ...]:
    seeds = tuple(value)
    if not seeds:
        raise IdentificationProtocolError(f"{label} must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise IdentificationProtocolError(f"{label} must contain integers")
    if len(set(seeds)) != len(seeds):
        raise IdentificationProtocolError(f"{label} seeds must be unique")
    return seeds


def _seed_blocks(value: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    blocks = tuple(_seed_plan(block, label="calibration seed block") for block in value)
    if not blocks:
        raise IdentificationProtocolError("calibration seed blocks must be non-empty")
    if len(set(blocks)) != len(blocks):
        raise IdentificationProtocolError("calibration seed blocks must be unique")
    return blocks


def _hash_tuple(value: Iterable[str], *, label: str) -> tuple[str, ...]:
    hashes = tuple(_hash(item, label=label) for item in value)
    if not hashes:
        raise IdentificationProtocolError(f"{label} set must be non-empty")
    if len(set(hashes)) != len(hashes):
        raise IdentificationProtocolError(f"{label} values must be unique")
    return tuple(sorted(hashes))


@dataclass(frozen=True)
class ParameterRecoveryExperiment:
    name: str
    model_identity: Mapping[str, object]
    true_parameters: Mapping[str, float] | ParameterTuple
    parameter_grid: Mapping[str, Iterable[float]]
    target_coordinates: tuple[str, ...]
    cases: tuple[Scenario, ...]
    generation_seeds: tuple[int, ...]
    calibration_seed_blocks: tuple[tuple[int, ...], ...]
    acceptance_loss_delta: float
    min_acceptance_fraction: float
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    expected_status: IdentificationStatus | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="recovery experiment name"))
        object.__setattr__(
            self,
            "model_identity",
            _freeze_mapping(self.model_identity, label="recovery model identity"),
        )
        grid = _parameter_grid(self.parameter_grid)
        object.__setattr__(self, "parameter_grid", grid)
        truth = _parameter_tuple(self.true_parameters, label="true parameters")
        if tuple(name for name, _ in truth) != tuple(grid):
            raise IdentificationProtocolError(
                "true parameters must have exactly the parameter-grid schema"
            )
        for name, value in truth:
            if value not in grid[name]:
                raise IdentificationProtocolError(
                    "true parameters must belong to the complete candidate grid"
                )
        object.__setattr__(self, "true_parameters", truth)

        coordinates = tuple(
            _text(item, label="target coordinate") for item in self.target_coordinates
        )
        if not coordinates or len(set(coordinates)) != len(coordinates):
            raise IdentificationProtocolError(
                "target coordinates must be non-empty and unique"
            )
        if not set(coordinates) <= set(grid):
            raise IdentificationProtocolError(
                "target coordinates must belong to the parameter schema"
            )
        object.__setattr__(self, "target_coordinates", tuple(sorted(coordinates)))

        cases = tuple(self.cases)
        if not cases or any(not isinstance(case, Scenario) for case in cases):
            raise IdentificationProtocolError("recovery cases must contain Scenario values")
        if len({case.content_hash for case in cases}) != len(cases):
            raise IdentificationProtocolError("recovery cases must be unique")
        object.__setattr__(
            self,
            "cases",
            tuple(sorted(cases, key=lambda case: (case.id, case.content_hash))),
        )
        object.__setattr__(
            self,
            "generation_seeds",
            _seed_plan(self.generation_seeds, label="generation seeds"),
        )
        object.__setattr__(
            self,
            "calibration_seed_blocks",
            _seed_blocks(self.calibration_seed_blocks),
        )
        delta = _finite(self.acceptance_loss_delta, label="acceptance loss delta")
        if delta < 0.0:
            raise IdentificationProtocolError("acceptance loss delta must be non-negative")
        object.__setattr__(self, "acceptance_loss_delta", delta)
        fraction = _finite(
            self.min_acceptance_fraction,
            label="minimum acceptance fraction",
        )
        if not 0.0 < fraction <= 1.0:
            raise IdentificationProtocolError(
                "minimum acceptance fraction must be in (0, 1]"
            )
        object.__setattr__(self, "min_acceptance_fraction", fraction)
        object.__setattr__(
            self,
            "metric_identity",
            _freeze_mapping(self.metric_identity, label="recovery metric identity"),
        )
        object.__setattr__(
            self,
            "loss_identity",
            _freeze_mapping(self.loss_identity, label="recovery loss identity"),
        )
        if self.expected_status is None:
            status = None
        else:
            try:
                status = (
                    self.expected_status
                    if isinstance(self.expected_status, IdentificationStatus)
                    else IdentificationStatus(self.expected_status)
                )
            except (TypeError, ValueError) as error:
                raise IdentificationProtocolError(
                    "expected identification status is unsupported"
                ) from error
        object.__setattr__(self, "expected_status", status)

    @property
    def candidate_parameters(self) -> tuple[ParameterTuple, ...]:
        names = tuple(self.parameter_grid)
        return tuple(
            tuple((name, float(value)) for name, value in zip(names, values, strict=True))
            for values in product(*(self.parameter_grid[name] for name in names))
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_identity": self.model_identity,
            "true_parameters": self.true_parameters,
            "parameter_grid": tuple(
                (name, self.parameter_grid[name]) for name in self.parameter_grid
            ),
            "target_coordinates": self.target_coordinates,
            "cases": tuple(
                {"id": case.id, "content_hash": case.content_hash}
                for case in self.cases
            ),
            "generation_seeds": self.generation_seeds,
            "calibration_seed_blocks": self.calibration_seed_blocks,
            "acceptance_loss_delta": self.acceptance_loss_delta,
            "min_acceptance_fraction": self.min_acceptance_fraction,
            "metric_identity": self.metric_identity,
            "loss_identity": self.loss_identity,
            "expected_status": (
                None if self.expected_status is None else self.expected_status.value
            ),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ParameterCandidateLoss:
    parameters: ParameterTuple
    block_losses: tuple[float, ...]
    mean_loss: float
    accepted_blocks: int
    acceptance_fraction: float

    def __post_init__(self) -> None:
        try:
            parameters = _parameter_tuple(self.parameters, label="candidate parameters")
            losses = tuple(
                _finite(item, label="candidate block loss") for item in self.block_losses
            )
            mean_loss = _finite(self.mean_loss, label="candidate mean loss")
            fraction = _finite(
                self.acceptance_fraction,
                label="candidate acceptance fraction",
            )
        except IdentificationProtocolError as error:
            raise SyntheticIdentificationError(str(error)) from error
        if not losses:
            raise SyntheticIdentificationError("candidate block losses must be non-empty")
        if any(loss < 0.0 for loss in losses) or mean_loss < 0.0:
            raise SyntheticIdentificationError("candidate losses must be non-negative")
        if not isinstance(self.accepted_blocks, int) or isinstance(self.accepted_blocks, bool):
            raise SyntheticIdentificationError("accepted block count must be an integer")
        if not 0 <= self.accepted_blocks <= len(losses):
            raise SyntheticIdentificationError("accepted block count is out of range")
        if not 0.0 <= fraction <= 1.0:
            raise SyntheticIdentificationError("acceptance fraction must be in [0, 1]")
        expected_fraction = self.accepted_blocks / len(losses)
        if not math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-15):
            raise SyntheticIdentificationError(
                "acceptance fraction must match accepted block count"
            )
        if not math.isclose(mean_loss, fmean(losses), rel_tol=0.0, abs_tol=1e-15):
            raise SyntheticIdentificationError("candidate mean loss must match block losses")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "block_losses", losses)
        object.__setattr__(self, "mean_loss", mean_loss)
        object.__setattr__(self, "acceptance_fraction", fraction)

    def identity_payload(self) -> dict[str, object]:
        return {
            "parameters": self.parameters,
            "block_losses": self.block_losses,
            "mean_loss": self.mean_loss,
            "accepted_blocks": self.accepted_blocks,
            "acceptance_fraction": self.acceptance_fraction,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ParameterIdentificationFinding:
    experiment_hash: str
    true_parameters: ParameterTuple
    candidate_losses: tuple[ParameterCandidateLoss, ...]
    accepted_parameters: ParameterAcceptanceSet
    truth_retained: bool
    identifiability: IdentifiabilityReport
    status: IdentificationStatus
    parent_manifest_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_hash",
            _hash(self.experiment_hash, label="identification experiment hash"),
        )
        object.__setattr__(
            self,
            "true_parameters",
            _parameter_tuple(self.true_parameters, label="finding true parameters"),
        )
        rows = tuple(sorted(self.candidate_losses, key=lambda item: item.parameters))
        if not rows or any(not isinstance(item, ParameterCandidateLoss) for item in rows):
            raise IdentificationRecoveryError(
                "identification finding requires candidate-loss rows"
            )
        if len({row.parameters for row in rows}) != len(rows):
            raise IdentificationRecoveryError(
                "identification finding candidate rows must be unique"
            )
        object.__setattr__(self, "candidate_losses", rows)
        if not isinstance(self.accepted_parameters, ParameterAcceptanceSet):
            raise IdentificationRecoveryError(
                "identification finding requires ParameterAcceptanceSet"
            )
        if not isinstance(self.truth_retained, bool):
            raise IdentificationRecoveryError("truth-retained flag must be boolean")
        if not isinstance(self.identifiability, IdentifiabilityReport):
            raise IdentificationRecoveryError(
                "identification finding requires IdentifiabilityReport"
            )
        if not isinstance(self.status, IdentificationStatus):
            raise IdentificationRecoveryError(
                "identification finding requires IdentificationStatus"
            )
        parents = tuple(sorted({_hash(item, label="finding parent manifest hash") for item in self.parent_manifest_hashes}))
        object.__setattr__(self, "parent_manifest_hashes", parents)

    def identity_payload(self) -> dict[str, object]:
        return {
            "experiment_hash": self.experiment_hash,
            "true_parameters": self.true_parameters,
            "candidate_losses": tuple(row.identity_payload() for row in self.candidate_losses),
            "accepted_parameters_hash": self.accepted_parameters.content_hash,
            "truth_retained": self.truth_retained,
            "identifiability": {
                "accepted_parameters": self.identifiability.accepted_parameters,
                "parameters": tuple(
                    {
                        "name": item.name,
                        "values": item.values,
                        "identified": item.identified,
                    }
                    for item in self.identifiability.parameters
                ),
            },
            "status": self.status.value,
            "parent_manifest_hashes": self.parent_manifest_hashes,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def interpret_parameter_identification(
    experiment: ParameterRecoveryExperiment,
    candidate_losses: tuple[ParameterCandidateLoss, ...],
    *,
    parent_manifest_hashes: tuple[str, ...] = (),
) -> ParameterIdentificationFinding:
    if not isinstance(experiment, ParameterRecoveryExperiment):
        raise IdentificationRecoveryError(
            "parameter identification requires ParameterRecoveryExperiment"
        )
    rows = tuple(candidate_losses)
    if any(not isinstance(row, ParameterCandidateLoss) for row in rows):
        raise IdentificationRecoveryError(
            "parameter identification requires ParameterCandidateLoss rows"
        )
    expected = experiment.candidate_parameters
    if len(rows) != len(expected) or {row.parameters for row in rows} != set(expected):
        raise IdentificationRecoveryError(
            "candidate-loss table must cover the complete parameter grid exactly once"
        )
    if len({row.parameters for row in rows}) != len(rows):
        raise IdentificationRecoveryError("candidate-loss table cannot contain duplicates")
    accepted = tuple(
        row.parameters
        for row in rows
        if row.acceptance_fraction >= experiment.min_acceptance_fraction
    )
    if not accepted:
        raise IdentificationRecoveryError("synthetic identification accepted set is empty")
    truth = experiment.true_parameters
    if truth not in accepted:
        raise IdentificationRecoveryError(
            "synthetic identification did not retain its true parameters"
        )
    report = diagnose_identifiability(accepted)
    by_name = {item.name: item for item in report.parameters}
    identified = all(by_name[name].identified for name in experiment.target_coordinates)
    if identified:
        truth_map = dict(truth)
        for name in experiment.target_coordinates:
            if by_name[name].values != (truth_map[name],):
                raise IdentificationRecoveryError(
                    "identified coordinate does not equal synthetic truth"
                )
    status = (
        IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL
        if identified
        else IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL
    )
    if experiment.expected_status is not None and status is not experiment.expected_status:
        raise IdentificationRecoveryError(
            "synthetic identification result violates preregistered expectation"
        )
    parents = tuple(sorted({_hash(item, label="recovery parent manifest hash") for item in parent_manifest_hashes}))
    accepted_set = ParameterAcceptanceSet.from_parameters(
        accepted,
        source_manifest_hashes=parents,
    )
    return ParameterIdentificationFinding(
        experiment_hash=experiment.content_hash,
        true_parameters=truth,
        candidate_losses=tuple(sorted(rows, key=lambda row: row.parameters)),
        accepted_parameters=accepted_set,
        truth_retained=True,
        identifiability=report,
        status=status,
        parent_manifest_hashes=parents,
    )


@dataclass(frozen=True)
class InformationInterventionPair:
    name: str
    baseline: Scenario
    intervention: Scenario
    allowed_information_path: tuple[str, ...]
    frozen_paths: tuple[tuple[str, ...], ...]
    stratum: str
    expected_relationship: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _text(self.name, label="information intervention name"),
        )
        if not isinstance(self.baseline, Scenario) or not isinstance(
            self.intervention, Scenario
        ):
            raise IdentificationProtocolError(
                "information intervention requires Scenario values"
            )
        if set(self.baseline.payload) != set(self.intervention.payload):
            raise IdentificationProtocolError(
                "information intervention scenarios must share one payload schema"
            )
        allowed = tuple(
            _text(item, label="allowed information path segment")
            for item in self.allowed_information_path
        )
        if len(allowed) != 1 or allowed[0] not in self.baseline.payload:
            raise IdentificationProtocolError(
                "V1 information interventions allow exactly one top-level payload path"
            )
        object.__setattr__(self, "allowed_information_path", allowed)
        frozen = tuple(
            tuple(_text(segment, label="frozen path segment") for segment in path)
            for path in self.frozen_paths
        )
        if any(len(path) != 1 for path in frozen):
            raise IdentificationProtocolError(
                "V1 information intervention frozen paths must be top-level"
            )
        if len(set(frozen)) != len(frozen):
            raise IdentificationProtocolError("frozen information paths must be unique")
        expected_frozen = tuple(
            (key,)
            for key in sorted(self.baseline.payload)
            if key != allowed[0]
        )
        if tuple(sorted(frozen)) != expected_frozen:
            raise IdentificationProtocolError(
                "frozen paths must cover every payload field except the allowed information path"
            )
        object.__setattr__(self, "frozen_paths", expected_frozen)
        object.__setattr__(self, "stratum", _text(self.stratum, label="intervention stratum"))
        relationship = self.expected_relationship
        if relationship is not None:
            relationship = _text(
                relationship,
                label="expected intervention relationship",
            )
            if relationship not in _ALLOWED_RELATIONSHIPS:
                raise IdentificationProtocolError(
                    "expected intervention relationship is unsupported"
                )
        object.__setattr__(self, "expected_relationship", relationship)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline": {
                "id": self.baseline.id,
                "content_hash": self.baseline.content_hash,
            },
            "intervention": {
                "id": self.intervention.id,
                "content_hash": self.intervention.content_hash,
            },
            "allowed_information_path": self.allowed_information_path,
            "frozen_paths": self.frozen_paths,
            "stratum": self.stratum,
            "expected_relationship": self.expected_relationship,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class SyntheticIdentificationProtocol:
    name: str
    version: str
    claim_scope: str
    fixture_hash: str
    adapter_identities: tuple[tuple[str, Mapping[str, object]], ...]
    recovery_experiments: tuple[ParameterRecoveryExperiment, ...]
    intervention_pairs: tuple[InformationInterventionPair, ...]
    train_partition_hash: str
    selection_partition_hash: str
    final_partition_hash: str
    train_case_hashes: tuple[str, ...]
    selection_case_hashes: tuple[str, ...]
    final_case_hashes: tuple[str, ...]
    target_spec_hash: str
    metric_identity: Mapping[str, object]
    brier_loss_identity: Mapping[str, object]
    log_loss_identity: Mapping[str, object]
    brier_training_seeds: tuple[int, ...]
    brier_selection_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    equivalence_tolerance: float = _EQUIVALENCE_TOLERANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="protocol name"))
        object.__setattr__(self, "version", _text(self.version, label="protocol version"))
        if self.claim_scope != _ALLOWED_CLAIM_SCOPE:
            raise IdentificationProtocolError(
                "synthetic identification claim scope must be synthetic_protocol_only"
            )
        object.__setattr__(
            self,
            "fixture_hash",
            _hash(self.fixture_hash, label="protocol fixture hash"),
        )

        adapters: list[tuple[str, Mapping[str, object]]] = []
        seen_adapters: set[str] = set()
        for raw_name, raw_identity in self.adapter_identities:
            name = _text(raw_name, label="adapter name")
            if name in seen_adapters:
                raise IdentificationProtocolError("adapter names must be unique")
            seen_adapters.add(name)
            adapters.append(
                (name, _freeze_mapping(raw_identity, label=f"adapter identity {name!r}"))
            )
        if not adapters:
            raise IdentificationProtocolError("synthetic protocol requires adapters")
        object.__setattr__(self, "adapter_identities", tuple(sorted(adapters)))

        experiments = tuple(self.recovery_experiments)
        if not experiments or any(
            not isinstance(item, ParameterRecoveryExperiment) for item in experiments
        ):
            raise IdentificationProtocolError(
                "synthetic protocol requires recovery experiments"
            )
        experiment_names = tuple(item.name for item in experiments)
        if len(set(experiment_names)) != len(experiment_names):
            raise IdentificationProtocolError("recovery experiment names must be unique")
        object.__setattr__(
            self,
            "recovery_experiments",
            tuple(sorted(experiments, key=lambda item: item.name)),
        )

        pairs = tuple(self.intervention_pairs)
        if not pairs or any(not isinstance(item, InformationInterventionPair) for item in pairs):
            raise IdentificationProtocolError(
                "synthetic protocol requires information intervention pairs"
            )
        pair_names = tuple(item.name for item in pairs)
        if len(set(pair_names)) != len(pair_names):
            raise IdentificationProtocolError("intervention pair names must be unique")
        object.__setattr__(
            self,
            "intervention_pairs",
            tuple(sorted(pairs, key=lambda item: item.name)),
        )

        for attribute, label in (
            ("train_partition_hash", "train partition hash"),
            ("selection_partition_hash", "selection partition hash"),
            ("final_partition_hash", "final partition hash"),
            ("target_spec_hash", "target spec hash"),
        ):
            object.__setattr__(
                self,
                attribute,
                _hash(getattr(self, attribute), label=label),
            )
        train_cases = _hash_tuple(self.train_case_hashes, label="train case hash")
        selection_cases = _hash_tuple(
            self.selection_case_hashes,
            label="selection case hash",
        )
        final_cases = _hash_tuple(self.final_case_hashes, label="final case hash")
        if (
            set(train_cases) & set(selection_cases)
            or set(train_cases) & set(final_cases)
            or set(selection_cases) & set(final_cases)
        ):
            raise IdentificationProtocolError(
                "synthetic protocol case partitions must be disjoint"
            )
        object.__setattr__(self, "train_case_hashes", train_cases)
        object.__setattr__(self, "selection_case_hashes", selection_cases)
        object.__setattr__(self, "final_case_hashes", final_cases)
        all_case_hashes = set(train_cases) | set(selection_cases) | set(final_cases)
        for experiment in self.recovery_experiments:
            if not {case.content_hash for case in experiment.cases} <= all_case_hashes:
                raise IdentificationProtocolError(
                    "recovery experiment contains a scenario outside the frozen dataset"
                )
        for pair in self.intervention_pairs:
            if pair.baseline.content_hash not in set(final_cases) or pair.intervention.content_hash not in set(final_cases):
                raise IdentificationProtocolError(
                    "information intervention pairs must use frozen final-test scenarios"
                )

        object.__setattr__(
            self,
            "metric_identity",
            _freeze_mapping(self.metric_identity, label="protocol metric identity"),
        )
        brier_identity = _freeze_mapping(
            self.brier_loss_identity,
            label="protocol Brier loss identity",
        )
        log_identity = _freeze_mapping(
            self.log_loss_identity,
            label="protocol Log loss identity",
        )
        if brier_identity.get("name") != "categorical_brier":
            raise IdentificationProtocolError(
                "synthetic protocol Brier loss must be categorical_brier"
            )
        if log_identity.get("name") != "categorical_log":
            raise IdentificationProtocolError(
                "synthetic protocol Log loss must be categorical_log"
            )
        object.__setattr__(self, "brier_loss_identity", brier_identity)
        object.__setattr__(self, "log_loss_identity", log_identity)
        object.__setattr__(
            self,
            "brier_training_seeds",
            _seed_plan(self.brier_training_seeds, label="Brier training seeds"),
        )
        object.__setattr__(
            self,
            "brier_selection_seeds",
            _seed_plan(self.brier_selection_seeds, label="Brier selection seeds"),
        )
        object.__setattr__(
            self,
            "final_seeds",
            _seed_plan(self.final_seeds, label="final seeds"),
        )
        tolerance = _finite(
            self.equivalence_tolerance,
            label="equivalence tolerance",
        )
        if tolerance != _EQUIVALENCE_TOLERANCE:
            raise IdentificationProtocolError(
                "equivalence tolerance must be exactly 1e-12"
            )
        object.__setattr__(self, "equivalence_tolerance", tolerance)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        claim_scope: str,
        dataset: ObservationDataset,
        adapter_sources: tuple[object, ...],
        recovery_experiments: tuple[ParameterRecoveryExperiment, ...],
        intervention_pairs: tuple[InformationInterventionPair, ...],
        target_spec: CategoricalTargetSpec,
        extractor: object,
        brier_loss: MetricLoss,
        log_loss: MetricLoss,
        brier_training_seeds: tuple[int, ...],
        brier_selection_seeds: tuple[int, ...],
        final_seeds: tuple[int, ...],
        equivalence_tolerance: float = _EQUIVALENCE_TOLERANCE,
    ) -> "SyntheticIdentificationProtocol":
        if not isinstance(dataset, ObservationDataset):
            raise IdentificationProtocolError(
                "synthetic protocol dataset must be ObservationDataset"
            )
        if not isinstance(target_spec, CategoricalTargetSpec):
            raise IdentificationProtocolError(
                "synthetic protocol target spec must be CategoricalTargetSpec"
            )
        sources = tuple(adapter_sources)
        if not sources:
            raise IdentificationProtocolError("synthetic protocol requires adapter sources")
        identities: list[tuple[str, Mapping[str, object]]] = []
        for source in sources:
            source_name = _text(getattr(source, "name", None), label="adapter source name")
            identities.append((source_name, component_identity(source)))

        train = dataset.partition(ObservationPartitionRole.TRAIN)
        selection = dataset.partition(ObservationPartitionRole.SELECTION_VALIDATION)
        final = dataset.partition(ObservationPartitionRole.FINAL_TEST)
        if not train.records or not selection.records or not final.records:
            raise IdentificationProtocolError(
                "synthetic identification requires record-oriented dataset partitions"
            )
        return cls(
            name=name,
            version=version,
            claim_scope=claim_scope,
            fixture_hash=dataset.content_hash,
            adapter_identities=tuple(identities),
            recovery_experiments=tuple(recovery_experiments),
            intervention_pairs=tuple(intervention_pairs),
            train_partition_hash=train.content_hash,
            selection_partition_hash=selection.content_hash,
            final_partition_hash=final.content_hash,
            train_case_hashes=tuple(record.scenario.content_hash for record in train.records),
            selection_case_hashes=tuple(
                record.scenario.content_hash for record in selection.records
            ),
            final_case_hashes=tuple(record.scenario.content_hash for record in final.records),
            target_spec_hash=target_spec.content_hash,
            metric_identity=callable_identity(extractor),
            brier_loss_identity=metric_loss_identity(brier_loss),
            log_loss_identity=metric_loss_identity(log_loss),
            brier_training_seeds=brier_training_seeds,
            brier_selection_seeds=brier_selection_seeds,
            final_seeds=final_seeds,
            equivalence_tolerance=equivalence_tolerance,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "claim_scope": self.claim_scope,
            "fixture_hash": self.fixture_hash,
            "adapter_identities": self.adapter_identities,
            "recovery_experiments": tuple(
                item.identity_payload() for item in self.recovery_experiments
            ),
            "intervention_pairs": tuple(
                item.identity_payload() for item in self.intervention_pairs
            ),
            "train_partition_hash": self.train_partition_hash,
            "selection_partition_hash": self.selection_partition_hash,
            "final_partition_hash": self.final_partition_hash,
            "train_case_hashes": self.train_case_hashes,
            "selection_case_hashes": self.selection_case_hashes,
            "final_case_hashes": self.final_case_hashes,
            "target_spec_hash": self.target_spec_hash,
            "metric_identity": self.metric_identity,
            "brier_loss_identity": self.brier_loss_identity,
            "log_loss_identity": self.log_loss_identity,
            "brier_training_seeds": self.brier_training_seeds,
            "brier_selection_seeds": self.brier_selection_seeds,
            "final_seeds": self.final_seeds,
            "equivalence_tolerance": self.equivalence_tolerance,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def validate_sibling_final_protocols(
    brier: PreregisteredEvaluationProtocol,
    log: PreregisteredEvaluationProtocol,
) -> None:
    if not isinstance(brier, PreregisteredEvaluationProtocol) or not isinstance(
        log, PreregisteredEvaluationProtocol
    ):
        raise IdentificationComparisonError(
            "sibling final protocols must be PreregisteredEvaluationProtocol values"
        )
    if brier.loss_identity.get("name") != "categorical_brier":
        raise IdentificationComparisonError(
            "Brier sibling must use categorical_brier loss"
        )
    if log.loss_identity.get("name") != "categorical_log":
        raise IdentificationComparisonError(
            "Log sibling must use categorical_log loss"
        )
    if brier.loss_identity == log.loss_identity:
        raise IdentificationComparisonError("sibling final losses must be distinct")
    for attribute in (
        "version",
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
        "candidates",
    ):
        if getattr(brier, attribute) != getattr(log, attribute):
            raise IdentificationComparisonError(
                f"sibling final protocols drifted at {attribute}"
            )


def evaluate_parameter_recovery_experiment(*args, **kwargs):
    raise NotImplementedError("parameter recovery execution is implemented in Task 4")


def certify_observational_equivalence(*args, **kwargs):
    raise NotImplementedError("observational equivalence is implemented in Task 4")


def certify_information_intervention(*args, **kwargs):
    raise NotImplementedError("information intervention certification is implemented in Task 5")


def evaluate_information_intervention(*args, **kwargs):
    raise NotImplementedError("information intervention evaluation is implemented in Task 5")


def score_model_comparison_by_stratum(*args, **kwargs):
    raise NotImplementedError("stratified final scoring is implemented in Task 6")


def build_synthetic_identification_report(*args, **kwargs):
    raise NotImplementedError("aggregate identification reporting is implemented in Task 7")


__all__ = [
    "IdentificationComparisonError",
    "IdentificationProtocolError",
    "IdentificationRecoveryError",
    "IdentificationReportError",
    "IdentificationStatus",
    "InformationInterventionPair",
    "InterventionCertificationError",
    "ParameterCandidateLoss",
    "ParameterIdentificationFinding",
    "ParameterRecoveryExperiment",
    "SyntheticIdentificationError",
    "SyntheticIdentificationProtocol",
    "build_synthetic_identification_report",
    "certify_information_intervention",
    "certify_observational_equivalence",
    "evaluate_information_intervention",
    "evaluate_parameter_recovery_experiment",
    "interpret_parameter_identification",
    "score_model_comparison_by_stratum",
    "validate_sibling_final_protocols",
]
