from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
import math
import re
from statistics import fmean
from types import MappingProxyType

from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.losses import MetricLoss, metric_loss_identity
from narrative_dynamics.manifest import (
    callable_identity,
    component_identity,
    required_manifest_hash,
)
from narrative_dynamics.metrics import aggregate_metrics
from narrative_dynamics.observations.dataset import (
    ObservationDataset,
    ObservationPartitionRole,
)
from narrative_dynamics.observations.preregistration import (
    PreregisteredEvaluationProtocol,
)
from narrative_dynamics.observations.targets import CategoricalTargetSpec
from narrative_dynamics.simulation import SimulationRunner
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


PolicyTuple = tuple[tuple[str, float], ...]
PolicyComparisonRow = tuple[str, PolicyTuple, PolicyTuple]


def _policy_tuple(
    value: Mapping[str, float] | Iterable[tuple[str, float]],
    *,
    action_keys: tuple[str, ...],
    label: str,
) -> PolicyTuple:
    items = value.items() if isinstance(value, Mapping) else value
    probabilities: dict[str, float] = {}
    try:
        for raw_key, raw_probability in items:
            key = _text(raw_key, label=f"{label} action key")
            if key in probabilities:
                raise IdentificationRecoveryError(
                    f"{label} action keys must be unique"
                )
            probability = _finite(
                raw_probability,
                label=f"{label} probability for {key!r}",
            )
            if not 0.0 <= probability <= 1.0:
                raise IdentificationRecoveryError(
                    f"{label} probabilities must be in [0, 1]"
                )
            probabilities[key] = probability
    except IdentificationProtocolError as error:
        raise IdentificationRecoveryError(str(error)) from error
    if set(probabilities) != set(action_keys):
        raise IdentificationRecoveryError(
            f"{label} must cover the declared observable action keys exactly"
        )
    if not math.isclose(
        math.fsum(probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=_EQUIVALENCE_TOLERANCE,
    ):
        raise IdentificationRecoveryError(f"{label} must sum to one")
    return tuple((key, probabilities[key]) for key in action_keys)


@dataclass(frozen=True)
class ObservationalEquivalenceFinding:
    name: str
    left_model_identity: Mapping[str, object]
    right_model_identity: Mapping[str, object]
    case_hashes: tuple[str, ...]
    action_keys: tuple[str, ...]
    policies: tuple[PolicyComparisonRow, ...]
    max_abs_policy_delta: float
    tolerance: float
    equivalent: bool
    parent_manifest_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            name = _text(self.name, label="observational equivalence name")
            left_identity = _freeze_mapping(
                self.left_model_identity,
                label="left observational equivalence model identity",
            )
            right_identity = _freeze_mapping(
                self.right_model_identity,
                label="right observational equivalence model identity",
            )
            raw_case_hashes = tuple(self.case_hashes)
            case_hashes = tuple(
                sorted(
                    _hash(item, label="observational equivalence case hash")
                    for item in raw_case_hashes
                )
            )
            if not case_hashes:
                raise IdentificationRecoveryError(
                    "observational equivalence requires at least one case"
                )
            if len(set(case_hashes)) != len(case_hashes):
                raise IdentificationRecoveryError(
                    "observational equivalence case hashes must be unique"
                )
            raw_action_keys = tuple(self.action_keys)
            action_keys = tuple(
                sorted(
                    _text(item, label="observational equivalence action key")
                    for item in raw_action_keys
                )
            )
            if not action_keys:
                raise IdentificationRecoveryError(
                    "observational equivalence requires observable action keys"
                )
            if len(set(action_keys)) != len(action_keys):
                raise IdentificationRecoveryError(
                    "observational equivalence action keys must be unique"
                )
            tolerance = _finite(
                self.tolerance,
                label="observational equivalence tolerance",
            )
            maximum = _finite(
                self.max_abs_policy_delta,
                label="observational equivalence maximum policy delta",
            )
        except IdentificationProtocolError as error:
            raise IdentificationRecoveryError(str(error)) from error
        if tolerance < 0.0 or maximum < 0.0:
            raise IdentificationRecoveryError(
                "observational equivalence tolerance and delta must be non-negative"
            )
        if not isinstance(self.equivalent, bool):
            raise IdentificationRecoveryError(
                "observational equivalence flag must be boolean"
            )
        if not isinstance(self.policies, tuple):
            raise IdentificationRecoveryError(
                "observational equivalence policies must be a tuple"
            )

        rows: list[PolicyComparisonRow] = []
        seen_cases: set[str] = set()
        for raw_row in self.policies:
            if not isinstance(raw_row, tuple) or len(raw_row) != 3:
                raise IdentificationRecoveryError(
                    "observational equivalence policy rows must be triples"
                )
            raw_case_hash, raw_left, raw_right = raw_row
            try:
                case_hash = _hash(
                    raw_case_hash,
                    label="observational equivalence policy case hash",
                )
            except IdentificationProtocolError as error:
                raise IdentificationRecoveryError(str(error)) from error
            if case_hash in seen_cases:
                raise IdentificationRecoveryError(
                    "observational equivalence policies must contain one row per case"
                )
            seen_cases.add(case_hash)
            left_policy = _policy_tuple(
                raw_left,
                action_keys=action_keys,
                label="left observational policy",
            )
            right_policy = _policy_tuple(
                raw_right,
                action_keys=action_keys,
                label="right observational policy",
            )
            rows.append((case_hash, left_policy, right_policy))
        if seen_cases != set(case_hashes):
            raise IdentificationRecoveryError(
                "observational equivalence policies must cover the frozen cases exactly"
            )
        rows_tuple = tuple(sorted(rows, key=lambda row: row[0]))
        computed_maximum = max(
            abs(dict(left)[key] - dict(right)[key])
            for _case_hash, left, right in rows_tuple
            for key in action_keys
        )
        if not math.isclose(
            maximum,
            computed_maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise IdentificationRecoveryError(
                "observational equivalence maximum delta must match complete policies"
            )
        if self.equivalent is not (computed_maximum <= tolerance):
            raise IdentificationRecoveryError(
                "observational equivalence flag must match complete-policy delta"
            )
        try:
            parents = tuple(
                sorted(
                    {
                        _hash(
                            item,
                            label="observational equivalence parent manifest hash",
                        )
                        for item in self.parent_manifest_hashes
                    }
                )
            )
        except IdentificationProtocolError as error:
            raise IdentificationRecoveryError(str(error)) from error
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "left_model_identity", left_identity)
        object.__setattr__(self, "right_model_identity", right_identity)
        object.__setattr__(self, "case_hashes", case_hashes)
        object.__setattr__(self, "action_keys", action_keys)
        object.__setattr__(self, "policies", rows_tuple)
        object.__setattr__(self, "max_abs_policy_delta", maximum)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "parent_manifest_hashes", parents)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "left_model_identity": self.left_model_identity,
            "right_model_identity": self.right_model_identity,
            "case_hashes": self.case_hashes,
            "action_keys": self.action_keys,
            "policies": tuple(
                {
                    "case_hash": case_hash,
                    "left": left,
                    "right": right,
                }
                for case_hash, left, right in self.policies
            ),
            "max_abs_policy_delta": self.max_abs_policy_delta,
            "tolerance": self.tolerance,
            "equivalent": self.equivalent,
            "parent_manifest_hashes": self.parent_manifest_hashes,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _recovery_identity_preflight(
    *,
    model: object,
    experiment: ParameterRecoveryExperiment,
    extractor: object,
    loss: MetricLoss,
) -> None:
    try:
        model_identity = component_identity(model)
        metric_identity = callable_identity(extractor)
        actual_loss_identity = metric_loss_identity(loss)
    except (TypeError, ValueError) as error:
        raise IdentificationRecoveryError(
            "parameter recovery identity preflight failed"
        ) from error
    if model_identity != experiment.model_identity:
        raise IdentificationRecoveryError(
            "parameter recovery model identity drifted from its declaration"
        )
    if metric_identity != experiment.metric_identity:
        raise IdentificationRecoveryError(
            "parameter recovery metric identity drifted from its declaration"
        )
    if actual_loss_identity != experiment.loss_identity:
        raise IdentificationRecoveryError(
            "parameter recovery loss identity drifted from its declaration"
        )


def evaluate_parameter_recovery_experiment(
    *,
    runner: SimulationRunner,
    model: object,
    experiment: ParameterRecoveryExperiment,
    extractor: object,
    loss: MetricLoss,
) -> ParameterIdentificationFinding:
    if not isinstance(runner, SimulationRunner):
        raise IdentificationRecoveryError(
            "parameter recovery requires SimulationRunner"
        )
    if not isinstance(experiment, ParameterRecoveryExperiment):
        raise IdentificationRecoveryError(
            "parameter recovery requires ParameterRecoveryExperiment"
        )
    _recovery_identity_preflight(
        model=model,
        experiment=experiment,
        extractor=extractor,
        loss=loss,
    )

    try:
        truth_parameters = dict(experiment.true_parameters)
        targets: dict[str, Mapping[str, float]] = {}
        parent_hashes: set[str] = set()
        for case in experiment.cases:
            truth_traces = runner.run_batch(
                model,
                case,
                truth_parameters,
                seeds=experiment.generation_seeds,
            )
            targets[case.content_hash] = aggregate_metrics(
                truth_traces,
                extractor,
            )
            parent_hashes.update(
                required_manifest_hash(
                    trace,
                    label="synthetic recovery generation trace",
                )
                for trace in truth_traces
            )

        candidates = experiment.candidate_parameters
        expected_candidates = set(candidates)
        block_losses: dict[ParameterTuple, list[float]] = {
            candidate: [] for candidate in candidates
        }
        accepted_blocks: dict[ParameterTuple, int] = {
            candidate: 0 for candidate in candidates
        }

        for seed_block in experiment.calibration_seed_blocks:
            case_losses: list[Mapping[ParameterTuple, float]] = []
            for case in experiment.cases:
                calibration = calibrate_grid(
                    runner=runner,
                    model=model,
                    scenario=case,
                    parameter_grid=experiment.parameter_grid,
                    seeds=seed_block,
                    extractor=extractor,
                    target=targets[case.content_hash],
                    loss=loss,
                )
                actual_candidates = tuple(
                    candidate.parameters for candidate in calibration.ranking
                )
                if (
                    len(actual_candidates) != len(candidates)
                    or set(actual_candidates) != expected_candidates
                ):
                    raise IdentificationRecoveryError(
                        "parameter recovery calibration changed the frozen candidate grid"
                    )
                parent_hashes.add(
                    required_manifest_hash(
                        calibration,
                        label="synthetic recovery calibration",
                    )
                )
                for candidate in calibration.ranking:
                    parent_hashes.update(candidate.run_manifest_hashes)
                case_losses.append(
                    {
                        candidate.parameters: candidate.loss
                        for candidate in calibration.ranking
                    }
                )

            aggregate_block = {
                candidate: fmean(
                    case_loss[candidate] for case_loss in case_losses
                )
                for candidate in candidates
            }
            if any(
                not math.isfinite(value) or value < 0.0
                for value in aggregate_block.values()
            ):
                raise IdentificationRecoveryError(
                    "parameter recovery candidate loss must be finite and non-negative"
                )
            best_loss = min(aggregate_block.values())
            for candidate in candidates:
                candidate_loss = aggregate_block[candidate]
                block_losses[candidate].append(candidate_loss)
                if candidate_loss <= best_loss + experiment.acceptance_loss_delta:
                    accepted_blocks[candidate] += 1

        block_count = len(experiment.calibration_seed_blocks)
        rows = tuple(
            ParameterCandidateLoss(
                parameters=candidate,
                block_losses=tuple(block_losses[candidate]),
                mean_loss=fmean(block_losses[candidate]),
                accepted_blocks=accepted_blocks[candidate],
                acceptance_fraction=accepted_blocks[candidate] / block_count,
            )
            for candidate in candidates
        )
        return interpret_parameter_identification(
            experiment,
            rows,
            parent_manifest_hashes=tuple(sorted(parent_hashes)),
        )
    except SyntheticIdentificationError:
        raise
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise IdentificationRecoveryError(
            "parameter recovery execution failed"
        ) from error


def certify_observational_equivalence(
    *,
    name: str,
    runner: SimulationRunner,
    left_model: object,
    right_model: object,
    left_parameters: Mapping[str, float],
    right_parameters: Mapping[str, float],
    cases: tuple[Scenario, ...],
    seeds: tuple[int, ...],
    extractor: object,
    action_keys: tuple[str, ...],
    tolerance: float = _EQUIVALENCE_TOLERANCE,
) -> ObservationalEquivalenceFinding:
    if not isinstance(runner, SimulationRunner):
        raise IdentificationRecoveryError(
            "observational equivalence requires SimulationRunner"
        )
    try:
        finding_name = _text(name, label="observational equivalence name")
        case_tuple = tuple(cases)
        if not case_tuple or any(
            not isinstance(case, Scenario) for case in case_tuple
        ):
            raise IdentificationRecoveryError(
                "observational equivalence cases must contain Scenario values"
            )
        if len({case.content_hash for case in case_tuple}) != len(case_tuple):
            raise IdentificationRecoveryError(
                "observational equivalence cases must be unique"
            )
        case_tuple = tuple(
            sorted(case_tuple, key=lambda case: (case.id, case.content_hash))
        )
        seed_tuple = _seed_plan(seeds, label="observational equivalence seeds")
        keys = tuple(
            sorted(
                _text(item, label="observational equivalence action key")
                for item in action_keys
            )
        )
        if not keys or len(set(keys)) != len(keys):
            raise IdentificationRecoveryError(
                "observational equivalence action keys must be non-empty and unique"
            )
        tolerance_value = _finite(
            tolerance,
            label="observational equivalence tolerance",
        )
        if tolerance_value < 0.0:
            raise IdentificationRecoveryError(
                "observational equivalence tolerance must be non-negative"
            )
        left_parameter_tuple = _parameter_tuple(
            left_parameters,
            label="left observational equivalence parameters",
        )
        right_parameter_tuple = _parameter_tuple(
            right_parameters,
            label="right observational equivalence parameters",
        )
    except IdentificationProtocolError as error:
        raise IdentificationRecoveryError(str(error)) from error

    try:
        rows: list[PolicyComparisonRow] = []
        parent_hashes: set[str] = set()
        maximum = 0.0
        for case in case_tuple:
            left_traces = runner.run_batch(
                left_model,
                case,
                dict(left_parameter_tuple),
                seeds=seed_tuple,
            )
            right_traces = runner.run_batch(
                right_model,
                case,
                dict(right_parameter_tuple),
                seeds=seed_tuple,
            )
            left_policy = _policy_tuple(
                aggregate_metrics(left_traces, extractor),
                action_keys=keys,
                label="left observational policy",
            )
            right_policy = _policy_tuple(
                aggregate_metrics(right_traces, extractor),
                action_keys=keys,
                label="right observational policy",
            )
            rows.append((case.content_hash, left_policy, right_policy))
            left_map = dict(left_policy)
            right_map = dict(right_policy)
            maximum = max(
                maximum,
                max(abs(left_map[key] - right_map[key]) for key in keys),
            )
            parent_hashes.update(
                required_manifest_hash(
                    trace,
                    label="left observational equivalence trace",
                )
                for trace in left_traces
            )
            parent_hashes.update(
                required_manifest_hash(
                    trace,
                    label="right observational equivalence trace",
                )
                for trace in right_traces
            )

        return ObservationalEquivalenceFinding(
            name=finding_name,
            left_model_identity=component_identity(left_model),
            right_model_identity=component_identity(right_model),
            case_hashes=tuple(case.content_hash for case in case_tuple),
            action_keys=keys,
            policies=tuple(rows),
            max_abs_policy_delta=maximum,
            tolerance=tolerance_value,
            equivalent=maximum <= tolerance_value,
            parent_manifest_hashes=tuple(sorted(parent_hashes)),
        )
    except SyntheticIdentificationError:
        raise
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise IdentificationRecoveryError(
            "observational equivalence execution failed"
        ) from error


ScenarioDiff = tuple[tuple[str, ...], object, object]
FamilyPolicyRow = tuple[str, PolicyTuple, PolicyTuple]
FamilyDeltaRow = tuple[str, float]
CrossFamilyContrastRow = tuple[str, str, float, float]


def _scenario_payload_diff(
    left: object,
    right: object,
    prefix: tuple[str, ...] = (),
) -> tuple[ScenarioDiff, ...]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = tuple(sorted(set(left) | set(right)))
        diffs: list[ScenarioDiff] = []
        missing = object()
        for key in keys:
            if not isinstance(key, str) or not key:
                raise InterventionCertificationError(
                    "information intervention payload keys must be non-empty strings"
                )
            left_value = left.get(key, missing)
            right_value = right.get(key, missing)
            if left_value is missing or right_value is missing:
                raise InterventionCertificationError(
                    "information intervention payload schemas must match recursively"
                )
            diffs.extend(
                _scenario_payload_diff(
                    left_value,
                    right_value,
                    prefix + (key,),
                )
            )
        return tuple(diffs)
    if type(left) is type(right) and left == right:
        return ()
    return ((prefix, left, right),)


def certify_information_intervention(
    pair: InformationInterventionPair,
) -> ScenarioDiff:
    if not isinstance(pair, InformationInterventionPair):
        raise InterventionCertificationError(
            "information intervention certification requires InformationInterventionPair"
        )
    diffs = _scenario_payload_diff(
        pair.baseline.payload,
        pair.intervention.payload,
    )
    if len(diffs) != 1:
        raise InterventionCertificationError(
            "information intervention must change exactly one payload leaf"
        )
    path, before, after = diffs[0]
    if path != pair.allowed_information_path:
        raise InterventionCertificationError(
            "information intervention changed an undeclared field"
        )
    if path in pair.frozen_paths:
        raise InterventionCertificationError(
            "information intervention changed a frozen field"
        )
    return path, before, after


def _intervention_policy(
    value: Mapping[str, float] | Iterable[tuple[str, float]],
    *,
    label: str,
) -> PolicyTuple:
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    raw_keys = tuple(item[0] for item in items)
    try:
        keys = tuple(sorted(_text(key, label=f"{label} action key") for key in raw_keys))
    except IdentificationProtocolError as error:
        raise InterventionCertificationError(str(error)) from error
    if not keys or len(set(keys)) != len(keys):
        raise InterventionCertificationError(
            f"{label} action keys must be non-empty and unique"
        )
    try:
        return _policy_tuple(items, action_keys=keys, label=label)
    except IdentificationRecoveryError as error:
        raise InterventionCertificationError(str(error)) from error


@dataclass(frozen=True)
class InformationInterventionFinding:
    name: str
    pair_hash: str
    certified_diff: ScenarioDiff
    family_policies: tuple[FamilyPolicyRow, ...]
    within_family_max_deltas: tuple[FamilyDeltaRow, ...]
    cross_family_contrasts: tuple[CrossFamilyContrastRow, ...]
    tolerance: float
    discriminating: bool
    parent_manifest_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            name = _text(self.name, label="information intervention finding name")
            pair_hash = _hash(self.pair_hash, label="information intervention pair hash")
            tolerance = _finite(
                self.tolerance,
                label="information intervention tolerance",
            )
        except IdentificationProtocolError as error:
            raise InterventionCertificationError(str(error)) from error
        if tolerance != _EQUIVALENCE_TOLERANCE:
            raise InterventionCertificationError(
                "information intervention tolerance must be exactly 1e-12"
            )
        if not isinstance(self.discriminating, bool):
            raise InterventionCertificationError(
                "information intervention discriminating flag must be boolean"
            )
        diff = self.certified_diff
        if not isinstance(diff, tuple) or len(diff) != 3:
            raise InterventionCertificationError(
                "certified information intervention diff must be one leaf triple"
            )
        raw_path, before, after = diff
        if not isinstance(raw_path, tuple):
            raise InterventionCertificationError(
                "certified information intervention path must be a tuple"
            )
        try:
            path = tuple(
                _text(item, label="certified information path segment")
                for item in raw_path
            )
            frozen_before = _freeze(before, label="certified information before value")
            frozen_after = _freeze(after, label="certified information after value")
        except IdentificationProtocolError as error:
            raise InterventionCertificationError(str(error)) from error
        if not path:
            raise InterventionCertificationError(
                "certified information intervention path must be non-empty"
            )
        if type(frozen_before) is type(frozen_after) and frozen_before == frozen_after:
            raise InterventionCertificationError(
                "certified information intervention must change its leaf value"
            )

        rows: list[FamilyPolicyRow] = []
        seen_families: set[str] = set()
        action_keys: tuple[str, ...] | None = None
        for raw_row in self.family_policies:
            if not isinstance(raw_row, tuple) or len(raw_row) != 3:
                raise InterventionCertificationError(
                    "information intervention family policies must be triples"
                )
            raw_family, raw_baseline, raw_intervention = raw_row
            try:
                family = _text(
                    raw_family,
                    label="information intervention family name",
                )
            except IdentificationProtocolError as error:
                raise InterventionCertificationError(str(error)) from error
            if family in seen_families:
                raise InterventionCertificationError(
                    "information intervention family names must be unique"
                )
            seen_families.add(family)
            baseline_policy = _intervention_policy(
                raw_baseline,
                label=f"{family} baseline policy",
            )
            intervention_policy = _intervention_policy(
                raw_intervention,
                label=f"{family} intervention policy",
            )
            baseline_keys = tuple(key for key, _ in baseline_policy)
            intervention_keys = tuple(key for key, _ in intervention_policy)
            if baseline_keys != intervention_keys:
                raise InterventionCertificationError(
                    "information intervention policies must share one action schema"
                )
            if action_keys is None:
                action_keys = baseline_keys
            elif baseline_keys != action_keys:
                raise InterventionCertificationError(
                    "all intervention families must share one action schema"
                )
            rows.append((family, baseline_policy, intervention_policy))
        if len(rows) < 2:
            raise InterventionCertificationError(
                "information intervention finding requires at least two families"
            )
        rows_tuple = tuple(sorted(rows, key=lambda row: row[0]))
        assert action_keys is not None

        computed_deltas = tuple(
            (
                family,
                max(
                    abs(dict(baseline)[key] - dict(intervention)[key])
                    for key in action_keys
                ),
            )
            for family, baseline, intervention in rows_tuple
        )
        supplied_deltas: list[FamilyDeltaRow] = []
        try:
            raw_deltas = tuple(self.within_family_max_deltas)
            for raw in raw_deltas:
                if not isinstance(raw, tuple) or len(raw) != 2:
                    raise InterventionCertificationError(
                        "within-family intervention deltas must be pairs"
                    )
                raw_family, raw_delta = raw
                family = _text(
                    raw_family,
                    label="within-family intervention name",
                )
                delta = _finite(
                    raw_delta,
                    label=f"within-family intervention delta for {family}",
                )
                if delta < 0.0:
                    raise InterventionCertificationError(
                        "within-family intervention deltas must be non-negative"
                    )
                supplied_deltas.append((family, delta))
        except IdentificationProtocolError as error:
            raise InterventionCertificationError(str(error)) from error
        supplied_deltas_tuple = tuple(sorted(supplied_deltas))
        if tuple(family for family, _ in supplied_deltas_tuple) != tuple(
            family for family, _ in computed_deltas
        ):
            raise InterventionCertificationError(
                "within-family intervention deltas must cover every family exactly once"
            )
        for (family, supplied), (_computed_family, computed) in zip(
            supplied_deltas_tuple,
            computed_deltas,
            strict=True,
        ):
            if not math.isclose(supplied, computed, rel_tol=0.0, abs_tol=1e-15):
                raise InterventionCertificationError(
                    f"within-family intervention delta for {family} must match policies"
                )

        computed_contrasts: list[CrossFamilyContrastRow] = []
        for index, (left_family, left_baseline, left_intervention) in enumerate(rows_tuple):
            for right_family, right_baseline, right_intervention in rows_tuple[index + 1 :]:
                computed_contrasts.append(
                    (
                        left_family,
                        right_family,
                        max(
                            abs(dict(left_baseline)[key] - dict(right_baseline)[key])
                            for key in action_keys
                        ),
                        max(
                            abs(
                                dict(left_intervention)[key]
                                - dict(right_intervention)[key]
                            )
                            for key in action_keys
                        ),
                    )
                )
        computed_contrasts_tuple = tuple(computed_contrasts)
        supplied_contrasts: list[CrossFamilyContrastRow] = []
        for raw in self.cross_family_contrasts:
            if not isinstance(raw, tuple) or len(raw) != 4:
                raise InterventionCertificationError(
                    "cross-family intervention contrasts must be quadruples"
                )
            raw_left, raw_right, raw_baseline_delta, raw_intervention_delta = raw
            try:
                left = _text(raw_left, label="cross-family left name")
                right = _text(raw_right, label="cross-family right name")
                baseline_delta = _finite(
                    raw_baseline_delta,
                    label="cross-family baseline delta",
                )
                intervention_delta = _finite(
                    raw_intervention_delta,
                    label="cross-family intervention delta",
                )
            except IdentificationProtocolError as error:
                raise InterventionCertificationError(str(error)) from error
            if left >= right:
                raise InterventionCertificationError(
                    "cross-family contrast names must be in canonical order"
                )
            if baseline_delta < 0.0 or intervention_delta < 0.0:
                raise InterventionCertificationError(
                    "cross-family intervention deltas must be non-negative"
                )
            supplied_contrasts.append(
                (left, right, baseline_delta, intervention_delta)
            )
        supplied_contrasts_tuple = tuple(sorted(supplied_contrasts))
        if len(supplied_contrasts_tuple) != len(computed_contrasts_tuple):
            raise InterventionCertificationError(
                "cross-family contrasts must cover every family pair exactly once"
            )
        for supplied, computed in zip(
            supplied_contrasts_tuple,
            computed_contrasts_tuple,
            strict=True,
        ):
            if supplied[:2] != computed[:2] or any(
                not math.isclose(
                    supplied[value_index],
                    computed[value_index],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for value_index in (2, 3)
            ):
                raise InterventionCertificationError(
                    "cross-family intervention contrasts must match complete policies"
                )

        try:
            parents = tuple(
                sorted(
                    {
                        _hash(
                            item,
                            label="information intervention parent manifest hash",
                        )
                        for item in self.parent_manifest_hashes
                    }
                )
            )
        except IdentificationProtocolError as error:
            raise InterventionCertificationError(str(error)) from error

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pair_hash", pair_hash)
        object.__setattr__(
            self,
            "certified_diff",
            (path, frozen_before, frozen_after),
        )
        object.__setattr__(self, "family_policies", rows_tuple)
        object.__setattr__(self, "within_family_max_deltas", computed_deltas)
        object.__setattr__(
            self,
            "cross_family_contrasts",
            computed_contrasts_tuple,
        )
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "parent_manifest_hashes", parents)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pair_hash": self.pair_hash,
            "certified_diff": self.certified_diff,
            "family_policies": tuple(
                {
                    "family": family,
                    "baseline": baseline,
                    "intervention": intervention,
                }
                for family, baseline, intervention in self.family_policies
            ),
            "within_family_max_deltas": self.within_family_max_deltas,
            "cross_family_contrasts": self.cross_family_contrasts,
            "tolerance": self.tolerance,
            "discriminating": self.discriminating,
            "parent_manifest_hashes": self.parent_manifest_hashes,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def evaluate_information_intervention(
    *,
    pair: InformationInterventionPair,
    runner: SimulationRunner,
    families: tuple[tuple[str, object, Mapping[str, float]], ...],
    seeds: tuple[int, ...],
    extractor: object,
    action_keys: tuple[str, ...],
    tolerance: float = _EQUIVALENCE_TOLERANCE,
) -> InformationInterventionFinding:
    certified_diff = certify_information_intervention(pair)
    if not isinstance(runner, SimulationRunner):
        raise InterventionCertificationError(
            "information intervention evaluation requires SimulationRunner"
        )
    try:
        seed_tuple = _seed_plan(seeds, label="information intervention seeds")
        keys = tuple(
            sorted(
                _text(item, label="information intervention action key")
                for item in action_keys
            )
        )
        tolerance_value = _finite(
            tolerance,
            label="information intervention tolerance",
        )
    except IdentificationProtocolError as error:
        raise InterventionCertificationError(str(error)) from error
    if not keys or len(set(keys)) != len(keys):
        raise InterventionCertificationError(
            "information intervention action keys must be non-empty and unique"
        )
    if tolerance_value != _EQUIVALENCE_TOLERANCE:
        raise InterventionCertificationError(
            "information intervention tolerance must be exactly 1e-12"
        )

    canonical_families: list[tuple[str, object, ParameterTuple]] = []
    seen: set[str] = set()
    for raw_family, model, raw_parameters in families:
        try:
            family = _text(raw_family, label="information intervention family name")
            parameters = _parameter_tuple(
                raw_parameters,
                label=f"information intervention {family} parameters",
            )
        except IdentificationProtocolError as error:
            raise InterventionCertificationError(str(error)) from error
        if family in seen:
            raise InterventionCertificationError(
                "information intervention family names must be unique"
            )
        seen.add(family)
        canonical_families.append((family, model, parameters))
    canonical_families.sort(key=lambda item: item[0])

    if pair.expected_relationship == "intentional_vs_reactive":
        required_families = {"intentional", "reactive"}
        stable_family = "reactive"
        responsive_family = "intentional"
    elif pair.expected_relationship == "planning_vs_intentional":
        required_families = {"intentional", "planning"}
        stable_family = "intentional"
        responsive_family = "planning"
    else:
        raise IdentificationProtocolError(
            "information intervention relationship is unsupported"
        )
    if {family for family, _model, _parameters in canonical_families} != required_families:
        raise InterventionCertificationError(
            "information intervention families must match the preregistered relationship"
        )

    try:
        family_policies: list[FamilyPolicyRow] = []
        parent_hashes: set[str] = set()
        for family, model, parameters in canonical_families:
            baseline_traces = runner.run_batch(
                model,
                pair.baseline,
                dict(parameters),
                seeds=seed_tuple,
            )
            intervention_traces = runner.run_batch(
                model,
                pair.intervention,
                dict(parameters),
                seeds=seed_tuple,
            )
            baseline_policy = _policy_tuple(
                aggregate_metrics(baseline_traces, extractor),
                action_keys=keys,
                label=f"{family} baseline intervention policy",
            )
            intervention_policy = _policy_tuple(
                aggregate_metrics(intervention_traces, extractor),
                action_keys=keys,
                label=f"{family} intervention policy",
            )
            family_policies.append(
                (family, baseline_policy, intervention_policy)
            )
            parent_hashes.update(
                required_manifest_hash(
                    trace,
                    label=f"{family} baseline intervention trace",
                )
                for trace in baseline_traces
            )
            parent_hashes.update(
                required_manifest_hash(
                    trace,
                    label=f"{family} intervention trace",
                )
                for trace in intervention_traces
            )
        family_policies_tuple = tuple(family_policies)
        delta_map = {
            family: max(
                abs(dict(baseline)[key] - dict(intervention)[key])
                for key in keys
            )
            for family, baseline, intervention in family_policies_tuple
        }
        within_deltas = tuple(sorted(delta_map.items()))
        contrasts: list[CrossFamilyContrastRow] = []
        for index, (left_family, left_baseline, left_intervention) in enumerate(
            family_policies_tuple
        ):
            for (
                right_family,
                right_baseline,
                right_intervention,
            ) in family_policies_tuple[index + 1 :]:
                contrasts.append(
                    (
                        left_family,
                        right_family,
                        max(
                            abs(dict(left_baseline)[key] - dict(right_baseline)[key])
                            for key in keys
                        ),
                        max(
                            abs(
                                dict(left_intervention)[key]
                                - dict(right_intervention)[key]
                            )
                            for key in keys
                        ),
                    )
                )
        discriminating = (
            delta_map[stable_family] <= tolerance_value
            and delta_map[responsive_family] > tolerance_value
        )
        return InformationInterventionFinding(
            name=pair.name,
            pair_hash=pair.content_hash,
            certified_diff=certified_diff,
            family_policies=family_policies_tuple,
            within_family_max_deltas=within_deltas,
            cross_family_contrasts=tuple(contrasts),
            tolerance=tolerance_value,
            discriminating=discriminating,
            parent_manifest_hashes=tuple(sorted(parent_hashes)),
        )
    except SyntheticIdentificationError:
        raise
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise InterventionCertificationError(
            "information intervention evaluation failed"
        ) from error


GlobalModelScore = tuple[str, float, float]
PairwiseMeanDelta = tuple[str, str, float]
_EXPECTED_FINAL_STRATA = frozenset(
    {"observational_equivalence", "memory_evidence", "future_information"}
)


@dataclass(frozen=True)
class StratumModelScore:
    stratum: str
    model_name: str
    mean_loss: float
    worst_loss: float
    case_names: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            stratum = _text(self.stratum, label="comparison stratum")
            model_name = _text(self.model_name, label="stratum model name")
            mean_loss = _finite(self.mean_loss, label="stratum mean loss")
            worst_loss = _finite(self.worst_loss, label="stratum worst loss")
            case_names = tuple(
                sorted(_text(item, label="stratum case name") for item in self.case_names)
            )
        except IdentificationProtocolError as error:
            raise IdentificationComparisonError(str(error)) from error
        if mean_loss < 0.0 or worst_loss < 0.0:
            raise IdentificationComparisonError("stratum losses must be non-negative")
        if not case_names or len(set(case_names)) != len(case_names):
            raise IdentificationComparisonError(
                "stratum case names must be non-empty and unique"
            )
        if worst_loss + 1e-15 < mean_loss:
            raise IdentificationComparisonError(
                "stratum worst loss cannot be below its mean loss"
            )
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "mean_loss", mean_loss)
        object.__setattr__(self, "worst_loss", worst_loss)
        object.__setattr__(self, "case_names", case_names)

    def identity_payload(self) -> dict[str, object]:
        return {
            "stratum": self.stratum,
            "model_name": self.model_name,
            "mean_loss": self.mean_loss,
            "worst_loss": self.worst_loss,
            "case_names": self.case_names,
        }


@dataclass(frozen=True)
class ScoredModelComparisonFinding:
    protocol_hash: str
    loss_identity: Mapping[str, object]
    candidate_hashes: tuple[str, ...]
    final_partition_hash: str
    final_target_hash: str
    final_seeds: tuple[int, ...]
    global_scores: tuple[GlobalModelScore, ...]
    stratum_scores: tuple[StratumModelScore, ...]
    pairwise_mean_deltas: tuple[PairwiseMeanDelta, ...]
    parent_comparison_manifest_hash: str

    def __post_init__(self) -> None:
        try:
            protocol_hash = _hash(self.protocol_hash, label="comparison protocol hash")
            loss_identity = _freeze_mapping(
                self.loss_identity,
                label="comparison loss identity",
            )
            candidate_hashes = tuple(
                _hash(item, label="comparison candidate hash")
                for item in self.candidate_hashes
            )
            final_partition_hash = _hash(
                self.final_partition_hash,
                label="comparison final partition hash",
            )
            final_target_hash = _hash(
                self.final_target_hash,
                label="comparison final target hash",
            )
            final_seeds = _seed_plan(
                self.final_seeds,
                label="comparison final seeds",
            )
            parent_hash = _hash(
                self.parent_comparison_manifest_hash,
                label="comparison parent manifest hash",
            )
        except IdentificationProtocolError as error:
            raise IdentificationComparisonError(str(error)) from error
        if loss_identity.get("name") not in {"categorical_brier", "categorical_log"}:
            raise IdentificationComparisonError(
                "synthetic comparison finding requires Brier or Log loss"
            )
        if not candidate_hashes or len(set(candidate_hashes)) != len(candidate_hashes):
            raise IdentificationComparisonError(
                "comparison candidate hashes must be non-empty and unique"
            )

        global_rows: list[GlobalModelScore] = []
        seen_models: set[str] = set()
        for raw in self.global_scores:
            if not isinstance(raw, tuple) or len(raw) != 3:
                raise IdentificationComparisonError(
                    "global model scores must be triples"
                )
            raw_name, raw_mean, raw_worst = raw
            try:
                name = _text(raw_name, label="global score model name")
                mean_loss = _finite(raw_mean, label=f"{name} global mean loss")
                worst_loss = _finite(raw_worst, label=f"{name} global worst loss")
            except IdentificationProtocolError as error:
                raise IdentificationComparisonError(str(error)) from error
            if name in seen_models:
                raise IdentificationComparisonError(
                    "global score model names must be unique"
                )
            seen_models.add(name)
            if mean_loss < 0.0 or worst_loss < 0.0 or worst_loss + 1e-15 < mean_loss:
                raise IdentificationComparisonError(
                    "global comparison losses must be finite non-negative mean/worst values"
                )
            global_rows.append((name, mean_loss, worst_loss))
        if not global_rows or len(global_rows) != len(candidate_hashes):
            raise IdentificationComparisonError(
                "global scores must cover every frozen candidate exactly once"
            )
        global_tuple = tuple(sorted(global_rows))

        strata = tuple(self.stratum_scores)
        if not strata or any(not isinstance(item, StratumModelScore) for item in strata):
            raise IdentificationComparisonError(
                "comparison finding requires stratum model scores"
            )
        if len({(item.stratum, item.model_name) for item in strata}) != len(strata):
            raise IdentificationComparisonError(
                "stratum scores must contain one row per stratum/model pair"
            )
        if {item.stratum for item in strata} != _EXPECTED_FINAL_STRATA:
            raise IdentificationComparisonError(
                "synthetic comparison must cover the exact final strata"
            )
        if {item.model_name for item in strata} != seen_models:
            raise IdentificationComparisonError(
                "every comparison stratum must use the frozen model set"
            )
        for stratum in _EXPECTED_FINAL_STRATA:
            if {item.model_name for item in strata if item.stratum == stratum} != seen_models:
                raise IdentificationComparisonError(
                    "every final stratum must score every frozen model"
                )
        strata_tuple = tuple(sorted(strata, key=lambda item: (item.stratum, item.model_name)))

        means = {name: mean_loss for name, mean_loss, _worst in global_tuple}
        expected_pairs = tuple(
            (left, right, means[left] - means[right])
            for index, left in enumerate(sorted(means))
            for right in sorted(means)[index + 1 :]
        )
        supplied_pairs: list[PairwiseMeanDelta] = []
        for raw in self.pairwise_mean_deltas:
            if not isinstance(raw, tuple) or len(raw) != 3:
                raise IdentificationComparisonError(
                    "pairwise mean deltas must be triples"
                )
            raw_left, raw_right, raw_delta = raw
            try:
                left = _text(raw_left, label="pairwise left model")
                right = _text(raw_right, label="pairwise right model")
                delta = _finite(raw_delta, label="pairwise mean delta")
            except IdentificationProtocolError as error:
                raise IdentificationComparisonError(str(error)) from error
            if left >= right:
                raise IdentificationComparisonError(
                    "pairwise model names must use canonical order"
                )
            supplied_pairs.append((left, right, delta))
        supplied_tuple = tuple(sorted(supplied_pairs))
        if len(supplied_tuple) != len(expected_pairs):
            raise IdentificationComparisonError(
                "pairwise mean deltas must cover every frozen model pair"
            )
        for supplied, expected in zip(supplied_tuple, expected_pairs, strict=True):
            if supplied[:2] != expected[:2] or not math.isclose(
                supplied[2], expected[2], rel_tol=0.0, abs_tol=1e-15
            ):
                raise IdentificationComparisonError(
                    "pairwise mean deltas must match global model means"
                )

        object.__setattr__(self, "protocol_hash", protocol_hash)
        object.__setattr__(self, "loss_identity", loss_identity)
        object.__setattr__(self, "candidate_hashes", candidate_hashes)
        object.__setattr__(self, "final_partition_hash", final_partition_hash)
        object.__setattr__(self, "final_target_hash", final_target_hash)
        object.__setattr__(self, "final_seeds", final_seeds)
        object.__setattr__(self, "global_scores", global_tuple)
        object.__setattr__(self, "stratum_scores", strata_tuple)
        object.__setattr__(self, "pairwise_mean_deltas", expected_pairs)
        object.__setattr__(self, "parent_comparison_manifest_hash", parent_hash)

    def identity_payload(self) -> dict[str, object]:
        return {
            "protocol_hash": self.protocol_hash,
            "loss_identity": self.loss_identity,
            "candidate_hashes": self.candidate_hashes,
            "final_partition_hash": self.final_partition_hash,
            "final_target_hash": self.final_target_hash,
            "final_seeds": self.final_seeds,
            "global_scores": self.global_scores,
            "stratum_scores": tuple(item.identity_payload() for item in self.stratum_scores),
            "pairwise_mean_deltas": self.pairwise_mean_deltas,
            "parent_comparison_manifest_hash": self.parent_comparison_manifest_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def score_model_comparison_by_stratum(
    *,
    dataset: ObservationDataset,
    protocol: PreregisteredEvaluationProtocol,
    report: object,
) -> ScoredModelComparisonFinding:
    from narrative_dynamics.model_comparison import ModelComparisonReport

    if not isinstance(dataset, ObservationDataset):
        raise IdentificationComparisonError(
            "stratified comparison requires ObservationDataset"
        )
    if not isinstance(protocol, PreregisteredEvaluationProtocol):
        raise IdentificationComparisonError(
            "stratified comparison requires PreregisteredEvaluationProtocol"
        )
    if not isinstance(report, ModelComparisonReport):
        raise IdentificationComparisonError(
            "stratified comparison requires ModelComparisonReport"
        )
    if dataset.content_hash != protocol.dataset_hash:
        raise IdentificationComparisonError(
            "comparison dataset drifted from final protocol"
        )
    final_partition = dataset.partition(ObservationPartitionRole.FINAL_TEST)
    if not final_partition.records:
        raise IdentificationComparisonError(
            "synthetic comparison requires record-oriented final partition"
        )
    if final_partition.content_hash != protocol.final_partition_hash:
        raise IdentificationComparisonError(
            "comparison final partition drifted from protocol"
        )
    record_by_name = {record.id: record for record in final_partition.records}
    if len(record_by_name) != len(final_partition.records):
        raise IdentificationComparisonError(
            "final comparison case names must be unique"
        )
    stratum_by_name: dict[str, str] = {}
    for name, record in record_by_name.items():
        raw_stratum = record.metadata.get("stratum")
        try:
            stratum = _text(raw_stratum, label=f"final stratum for {name}")
        except IdentificationProtocolError as error:
            raise IdentificationComparisonError(str(error)) from error
        stratum_by_name[name] = stratum
    if set(stratum_by_name.values()) != _EXPECTED_FINAL_STRATA:
        raise IdentificationComparisonError(
            "final dataset must contain exactly the preregistered synthetic strata"
        )

    if report.baseline_name != protocol.baseline_name:
        raise IdentificationComparisonError(
            "comparison report baseline drifted from protocol"
        )
    if report.manifest.inputs.get("protocol_hash") != protocol.content_hash:
        raise IdentificationComparisonError(
            "comparison report does not bind the exact final protocol"
        )
    if report.manifest.inputs.get("loss_identity") != protocol.loss_identity:
        raise IdentificationComparisonError(
            "comparison report loss drifted from final protocol"
        )

    candidates = tuple(protocol.candidates)
    candidate_names = tuple(candidate.name for candidate in candidates)
    candidate_hashes = tuple(candidate.content_hash for candidate in candidates)
    if len(set(candidate_names)) != len(candidate_names):
        raise IdentificationComparisonError(
            "final protocol candidate names must be unique"
        )
    entries = {entry.name: entry for entry in report.ranking}
    if len(entries) != len(report.ranking) or set(entries) != set(candidate_names):
        raise IdentificationComparisonError(
            "comparison report must cover the frozen candidates exactly"
        )
    candidate_by_name = {candidate.name: candidate for candidate in candidates}

    global_scores: list[GlobalModelScore] = []
    stratum_scores: list[StratumModelScore] = []
    expected_case_names = set(record_by_name)
    for model_name in sorted(entries):
        entry = entries[model_name]
        if entry.parameters != candidate_by_name[model_name].parameters:
            raise IdentificationComparisonError(
                "comparison report parameters drifted from frozen candidate"
            )
        evaluations = tuple(entry.final_test.validation.cases)
        if len({case.name for case in evaluations}) != len(evaluations):
            raise IdentificationComparisonError(
                "comparison report case names must be unique per model"
            )
        case_loss = {case.name: case.loss for case in evaluations}
        if set(case_loss) != expected_case_names:
            raise IdentificationComparisonError(
                "comparison report must score every frozen final case exactly once"
            )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in case_loss.values()
        ):
            raise IdentificationComparisonError(
                "comparison case losses must be finite and non-negative"
            )
        losses = tuple(float(case_loss[name]) for name in sorted(case_loss))
        computed_mean = fmean(losses)
        computed_worst = max(losses)
        if not math.isclose(
            computed_mean, entry.mean_loss, rel_tol=0.0, abs_tol=1e-15
        ) or not math.isclose(
            computed_worst, entry.worst_loss, rel_tol=0.0, abs_tol=1e-15
        ):
            raise IdentificationComparisonError(
                "comparison global scores must match complete final case losses"
            )
        global_scores.append((model_name, computed_mean, computed_worst))
        for stratum in sorted(_EXPECTED_FINAL_STRATA):
            names = tuple(
                sorted(
                    name
                    for name, item_stratum in stratum_by_name.items()
                    if item_stratum == stratum
                )
            )
            stratum_losses = tuple(float(case_loss[name]) for name in names)
            if not stratum_losses:
                raise IdentificationComparisonError(
                    "every synthetic final stratum must contain at least one case"
                )
            stratum_scores.append(
                StratumModelScore(
                    stratum=stratum,
                    model_name=model_name,
                    mean_loss=fmean(stratum_losses),
                    worst_loss=max(stratum_losses),
                    case_names=names,
                )
            )

    mean_map = {name: mean for name, mean, _worst in global_scores}
    pairwise = tuple(
        (left, right, mean_map[left] - mean_map[right])
        for index, left in enumerate(sorted(mean_map))
        for right in sorted(mean_map)[index + 1 :]
    )
    return ScoredModelComparisonFinding(
        protocol_hash=protocol.content_hash,
        loss_identity=protocol.loss_identity,
        candidate_hashes=candidate_hashes,
        final_partition_hash=protocol.final_partition_hash,
        final_target_hash=protocol.final_target_hash,
        final_seeds=protocol.simulation_seeds,
        global_scores=tuple(global_scores),
        stratum_scores=tuple(stratum_scores),
        pairwise_mean_deltas=pairwise,
        parent_comparison_manifest_hash=required_manifest_hash(
            report,
            label="synthetic model comparison report",
        ),
    )


def build_synthetic_identification_report(*args, **kwargs):
    raise NotImplementedError("aggregate identification reporting is implemented in Task 7")


__all__ = [
    "IdentificationComparisonError",
    "IdentificationProtocolError",
    "IdentificationRecoveryError",
    "IdentificationReportError",
    "IdentificationStatus",
    "InformationInterventionFinding",
    "InformationInterventionPair",
    "InterventionCertificationError",
    "ObservationalEquivalenceFinding",
    "ParameterCandidateLoss",
    "ParameterIdentificationFinding",
    "ParameterRecoveryExperiment",
    "ScoredModelComparisonFinding",
    "StratumModelScore",
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
