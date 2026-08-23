from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.losses import MetricLoss, metric_loss_identity
from narrative_dynamics.manifest import callable_identity, component_identity, required_manifest_hash
from narrative_dynamics.validation import SelectionValidationReport

from .dataset import ObservationDataset, ObservationPartitionRole, _freeze_mapping
from .targets import CategoricalTargetSpec, TargetConstructionReport, construct_categorical_targets


_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_RANKING_RULE = ("mean_loss", "worst_loss", "model_name")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _parameters(
    values: Mapping[str, float] | tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_name, raw_value in items:
        name = _text(raw_name, label="frozen parameter name")
        if name in seen:
            raise ValueError("frozen parameter names must be unique")
        seen.add(name)
        if isinstance(raw_value, bool):
            raise ValueError("frozen parameter values must be numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError("frozen parameter values must be numeric") from error
        if not math.isfinite(value):
            raise ValueError("frozen parameter values must be finite")
        canonical.append((name, value))
    if not canonical:
        raise ValueError("frozen model requires at least one parameter")
    return tuple(sorted(canonical))


def _seed_plan(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError("preregistered simulation seeds must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("preregistered simulation seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("preregistered simulation seeds must be unique")
    return seeds


@dataclass(frozen=True)
class FrozenModelSpec:
    name: str
    model_identity: Mapping[str, object]
    parameters: tuple[tuple[str, float], ...]
    selection_manifest_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="frozen model name"))
        object.__setattr__(
            self,
            "model_identity",
            _freeze_mapping(self.model_identity, label="frozen model identity"),
        )
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        object.__setattr__(
            self,
            "selection_manifest_hash",
            _hash(self.selection_manifest_hash, label="selection manifest hash"),
        )

    @classmethod
    def from_selection(
        cls,
        name: str,
        model: object,
        report: SelectionValidationReport,
    ) -> "FrozenModelSpec":
        if not isinstance(report, SelectionValidationReport):
            raise TypeError("frozen model selection provenance must be a SelectionValidationReport")
        if report.manifest is None:
            raise ValueError("selection report is missing a manifest")
        return cls(
            name=name,
            model_identity=component_identity(model),
            parameters=report.selected_parameters,
            selection_manifest_hash=required_manifest_hash(
                report,
                label="frozen model selection report",
            ),
        )

    @classmethod
    def freeze(
        cls,
        *,
        name: str,
        model: object,
        parameters: Mapping[str, float] | tuple[tuple[str, float], ...],
        selection_manifest_hash: str,
    ) -> "FrozenModelSpec":
        return cls(
            name=name,
            model_identity=component_identity(model),
            parameters=_parameters(parameters),
            selection_manifest_hash=selection_manifest_hash,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_identity": self.model_identity,
            "parameters": self.parameters,
            "selection_manifest_hash": self.selection_manifest_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


FrozenModelCandidate = FrozenModelSpec


@dataclass(frozen=True)
class AdequacyThresholds:
    max_mean_loss: float
    max_worst_loss: float

    def __post_init__(self) -> None:
        for attribute, label in (
            ("max_mean_loss", "maximum mean loss"),
            ("max_worst_loss", "maximum worst loss"),
        ):
            raw_value = getattr(self, attribute)
            if isinstance(raw_value, bool):
                raise ValueError(f"{label} must be numeric")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} must be numeric") from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
            object.__setattr__(self, attribute, value)

    def identity_payload(self) -> dict[str, object]:
        return {
            "max_mean_loss": self.max_mean_loss,
            "max_worst_loss": self.max_worst_loss,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ComparisonPreregistration:
    name: str
    version: str
    dataset_hash: str
    final_partition_hash: str
    target_construction_manifest_hash: str
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    thresholds: AdequacyThresholds
    models: tuple[FrozenModelSpec, ...]
    ranking_rule: tuple[str, ...] = _RANKING_RULE

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="comparison registration name"))
        object.__setattr__(self, "version", _text(self.version, label="comparison registration version"))
        for attribute, label in (
            ("dataset_hash", "comparison dataset hash"),
            ("final_partition_hash", "comparison final partition hash"),
            ("target_construction_manifest_hash", "comparison target-construction manifest hash"),
        ):
            object.__setattr__(self, attribute, _hash(getattr(self, attribute), label=label))
        object.__setattr__(self, "metric_identity", _freeze_mapping(self.metric_identity, label="comparison metric identity"))
        object.__setattr__(self, "loss_identity", _freeze_mapping(self.loss_identity, label="comparison loss identity"))
        if not isinstance(self.thresholds, AdequacyThresholds):
            raise TypeError("comparison thresholds must be AdequacyThresholds")
        models = tuple(sorted(self.models, key=lambda model: model.name))
        if not models:
            raise ValueError("comparison registration requires frozen models")
        if any(not isinstance(model, FrozenModelSpec) for model in models):
            raise TypeError("comparison registration models must be FrozenModelSpec values")
        names = tuple(model.name for model in models)
        if len(set(names)) != len(names):
            raise ValueError("comparison registration model names must be unique")
        object.__setattr__(self, "models", models)
        if tuple(self.ranking_rule) != _RANKING_RULE:
            raise ValueError("comparison ranking rule is fixed")
        object.__setattr__(self, "ranking_rule", _RANKING_RULE)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        target_report: TargetConstructionReport,
        extractor: object,
        loss: MetricLoss,
        thresholds: AdequacyThresholds,
        models: tuple[FrozenModelSpec, ...],
    ) -> "ComparisonPreregistration":
        if not isinstance(target_report, TargetConstructionReport):
            raise TypeError("comparison registration target must be a target-construction report")
        if target_report.role is not ObservationPartitionRole.FINAL_TEST:
            raise ValueError("comparison registration requires final-test targets")
        if target_report.manifest is None:
            raise ValueError("target-construction report is missing a manifest")
        return cls(
            name=name,
            version=version,
            dataset_hash=target_report.dataset_hash,
            final_partition_hash=target_report.partition_hash,
            target_construction_manifest_hash=target_report.manifest.content_hash,
            metric_identity=callable_identity(extractor),
            loss_identity=metric_loss_identity(loss),
            thresholds=thresholds,
            models=models,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "dataset_hash": self.dataset_hash,
            "final_partition_hash": self.final_partition_hash,
            "target_construction_manifest_hash": self.target_construction_manifest_hash,
            "metric_identity": self.metric_identity,
            "loss_identity": self.loss_identity,
            "thresholds": self.thresholds.identity_payload(),
            "models": tuple(model.identity_payload() for model in self.models),
            "ranking_rule": self.ranking_rule,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class PreregisteredEvaluationProtocol:
    name: str
    version: str
    dataset_hash: str
    train_partition_hash: str
    selection_partition_hash: str
    final_partition_hash: str
    target_spec_hash: str
    final_target_hash: str
    final_target_manifest_hash: str
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    simulation_seeds: tuple[int, ...]
    baseline_name: str
    candidates: tuple[FrozenModelCandidate, ...]
    thresholds: AdequacyThresholds
    declared_precommitment_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="evaluation protocol name"))
        object.__setattr__(self, "version", _text(self.version, label="evaluation protocol version"))
        for attribute, label in (
            ("dataset_hash", "protocol dataset hash"),
            ("train_partition_hash", "protocol train partition hash"),
            ("selection_partition_hash", "protocol selection partition hash"),
            ("final_partition_hash", "protocol final partition hash"),
            ("target_spec_hash", "protocol target spec hash"),
            ("final_target_hash", "protocol final target hash"),
            ("final_target_manifest_hash", "protocol target-construction manifest hash"),
        ):
            object.__setattr__(self, attribute, _hash(getattr(self, attribute), label=label))
        object.__setattr__(
            self,
            "metric_identity",
            _freeze_mapping(self.metric_identity, label="protocol metric identity"),
        )
        object.__setattr__(
            self,
            "loss_identity",
            _freeze_mapping(self.loss_identity, label="protocol loss identity"),
        )
        object.__setattr__(self, "simulation_seeds", _seed_plan(self.simulation_seeds))
        object.__setattr__(self, "baseline_name", _text(self.baseline_name, label="protocol baseline name"))
        candidates = tuple(sorted(self.candidates, key=lambda item: item.name))
        if not candidates:
            raise ValueError("evaluation protocol requires candidates")
        if any(not isinstance(item, FrozenModelSpec) for item in candidates):
            raise TypeError("evaluation protocol candidates must be FrozenModelCandidate values")
        names = tuple(item.name for item in candidates)
        if len(set(names)) != len(names):
            raise ValueError("evaluation protocol candidate names must be unique")
        if self.baseline_name not in names:
            raise ValueError("evaluation protocol baseline must be a candidate")
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.thresholds, AdequacyThresholds):
            raise TypeError("evaluation protocol thresholds are invalid")
        declared = _hash(self.declared_precommitment_hash, label="declared evaluation precommitment hash")
        expected = stable_content_hash(self.identity_payload())
        if declared != expected:
            raise ValueError("declared evaluation precommitment hash does not match protocol")
        object.__setattr__(self, "declared_precommitment_hash", declared)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        dataset: ObservationDataset,
        target_spec: CategoricalTargetSpec,
        extractor: object,
        loss: MetricLoss,
        simulation_seeds: tuple[int, ...],
        baseline_name: str,
        candidates: tuple[FrozenModelCandidate, ...],
        thresholds: AdequacyThresholds,
    ) -> "PreregisteredEvaluationProtocol":
        if not isinstance(dataset, ObservationDataset):
            raise TypeError("evaluation protocol dataset must be ObservationDataset")
        if not isinstance(target_spec, CategoricalTargetSpec):
            raise TypeError("evaluation protocol target spec must be CategoricalTargetSpec")
        final_targets = construct_categorical_targets(
            dataset,
            role=ObservationPartitionRole.FINAL_TEST,
            spec=target_spec,
        )
        canonical_name = _text(name, label="evaluation protocol name")
        canonical_version = _text(version, label="evaluation protocol version")
        canonical_metric_identity = callable_identity(extractor)
        canonical_loss_identity = metric_loss_identity(loss)
        canonical_seeds = _seed_plan(simulation_seeds)
        canonical_baseline = _text(baseline_name, label="evaluation protocol baseline name")
        canonical_candidates = tuple(sorted(candidates, key=lambda item: item.name))
        identity_payload = {
            "name": canonical_name,
            "version": canonical_version,
            "dataset_hash": dataset.content_hash,
            "train_partition_hash": dataset.partition(ObservationPartitionRole.TRAIN).content_hash,
            "selection_partition_hash": dataset.partition(ObservationPartitionRole.SELECTION_VALIDATION).content_hash,
            "final_partition_hash": dataset.partition(ObservationPartitionRole.FINAL_TEST).content_hash,
            "target_spec_hash": target_spec.content_hash,
            "final_target_hash": final_targets.content_hash,
            "final_target_manifest_hash": final_targets.manifest.content_hash,
            "metric_identity": canonical_metric_identity,
            "loss_identity": canonical_loss_identity,
            "simulation_seeds": canonical_seeds,
            "baseline_name": canonical_baseline,
            "candidates": tuple(candidate.identity_payload() for candidate in canonical_candidates),
            "thresholds": thresholds.identity_payload(),
            "ranking_rule": _RANKING_RULE,
        }
        return cls(
            name=canonical_name,
            version=canonical_version,
            dataset_hash=dataset.content_hash,
            train_partition_hash=dataset.partition(ObservationPartitionRole.TRAIN).content_hash,
            selection_partition_hash=dataset.partition(ObservationPartitionRole.SELECTION_VALIDATION).content_hash,
            final_partition_hash=dataset.partition(ObservationPartitionRole.FINAL_TEST).content_hash,
            target_spec_hash=target_spec.content_hash,
            final_target_hash=final_targets.content_hash,
            final_target_manifest_hash=final_targets.manifest.content_hash,
            metric_identity=canonical_metric_identity,
            loss_identity=canonical_loss_identity,
            simulation_seeds=canonical_seeds,
            baseline_name=canonical_baseline,
            candidates=canonical_candidates,
            thresholds=thresholds,
            declared_precommitment_hash=stable_content_hash(identity_payload),
        )

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "dataset_hash": self.dataset_hash,
            "train_partition_hash": self.train_partition_hash,
            "selection_partition_hash": self.selection_partition_hash,
            "final_partition_hash": self.final_partition_hash,
            "target_spec_hash": self.target_spec_hash,
            "final_target_hash": self.final_target_hash,
            "final_target_manifest_hash": self.final_target_manifest_hash,
            "metric_identity": self.metric_identity,
            "loss_identity": self.loss_identity,
            "simulation_seeds": self.simulation_seeds,
            "baseline_name": self.baseline_name,
            "candidates": tuple(candidate.identity_payload() for candidate in self.candidates),
            "thresholds": self.thresholds.identity_payload(),
            "ranking_rule": _RANKING_RULE,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


__all__ = [
    "AdequacyThresholds",
    "ComparisonPreregistration",
    "FrozenModelCandidate",
    "FrozenModelSpec",
    "PreregisteredEvaluationProtocol",
]
