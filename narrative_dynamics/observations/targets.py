from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage, Scenario, stable_content_hash
from narrative_dynamics.losses import CategoricalMetricGroup
from narrative_dynamics.manifest import scenario_identity
from narrative_dynamics.validation import EvaluationRole, HeldOutCase, HeldOutSuite

from .dataset import (
    ObservationCase,
    ObservationDataset,
    ObservationPartitionRole,
    ObservationRecord,
)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _role(value: ObservationPartitionRole | str) -> ObservationPartitionRole:
    try:
        return value if isinstance(value, ObservationPartitionRole) else ObservationPartitionRole(value)
    except (TypeError, ValueError) as error:
        raise ValueError("target-construction role must be supported") from error


def _groups(values: tuple[CategoricalMetricGroup, ...]) -> tuple[CategoricalMetricGroup, ...]:
    groups = tuple(values)
    if not groups:
        raise ValueError("categorical target construction requires metric groups")
    if any(not isinstance(group, CategoricalMetricGroup) for group in groups):
        raise TypeError("categorical target groups must be CategoricalMetricGroup values")
    names = tuple(group.name for group in groups)
    keys = tuple(key for group in groups for key in group.keys)
    if len(set(names)) != len(names):
        raise ValueError("categorical target group names must be unique")
    if len(set(keys)) != len(keys):
        raise ValueError("categorical target keys cannot appear in multiple groups")
    return groups


def _seed_tuple(values: object, *, label: str) -> tuple[int, ...]:
    try:
        seeds = tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an iterable of integers") from error
    if not seeds:
        raise ValueError(f"{label} must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError(f"{label} values must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} values must be unique")
    return seeds


@dataclass(frozen=True)
class CategoricalTargetPlan:
    name: str
    version: str
    role: ObservationPartitionRole | str
    groups: tuple[CategoricalMetricGroup, ...]
    seeds_by_case: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="target plan name"))
        object.__setattr__(self, "version", _text(self.version, label="target plan version"))
        object.__setattr__(self, "role", _role(self.role))
        object.__setattr__(self, "groups", _groups(self.groups))
        if not isinstance(self.seeds_by_case, Mapping):
            raise TypeError("target plan seeds_by_case must be a mapping")
        canonical: dict[str, tuple[int, ...]] = {}
        for case_name in sorted(self.seeds_by_case):
            canonical[_text(case_name, label="target plan case name")] = _seed_tuple(
                self.seeds_by_case[case_name],
                label=f"target plan seeds for {case_name!r}",
            )
        object.__setattr__(self, "seeds_by_case", MappingProxyType(canonical))

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "role": self.role.value,
            "groups": tuple(group.manifest_identity() for group in self.groups),
            "seeds_by_case": self.seeds_by_case,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class CategoricalTargetSpec:
    name: str
    version: str
    categories: tuple[str, ...]
    metric_prefix: str
    pseudocount: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="target spec name"))
        object.__setattr__(self, "version", _text(self.version, label="target spec version"))
        categories = tuple(_text(category, label="target category") for category in self.categories)
        if not categories:
            raise ValueError("target spec requires at least one category")
        if len(set(categories)) != len(categories):
            raise ValueError("target spec categories must be unique")
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "metric_prefix", _text(self.metric_prefix, label="target metric prefix"))
        if (
            isinstance(self.pseudocount, bool)
            or not isinstance(self.pseudocount, (int, float))
            or not math.isfinite(float(self.pseudocount))
            or float(self.pseudocount) < 0.0
        ):
            raise ValueError("target pseudocount must be finite and non-negative")
        object.__setattr__(self, "pseudocount", float(self.pseudocount))

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return tuple(f"{self.metric_prefix}.{category}" for category in self.categories)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "categories": self.categories,
            "metric_prefix": self.metric_prefix,
            "pseudocount": self.pseudocount,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ConstructedObservationCase:
    name: str
    scenario: Scenario
    counts: tuple[tuple[str, int], ...]
    targets: tuple[tuple[str, float], ...]
    observation_ids: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)
    record_hash: str | None = None
    observation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="constructed case name"))
        if not isinstance(self.scenario, Scenario):
            raise TypeError("constructed target scenario must be a Scenario")
        counts = tuple(sorted(self.counts))
        targets = tuple(sorted(self.targets))
        if not counts or not targets:
            raise ValueError("constructed target cases require counts and targets")
        if len({key for key, _ in counts}) != len(counts):
            raise ValueError("constructed target count keys must be unique")
        if len({key for key, _ in targets}) != len(targets):
            raise ValueError("constructed target probability keys must be unique")
        if any(
            not isinstance(key, str) or not key or isinstance(value, bool)
            or not isinstance(value, int) or value < 0
            for key, value in counts
        ):
            raise ValueError("constructed target counts are invalid")
        if any(
            not isinstance(key, str) or not key or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            or float(value) < 0.0
            for key, value in targets
        ):
            raise ValueError("constructed target probabilities are invalid")
        observation_ids = tuple(self.observation_ids)
        seeds = tuple(self.seeds)
        if any(not isinstance(value, str) or not value for value in observation_ids):
            raise ValueError("constructed observation ids must be non-empty strings")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("constructed observation ids must be unique")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError("constructed simulation seeds must be integers")
        if seeds and len(set(seeds)) != len(seeds):
            raise ValueError("constructed simulation seeds must be unique")
        if (
            not isinstance(self.observation_count, int)
            or isinstance(self.observation_count, bool)
            or self.observation_count <= 0
        ):
            raise ValueError("constructed observation count must be positive")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "targets", tuple((key, float(value)) for key, value in targets))
        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def count_map(self) -> dict[str, int]:
        return dict(self.counts)

    @property
    def target_map(self) -> dict[str, float]:
        return dict(self.targets)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scenario": scenario_identity(self.scenario),
            "counts": self.counts,
            "targets": self.targets,
            "observation_ids": self.observation_ids,
            "seeds": self.seeds,
            "provenance": self.provenance,
            "record_hash": self.record_hash,
            "observation_count": self.observation_count,
        }


@dataclass(frozen=True)
class TargetConstructionReport:
    dataset_hash: str
    partition_hash: str
    role: ObservationPartitionRole
    cases: tuple[ConstructedObservationCase, ...]
    manifest: ExperimentManifest
    plan: CategoricalTargetPlan | None = None
    plan_hash: str | None = None
    spec: CategoricalTargetSpec | None = None
    spec_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _role(self.role))
        cases = tuple(sorted(self.cases, key=lambda case: case.name))
        if not cases:
            raise ValueError("target construction report requires cases")
        object.__setattr__(self, "cases", cases)
        if not isinstance(self.manifest, ExperimentManifest):
            raise TypeError("target construction manifest must be an ExperimentManifest")
        if bool(self.plan) == bool(self.spec):
            raise ValueError("target report requires exactly one plan or spec")
        if self.plan is not None and self.plan_hash != self.plan.content_hash:
            raise ValueError("target report plan hash does not match plan")
        if self.spec is not None and self.spec_hash != self.spec.content_hash:
            raise ValueError("target report spec hash does not match spec")

    @property
    def content_hash(self) -> str:
        return stable_content_hash({
            "dataset_hash": self.dataset_hash,
            "partition_hash": self.partition_hash,
            "role": self.role.value,
            "plan_hash": self.plan_hash,
            "spec_hash": self.spec_hash,
            "cases": tuple(case.identity_payload() for case in self.cases),
            "manifest_hash": self.manifest.content_hash,
        })

    def as_held_out_suite(self) -> HeldOutSuite:
        if self.role is ObservationPartitionRole.TRAIN:
            raise ValueError("training targets cannot be exposed as held-out data")
        evaluation_role = (
            EvaluationRole.SELECTION_VALIDATION
            if self.role is ObservationPartitionRole.SELECTION_VALIDATION
            else EvaluationRole.FINAL_TEST
        )
        held_out_cases: list[HeldOutCase] = []
        for case in self.cases:
            if not case.seeds:
                raise ValueError("target set has no per-case simulation seed plan")
            held_out_cases.append(HeldOutCase(
                scenario=case.scenario,
                seeds=case.seeds,
                target=case.target_map,
                name=case.name,
            ))
        suite_name = self.plan.name if self.plan is not None else self.spec.name
        return HeldOutSuite(
            name=f"{suite_name}:{self.role.value}",
            role=evaluation_role,
            cases=tuple(held_out_cases),
        )


ConstructedTargetSet = TargetConstructionReport


def _old_case_targets(case: ObservationCase, groups: tuple[CategoricalMetricGroup, ...]) -> tuple[tuple[str, float], ...]:
    counts = case.count_map
    declared = tuple(key for group in groups for key in group.keys)
    if set(declared) != set(counts) or len(declared) != len(set(declared)):
        raise ValueError("categorical target groups must cover each count coordinate exactly once")
    targets: dict[str, float] = {}
    for group in groups:
        total = sum(counts[key] for key in group.keys)
        if total <= 0:
            raise ValueError("categorical target group count total must be positive")
        if total != len(case.observation_ids):
            raise ValueError("categorical target group totals must equal source observation count")
        for key in group.keys:
            targets[key] = counts[key] / total
    return tuple(sorted(targets.items()))


def _record_targets(record: ObservationRecord, spec: CategoricalTargetSpec) -> tuple[tuple[str, float], ...]:
    counts = record.count_map
    if set(counts) != set(spec.categories):
        raise ValueError("target spec categories must exactly match observation count schema")
    denominator = sum(counts.values()) + spec.pseudocount * len(spec.categories)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("categorical target denominator must be positive and finite")
    return tuple(
        (f"{spec.metric_prefix}.{category}", (counts[category] + spec.pseudocount) / denominator)
        for category in spec.categories
    )


def construct_categorical_targets(
    dataset: ObservationDataset,
    plan: CategoricalTargetPlan | None = None,
    *,
    role: ObservationPartitionRole | str | None = None,
    spec: CategoricalTargetSpec | None = None,
) -> TargetConstructionReport:
    if not isinstance(dataset, ObservationDataset):
        raise TypeError("categorical targets require an ObservationDataset")
    if plan is not None:
        if role is not None or spec is not None:
            raise ValueError("target construction cannot mix plan and spec APIs")
        if not isinstance(plan, CategoricalTargetPlan):
            raise TypeError("target construction plan must be CategoricalTargetPlan")
        selected_role = plan.role
        partition = dataset.partition(selected_role)
        if not partition.cases:
            raise ValueError("case-oriented target plan requires case-oriented dataset")
        case_names = {case.name for case in partition.cases}
        if set(plan.seeds_by_case) != case_names:
            raise ValueError("target seed plan must exactly match selected partition cases")
        built = tuple(
            ConstructedObservationCase(
                name=case.name,
                scenario=case.scenario,
                counts=tuple(case.counts.items()),
                targets=_old_case_targets(case, plan.groups),
                observation_ids=case.observation_ids,
                seeds=plan.seeds_by_case[case.name],
                provenance=case.provenance,
                observation_count=len(case.observation_ids),
            )
            for case in partition.cases
        )
        manifest = ExperimentManifest(
            stage=ExperimentStage.TARGET_CONSTRUCTION,
            inputs={
                "dataset_hash": dataset.content_hash,
                "partition_hash": partition.content_hash,
                "role": selected_role.value,
                "plan": plan.identity_payload(),
                "cases": tuple(case.identity_payload() for case in built),
            },
        )
        return TargetConstructionReport(
            dataset_hash=dataset.content_hash,
            partition_hash=partition.content_hash,
            role=selected_role,
            cases=built,
            manifest=manifest,
            plan=plan,
            plan_hash=plan.content_hash,
        )

    if role is None or spec is None:
        raise ValueError("target construction requires either plan or role/spec")
    if not isinstance(spec, CategoricalTargetSpec):
        raise TypeError("target construction spec must be CategoricalTargetSpec")
    selected_role = _role(role)
    partition = dataset.partition(selected_role)
    if not partition.records:
        raise ValueError("record-oriented target spec requires record-oriented dataset")
    built = tuple(
        ConstructedObservationCase(
            name=record.id,
            scenario=record.scenario,
            counts=tuple(record.counts.items()),
            targets=_record_targets(record, spec),
            observation_ids=(record.id,),
            provenance=record.metadata,
            record_hash=record.content_hash,
            observation_count=sum(record.counts.values()),
        )
        for record in partition.records
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.OBSERVATION_TARGET_CONSTRUCTION,
        inputs={
            "dataset_hash": dataset.content_hash,
            "partition_hash": partition.content_hash,
            "role": selected_role.value,
            "spec": spec.identity_payload(),
            "record_hashes": tuple(record.content_hash for record in partition.records),
            "cases": tuple(case.identity_payload() for case in built),
        },
    )
    return TargetConstructionReport(
        dataset_hash=dataset.content_hash,
        partition_hash=partition.content_hash,
        role=selected_role,
        cases=built,
        manifest=manifest,
        spec=spec,
        spec_hash=spec.content_hash,
    )


__all__ = [
    "CategoricalTargetPlan",
    "CategoricalTargetSpec",
    "ConstructedObservationCase",
    "ConstructedTargetSet",
    "TargetConstructionReport",
    "construct_categorical_targets",
]
