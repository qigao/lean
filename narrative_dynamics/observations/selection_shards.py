from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from statistics import fmean

from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    _validated_content_hash,
    stable_content_hash,
)
from narrative_dynamics.losses import (
    DEFAULT_METRIC_LOSS,
    MetricLoss,
    evaluate_metric_loss,
    metric_loss_identity,
)
from narrative_dynamics.manifest import (
    callable_identity,
    component_identity,
    required_manifest_hash,
    scenario_identity,
)
from narrative_dynamics.simulation import ModelSource, SimulationRunner
from narrative_dynamics.uncertainty import (
    ParameterAcceptanceSet,
    ParameterTuple,
    _canonical_parameter_tuple,
)
from narrative_dynamics.validation import (
    AcceptedParameterHeldOutEvaluation,
    EvaluationRole,
    HeldOutAcceptanceReport,
    HeldOutCaseEvaluation,
    HeldOutSuite,
    HeldOutValidationReport,
    SelectionValidationReport,
    _optional_loss_limit,
    validate_held_out,
)

from .dataset import _freeze_mapping, _thaw


SELECTION_CANDIDATE_SHARD_SCHEMA_VERSION = 1


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _metrics(values: Iterable[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_name, raw_value in values:
        name = _text(raw_name, label="selection shard metric name")
        if name in seen:
            raise ValueError("selection shard metric names must be unique")
        seen.add(name)
        canonical.append(
            (name, _finite(raw_value, label=f"selection shard metric {name!r}"))
        )
    if not canonical:
        raise ValueError("selection shard metrics must be non-empty")
    return tuple(sorted(canonical))


def _selected_loss(loss: MetricLoss | None) -> MetricLoss:
    return DEFAULT_METRIC_LOSS if loss is None else loss


def _acceptance_set(
    values: ParameterAcceptanceSet | Iterable[Iterable[tuple[str, float]]],
) -> ParameterAcceptanceSet:
    result = (
        values
        if isinstance(values, ParameterAcceptanceSet)
        else ParameterAcceptanceSet.from_parameters(values)
    )
    if not result.parameters:
        raise ValueError("selection candidate shards require accepted parameters")
    return result


def _selection_suite(suite: HeldOutSuite) -> HeldOutSuite:
    if not isinstance(suite, HeldOutSuite):
        raise TypeError("selection candidate shards require HeldOutSuite")
    if suite.role is not EvaluationRole.SELECTION_VALIDATION:
        raise ValueError("selection candidate shards require selection-validation suite")
    return suite


def _suite_payload(suite: HeldOutSuite) -> dict[str, object]:
    return {
        "name": suite.name,
        "role": suite.role.value,
        "cases": tuple(
            {
                "name": case.effective_name,
                "scenario": scenario_identity(case.scenario),
                "seeds": tuple(case.seeds),
                "target_hash": stable_content_hash(case.target),
                "weights_hash": (
                    None
                    if case.weights is None
                    else stable_content_hash(case.weights)
                ),
            }
            for case in suite.cases
        ),
    }


def _suite_hash(suite: HeldOutSuite) -> str:
    return stable_content_hash(_suite_payload(suite))


@dataclass(frozen=True)
class SelectionShardCase:
    name: str
    metrics: tuple[tuple[str, float], ...]
    loss: float
    run_manifest_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="selection shard case name"))
        object.__setattr__(self, "metrics", _metrics(self.metrics))
        object.__setattr__(
            self,
            "loss",
            _finite(self.loss, label="selection shard case loss"),
        )
        hashes = tuple(
            _validated_content_hash(value, label="selection shard run manifest hash")
            for value in self.run_manifest_hashes
        )
        if not hashes:
            raise ValueError("selection shard case requires run-manifest hashes")
        object.__setattr__(self, "run_manifest_hashes", hashes)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "loss": self.loss,
            "run_manifest_hashes": self.run_manifest_hashes,
        }


@dataclass(frozen=True)
class SelectionCandidateShard:
    parameters: ParameterTuple
    accepted_parameter_set_hash: str
    suite_hash: str
    suite_name: str
    suite_role: str
    model_identity: Mapping[str, object]
    repository_identity: Mapping[str, object]
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    cases: tuple[SelectionShardCase, ...]
    schema_version: int = SELECTION_CANDIDATE_SHARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != SELECTION_CANDIDATE_SHARD_SCHEMA_VERSION
        ):
            raise ValueError("unsupported selection candidate shard schema version")
        object.__setattr__(
            self,
            "parameters",
            _canonical_parameter_tuple(self.parameters),
        )
        for attribute, label in (
            ("accepted_parameter_set_hash", "selection shard accepted-set hash"),
            ("suite_hash", "selection shard suite hash"),
        ):
            object.__setattr__(
                self,
                attribute,
                _validated_content_hash(getattr(self, attribute), label=label),
            )
        object.__setattr__(
            self,
            "suite_name",
            _text(self.suite_name, label="selection shard suite name"),
        )
        if self.suite_role != EvaluationRole.SELECTION_VALIDATION.value:
            raise ValueError("selection shard role must be selection_validation")
        object.__setattr__(
            self,
            "model_identity",
            _freeze_mapping(self.model_identity, label="selection shard model identity"),
        )
        object.__setattr__(
            self,
            "repository_identity",
            _freeze_mapping(
                self.repository_identity,
                label="selection shard repository identity",
            ),
        )
        object.__setattr__(
            self,
            "metric_identity",
            _freeze_mapping(self.metric_identity, label="selection shard metric identity"),
        )
        object.__setattr__(
            self,
            "loss_identity",
            _freeze_mapping(self.loss_identity, label="selection shard loss identity"),
        )
        cases = tuple(self.cases)
        if not cases or any(not isinstance(case, SelectionShardCase) for case in cases):
            raise TypeError("selection shard cases must contain SelectionShardCase values")
        names = tuple(case.name for case in cases)
        if len(set(names)) != len(names):
            raise ValueError("selection shard case names must be unique")
        object.__setattr__(self, "cases", cases)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parameters": self.parameters,
            "accepted_parameter_set_hash": self.accepted_parameter_set_hash,
            "suite_hash": self.suite_hash,
            "suite_name": self.suite_name,
            "suite_role": self.suite_role,
            "model_identity": self.model_identity,
            "repository_identity": self.repository_identity,
            "metric_identity": self.metric_identity,
            "loss_identity": self.loss_identity,
            "cases": tuple(case.identity_payload() for case in self.cases),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    def to_payload(self) -> dict[str, object]:
        payload = _thaw(self.identity_payload())
        assert isinstance(payload, dict)
        return {**payload, "content_hash": self.content_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SelectionCandidateShard":
        if not isinstance(payload, Mapping):
            raise TypeError("selection candidate shard payload must be a mapping")
        required = {
            "schema_version",
            "parameters",
            "accepted_parameter_set_hash",
            "suite_hash",
            "suite_name",
            "suite_role",
            "model_identity",
            "repository_identity",
            "metric_identity",
            "loss_identity",
            "cases",
            "content_hash",
        }
        if set(payload) != required:
            raise ValueError("selection candidate shard payload fields must match exactly")
        raw_cases = payload["cases"]
        if not isinstance(raw_cases, (list, tuple)):
            raise ValueError("selection candidate shard cases must be a sequence")
        cases: list[SelectionShardCase] = []
        case_fields = {"name", "metrics", "loss", "run_manifest_hashes"}
        for raw_case in raw_cases:
            if not isinstance(raw_case, Mapping) or set(raw_case) != case_fields:
                raise ValueError("selection candidate shard case fields must match exactly")
            cases.append(
                SelectionShardCase(
                    name=raw_case["name"],
                    metrics=tuple(tuple(item) for item in raw_case["metrics"]),
                    loss=raw_case["loss"],
                    run_manifest_hashes=tuple(raw_case["run_manifest_hashes"]),
                )
            )
        shard = cls(
            schema_version=payload["schema_version"],
            parameters=tuple(tuple(item) for item in payload["parameters"]),
            accepted_parameter_set_hash=payload["accepted_parameter_set_hash"],
            suite_hash=payload["suite_hash"],
            suite_name=payload["suite_name"],
            suite_role=payload["suite_role"],
            model_identity=payload["model_identity"],
            repository_identity=payload["repository_identity"],
            metric_identity=payload["metric_identity"],
            loss_identity=payload["loss_identity"],
            cases=tuple(cases),
        )
        declared = _validated_content_hash(
            payload["content_hash"],
            label="declared selection candidate shard content hash",
        )
        if declared != shard.content_hash:
            raise ValueError(
                "declared selection candidate shard content hash does not match payload"
            )
        return shard


def evaluate_selection_candidate(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    accepted_parameter_set: ParameterAcceptanceSet
    | Iterable[Iterable[tuple[str, float]]],
    parameters: Iterable[tuple[str, float]],
    suite: HeldOutSuite,
    extractor: object,
    loss: MetricLoss | None = None,
) -> SelectionCandidateShard:
    if not isinstance(runner, SimulationRunner):
        raise TypeError("selection candidate evaluation requires SimulationRunner")
    accepted = _acceptance_set(accepted_parameter_set)
    selected_suite = _selection_suite(suite)
    canonical_parameters = _canonical_parameter_tuple(parameters)
    if canonical_parameters not in accepted.parameters:
        raise ValueError("selection candidate is not in the accepted parameter set")
    selected_loss = _selected_loss(loss)
    validation = validate_held_out(
        runner=runner,
        model=model,
        parameters=dict(canonical_parameters),
        cases=selected_suite.cases,
        extractor=extractor,
        loss=selected_loss,
    )
    return SelectionCandidateShard(
        parameters=canonical_parameters,
        accepted_parameter_set_hash=accepted.content_hash,
        suite_hash=_suite_hash(selected_suite),
        suite_name=selected_suite.name,
        suite_role=selected_suite.role.value,
        model_identity=component_identity(model),
        repository_identity=runner.repository_identity.manifest_identity(),
        metric_identity=callable_identity(extractor),
        loss_identity=metric_loss_identity(selected_loss),
        cases=tuple(
            SelectionShardCase(
                name=case.name,
                metrics=case.metrics,
                loss=case.loss,
                run_manifest_hashes=case.run_manifest_hashes,
            )
            for case in validation.cases
        ),
    )


def assemble_selection_validation_report(
    *,
    model: ModelSource,
    accepted_parameters: ParameterAcceptanceSet
    | Iterable[Iterable[tuple[str, float]]],
    suite: HeldOutSuite,
    extractor: object,
    loss: MetricLoss | None = None,
    max_mean_loss: float | None = None,
    max_worst_loss: float | None = None,
    shards: Iterable[SelectionCandidateShard],
) -> SelectionValidationReport:
    accepted = _acceptance_set(accepted_parameters)
    selected_suite = _selection_suite(suite)
    selected_loss = _selected_loss(loss)
    mean_limit = _optional_loss_limit(max_mean_loss, label="maximum mean loss")
    worst_limit = _optional_loss_limit(max_worst_loss, label="maximum worst loss")
    model_identity = component_identity(model)
    metric_identity = callable_identity(extractor)
    loss_identity = metric_loss_identity(selected_loss)
    suite_hash = _suite_hash(selected_suite)

    shard_values = tuple(shards)
    if any(not isinstance(shard, SelectionCandidateShard) for shard in shard_values):
        raise TypeError("selection shard assembly requires SelectionCandidateShard values")
    if len(shard_values) != len(accepted.parameters):
        raise ValueError("selection shard set does not match accepted candidate count")

    shard_by_parameters: dict[ParameterTuple, SelectionCandidateShard] = {}
    repository_identity: Mapping[str, object] | None = None
    expected_case_names = tuple(case.effective_name for case in selected_suite.cases)
    for shard in shard_values:
        if shard.parameters in shard_by_parameters:
            raise ValueError("selection shard set contains a duplicate candidate")
        shard_by_parameters[shard.parameters] = shard
        if shard.parameters not in accepted.parameters:
            raise ValueError("selection shard set contains an undeclared candidate")
        if shard.accepted_parameter_set_hash != accepted.content_hash:
            raise ValueError("selection shard accepted-set identity changed")
        if (
            shard.suite_hash != suite_hash
            or shard.suite_name != selected_suite.name
            or shard.suite_role != selected_suite.role.value
        ):
            raise ValueError("selection shard suite identity changed")
        if shard.model_identity != model_identity:
            raise ValueError("selection shard model identity changed")
        if shard.metric_identity != metric_identity:
            raise ValueError("selection shard metric identity changed")
        if shard.loss_identity != loss_identity:
            raise ValueError("selection shard loss identity changed")
        if tuple(case.name for case in shard.cases) != expected_case_names:
            raise ValueError("selection shard case coverage or order changed")
        if repository_identity is None:
            repository_identity = shard.repository_identity
        elif shard.repository_identity != repository_identity:
            raise ValueError("selection shards disagree on repository identity")
        for shard_case, suite_case in zip(
            shard.cases,
            selected_suite.cases,
            strict=True,
        ):
            if len(shard_case.run_manifest_hashes) != len(tuple(suite_case.seeds)):
                raise ValueError("selection shard run lineage does not match case seeds")

    if set(shard_by_parameters) != set(accepted.parameters):
        raise ValueError("selection shard set is incomplete")

    evaluations: list[AcceptedParameterHeldOutEvaluation] = []
    for parameters in accepted.parameters:
        shard = shard_by_parameters[parameters]
        held_out_cases: list[HeldOutCaseEvaluation] = []
        case_inputs: list[dict[str, object]] = []
        parent_hashes: list[str] = []
        for shard_case, suite_case in zip(
            shard.cases,
            selected_suite.cases,
            strict=True,
        ):
            recalculated_loss = evaluate_metric_loss(
                selected_loss,
                dict(shard_case.metrics),
                suite_case.target,
                weights=suite_case.weights,
            )
            if recalculated_loss != shard_case.loss:
                raise ValueError("selection shard case loss does not match its metrics")
            held_out_cases.append(
                HeldOutCaseEvaluation(
                    name=suite_case.effective_name,
                    scenario_id=suite_case.scenario.id,
                    metrics=shard_case.metrics,
                    target=tuple(
                        sorted(
                            (metric_name, float(value))
                            for metric_name, value in suite_case.target.items()
                        )
                    ),
                    loss=shard_case.loss,
                    run_manifest_hashes=shard_case.run_manifest_hashes,
                )
            )
            seeds = tuple(suite_case.seeds)
            case_inputs.append(
                {
                    "name": suite_case.effective_name,
                    "scenario": scenario_identity(suite_case.scenario),
                    "seeds": seeds,
                    "target_hash": stable_content_hash(suite_case.target),
                    "weights_hash": (
                        None
                        if suite_case.weights is None
                        else stable_content_hash(suite_case.weights)
                    ),
                }
            )
            parent_hashes.extend(shard_case.run_manifest_hashes)
        losses = tuple(case.loss for case in held_out_cases)
        validation_manifest = ExperimentManifest(
            stage=ExperimentStage.HELD_OUT_VALIDATION,
            inputs={
                "model": model_identity,
                "parameters": parameters,
                "cases": tuple(case_inputs),
                "metric": metric_identity,
                "loss": loss_identity,
            },
            parent_hashes=tuple(parent_hashes),
        )
        validation = HeldOutValidationReport(
            parameters=parameters,
            cases=tuple(held_out_cases),
            mean_loss=fmean(losses),
            worst_loss=max(losses),
            manifest=validation_manifest,
        )
        evaluations.append(
            AcceptedParameterHeldOutEvaluation(
                parameters=parameters,
                validation=validation,
            )
        )

    ordered_evaluations = tuple(
        sorted(
            evaluations,
            key=lambda evaluation: (
                evaluation.validation.mean_loss,
                evaluation.validation.worst_loss,
                evaluation.parameters,
            ),
        )
    )
    retained = tuple(
        evaluation.parameters
        for evaluation in ordered_evaluations
        if (mean_limit is None or evaluation.validation.mean_loss <= mean_limit)
        and (worst_limit is None or evaluation.validation.worst_loss <= worst_limit)
    )
    validation_hashes = tuple(
        required_manifest_hash(
            evaluation.validation,
            label="accepted-parameter held-out validation",
        )
        for evaluation in ordered_evaluations
    )
    acceptance_manifest = ExperimentManifest(
        stage=ExperimentStage.ACCEPTANCE_VALIDATION,
        inputs={
            "accepted_parameters": accepted.parameters,
            "accepted_parameter_set_hash": accepted.content_hash,
            "max_mean_loss": mean_limit,
            "max_worst_loss": worst_limit,
            "loss": loss_identity,
        },
        parent_hashes=accepted.source_manifest_hashes + validation_hashes,
    )
    candidate_report = HeldOutAcceptanceReport(
        evaluations=ordered_evaluations,
        retained_parameters=ParameterAcceptanceSet.from_parameters(
            retained,
            source_manifest_hashes=(acceptance_manifest.content_hash,),
        ),
        best=ordered_evaluations[0],
        manifest=acceptance_manifest,
    )

    retained_set = set(candidate_report.retained_parameters.parameters)
    if not retained_set:
        raise ValueError("selection-validation thresholds retained no candidates")
    selected_parameters = next(
        evaluation.parameters
        for evaluation in candidate_report.evaluations
        if evaluation.parameters in retained_set
    )
    candidate_hash = required_manifest_hash(
        candidate_report,
        label="selection candidate report",
    )
    selection_manifest = ExperimentManifest(
        stage=ExperimentStage.SELECTION_VALIDATION,
        inputs={
            "suite_name": selected_suite.name,
            "role": selected_suite.role.value,
            "max_mean_loss": max_mean_loss,
            "max_worst_loss": max_worst_loss,
            "loss": loss_identity,
        },
        parent_hashes=(candidate_hash,),
    )
    return SelectionValidationReport(
        suite_name=selected_suite.name,
        role=selected_suite.role,
        candidate_report=candidate_report,
        selected_parameters=selected_parameters,
        manifest=selection_manifest,
    )


__all__ = [
    "SELECTION_CANDIDATE_SHARD_SCHEMA_VERSION",
    "SelectionCandidateShard",
    "SelectionShardCase",
    "assemble_selection_validation_report",
    "evaluate_selection_candidate",
]
