from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from statistics import fmean

from narrative_dynamics.calibration import CalibrationResult, CandidateEvaluation
from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    _validated_content_hash,
    stable_content_hash,
)
from narrative_dynamics.losses import (
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
from narrative_dynamics.metrics import aggregate_metrics
from narrative_dynamics.simulation import ModelSource, SimulationRunner
from narrative_dynamics.uncertainty import ParameterTuple, _canonical_parameter_tuple

from .dataset import ObservationPartitionRole, _freeze_mapping, _thaw
from .targets import TargetConstructionReport
from .training import TrainingCandidateFit, TrainingCaseFit, TrainingFitReport


TRAINING_CANDIDATE_SHARD_SCHEMA_VERSION = 1


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


def _seeds(values: Iterable[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError("training candidate shard requires simulation seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise TypeError("training candidate shard seeds must be integers")
    return seeds


def _metrics(values: Iterable[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_name, raw_value in values:
        name = _text(raw_name, label="training shard metric name")
        if name in seen:
            raise ValueError("training shard metric names must be unique")
        seen.add(name)
        canonical.append(
            (name, _finite(raw_value, label=f"training shard metric {name!r}"))
        )
    if not canonical:
        raise ValueError("training shard metrics must be non-empty")
    return tuple(sorted(canonical))


@dataclass(frozen=True)
class TrainingShardCase:
    name: str
    metrics: tuple[tuple[str, float], ...]
    loss: float
    run_manifest_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="training shard case name"))
        object.__setattr__(self, "metrics", _metrics(self.metrics))
        object.__setattr__(
            self,
            "loss",
            _finite(self.loss, label="training shard case loss"),
        )
        hashes = tuple(
            _validated_content_hash(value, label="training shard run manifest hash")
            for value in self.run_manifest_hashes
        )
        if not hashes:
            raise ValueError("training shard case requires run-manifest hashes")
        object.__setattr__(self, "run_manifest_hashes", hashes)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "loss": self.loss,
            "run_manifest_hashes": self.run_manifest_hashes,
        }


@dataclass(frozen=True)
class TrainingCandidateShard:
    parameters: ParameterTuple
    target_report_hash: str
    model_identity: Mapping[str, object]
    repository_identity: Mapping[str, object]
    simulation_seeds: tuple[int, ...]
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    cases: tuple[TrainingShardCase, ...]
    schema_version: int = TRAINING_CANDIDATE_SHARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != TRAINING_CANDIDATE_SHARD_SCHEMA_VERSION
        ):
            raise ValueError("unsupported training candidate shard schema version")
        object.__setattr__(
            self,
            "parameters",
            _canonical_parameter_tuple(self.parameters),
        )
        object.__setattr__(
            self,
            "target_report_hash",
            _validated_content_hash(
                self.target_report_hash,
                label="training shard target report hash",
            ),
        )
        object.__setattr__(
            self,
            "model_identity",
            _freeze_mapping(self.model_identity, label="training shard model identity"),
        )
        object.__setattr__(
            self,
            "repository_identity",
            _freeze_mapping(
                self.repository_identity,
                label="training shard repository identity",
            ),
        )
        object.__setattr__(self, "simulation_seeds", _seeds(self.simulation_seeds))
        object.__setattr__(
            self,
            "metric_identity",
            _freeze_mapping(self.metric_identity, label="training shard metric identity"),
        )
        object.__setattr__(
            self,
            "loss_identity",
            _freeze_mapping(self.loss_identity, label="training shard loss identity"),
        )
        cases = tuple(self.cases)
        if not cases or any(not isinstance(case, TrainingShardCase) for case in cases):
            raise TypeError("training shard cases must contain TrainingShardCase values")
        names = tuple(case.name for case in cases)
        if len(set(names)) != len(names):
            raise ValueError("training shard case names must be unique")
        if any(len(case.run_manifest_hashes) != len(self.simulation_seeds) for case in cases):
            raise ValueError("training shard run lineage must match the seed plan")
        for case in cases:
            seed_by_run_hash: dict[str, int] = {}
            for seed, run_hash in zip(
                self.simulation_seeds,
                case.run_manifest_hashes,
                strict=True,
            ):
                prior_seed = seed_by_run_hash.setdefault(run_hash, seed)
                if prior_seed != seed:
                    raise ValueError(
                        "training shard run lineage reuses one manifest across distinct seeds"
                    )
        object.__setattr__(self, "cases", cases)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parameters": self.parameters,
            "target_report_hash": self.target_report_hash,
            "model_identity": self.model_identity,
            "repository_identity": self.repository_identity,
            "simulation_seeds": self.simulation_seeds,
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
    def from_payload(cls, payload: Mapping[str, object]) -> "TrainingCandidateShard":
        if not isinstance(payload, Mapping):
            raise TypeError("training candidate shard payload must be a mapping")
        required = {
            "schema_version",
            "parameters",
            "target_report_hash",
            "model_identity",
            "repository_identity",
            "simulation_seeds",
            "metric_identity",
            "loss_identity",
            "cases",
            "content_hash",
        }
        if set(payload) != required:
            raise ValueError("training candidate shard payload fields must match exactly")
        raw_cases = payload["cases"]
        if not isinstance(raw_cases, (list, tuple)):
            raise ValueError("training candidate shard cases must be a sequence")
        cases: list[TrainingShardCase] = []
        case_fields = {"name", "metrics", "loss", "run_manifest_hashes"}
        for raw_case in raw_cases:
            if not isinstance(raw_case, Mapping) or set(raw_case) != case_fields:
                raise ValueError("training candidate shard case fields must match exactly")
            cases.append(
                TrainingShardCase(
                    name=raw_case["name"],
                    metrics=tuple(tuple(item) for item in raw_case["metrics"]),
                    loss=raw_case["loss"],
                    run_manifest_hashes=tuple(raw_case["run_manifest_hashes"]),
                )
            )
        shard = cls(
            schema_version=payload["schema_version"],
            parameters=tuple(tuple(item) for item in payload["parameters"]),
            target_report_hash=payload["target_report_hash"],
            model_identity=payload["model_identity"],
            repository_identity=payload["repository_identity"],
            simulation_seeds=tuple(payload["simulation_seeds"]),
            metric_identity=payload["metric_identity"],
            loss_identity=payload["loss_identity"],
            cases=tuple(cases),
        )
        declared = _validated_content_hash(
            payload["content_hash"],
            label="declared training candidate shard content hash",
        )
        if declared != shard.content_hash:
            raise ValueError(
                "declared training candidate shard content hash does not match payload"
            )
        return shard


def _target_report(value: TargetConstructionReport) -> TargetConstructionReport:
    if not isinstance(value, TargetConstructionReport):
        raise TypeError("training candidate shard requires TargetConstructionReport")
    if value.role is not ObservationPartitionRole.TRAIN:
        raise ValueError("training candidate shard requires train targets")
    if value.manifest is None:
        raise ValueError("training target report is missing a manifest")
    if not value.cases:
        raise ValueError("training target report requires at least one case")
    return value


def _candidate_sequence(
    values: Iterable[Iterable[tuple[str, float]]],
) -> tuple[ParameterTuple, ...]:
    candidates = tuple(_canonical_parameter_tuple(candidate) for candidate in values)
    if not candidates:
        raise ValueError("training shard assembly requires parameter candidates")
    if len(set(candidates)) != len(candidates):
        raise ValueError("training shard parameter candidates must be unique")
    schema = tuple(name for name, _ in candidates[0])
    if any(tuple(name for name, _ in candidate) != schema for candidate in candidates[1:]):
        raise ValueError("training shard parameter candidates must share one schema")
    return candidates


def evaluate_training_candidate(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    target_report: TargetConstructionReport,
    parameters: Iterable[tuple[str, float]],
    simulation_seeds: Iterable[int],
    extractor: object,
    loss: MetricLoss,
) -> TrainingCandidateShard:
    if not isinstance(runner, SimulationRunner):
        raise TypeError("training candidate evaluation requires SimulationRunner")
    report = _target_report(target_report)
    canonical_parameters = _canonical_parameter_tuple(parameters)
    seeds = _seeds(simulation_seeds)
    cases: list[TrainingShardCase] = []
    for case in report.cases:
        traces = runner.run_batch(
            model,
            case.scenario,
            dict(canonical_parameters),
            seeds=seeds,
        )
        if not traces or traces[0].parameters != canonical_parameters:
            raise RuntimeError("training candidate execution changed parameter metadata")
        if any(trace.parameters != canonical_parameters for trace in traces[1:]):
            raise RuntimeError("training candidate batch changed parameter metadata")
        run_hashes = tuple(
            required_manifest_hash(trace, label="training candidate trace")
            for trace in traces
        )
        metrics = aggregate_metrics(traces, extractor)
        case_loss = evaluate_metric_loss(loss, metrics, case.target_map)
        cases.append(
            TrainingShardCase(
                name=case.name,
                metrics=tuple(sorted(metrics.items())),
                loss=case_loss,
                run_manifest_hashes=run_hashes,
            )
        )
    return TrainingCandidateShard(
        parameters=canonical_parameters,
        target_report_hash=report.content_hash,
        model_identity=component_identity(model),
        repository_identity=runner.repository_identity.manifest_identity(),
        simulation_seeds=seeds,
        metric_identity=callable_identity(extractor),
        loss_identity=metric_loss_identity(loss),
        cases=tuple(cases),
    )


def assemble_training_fit_report(
    *,
    model: ModelSource,
    target_report: TargetConstructionReport,
    parameter_candidates: Iterable[Iterable[tuple[str, float]]],
    simulation_seeds: Iterable[int],
    extractor: object,
    loss: MetricLoss,
    shards: Iterable[TrainingCandidateShard],
) -> TrainingFitReport:
    report = _target_report(target_report)
    declared_candidates = _candidate_sequence(parameter_candidates)
    canonical_candidates = tuple(sorted(declared_candidates))
    seeds = _seeds(simulation_seeds)
    model_identity = component_identity(model)
    metric_identity = callable_identity(extractor)
    loss_identity = metric_loss_identity(loss)
    shard_values = tuple(shards)
    if any(not isinstance(shard, TrainingCandidateShard) for shard in shard_values):
        raise TypeError("training shard assembly requires TrainingCandidateShard values")
    if len(shard_values) != len(declared_candidates):
        raise ValueError("training shard set does not match declared candidate count")

    shard_by_parameters: dict[ParameterTuple, TrainingCandidateShard] = {}
    repository_identity: Mapping[str, object] | None = None
    expected_case_names = tuple(case.name for case in report.cases)
    for shard in shard_values:
        if shard.parameters in shard_by_parameters:
            raise ValueError("training shard set contains a duplicate candidate")
        shard_by_parameters[shard.parameters] = shard
        if shard.parameters not in declared_candidates:
            raise ValueError("training shard set contains an undeclared candidate")
        if shard.target_report_hash != report.content_hash:
            raise ValueError("training shard target-report identity changed")
        if shard.model_identity != model_identity:
            raise ValueError("training shard model identity changed")
        if shard.simulation_seeds != seeds:
            raise ValueError("training shard seed plan changed")
        if shard.metric_identity != metric_identity:
            raise ValueError("training shard metric identity changed")
        if shard.loss_identity != loss_identity:
            raise ValueError("training shard loss identity changed")
        if tuple(case.name for case in shard.cases) != expected_case_names:
            raise ValueError("training shard case coverage or order changed")
        if repository_identity is None:
            repository_identity = shard.repository_identity
        elif shard.repository_identity != repository_identity:
            raise ValueError("training shards disagree on repository identity")

    if set(shard_by_parameters) != set(declared_candidates):
        raise ValueError("training shard set is incomplete")

    calibrations: list[CalibrationResult] = []
    for case_index, target_case in enumerate(report.cases):
        candidate_evaluations: list[CandidateEvaluation] = []
        parent_hashes: list[str] = []
        for parameters in declared_candidates:
            shard_case = shard_by_parameters[parameters].cases[case_index]
            recalculated_loss = evaluate_metric_loss(
                loss,
                dict(shard_case.metrics),
                target_case.target_map,
            )
            if recalculated_loss != shard_case.loss:
                raise ValueError("training shard case loss does not match its metrics")
            candidate_evaluations.append(
                CandidateEvaluation(
                    parameters=parameters,
                    metrics=shard_case.metrics,
                    loss=shard_case.loss,
                    run_manifest_hashes=shard_case.run_manifest_hashes,
                )
            )
            parent_hashes.extend(shard_case.run_manifest_hashes)
        ranking = tuple(
            sorted(
                candidate_evaluations,
                key=lambda candidate: (candidate.loss, candidate.parameters),
            )
        )
        calibration_manifest = ExperimentManifest(
            stage=ExperimentStage.GRID_CALIBRATION,
            inputs={
                "model": model_identity,
                "scenario": scenario_identity(target_case.scenario),
                "parameter_grid": declared_candidates,
                "seeds": seeds,
                "metric": metric_identity,
                "target_hash": stable_content_hash(target_case.target_map),
                "weights_hash": None,
                "loss": loss_identity,
            },
            parent_hashes=tuple(parent_hashes),
        )
        calibrations.append(
            CalibrationResult(
                best=ranking[0],
                ranking=ranking,
                manifest=calibration_manifest,
            )
        )

    calibration_hashes = tuple(
        required_manifest_hash(calibration, label="training calibration")
        for calibration in calibrations
    )
    ranking_entries: list[TrainingCandidateFit] = []
    for parameters in canonical_candidates:
        case_fits: list[TrainingCaseFit] = []
        for target_case, calibration, calibration_hash in zip(
            report.cases,
            calibrations,
            calibration_hashes,
            strict=True,
        ):
            candidate = next(
                item for item in calibration.ranking if item.parameters == parameters
            )
            case_fits.append(
                TrainingCaseFit(
                    name=target_case.name,
                    loss=candidate.loss,
                    calibration_manifest_hash=calibration_hash,
                    run_manifest_hashes=candidate.run_manifest_hashes,
                )
            )
        losses = tuple(case_fit.loss for case_fit in case_fits)
        ranking_entries.append(
            TrainingCandidateFit(
                parameters=parameters,
                cases=tuple(case_fits),
                mean_loss=fmean(losses),
                worst_loss=max(losses),
            )
        )
    ranking = tuple(
        sorted(
            ranking_entries,
            key=lambda candidate: (
                candidate.mean_loss,
                candidate.worst_loss,
                candidate.parameters,
            ),
        )
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.TRAINING_TARGET_FIT,
        inputs={
            "model": model_identity,
            "target_report_hash": report.content_hash,
            "simulation_seeds": seeds,
            "metric": metric_identity,
            "loss": loss_identity,
            "parameter_candidates": canonical_candidates,
            "cases": tuple(
                {
                    "name": case.name,
                    "scenario": scenario_identity(case.scenario),
                    "target_hash": stable_content_hash(case.target_map),
                }
                for case in report.cases
            ),
        },
        parent_hashes=(report.manifest.content_hash,) + calibration_hashes,
    )
    return TrainingFitReport(
        target_report_hash=report.content_hash,
        ranking=ranking,
        manifest=manifest,
    )


__all__ = [
    "TRAINING_CANDIDATE_SHARD_SCHEMA_VERSION",
    "TrainingCandidateShard",
    "TrainingShardCase",
    "assemble_training_fit_report",
    "evaluate_training_candidate",
]
