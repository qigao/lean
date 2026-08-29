from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from statistics import fmean

from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    stable_content_hash,
)
from narrative_dynamics.external_validation import (
    ExternalReleasePreflight,
    ExternalValidationPreregistration,
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
)
from narrative_dynamics.model_comparison import (
    ComparisonModel,
    ModelComparisonEntry,
    ModelComparisonReport,
)
from narrative_dynamics.observations.dataset import (
    ObservationPartitionRole,
    _freeze_mapping,
)
from narrative_dynamics.observations.preregistration import (
    FrozenModelSpec,
    PreregisteredEvaluationProtocol,
)
from narrative_dynamics.observations.release import (
    ReleasedModelComparisonReport,
    VerifiedProtocolRelease,
)
from narrative_dynamics.observations.targets import TargetConstructionReport
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.validation import (
    EvaluationRole,
    FinalTestReport,
    HeldOutCaseEvaluation,
    HeldOutValidationReport,
)


_NUMERIC_ZERO_ABS_TOL = 1e-15
_FRESH_SOURCE_LIFECYCLES = frozenset(
    {"fresh_per_batch", "fresh_process_per_run"}
)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise ValueError(f"{label} must be a sha256 content hash")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a sha256 content hash") from error
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


def _parameters(
    values: object,
    *,
    label: str,
) -> tuple[tuple[str, float], ...]:
    try:
        rows = tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be iterable") from error
    if not rows:
        raise ValueError(f"{label} must be non-empty")
    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(f"{label} rows must be name/value pairs")
        name = _text(row[0], label=f"{label} name")
        if name in seen:
            raise ValueError(f"{label} names must be unique")
        seen.add(name)
        canonical.append(
            (name, _finite(row[1], label=f"{label} value for {name!r}"))
        )
    return tuple(sorted(canonical))


def _metrics(
    values: object,
    *,
    label: str,
) -> tuple[tuple[str, float], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    try:
        rows = tuple(items)
    except TypeError as error:
        raise TypeError(f"{label} must be a mapping or iterable of pairs") from error
    if not rows:
        raise ValueError(f"{label} must be non-empty")
    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(f"{label} rows must be name/value pairs")
        name = _text(row[0], label=f"{label} name")
        if name in seen:
            raise ValueError(f"{label} names must be unique")
        seen.add(name)
        canonical.append(
            (name, _finite(row[1], label=f"{label} value for {name!r}"))
        )
    return tuple(sorted(canonical))


def _seeds(values: object) -> tuple[int, ...]:
    try:
        seeds = tuple(values)
    except TypeError as error:
        raise TypeError("external prediction seeds must be iterable") from error
    if not seeds:
        raise ValueError("external prediction seeds must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("external prediction seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("external prediction seeds must be unique")
    return seeds


def _canonical_loss(value: float) -> float:
    numeric = float(value)
    return (
        0.0
        if math.isclose(
            numeric,
            0.0,
            rel_tol=0.0,
            abs_tol=_NUMERIC_ZERO_ABS_TOL,
        )
        else numeric
    )


def _runtime_source_is_reusable(source: object) -> bool:
    if callable(getattr(source, "instantiate", None)):
        return True
    return getattr(source, "lifecycle", None) in _FRESH_SOURCE_LIFECYCLES


@dataclass(frozen=True)
class ExternalSeedPrediction:
    case_name: str
    scenario_id: str
    seed: int
    metrics: tuple[tuple[str, float], ...]
    run_manifest_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_name",
            _text(self.case_name, label="external prediction case name"),
        )
        object.__setattr__(
            self,
            "scenario_id",
            _text(self.scenario_id, label="external prediction scenario id"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("external prediction seed must be an integer")
        object.__setattr__(
            self,
            "metrics",
            _metrics(self.metrics, label="external prediction metrics"),
        )
        object.__setattr__(
            self,
            "run_manifest_hash",
            _hash(
                self.run_manifest_hash,
                label="external prediction run manifest hash",
            ),
        )

    @property
    def metric_map(self) -> dict[str, float]:
        return dict(self.metrics)

    def identity_payload(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "metrics": self.metrics,
            "run_manifest_hash": self.run_manifest_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ExternalModelPrediction:
    model_name: str
    frozen_model_hash: str
    parameters: tuple[tuple[str, float], ...]
    predictions: tuple[ExternalSeedPrediction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_name",
            _text(self.model_name, label="external prediction model name"),
        )
        object.__setattr__(
            self,
            "frozen_model_hash",
            _hash(
                self.frozen_model_hash,
                label="external prediction frozen model hash",
            ),
        )
        object.__setattr__(
            self,
            "parameters",
            _parameters(
                self.parameters,
                label="external prediction model parameters",
            ),
        )
        predictions = tuple(self.predictions)
        if not predictions:
            raise ValueError("external model prediction requires predictions")
        if any(
            not isinstance(item, ExternalSeedPrediction)
            for item in predictions
        ):
            raise TypeError(
                "external model predictions must be ExternalSeedPrediction values"
            )
        keys = tuple((item.case_name, item.seed) for item in predictions)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "external model prediction case/seed pairs must be unique"
            )
        metric_schema = tuple(name for name, _ in predictions[0].metrics)
        if any(
            tuple(name for name, _ in item.metrics) != metric_schema
            for item in predictions[1:]
        ):
            raise ValueError(
                "external model predictions must share one metric schema"
            )
        object.__setattr__(self, "predictions", predictions)

    @property
    def prediction_map(
        self,
    ) -> dict[tuple[str, int], ExternalSeedPrediction]:
        return {
            (item.case_name, item.seed): item
            for item in self.predictions
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "frozen_model_hash": self.frozen_model_hash,
            "parameters": self.parameters,
            "predictions": tuple(
                item.identity_payload() for item in self.predictions
            ),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class ExternalFinalPredictionArtifact:
    preregistration_hash: str
    preflight_hash: str
    brier_protocol_hash: str
    log_protocol_hash: str
    repository_revision: str
    dataset_hash: str
    final_partition_hash: str
    final_target_hash: str
    metric_identity: Mapping[str, object]
    simulation_seeds: tuple[int, ...]
    model_predictions: tuple[ExternalModelPrediction, ...]
    manifest: ExperimentManifest

    def __post_init__(self) -> None:
        for field_name, label in (
            ("preregistration_hash", "external prediction preregistration hash"),
            ("preflight_hash", "external prediction preflight hash"),
            ("brier_protocol_hash", "external prediction Brier protocol hash"),
            ("log_protocol_hash", "external prediction Log protocol hash"),
            ("dataset_hash", "external prediction dataset hash"),
            ("final_partition_hash", "external prediction final partition hash"),
            ("final_target_hash", "external prediction final target hash"),
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=label),
            )
        object.__setattr__(
            self,
            "repository_revision",
            _text(
                self.repository_revision,
                label="external prediction repository revision",
            ),
        )
        object.__setattr__(
            self,
            "metric_identity",
            _freeze_mapping(
                self.metric_identity,
                label="external prediction metric identity",
            ),
        )
        object.__setattr__(
            self,
            "simulation_seeds",
            _seeds(self.simulation_seeds),
        )
        models = tuple(
            sorted(tuple(self.model_predictions), key=lambda item: item.model_name)
        )
        if not models:
            raise ValueError("external prediction artifact requires model predictions")
        if any(not isinstance(item, ExternalModelPrediction) for item in models):
            raise TypeError(
                "external prediction artifact models must be ExternalModelPrediction values"
            )
        names = tuple(item.model_name for item in models)
        if len(set(names)) != len(names):
            raise ValueError("external prediction artifact model names must be unique")
        coverage = tuple(
            (item.case_name, item.scenario_id, item.seed)
            for item in models[0].predictions
        )
        metric_schema = tuple(
            name for name, _ in models[0].predictions[0].metrics
        )
        for model in models[1:]:
            if tuple(
                (item.case_name, item.scenario_id, item.seed)
                for item in model.predictions
            ) != coverage:
                raise ValueError(
                    "external prediction artifact models must share exact case/seed coverage"
                )
            if tuple(
                name for name, _ in model.predictions[0].metrics
            ) != metric_schema:
                raise ValueError(
                    "external prediction artifact models must share one metric schema"
                )
        object.__setattr__(self, "model_predictions", models)
        if not isinstance(self.manifest, ExperimentManifest):
            raise TypeError(
                "external prediction artifact manifest must be ExperimentManifest"
            )
        if self.manifest.stage is not ExperimentStage.EXTERNAL_PREDICTION:
            raise ValueError(
                "external prediction artifact manifest stage must be external_prediction"
            )

    @property
    def model_prediction_map(self) -> dict[str, ExternalModelPrediction]:
        return {item.model_name: item for item in self.model_predictions}

    def identity_payload(self) -> dict[str, object]:
        return {
            "preregistration_hash": self.preregistration_hash,
            "preflight_hash": self.preflight_hash,
            "brier_protocol_hash": self.brier_protocol_hash,
            "log_protocol_hash": self.log_protocol_hash,
            "repository_revision": self.repository_revision,
            "dataset_hash": self.dataset_hash,
            "final_partition_hash": self.final_partition_hash,
            "final_target_hash": self.final_target_hash,
            "metric_identity": self.metric_identity,
            "simulation_seeds": self.simulation_seeds,
            "model_predictions": tuple(
                item.identity_payload() for item in self.model_predictions
            ),
            "manifest_hash": self.manifest.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _require_target(
    target_set: TargetConstructionReport,
    protocol: PreregisteredEvaluationProtocol,
) -> None:
    if not isinstance(target_set, TargetConstructionReport):
        raise TypeError("external prediction requires constructed final targets")
    if target_set.role is not ObservationPartitionRole.FINAL_TEST:
        raise ValueError("external prediction requires final-test targets")
    if target_set.dataset_hash != protocol.dataset_hash:
        raise ValueError("external prediction dataset changed")
    if target_set.partition_hash != protocol.final_partition_hash:
        raise ValueError("external prediction final partition changed")
    if target_set.spec_hash != protocol.target_spec_hash:
        raise ValueError("external prediction target spec changed")
    if target_set.content_hash != protocol.final_target_hash:
        raise ValueError("external prediction target payload changed")
    if target_set.manifest.content_hash != protocol.final_target_manifest_hash:
        raise ValueError("external prediction target lineage changed")


def _require_sibling_protocols(
    preregistration: ExternalValidationPreregistration,
    preflight: ExternalReleasePreflight,
    brier_protocol: PreregisteredEvaluationProtocol,
    log_protocol: PreregisteredEvaluationProtocol,
    final_targets: TargetConstructionReport,
    extractor: object,
) -> None:
    if not isinstance(preregistration, ExternalValidationPreregistration):
        raise TypeError("external prediction requires P3 preregistration")
    if not isinstance(preflight, ExternalReleasePreflight):
        raise TypeError("external prediction requires external release preflight")
    if not isinstance(brier_protocol, PreregisteredEvaluationProtocol):
        raise TypeError("external prediction Brier protocol is invalid")
    if not isinstance(log_protocol, PreregisteredEvaluationProtocol):
        raise TypeError("external prediction Log protocol is invalid")

    if preflight.preregistration_hash != preregistration.content_hash:
        raise ValueError("external prediction preflight preregistration changed")
    if preregistration.brier_protocol_hash != brier_protocol.content_hash:
        raise ValueError("external prediction Brier preregistration binding changed")
    if preregistration.log_protocol_hash != log_protocol.content_hash:
        raise ValueError("external prediction Log preregistration binding changed")
    if preflight.brier_protocol_hash != brier_protocol.content_hash:
        raise ValueError("external prediction Brier preflight binding changed")
    if preflight.log_protocol_hash != log_protocol.content_hash:
        raise ValueError("external prediction Log preflight binding changed")

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
            raise ValueError(
                f"external prediction sibling protocols disagree on {field_name}"
            )
    if tuple(
        candidate.content_hash for candidate in brier_protocol.candidates
    ) != tuple(
        candidate.content_hash for candidate in log_protocol.candidates
    ):
        raise ValueError(
            "external prediction sibling protocols disagree on frozen candidates"
        )

    _require_target(final_targets, brier_protocol)
    _require_target(final_targets, log_protocol)
    if callable_identity(extractor) != dict(brier_protocol.metric_identity):
        raise ValueError("external prediction metric extractor changed")
    if callable_identity(extractor) != dict(log_protocol.metric_identity):
        raise ValueError("external prediction sibling metric extractor changed")


def _resolve_models(
    models: tuple[ComparisonModel, ...],
    protocol: PreregisteredEvaluationProtocol,
) -> tuple[ComparisonModel, ...]:
    resolved = tuple(sorted(tuple(models), key=lambda item: item.frozen.name))
    if not resolved:
        raise ValueError("external prediction requires models")
    if any(not isinstance(item, ComparisonModel) for item in resolved):
        raise TypeError(
            "external prediction models must be ComparisonModel values"
        )
    if tuple(item.frozen.name for item in resolved) != protocol.candidate_names:
        raise ValueError(
            "external prediction requires exactly the preregistered candidates"
        )

    frozen_by_name = {
        candidate.name: candidate for candidate in protocol.candidates
    }
    seen_runtime_sources: set[int] = set()
    for item in resolved:
        if not _runtime_source_is_reusable(item.model):
            source_identity = id(item.model)
            if source_identity in seen_runtime_sources:
                raise ValueError(
                    "one mutable runtime model instance cannot serve multiple external candidates"
                )
            seen_runtime_sources.add(source_identity)
        expected = frozen_by_name[item.frozen.name]
        if item.frozen.content_hash != expected.content_hash:
            raise ValueError(
                f"external prediction frozen candidate changed for {item.frozen.name!r}"
            )
        if component_identity(item.model) != dict(expected.model_identity):
            raise ValueError(
                f"external prediction runtime model identity changed for {item.frozen.name!r}"
            )
    return resolved


def _extract_metrics(extractor: object, trace: object) -> tuple[tuple[str, float], ...]:
    if not callable(extractor):
        raise TypeError("external prediction metric extractor must be callable")
    observed = extractor(trace)
    if not isinstance(observed, Mapping):
        raise TypeError("external prediction metric extractor must return a mapping")
    return _metrics(observed, label="external prediction extracted metrics")


def predict_external_final_once(
    *,
    runner: SimulationRunner,
    preregistration: ExternalValidationPreregistration,
    preflight: ExternalReleasePreflight,
    brier_protocol: PreregisteredEvaluationProtocol,
    log_protocol: PreregisteredEvaluationProtocol,
    models: tuple[ComparisonModel, ...],
    final_targets: TargetConstructionReport,
    extractor: object,
    repository_revision: str,
) -> ExternalFinalPredictionArtifact:
    if not isinstance(runner, SimulationRunner):
        raise TypeError("external prediction requires SimulationRunner")
    revision = _text(
        repository_revision,
        label="external prediction repository revision",
    )
    _require_sibling_protocols(
        preregistration,
        preflight,
        brier_protocol,
        log_protocol,
        final_targets,
        extractor,
    )
    resolved = _resolve_models(models, brier_protocol)

    model_predictions: list[ExternalModelPrediction] = []
    run_hashes: list[str] = []
    expected_metric_schema: tuple[str, ...] | None = None
    for item in resolved:
        predictions: list[ExternalSeedPrediction] = []
        for case in final_targets.cases:
            for seed in brier_protocol.simulation_seeds:
                trace = runner.run_once(
                    item.model,
                    case.scenario,
                    dict(item.frozen.parameters),
                    seed=seed,
                )
                if trace.scenario_id != case.scenario.id:
                    raise RuntimeError("external prediction trace scenario identity changed")
                if trace.seed != seed:
                    raise RuntimeError("external prediction trace seed changed")
                if trace.parameters != item.frozen.parameters:
                    raise RuntimeError("external prediction trace parameters changed")
                metrics = _extract_metrics(extractor, trace)
                metric_schema = tuple(name for name, _ in metrics)
                if expected_metric_schema is None:
                    expected_metric_schema = metric_schema
                elif metric_schema != expected_metric_schema:
                    raise ValueError(
                        "external prediction metric schema changed across runs"
                    )
                run_hash = required_manifest_hash(
                    trace,
                    label="external prediction trace",
                )
                run_hashes.append(run_hash)
                predictions.append(
                    ExternalSeedPrediction(
                        case_name=case.name,
                        scenario_id=case.scenario.id,
                        seed=seed,
                        metrics=metrics,
                        run_manifest_hash=run_hash,
                    )
                )
        model_predictions.append(
            ExternalModelPrediction(
                model_name=item.frozen.name,
                frozen_model_hash=item.frozen.content_hash,
                parameters=item.frozen.parameters,
                predictions=tuple(predictions),
            )
        )

    canonical_models = tuple(model_predictions)
    manifest = ExperimentManifest(
        stage=ExperimentStage.EXTERNAL_PREDICTION,
        inputs={
            "preregistration_hash": preregistration.content_hash,
            "preflight_hash": preflight.content_hash,
            "preflight": preflight.identity_payload(),
            "brier_protocol_hash": brier_protocol.content_hash,
            "log_protocol_hash": log_protocol.content_hash,
            "repository_revision": revision,
            "dataset_hash": brier_protocol.dataset_hash,
            "final_partition_hash": brier_protocol.final_partition_hash,
            "final_target_hash": final_targets.content_hash,
            "final_target_manifest_hash": final_targets.manifest.content_hash,
            "metric_identity": brier_protocol.metric_identity,
            "simulation_seeds": brier_protocol.simulation_seeds,
            "models": tuple(
                {
                    "name": item.model_name,
                    "frozen_model_hash": item.frozen_model_hash,
                    "parameters": item.parameters,
                }
                for item in canonical_models
            ),
            "cases": tuple(
                {
                    "name": case.name,
                    "scenario_id": case.scenario.id,
                    "scenario_hash": case.scenario.content_hash,
                }
                for case in final_targets.cases
            ),
        },
        parent_hashes=(
            tuple(run_hashes)
            + (final_targets.manifest.content_hash,)
        ),
    )
    return ExternalFinalPredictionArtifact(
        preregistration_hash=preregistration.content_hash,
        preflight_hash=preflight.content_hash,
        brier_protocol_hash=brier_protocol.content_hash,
        log_protocol_hash=log_protocol.content_hash,
        repository_revision=revision,
        dataset_hash=brier_protocol.dataset_hash,
        final_partition_hash=brier_protocol.final_partition_hash,
        final_target_hash=final_targets.content_hash,
        metric_identity=brier_protocol.metric_identity,
        simulation_seeds=brier_protocol.simulation_seeds,
        model_predictions=canonical_models,
        manifest=manifest,
    )


def _require_artifact_manifest(
    artifact: ExternalFinalPredictionArtifact,
) -> Mapping[str, object]:
    if artifact.manifest.stage is not ExperimentStage.EXTERNAL_PREDICTION:
        raise ValueError("external prediction artifact manifest stage changed")
    inputs = artifact.manifest.inputs
    for field_name in (
        "preregistration_hash",
        "preflight_hash",
        "brier_protocol_hash",
        "log_protocol_hash",
        "repository_revision",
        "dataset_hash",
        "final_partition_hash",
        "final_target_hash",
        "metric_identity",
        "simulation_seeds",
    ):
        if inputs.get(field_name) != getattr(artifact, field_name):
            raise ValueError(
                f"external prediction artifact manifest {field_name} changed"
            )
    preflight = inputs.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("external prediction artifact preflight binding is missing")
    return preflight


def _require_artifact_for_scoring(
    *,
    artifact: ExternalFinalPredictionArtifact,
    verified_release: VerifiedProtocolRelease,
    protocol: PreregisteredEvaluationProtocol,
    target_set: TargetConstructionReport,
    loss: MetricLoss,
) -> tuple[tuple[FrozenModelSpec, ExternalModelPrediction], ...]:
    if not isinstance(artifact, ExternalFinalPredictionArtifact):
        raise TypeError(
            "external prediction scoring requires ExternalFinalPredictionArtifact"
        )
    if not isinstance(verified_release, VerifiedProtocolRelease):
        raise TypeError(
            "external prediction scoring requires VerifiedProtocolRelease"
        )
    if not isinstance(protocol, PreregisteredEvaluationProtocol):
        raise TypeError(
            "external prediction scoring requires PreregisteredEvaluationProtocol"
        )
    verified_release.require_matches(protocol)
    preflight = _require_artifact_manifest(artifact)

    if protocol.content_hash == artifact.brier_protocol_hash:
        release_field = "brier_release_hash"
        verification_field = "brier_verification_hash"
    elif protocol.content_hash == artifact.log_protocol_hash:
        release_field = "log_release_hash"
        verification_field = "log_verification_hash"
    else:
        raise ValueError(
            "external prediction artifact does not bind the scoring protocol"
        )
    if preflight.get(release_field) != verified_release.release_hash:
        raise ValueError(
            "external prediction verified release does not match sealed preflight"
        )
    if preflight.get(verification_field) != verified_release.content_hash:
        raise ValueError(
            "external prediction verification does not match sealed preflight"
        )

    _require_target(target_set, protocol)
    if artifact.dataset_hash != protocol.dataset_hash:
        raise ValueError("external prediction artifact dataset changed")
    if artifact.final_partition_hash != protocol.final_partition_hash:
        raise ValueError("external prediction artifact final partition changed")
    if artifact.final_target_hash != target_set.content_hash:
        raise ValueError("external prediction artifact final target changed")
    if artifact.metric_identity != protocol.metric_identity:
        raise ValueError("external prediction artifact metric identity changed")
    if artifact.simulation_seeds != protocol.simulation_seeds:
        raise ValueError("external prediction artifact simulation seeds changed")
    if metric_loss_identity(loss) != dict(protocol.loss_identity):
        raise ValueError("external prediction scoring loss changed")

    artifact_models = artifact.model_prediction_map
    if tuple(sorted(artifact_models)) != protocol.candidate_names:
        raise ValueError(
            "external prediction artifact candidate set changed"
        )
    expected_cases = tuple(
        (case.name, case.scenario.id) for case in target_set.cases
    )
    expected_coverage = tuple(
        (case_name, scenario_id, seed)
        for case_name, scenario_id in expected_cases
        for seed in protocol.simulation_seeds
    )
    resolved: list[tuple[FrozenModelSpec, ExternalModelPrediction]] = []
    for frozen in protocol.candidates:
        model = artifact_models[frozen.name]
        if model.frozen_model_hash != frozen.content_hash:
            raise ValueError(
                f"external prediction frozen candidate changed for {frozen.name!r}"
            )
        if model.parameters != frozen.parameters:
            raise ValueError(
                f"external prediction parameters changed for {frozen.name!r}"
            )
        actual_coverage = tuple(
            (item.case_name, item.scenario_id, item.seed)
            for item in model.predictions
        )
        if actual_coverage != expected_coverage:
            raise ValueError(
                f"external prediction coverage changed for {frozen.name!r}"
            )
        for prediction in model.predictions:
            if prediction.run_manifest_hash not in artifact.manifest.parent_hashes:
                raise ValueError(
                    "external prediction run lineage is not sealed by the artifact manifest"
                )
        resolved.append((frozen, model))
    return tuple(resolved)


def score_external_prediction_artifact(
    *,
    artifact: ExternalFinalPredictionArtifact,
    verified_release: VerifiedProtocolRelease,
    protocol: PreregisteredEvaluationProtocol,
    target_set: TargetConstructionReport,
    loss: MetricLoss,
) -> ReleasedModelComparisonReport:
    resolved = _require_artifact_for_scoring(
        artifact=artifact,
        verified_release=verified_release,
        protocol=protocol,
        target_set=target_set,
        loss=loss,
    )
    target_by_name = {case.name: case for case in target_set.cases}

    raw_entries: list[tuple[FrozenModelSpec, FinalTestReport]] = []
    for frozen, model in resolved:
        prediction_map = model.prediction_map
        case_evaluations: list[HeldOutCaseEvaluation] = []
        for case in target_set.cases:
            sealed = tuple(
                prediction_map[(case.name, seed)]
                for seed in protocol.simulation_seeds
            )
            metric_names = tuple(name for name, _ in sealed[0].metrics)
            if any(
                tuple(name for name, _ in item.metrics) != metric_names
                for item in sealed[1:]
            ):
                raise ValueError(
                    f"external prediction metric schema changed for case {case.name!r}"
                )
            averaged = {
                name: fmean(item.metric_map[name] for item in sealed)
                for name in metric_names
            }
            case_loss = _canonical_loss(
                evaluate_metric_loss(
                    loss,
                    averaged,
                    target_by_name[case.name].target_map,
                )
            )
            case_evaluations.append(
                HeldOutCaseEvaluation(
                    name=case.name,
                    scenario_id=case.scenario.id,
                    metrics=tuple(sorted(averaged.items())),
                    target=tuple(sorted(case.target_map.items())),
                    loss=case_loss,
                    run_manifest_hashes=tuple(
                        item.run_manifest_hash for item in sealed
                    ),
                )
            )

        losses = tuple(item.loss for item in case_evaluations)
        validation_manifest = ExperimentManifest(
            stage=ExperimentStage.HELD_OUT_VALIDATION,
            inputs={
                "prediction_artifact_hash": artifact.content_hash,
                "model_name": frozen.name,
                "frozen_model_hash": frozen.content_hash,
                "parameters": frozen.parameters,
                "dataset_hash": protocol.dataset_hash,
                "final_partition_hash": protocol.final_partition_hash,
                "final_target_hash": protocol.final_target_hash,
                "metric_identity": protocol.metric_identity,
                "loss_identity": protocol.loss_identity,
                "simulation_seeds": protocol.simulation_seeds,
                "cases": tuple(case.name for case in target_set.cases),
            },
            parent_hashes=(artifact.manifest.content_hash,),
        )
        validation = HeldOutValidationReport(
            parameters=frozen.parameters,
            cases=tuple(case_evaluations),
            mean_loss=fmean(losses),
            worst_loss=max(losses),
            manifest=validation_manifest,
        )
        final_manifest = ExperimentManifest(
            stage=ExperimentStage.FINAL_TEST,
            inputs={
                "suite_name": f"{protocol.name}:final_test",
                "role": EvaluationRole.FINAL_TEST.value,
                "parameters": frozen.parameters,
                "loss": protocol.loss_identity,
                "prediction_artifact_hash": artifact.content_hash,
            },
            parent_hashes=(validation_manifest.content_hash,),
        )
        final_test = FinalTestReport(
            suite_name=f"{protocol.name}:final_test",
            role=EvaluationRole.FINAL_TEST,
            validation=validation,
            manifest=final_manifest,
        )
        raw_entries.append((frozen, final_test))

    baseline_report = next(
        report
        for frozen, report in raw_entries
        if frozen.name == protocol.baseline_name
    )
    baseline_mean = baseline_report.validation.mean_loss
    baseline_worst = baseline_report.validation.worst_loss
    entries = tuple(
        ModelComparisonEntry(
            name=frozen.name,
            parameters=frozen.parameters,
            final_test=report,
            mean_loss=report.validation.mean_loss,
            worst_loss=report.validation.worst_loss,
            delta_mean_from_baseline=report.validation.mean_loss - baseline_mean,
            delta_worst_from_baseline=report.validation.worst_loss - baseline_worst,
            adequate=(
                report.validation.mean_loss
                <= protocol.thresholds.max_mean_loss
                and report.validation.worst_loss
                <= protocol.thresholds.max_worst_loss
            ),
        )
        for frozen, report in raw_entries
    )
    ranking = tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.mean_loss,
                entry.worst_loss,
                entry.name,
            ),
        )
    )
    comparison_manifest = ExperimentManifest(
        stage=ExperimentStage.MODEL_COMPARISON,
        inputs={
            "protocol_hash": protocol.content_hash,
            "declared_precommitment_hash": protocol.declared_precommitment_hash,
            "dataset_hash": protocol.dataset_hash,
            "final_partition_hash": protocol.final_partition_hash,
            "final_target_hash": protocol.final_target_hash,
            "final_target_manifest_hash": protocol.final_target_manifest_hash,
            "metric_identity": protocol.metric_identity,
            "loss_identity": protocol.loss_identity,
            "simulation_seeds": protocol.simulation_seeds,
            "baseline_name": protocol.baseline_name,
            "numeric_zero_abs_tol": _NUMERIC_ZERO_ABS_TOL,
            "ranking_rule": ("mean_loss", "worst_loss", "model_name"),
            "prediction_artifact_hash": artifact.content_hash,
            "entries": tuple(
                {
                    "name": entry.name,
                    "parameters": entry.parameters,
                    "mean_loss": entry.mean_loss,
                    "worst_loss": entry.worst_loss,
                    "delta_mean_from_baseline": entry.delta_mean_from_baseline,
                    "delta_worst_from_baseline": entry.delta_worst_from_baseline,
                    "adequate": entry.adequate,
                }
                for entry in ranking
            ),
        },
        parent_hashes=(
            tuple(
                required_manifest_hash(
                    entry.final_test,
                    label=f"precomputed comparison {entry.name!r} final test",
                )
                for entry in ranking
            )
            + (artifact.manifest.content_hash,)
        ),
    )
    comparison = ModelComparisonReport(
        baseline_name=protocol.baseline_name,
        ranking=ranking,
        manifest=comparison_manifest,
    )
    released_manifest = ExperimentManifest(
        stage=ExperimentStage.RELEASED_MODEL_COMPARISON,
        inputs={
            "verified_release_hash": verified_release.content_hash,
            "release_hash": verified_release.release_hash,
            "protocol_hash": protocol.content_hash,
            "verifier_identity": verified_release.verifier_identity,
            "verified_receipt_hashes": verified_release.verified_receipt_hashes,
            "comparison_manifest_hash": comparison.manifest.content_hash,
            "prediction_artifact_hash": artifact.content_hash,
        },
        parent_hashes=(comparison.manifest.content_hash,),
    )
    return ReleasedModelComparisonReport(
        release_hash=verified_release.release_hash,
        verification_hash=verified_release.content_hash,
        comparison=comparison,
        manifest=released_manifest,
    )


def _install_task9_import_seam() -> None:
    """Expose an intentionally non-functional Task 9 symbol for staged RED tests."""

    import narrative_dynamics.external_validation as external_validation

    if hasattr(external_validation, "assemble_external_final_from_reports"):
        return

    def assemble_external_final_from_reports(**_kwargs):
        raise NotImplementedError(
            "precomputed external-final assembly is implemented in Task 9"
        )

    external_validation.assemble_external_final_from_reports = (
        assemble_external_final_from_reports
    )


_install_task9_import_seam()


__all__ = [
    "ExternalFinalPredictionArtifact",
    "ExternalModelPrediction",
    "ExternalSeedPrediction",
    "predict_external_final_once",
    "score_external_prediction_artifact",
]
