from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
import math

from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    Scenario,
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
    stable_content_hash,
)
from narrative_dynamics.metrics import MetricExtractor, aggregate_metrics
from narrative_dynamics.simulation import ModelSource, SimulationRunner


@dataclass(frozen=True)
class CandidateEvaluation:
    parameters: tuple[tuple[str, float], ...]
    metrics: tuple[tuple[str, float], ...]
    loss: float
    run_manifest_hashes: tuple[str, ...] = ()

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)

    @property
    def metric_map(self) -> dict[str, float]:
        return dict(self.metrics)


@dataclass(frozen=True)
class CalibrationResult:
    best: CandidateEvaluation
    ranking: tuple[CandidateEvaluation, ...]
    manifest: ExperimentManifest | None = None


def _grid_candidates(
    parameter_grid: Mapping[str, Iterable[float]],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    if not parameter_grid:
        raise ValueError("parameter grid must contain at least one dimension")

    names = tuple(sorted(parameter_grid))
    dimensions: list[tuple[float, ...]] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be non-empty strings")
        values: list[float] = []
        for raw_value in parameter_grid[name]:
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"grid value for {name!r} must be numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"grid value for {name!r} must be finite")
            values.append(value)
        if not values:
            raise ValueError(f"parameter grid dimension {name!r} must be non-empty")
        dimensions.append(tuple(values))

    return tuple(
        tuple(zip(names, combination, strict=True))
        for combination in product(*dimensions)
    )


def calibrate_grid(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    scenario: Scenario,
    parameter_grid: Mapping[str, Iterable[float]],
    seeds: Iterable[int],
    extractor: MetricExtractor,
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    loss: MetricLoss | None = None,
) -> CalibrationResult:
    """Exhaustively rank a finite parameter grid by a declared metric loss."""

    ordered_seeds = tuple(seeds)
    if not ordered_seeds:
        raise ValueError("calibration requires at least one simulation seed")

    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
    loss_identity = metric_loss_identity(selected_loss)
    candidates = _grid_candidates(parameter_grid)
    evaluations: list[CandidateEvaluation] = []
    run_parent_hashes: list[str] = []
    for parameters in candidates:
        traces = runner.run_batch(
            model,
            scenario,
            dict(parameters),
            seeds=ordered_seeds,
        )
        run_hashes = tuple(
            required_manifest_hash(trace, label="calibration trace")
            for trace in traces
        )
        run_parent_hashes.extend(run_hashes)
        metrics = aggregate_metrics(traces, extractor)
        candidate_loss = evaluate_metric_loss(
            selected_loss,
            metrics,
            target,
            weights=weights,
        )
        evaluations.append(
            CandidateEvaluation(
                parameters=parameters,
                metrics=tuple(sorted(metrics.items())),
                loss=candidate_loss,
                run_manifest_hashes=run_hashes,
            )
        )

    ranking = tuple(
        sorted(
            evaluations,
            key=lambda candidate: (candidate.loss, candidate.parameters),
        )
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.GRID_CALIBRATION,
        inputs={
            "model": component_identity(model),
            "scenario": scenario_identity(scenario),
            "parameter_grid": candidates,
            "seeds": ordered_seeds,
            "metric": callable_identity(extractor),
            "target_hash": stable_content_hash(target),
            "weights_hash": (
                None if weights is None else stable_content_hash(weights)
            ),
            "loss": loss_identity,
        },
        parent_hashes=tuple(run_parent_hashes),
    )
    return CalibrationResult(
        best=ranking[0],
        ranking=ranking,
        manifest=manifest,
    )
