from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
import math

from narrative_dynamics.contracts import Scenario, SimulatorModel
from narrative_dynamics.metrics import (
    MetricExtractor,
    aggregate_metrics,
    weighted_squared_error,
)
from narrative_dynamics.simulation import SimulationRunner


@dataclass(frozen=True)
class CandidateEvaluation:
    parameters: tuple[tuple[str, float], ...]
    metrics: tuple[tuple[str, float], ...]
    loss: float

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
    model: SimulatorModel,
    scenario: Scenario,
    parameter_grid: Mapping[str, Iterable[float]],
    seeds: Iterable[int],
    extractor: MetricExtractor,
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> CalibrationResult:
    """Exhaustively rank a finite parameter grid by trace-summary loss."""

    ordered_seeds = tuple(seeds)
    if not ordered_seeds:
        raise ValueError("calibration requires at least one simulation seed")

    evaluations: list[CandidateEvaluation] = []
    for parameters in _grid_candidates(parameter_grid):
        traces = runner.run_batch(
            model,
            scenario,
            dict(parameters),
            seeds=ordered_seeds,
        )
        metrics = aggregate_metrics(traces, extractor)
        loss = weighted_squared_error(metrics, target, weights=weights)
        evaluations.append(
            CandidateEvaluation(
                parameters=parameters,
                metrics=tuple(sorted(metrics.items())),
                loss=loss,
            )
        )

    ranking = tuple(
        sorted(
            evaluations,
            key=lambda candidate: (candidate.loss, candidate.parameters),
        )
    )
    return CalibrationResult(best=ranking[0], ranking=ranking)
