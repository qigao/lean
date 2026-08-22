from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from narrative_dynamics.calibration import CalibrationResult, calibrate_grid
from narrative_dynamics.contracts import Scenario, SimulatorModel
from narrative_dynamics.metrics import MetricExtractor, aggregate_metrics
from narrative_dynamics.simulation import SimulationRunner


@dataclass(frozen=True)
class SyntheticRecoveryReport:
    true_parameters: tuple[tuple[str, float], ...]
    target_metrics: tuple[tuple[str, float], ...]
    calibration: CalibrationResult
    recovered: bool


def synthetic_recovery(
    *,
    runner: SimulationRunner,
    model: SimulatorModel,
    scenario: Scenario,
    true_parameters: Mapping[str, float],
    parameter_grid: Mapping[str, Iterable[float]],
    observation_seeds: Iterable[int],
    calibration_seeds: Iterable[int],
    extractor: MetricExtractor,
    weights: Mapping[str, float] | None = None,
) -> SyntheticRecoveryReport:
    """Generate synthetic targets and test finite-grid parameter recovery."""

    observed_seed_tuple = tuple(observation_seeds)
    calibrated_seed_tuple = tuple(calibration_seeds)
    if not observed_seed_tuple:
        raise ValueError("synthetic recovery requires observation seeds")
    if not calibrated_seed_tuple:
        raise ValueError("synthetic recovery requires calibration seeds")

    observed_traces = runner.run_batch(
        model,
        scenario,
        true_parameters,
        seeds=observed_seed_tuple,
    )
    target = aggregate_metrics(observed_traces, extractor)
    calibration = calibrate_grid(
        runner=runner,
        model=model,
        scenario=scenario,
        parameter_grid=parameter_grid,
        seeds=calibrated_seed_tuple,
        extractor=extractor,
        target=target,
        weights=weights,
    )
    canonical_true = observed_traces[0].parameters
    return SyntheticRecoveryReport(
        true_parameters=canonical_true,
        target_metrics=tuple(sorted(target.items())),
        calibration=calibration,
        recovered=calibration.best.parameters == canonical_true,
    )
