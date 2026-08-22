from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean

from narrative_dynamics.calibration import CalibrationResult, calibrate_grid
from narrative_dynamics.contracts import Scenario, SimulatorModel
from narrative_dynamics.metrics import (
    MetricExtractor,
    aggregate_metrics,
    weighted_squared_error,
)
from narrative_dynamics.simulation import SimulationRunner


@dataclass(frozen=True)
class SyntheticRecoveryReport:
    true_parameters: tuple[tuple[str, float], ...]
    target_metrics: tuple[tuple[str, float], ...]
    calibration: CalibrationResult
    recovered: bool


@dataclass(frozen=True)
class HeldOutCase:
    """One external scenario and target excluded from parameter calibration.

    ``name`` is an experiment label. When omitted, the stable scenario id is
    used. Keeping both fields lets multiple validation suites describe cases
    without weakening the requirement that scenario ids remain unique.
    """

    scenario: Scenario
    seeds: tuple[int, ...]
    target: Mapping[str, float]
    weights: Mapping[str, float] | None = None
    name: str | None = None

    @property
    def effective_name(self) -> str:
        return self.scenario.id if self.name is None else self.name


@dataclass(frozen=True)
class HeldOutCaseEvaluation:
    name: str
    scenario_id: str
    metrics: tuple[tuple[str, float], ...]
    target: tuple[tuple[str, float], ...]
    loss: float

    @property
    def metric_map(self) -> dict[str, float]:
        return dict(self.metrics)

    @property
    def target_map(self) -> dict[str, float]:
        return dict(self.target)


@dataclass(frozen=True)
class HeldOutValidationReport:
    parameters: tuple[tuple[str, float], ...]
    cases: tuple[HeldOutCaseEvaluation, ...]
    mean_loss: float
    worst_loss: float

    @property
    def max_loss(self) -> float:
        """Alias emphasizing that the report retains the worst held-out case."""

        return self.worst_loss


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


def validate_held_out(
    *,
    runner: SimulationRunner,
    model: SimulatorModel,
    parameters: Mapping[str, float],
    cases: Iterable[HeldOutCase],
    extractor: MetricExtractor,
) -> HeldOutValidationReport:
    """Evaluate fixed parameters on scenarios excluded from calibration.

    The function never modifies or re-calibrates ``parameters``. It reports
    each case, the mean loss, and the worst loss so a single failing world
    cannot be hidden by an aggregate score.
    """

    held_out_cases = tuple(cases)
    if not held_out_cases:
        raise ValueError("held-out validation requires at least one case")

    scenario_ids = tuple(case.scenario.id for case in held_out_cases)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("held-out scenario ids must be unique")

    names = tuple(case.effective_name for case in held_out_cases)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("held-out case names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("held-out case names must be unique")

    evaluations: list[HeldOutCaseEvaluation] = []
    canonical_parameters: tuple[tuple[str, float], ...] | None = None
    for case in held_out_cases:
        seeds = tuple(case.seeds)
        if not seeds:
            raise ValueError("every held-out case must contain at least one seed")

        traces = runner.run_batch(
            model,
            case.scenario,
            parameters,
            seeds=seeds,
        )
        if canonical_parameters is None:
            canonical_parameters = traces[0].parameters
        elif traces[0].parameters != canonical_parameters:
            raise RuntimeError("held-out validation changed its parameter metadata")

        metrics = aggregate_metrics(traces, extractor)
        loss = weighted_squared_error(
            metrics,
            case.target,
            weights=case.weights,
        )
        evaluations.append(
            HeldOutCaseEvaluation(
                name=case.effective_name,
                scenario_id=case.scenario.id,
                metrics=tuple(sorted(metrics.items())),
                target=tuple(
                    sorted((name, float(value)) for name, value in case.target.items())
                ),
                loss=loss,
            )
        )

    assert canonical_parameters is not None
    losses = tuple(evaluation.loss for evaluation in evaluations)
    return HeldOutValidationReport(
        parameters=canonical_parameters,
        cases=tuple(evaluations),
        mean_loss=fmean(losses),
        worst_loss=max(losses),
    )
