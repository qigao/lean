from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from narrative_dynamics.contracts import Scenario, SimulatorModel
from narrative_dynamics.metrics import MetricExtractor, aggregate_metrics
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.validation import (
    HeldOutCase as ValidationHeldOutCase,
    validate_held_out as validate_held_out_cases,
)


@dataclass(frozen=True)
class HeldOutCase:
    """Named out-of-sample case used by diagnostic reports."""

    name: str
    scenario: Scenario
    seeds: tuple[int, ...]
    target: Mapping[str, float]
    weights: Mapping[str, float] | None = None


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
    max_loss: float

    @property
    def worst_loss(self) -> float:
        return self.max_loss


def validate_held_out(
    *,
    runner: SimulationRunner,
    model: SimulatorModel,
    parameters: Mapping[str, float],
    cases: Iterable[HeldOutCase],
    extractor: MetricExtractor,
) -> HeldOutValidationReport:
    """Evaluate fixed parameters on independent named scenarios.

    The underlying execution and loss semantics are delegated to the existing
    validation module. This layer only adds stable case names and a report
    shape suitable for cross-scenario diagnostics.
    """

    named_cases = tuple(cases)
    if not named_cases:
        raise ValueError("held-out diagnostics require at least one case")

    names = tuple(case.name for case in named_cases)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("held-out diagnostic names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("held-out diagnostic names must be unique")

    base_report = validate_held_out_cases(
        runner=runner,
        model=model,
        parameters=parameters,
        cases=tuple(
            ValidationHeldOutCase(
                scenario=case.scenario,
                seeds=tuple(case.seeds),
                target=case.target,
                weights=case.weights,
            )
            for case in named_cases
        ),
        extractor=extractor,
    )

    evaluations = tuple(
        HeldOutCaseEvaluation(
            name=case.name,
            scenario_id=evaluation.scenario_id,
            metrics=evaluation.metrics,
            target=evaluation.target,
            loss=evaluation.loss,
        )
        for case, evaluation in zip(named_cases, base_report.cases, strict=True)
    )
    return HeldOutValidationReport(
        parameters=base_report.parameters,
        cases=evaluations,
        mean_loss=base_report.mean_loss,
        max_loss=base_report.worst_loss,
    )


@dataclass(frozen=True)
class SensitivityEntry:
    parameter: str
    step: float
    minus_metrics: tuple[tuple[str, float], ...]
    plus_metrics: tuple[tuple[str, float], ...]
    derivatives: tuple[tuple[str, float], ...]

    @property
    def minus_metric_map(self) -> dict[str, float]:
        return dict(self.minus_metrics)

    @property
    def plus_metric_map(self) -> dict[str, float]:
        return dict(self.plus_metrics)

    @property
    def derivative_map(self) -> dict[str, float]:
        return dict(self.derivatives)


@dataclass(frozen=True)
class SensitivityReport:
    baseline_parameters: tuple[tuple[str, float], ...]
    baseline_metrics: tuple[tuple[str, float], ...]
    entries: tuple[SensitivityEntry, ...]

    @property
    def baseline_parameter_map(self) -> dict[str, float]:
        return dict(self.baseline_parameters)

    @property
    def baseline_metric_map(self) -> dict[str, float]:
        return dict(self.baseline_metrics)


def _validated_positive_step(parameter: str, raw_step: float) -> float:
    try:
        step = float(raw_step)
    except (TypeError, ValueError) as error:
        raise ValueError(f"sensitivity step for {parameter!r} must be numeric") from error
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("sensitivity steps must be positive and finite")
    return step


def central_difference_sensitivity(
    *,
    runner: SimulationRunner,
    model: SimulatorModel,
    scenario: Scenario,
    baseline_parameters: Mapping[str, float],
    steps: Mapping[str, float],
    seeds: Iterable[int],
    extractor: MetricExtractor,
) -> SensitivityReport:
    """Estimate local metric derivatives by common-seed central differences.

    This is a local numerical diagnostic, not a proof of global structural
    identifiability. Reusing the same seeds for the plus and minus runs reduces
    stochastic comparison noise without changing the model's RNG boundary.
    """

    ordered_seeds = tuple(seeds)
    if not ordered_seeds:
        raise ValueError("sensitivity analysis requires at least one seed")
    if not steps:
        raise ValueError("sensitivity analysis requires at least one step")

    baseline_traces = runner.run_batch(
        model,
        scenario,
        baseline_parameters,
        seeds=ordered_seeds,
    )
    canonical_parameters = baseline_traces[0].parameters
    baseline_map = dict(canonical_parameters)
    baseline_metrics = aggregate_metrics(baseline_traces, extractor)

    unknown = set(steps) - set(baseline_map)
    if unknown:
        raise ValueError("sensitivity steps contain an unknown parameter")

    entries: list[SensitivityEntry] = []
    for parameter in sorted(steps):
        step = _validated_positive_step(parameter, steps[parameter])
        minus_parameters = dict(baseline_map)
        plus_parameters = dict(baseline_map)
        minus_parameters[parameter] -= step
        plus_parameters[parameter] += step

        minus_metrics = aggregate_metrics(
            runner.run_batch(
                model,
                scenario,
                minus_parameters,
                seeds=ordered_seeds,
            ),
            extractor,
        )
        plus_metrics = aggregate_metrics(
            runner.run_batch(
                model,
                scenario,
                plus_parameters,
                seeds=ordered_seeds,
            ),
            extractor,
        )
        if set(minus_metrics) != set(baseline_metrics) or set(plus_metrics) != set(
            baseline_metrics
        ):
            raise ValueError("sensitivity runs changed the metric schema")

        derivatives = {
            name: (plus_metrics[name] - minus_metrics[name]) / (2.0 * step)
            for name in sorted(baseline_metrics)
        }
        if any(not math.isfinite(value) for value in derivatives.values()):
            raise ValueError("sensitivity derivative must be finite")

        entries.append(
            SensitivityEntry(
                parameter=parameter,
                step=step,
                minus_metrics=tuple(sorted(minus_metrics.items())),
                plus_metrics=tuple(sorted(plus_metrics.items())),
                derivatives=tuple(sorted(derivatives.items())),
            )
        )

    return SensitivityReport(
        baseline_parameters=canonical_parameters,
        baseline_metrics=tuple(sorted(baseline_metrics.items())),
        entries=tuple(entries),
    )
