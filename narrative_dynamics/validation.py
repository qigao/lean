from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from statistics import fmean

from narrative_dynamics.calibration import CalibrationResult, calibrate_grid
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
from narrative_dynamics.uncertainty import ParameterAcceptanceSet


@dataclass(frozen=True)
class SyntheticRecoveryReport:
    true_parameters: tuple[tuple[str, float], ...]
    target_metrics: tuple[tuple[str, float], ...]
    calibration: CalibrationResult
    recovered: bool
    manifest: ExperimentManifest | None = None


@dataclass(frozen=True)
class HeldOutCase:
    """One external scenario and target excluded from parameter calibration."""

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
    run_manifest_hashes: tuple[str, ...] = ()

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
    manifest: ExperimentManifest | None = None

    @property
    def max_loss(self) -> float:
        return self.worst_loss


@dataclass(frozen=True)
class AcceptedParameterHeldOutEvaluation:
    parameters: tuple[tuple[str, float], ...]
    validation: HeldOutValidationReport

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class HeldOutAcceptanceReport:
    evaluations: tuple[AcceptedParameterHeldOutEvaluation, ...]
    retained_parameters: ParameterAcceptanceSet
    best: AcceptedParameterHeldOutEvaluation
    manifest: ExperimentManifest | None = None


class EvaluationRole(str, Enum):
    SELECTION_VALIDATION = "selection_validation"
    FINAL_TEST = "final_test"


@dataclass(frozen=True)
class HeldOutSuite:
    name: str
    role: EvaluationRole
    cases: tuple[HeldOutCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("held-out suite name must be a non-empty string")
        try:
            role = (
                self.role
                if isinstance(self.role, EvaluationRole)
                else EvaluationRole(self.role)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "held-out suite role must be a supported evaluation role"
            ) from error
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("held-out suite must contain at least one case")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "cases", cases)


@dataclass(frozen=True)
class SelectionValidationReport:
    suite_name: str
    role: EvaluationRole
    candidate_report: HeldOutAcceptanceReport
    selected_parameters: tuple[tuple[str, float], ...]
    manifest: ExperimentManifest | None = None


@dataclass(frozen=True)
class FinalTestReport:
    suite_name: str
    role: EvaluationRole
    validation: HeldOutValidationReport
    manifest: ExperimentManifest | None = None


@dataclass(frozen=True)
class LocalParameterSensitivity:
    name: str
    baseline_value: float
    step: float
    baseline: HeldOutValidationReport
    lower: HeldOutValidationReport
    upper: HeldOutValidationReport

    @property
    def mean_loss_slope(self) -> float:
        return (self.upper.mean_loss - self.lower.mean_loss) / (2.0 * self.step)

    @property
    def mean_loss_curvature(self) -> float:
        return (
            self.upper.mean_loss
            - 2.0 * self.baseline.mean_loss
            + self.lower.mean_loss
        ) / (self.step * self.step)

    @property
    def worst_loss_slope(self) -> float:
        return (self.upper.worst_loss - self.lower.worst_loss) / (2.0 * self.step)

    @property
    def worst_loss_curvature(self) -> float:
        return (
            self.upper.worst_loss
            - 2.0 * self.baseline.worst_loss
            + self.lower.worst_loss
        ) / (self.step * self.step)

    @property
    def max_absolute_mean_loss_change(self) -> float:
        return max(
            abs(self.lower.mean_loss - self.baseline.mean_loss),
            abs(self.upper.mean_loss - self.baseline.mean_loss),
        )


@dataclass(frozen=True)
class LocalSensitivityReport:
    baseline: HeldOutValidationReport
    parameters: tuple[LocalParameterSensitivity, ...]
    manifest: ExperimentManifest | None = None


def synthetic_recovery(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    scenario: Scenario,
    true_parameters: Mapping[str, float],
    parameter_grid: Mapping[str, Iterable[float]],
    observation_seeds: Iterable[int],
    calibration_seeds: Iterable[int],
    extractor: MetricExtractor,
    weights: Mapping[str, float] | None = None,
    loss: MetricLoss | None = None,
) -> SyntheticRecoveryReport:
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
        loss=loss,
    )
    canonical_true = observed_traces[0].parameters
    recovered = calibration.best.parameters == canonical_true
    observation_hashes = tuple(
        required_manifest_hash(trace, label="synthetic observation trace")
        for trace in observed_traces
    )
    calibration_hash = required_manifest_hash(
        calibration,
        label="synthetic recovery calibration",
    )
    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
    manifest = ExperimentManifest(
        stage=ExperimentStage.SYNTHETIC_RECOVERY,
        inputs={
            "model": component_identity(model),
            "scenario": scenario_identity(scenario),
            "true_parameters": canonical_true,
            "observation_seeds": observed_seed_tuple,
            "calibration_seeds": calibrated_seed_tuple,
            "metric": callable_identity(extractor),
            "target_hash": stable_content_hash(target),
            "loss": metric_loss_identity(selected_loss),
        },
        parent_hashes=observation_hashes + (calibration_hash,),
    )
    return SyntheticRecoveryReport(
        true_parameters=canonical_true,
        target_metrics=tuple(sorted(target.items())),
        calibration=calibration,
        recovered=recovered,
        manifest=manifest,
    )


def validate_held_out(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    parameters: Mapping[str, float],
    cases: Iterable[HeldOutCase],
    extractor: MetricExtractor,
    loss: MetricLoss | None = None,
) -> HeldOutValidationReport:
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

    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
    loss_identity = metric_loss_identity(selected_loss)
    evaluations: list[HeldOutCaseEvaluation] = []
    case_inputs: list[dict[str, object]] = []
    run_parent_hashes: list[str] = []
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

        run_hashes = tuple(
            required_manifest_hash(trace, label="held-out trace")
            for trace in traces
        )
        run_parent_hashes.extend(run_hashes)
        metrics = aggregate_metrics(traces, extractor)
        case_loss = evaluate_metric_loss(
            selected_loss,
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
                    sorted(
                        (metric_name, float(value))
                        for metric_name, value in case.target.items()
                    )
                ),
                loss=case_loss,
                run_manifest_hashes=run_hashes,
            )
        )
        case_inputs.append(
            {
                "name": case.effective_name,
                "scenario": scenario_identity(case.scenario),
                "seeds": seeds,
                "target_hash": stable_content_hash(case.target),
                "weights_hash": (
                    None
                    if case.weights is None
                    else stable_content_hash(case.weights)
                ),
            }
        )

    assert canonical_parameters is not None
    losses = tuple(evaluation.loss for evaluation in evaluations)
    manifest = ExperimentManifest(
        stage=ExperimentStage.HELD_OUT_VALIDATION,
        inputs={
            "model": component_identity(model),
            "parameters": canonical_parameters,
            "cases": tuple(case_inputs),
            "metric": callable_identity(extractor),
            "loss": loss_identity,
        },
        parent_hashes=tuple(run_parent_hashes),
    )
    return HeldOutValidationReport(
        parameters=canonical_parameters,
        cases=tuple(evaluations),
        mean_loss=fmean(losses),
        worst_loss=max(losses),
        manifest=manifest,
    )


def _optional_loss_limit(value: float | None, *, label: str) -> float | None:
    if value is None:
        return None
    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(validated) or validated < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return validated


def validate_acceptance_set_held_out(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    accepted_parameters: ParameterAcceptanceSet
    | Iterable[Iterable[tuple[str, float]]],
    cases: Iterable[HeldOutCase],
    extractor: MetricExtractor,
    max_mean_loss: float | None = None,
    max_worst_loss: float | None = None,
    loss: MetricLoss | None = None,
) -> HeldOutAcceptanceReport:
    acceptance_set = (
        accepted_parameters
        if isinstance(accepted_parameters, ParameterAcceptanceSet)
        else ParameterAcceptanceSet.from_parameters(accepted_parameters)
    )
    if not acceptance_set.parameters:
        raise ValueError("held-out acceptance validation requires candidates")

    held_out_cases = tuple(cases)
    mean_limit = _optional_loss_limit(max_mean_loss, label="maximum mean loss")
    worst_limit = _optional_loss_limit(max_worst_loss, label="maximum worst loss")
    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss

    evaluations = tuple(
        sorted(
            (
                AcceptedParameterHeldOutEvaluation(
                    parameters=candidate_parameters,
                    validation=validate_held_out(
                        runner=runner,
                        model=model,
                        parameters=dict(candidate_parameters),
                        cases=held_out_cases,
                        extractor=extractor,
                        loss=selected_loss,
                    ),
                )
                for candidate_parameters in acceptance_set.parameters
            ),
            key=lambda evaluation: (
                evaluation.validation.mean_loss,
                evaluation.validation.worst_loss,
                evaluation.parameters,
            ),
        )
    )
    retained = tuple(
        evaluation.parameters
        for evaluation in evaluations
        if (mean_limit is None or evaluation.validation.mean_loss <= mean_limit)
        and (worst_limit is None or evaluation.validation.worst_loss <= worst_limit)
    )
    validation_hashes = tuple(
        required_manifest_hash(
            evaluation.validation,
            label="accepted-parameter held-out validation",
        )
        for evaluation in evaluations
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.ACCEPTANCE_VALIDATION,
        inputs={
            "accepted_parameters": acceptance_set.parameters,
            "accepted_parameter_set_hash": acceptance_set.content_hash,
            "max_mean_loss": mean_limit,
            "max_worst_loss": worst_limit,
            "loss": metric_loss_identity(selected_loss),
        },
        parent_hashes=(
            acceptance_set.source_manifest_hashes + validation_hashes
        ),
    )
    return HeldOutAcceptanceReport(
        evaluations=evaluations,
        retained_parameters=ParameterAcceptanceSet.from_parameters(
            retained,
            source_manifest_hashes=(manifest.content_hash,),
        ),
        best=evaluations[0],
        manifest=manifest,
    )


def select_on_validation_suite(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    accepted_parameters: ParameterAcceptanceSet
    | Iterable[Iterable[tuple[str, float]]],
    suite: HeldOutSuite,
    extractor: MetricExtractor,
    max_mean_loss: float | None = None,
    max_worst_loss: float | None = None,
    loss: MetricLoss | None = None,
) -> SelectionValidationReport:
    if suite.role is not EvaluationRole.SELECTION_VALIDATION:
        raise ValueError("candidate selection requires a selection-validation suite")
    candidate_report = validate_acceptance_set_held_out(
        runner=runner,
        model=model,
        accepted_parameters=accepted_parameters,
        cases=suite.cases,
        extractor=extractor,
        max_mean_loss=max_mean_loss,
        max_worst_loss=max_worst_loss,
        loss=loss,
    )
    retained = set(candidate_report.retained_parameters.parameters)
    if not retained:
        raise ValueError("selection-validation thresholds retained no candidates")
    selected_parameters = next(
        evaluation.parameters
        for evaluation in candidate_report.evaluations
        if evaluation.parameters in retained
    )
    candidate_hash = required_manifest_hash(
        candidate_report,
        label="selection candidate report",
    )
    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
    manifest = ExperimentManifest(
        stage=ExperimentStage.SELECTION_VALIDATION,
        inputs={
            "suite_name": suite.name,
            "role": suite.role.value,
            "max_mean_loss": max_mean_loss,
            "max_worst_loss": max_worst_loss,
            "loss": metric_loss_identity(selected_loss),
        },
        parent_hashes=(candidate_hash,),
    )
    return SelectionValidationReport(
        suite_name=suite.name,
        role=suite.role,
        candidate_report=candidate_report,
        selected_parameters=selected_parameters,
        manifest=manifest,
    )


def evaluate_on_final_test_suite(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    parameters: Mapping[str, float],
    suite: HeldOutSuite,
    extractor: MetricExtractor,
    loss: MetricLoss | None = None,
) -> FinalTestReport:
    if suite.role is not EvaluationRole.FINAL_TEST:
        raise ValueError("final-test evaluation requires a final-test suite")
    validation = validate_held_out(
        runner=runner,
        model=model,
        parameters=parameters,
        cases=suite.cases,
        extractor=extractor,
        loss=loss,
    )
    validation_hash = required_manifest_hash(
        validation,
        label="final-test validation",
    )
    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
    manifest = ExperimentManifest(
        stage=ExperimentStage.FINAL_TEST,
        inputs={
            "suite_name": suite.name,
            "role": suite.role.value,
            "parameters": validation.parameters,
            "loss": metric_loss_identity(selected_loss),
        },
        parent_hashes=(validation_hash,),
    )
    return FinalTestReport(
        suite_name=suite.name,
        role=suite.role,
        validation=validation,
        manifest=manifest,
    )


def local_sensitivity_report(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    parameters: Mapping[str, float],
    cases: Iterable[HeldOutCase],
    extractor: MetricExtractor,
    step_sizes: Mapping[str, float],
    loss: MetricLoss | None = None,
) -> LocalSensitivityReport:
    held_out_cases = tuple(cases)
    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
    baseline = validate_held_out(
        runner=runner,
        model=model,
        parameters=parameters,
        cases=held_out_cases,
        extractor=extractor,
        loss=selected_loss,
    )
    baseline_parameters = dict(baseline.parameters)
    if not step_sizes:
        raise ValueError("local sensitivity requires at least one step size")
    unknown = set(step_sizes) - set(baseline_parameters)
    if unknown:
        raise ValueError("local sensitivity contains an unknown parameter")

    sensitivities: list[LocalParameterSensitivity] = []
    parent_hashes: list[str] = [
        required_manifest_hash(baseline, label="sensitivity baseline")
    ]
    canonical_steps: list[tuple[str, float]] = []
    for name in sorted(step_sizes):
        try:
            step = float(step_sizes[name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"sensitivity step for {name!r} must be numeric"
            ) from error
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("local sensitivity steps must be finite and positive")

        baseline_value = baseline_parameters[name]
        lower_value = baseline_value - step
        upper_value = baseline_value + step
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            raise ValueError(
                "local sensitivity perturbations must remain finite"
            )

        lower_parameters = dict(baseline_parameters)
        upper_parameters = dict(baseline_parameters)
        lower_parameters[name] = lower_value
        upper_parameters[name] = upper_value
        lower = validate_held_out(
            runner=runner,
            model=model,
            parameters=lower_parameters,
            cases=held_out_cases,
            extractor=extractor,
            loss=selected_loss,
        )
        upper = validate_held_out(
            runner=runner,
            model=model,
            parameters=upper_parameters,
            cases=held_out_cases,
            extractor=extractor,
            loss=selected_loss,
        )
        parent_hashes.extend(
            (
                required_manifest_hash(
                    lower,
                    label="sensitivity lower report",
                ),
                required_manifest_hash(
                    upper,
                    label="sensitivity upper report",
                ),
            )
        )
        canonical_steps.append((name, step))
        sensitivities.append(
            LocalParameterSensitivity(
                name=name,
                baseline_value=baseline_value,
                step=step,
                baseline=baseline,
                lower=lower,
                upper=upper,
            )
        )

    manifest = ExperimentManifest(
        stage=ExperimentStage.LOCAL_SENSITIVITY,
        inputs={
            "baseline_parameters": baseline.parameters,
            "step_sizes": tuple(canonical_steps),
            "loss": metric_loss_identity(selected_loss),
        },
        parent_hashes=tuple(parent_hashes),
    )
    return LocalSensitivityReport(
        baseline=baseline,
        parameters=tuple(sensitivities),
        manifest=manifest,
    )
