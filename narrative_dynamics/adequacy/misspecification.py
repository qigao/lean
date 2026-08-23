from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from narrative_dynamics.calibration import _grid_candidates
from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage
from narrative_dynamics.losses import (
    DEFAULT_METRIC_LOSS,
    MetricLoss,
    metric_loss_identity,
)
from narrative_dynamics.manifest import (
    callable_identity,
    component_identity,
    required_manifest_hash,
)
from narrative_dynamics.metrics import MetricExtractor
from narrative_dynamics.simulation import ModelSource, SimulationRunner
from narrative_dynamics.validation import HeldOutCase, validate_held_out


ParameterTuple = tuple[tuple[str, float], ...]


def _validated_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _validated_limit(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class MisspecificationCandidate:
    """One shared parameter candidate evaluated over every external case."""

    parameters: ParameterTuple
    case_losses: tuple[tuple[str, float], ...]
    mean_loss: float
    worst_loss: float
    validation_manifest_hash: str

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)

    @property
    def case_loss_map(self) -> dict[str, float]:
        return dict(self.case_losses)


@dataclass(frozen=True)
class MisspecificationReport:
    """Finite-grid adequacy result under one shared parameter requirement."""

    suite_name: str
    best: MisspecificationCandidate
    ranking: tuple[MisspecificationCandidate, ...]
    max_mean_loss: float
    max_worst_loss: float
    adequate: bool
    manifest: ExperimentManifest

    @property
    def misspecified(self) -> bool:
        return not self.adequate


def assess_misspecification(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    suite_name: str,
    parameter_grid: Mapping[str, Iterable[float]],
    cases: Iterable[HeldOutCase],
    extractor: MetricExtractor,
    loss: MetricLoss | None = None,
    max_mean_loss: float,
    max_worst_loss: float,
) -> MisspecificationReport:
    """Ask whether one finite-grid candidate can explain all external cases.

    ``misspecified`` is deliberately local to the supplied model family, grid,
    cases, scoring rule, and thresholds. It is not a global impossibility claim.
    """

    name = _validated_name(suite_name, label="misspecification suite name")
    held_out_cases = tuple(cases)
    if not held_out_cases:
        raise ValueError("misspecification assessment requires at least one case")
    if any(not isinstance(case, HeldOutCase) for case in held_out_cases):
        raise TypeError("misspecification cases must be HeldOutCase values")

    mean_limit = _validated_limit(max_mean_loss, label="maximum mean loss")
    worst_limit = _validated_limit(max_worst_loss, label="maximum worst loss")
    candidates = _grid_candidates(parameter_grid)
    selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
    loss_identity = metric_loss_identity(selected_loss)

    evaluated: list[MisspecificationCandidate] = []
    parent_hashes: list[str] = []
    for parameters in candidates:
        validation = validate_held_out(
            runner=runner,
            model=model,
            parameters=dict(parameters),
            cases=held_out_cases,
            extractor=extractor,
            loss=selected_loss,
        )
        validation_hash = required_manifest_hash(
            validation,
            label="misspecification candidate validation",
        )
        parent_hashes.append(validation_hash)
        evaluated.append(
            MisspecificationCandidate(
                parameters=parameters,
                case_losses=tuple(
                    (case.name, case.loss) for case in validation.cases
                ),
                mean_loss=validation.mean_loss,
                worst_loss=validation.worst_loss,
                validation_manifest_hash=validation_hash,
            )
        )

    ranking = tuple(
        sorted(
            evaluated,
            key=lambda candidate: (
                candidate.mean_loss,
                candidate.worst_loss,
                candidate.parameters,
            ),
        )
    )
    best = ranking[0]
    adequate = best.mean_loss <= mean_limit and best.worst_loss <= worst_limit
    manifest = ExperimentManifest(
        stage=ExperimentStage.MODEL_MISSPECIFICATION,
        inputs={
            "suite_name": name,
            "model": component_identity(model),
            "parameter_grid": candidates,
            "case_names": tuple(case.effective_name for case in held_out_cases),
            "metric": callable_identity(extractor),
            "loss": loss_identity,
            "max_mean_loss": mean_limit,
            "max_worst_loss": worst_limit,
            "best_parameters": best.parameters,
            "adequate": adequate,
        },
        parent_hashes=tuple(parent_hashes),
    )
    return MisspecificationReport(
        suite_name=name,
        best=best,
        ranking=ranking,
        max_mean_loss=mean_limit,
        max_worst_loss=worst_limit,
        adequate=adequate,
        manifest=manifest,
    )


__all__ = [
    "MisspecificationCandidate",
    "MisspecificationReport",
    "assess_misspecification",
]
