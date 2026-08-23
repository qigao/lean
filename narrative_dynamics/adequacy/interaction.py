from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math

from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage, Scenario
from narrative_dynamics.manifest import (
    callable_identity,
    component_identity,
    required_manifest_hash,
    scenario_identity,
)
from narrative_dynamics.metrics import MetricExtractor, aggregate_metrics
from narrative_dynamics.simulation import ModelSource, SimulationRunner


class FactorSource(str, Enum):
    PARAMETER = "parameter"
    SCENARIO = "scenario"


def _validated_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class LocalFactor:
    source: FactorSource | str
    name: str
    step: float

    def __post_init__(self) -> None:
        try:
            source = (
                self.source
                if isinstance(self.source, FactorSource)
                else FactorSource(self.source)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("local factor source must be parameter or scenario") from error
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "name",
            _validated_name(self.name, label="local factor name"),
        )
        step = _finite_number(self.step, label="local factor step")
        if step <= 0.0:
            raise ValueError("local factor step must be positive")
        object.__setattr__(self, "step", step)

    @property
    def coordinate(self) -> tuple[str, str]:
        return self.source.value, self.name

    def manifest_identity(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "name": self.name,
            "step": self.step,
        }


@dataclass(frozen=True)
class InteractionCorner:
    label: str
    first_sign: int
    second_sign: int
    parameters: tuple[tuple[str, float], ...]
    scenario_content_hash: str
    metrics: tuple[tuple[str, float], ...]
    run_manifest_hashes: tuple[str, ...]

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)

    @property
    def metric_map(self) -> dict[str, float]:
        return dict(self.metrics)


@dataclass(frozen=True)
class MetricInteraction:
    name: str
    mixed_partial: float


@dataclass(frozen=True)
class FactorInteractionReport:
    first: LocalFactor
    second: LocalFactor
    corners: tuple[InteractionCorner, ...]
    interactions: tuple[MetricInteraction, ...]
    manifest: ExperimentManifest

    @property
    def corner_map(self) -> dict[str, InteractionCorner]:
        return {corner.label: corner for corner in self.corners}

    @property
    def interaction_map(self) -> dict[str, float]:
        return {
            interaction.name: interaction.mixed_partial
            for interaction in self.interactions
        }


def _factor_center(
    factor: LocalFactor,
    *,
    parameters: Mapping[str, float],
    scenario: Scenario,
) -> float:
    source = parameters if factor.source is FactorSource.PARAMETER else scenario.payload
    if factor.name not in source:
        raise ValueError(
            f"{factor.source.value} factor {factor.name!r} does not exist"
        )
    return _finite_number(
        source[factor.name],
        label=f"{factor.source.value} factor {factor.name!r}",
    )


def _perturb(
    *,
    base_parameters: Mapping[str, float],
    base_scenario: Scenario,
    first: LocalFactor,
    first_sign: int,
    second: LocalFactor,
    second_sign: int,
    first_center: float,
    second_center: float,
    label: str,
) -> tuple[dict[str, float], Scenario]:
    parameters = {name: float(value) for name, value in base_parameters.items()}
    payload = dict(base_scenario.payload)
    for factor, sign, center in (
        (first, first_sign, first_center),
        (second, second_sign, second_center),
    ):
        value = center + sign * factor.step
        if not math.isfinite(value):
            raise ValueError("interaction perturbations must remain finite")
        if factor.source is FactorSource.PARAMETER:
            parameters[factor.name] = value
        else:
            payload[factor.name] = value
    scenario = Scenario(
        id=f"{base_scenario.id}:interaction:{label}",
        payload=payload,
    )
    return parameters, scenario


def local_factor_interaction_report(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    scenario: Scenario,
    parameters: Mapping[str, float],
    seeds: Iterable[int],
    extractor: MetricExtractor,
    first: LocalFactor,
    second: LocalFactor,
) -> FactorInteractionReport:
    """Compute a paired central mixed finite difference over two coordinates."""

    if not isinstance(first, LocalFactor) or not isinstance(second, LocalFactor):
        raise TypeError("interaction factors must be LocalFactor values")
    if first.coordinate == second.coordinate:
        raise ValueError("interaction factors must identify distinct coordinates")
    ordered_seeds = tuple(seeds)
    if not ordered_seeds:
        raise ValueError("interaction sensitivity requires at least one seed")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in ordered_seeds):
        raise TypeError("interaction sensitivity seeds must be integers")

    first_center = _factor_center(
        first,
        parameters=parameters,
        scenario=scenario,
    )
    second_center = _factor_center(
        second,
        parameters=parameters,
        scenario=scenario,
    )

    corners: list[InteractionCorner] = []
    parent_hashes: list[str] = []
    expected_metric_names: tuple[str, ...] | None = None
    for first_sign, second_sign, label in (
        (-1, -1, "--"),
        (-1, 1, "-+"),
        (1, -1, "+-"),
        (1, 1, "++"),
    ):
        corner_parameters, corner_scenario = _perturb(
            base_parameters=parameters,
            base_scenario=scenario,
            first=first,
            first_sign=first_sign,
            second=second,
            second_sign=second_sign,
            first_center=first_center,
            second_center=second_center,
            label=label,
        )
        traces = runner.run_batch(
            model,
            corner_scenario,
            corner_parameters,
            seeds=ordered_seeds,
        )
        run_hashes = tuple(
            required_manifest_hash(trace, label="interaction corner trace")
            for trace in traces
        )
        parent_hashes.extend(run_hashes)
        metrics = aggregate_metrics(traces, extractor)
        metric_names = tuple(sorted(metrics))
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ValueError("interaction corners changed their metric schema")
        corners.append(
            InteractionCorner(
                label=label,
                first_sign=first_sign,
                second_sign=second_sign,
                parameters=tuple(sorted(corner_parameters.items())),
                scenario_content_hash=corner_scenario.content_hash,
                metrics=tuple(sorted(metrics.items())),
                run_manifest_hashes=run_hashes,
            )
        )

    assert expected_metric_names is not None
    corner_map = {corner.label: corner.metric_map for corner in corners}
    denominator = 4.0 * first.step * second.step
    interactions = tuple(
        MetricInteraction(
            name=name,
            mixed_partial=(
                corner_map["++"][name]
                - corner_map["+-"][name]
                - corner_map["-+"][name]
                + corner_map["--"][name]
            )
            / denominator,
        )
        for name in expected_metric_names
    )
    if any(not math.isfinite(item.mixed_partial) for item in interactions):
        raise ValueError("interaction sensitivity produced a non-finite derivative")

    manifest = ExperimentManifest(
        stage=ExperimentStage.INTERACTION_SENSITIVITY,
        inputs={
            "model": component_identity(model),
            "scenario": scenario_identity(scenario),
            "parameters": tuple(sorted((name, float(value)) for name, value in parameters.items())),
            "seeds": ordered_seeds,
            "metric": callable_identity(extractor),
            "first": first.manifest_identity(),
            "second": second.manifest_identity(),
            "corner_scenarios": tuple(
                (corner.label, corner.scenario_content_hash) for corner in corners
            ),
        },
        parent_hashes=tuple(parent_hashes),
    )
    return FactorInteractionReport(
        first=first,
        second=second,
        corners=tuple(corners),
        interactions=interactions,
        manifest=manifest,
    )


__all__ = [
    "FactorInteractionReport",
    "FactorSource",
    "InteractionCorner",
    "LocalFactor",
    "MetricInteraction",
    "local_factor_interaction_report",
]
