from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Protocol

from narrative_dynamics.contracts import SimulationTrace


class MetricExtractor(Protocol):
    def __call__(self, trace: SimulationTrace) -> Mapping[str, float]:
        ...


def _validated_metrics(
    metrics: Mapping[str, float],
    *,
    label: str,
) -> dict[str, float]:
    if not metrics:
        raise ValueError(f"{label} must contain at least one metric")
    validated: dict[str, float] = {}
    for name, raw_value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"metric {name!r} must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        validated[name] = value
    return validated


def aggregate_metrics(
    traces: Iterable[SimulationTrace],
    extractor: MetricExtractor,
) -> dict[str, float]:
    """Average one stable metric schema over a nonempty trace collection."""

    trace_batch = tuple(traces)
    if not trace_batch:
        raise ValueError("metric aggregation requires at least one trace")

    first = _validated_metrics(extractor(trace_batch[0]), label="metrics")
    keys = tuple(sorted(first))
    totals = {name: first[name] for name in keys}

    for trace in trace_batch[1:]:
        current = _validated_metrics(extractor(trace), label="metrics")
        if set(current) != set(keys):
            raise ValueError("metric extractor changed its output schema")
        for name in keys:
            totals[name] += current[name]

    count = float(len(trace_batch))
    return {name: totals[name] / count for name in keys}


def weighted_squared_error(
    observed: Mapping[str, float],
    target: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compare two equal metric schemas with optional nonnegative weights."""

    observed_values = _validated_metrics(observed, label="observed metrics")
    target_values = _validated_metrics(target, label="target metrics")
    if set(observed_values) != set(target_values):
        raise ValueError("observed and target metric schemas must match")

    weight_values: dict[str, float] = {}
    if weights is not None:
        unknown = set(weights) - set(observed_values)
        if unknown:
            raise ValueError("weights contain an unknown metric")
        for name, raw_weight in weights.items():
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as error:
                raise ValueError(f"weight for {name!r} must be numeric") from error
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("metric weights must be finite and non-negative")
            weight_values[name] = weight

    loss = sum(
        weight_values.get(name, 1.0)
        * (observed_values[name] - target_values[name]) ** 2
        for name in sorted(observed_values)
    )
    if not math.isfinite(loss):
        raise ValueError("metric loss must be finite")
    return loss
