from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import random

from narrative_dynamics.contracts import (
    ModelRun,
    Scenario,
    SimulationTrace,
    SimulatorModel,
)


def _canonical_parameters(
    parameters: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    canonical: list[tuple[str, float]] = []
    for name, raw_value in parameters.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be non-empty strings")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"parameter {name!r} must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"parameter {name!r} must be finite")
        canonical.append((name, value))
    canonical.sort(key=lambda item: item[0])
    return tuple(canonical)


class SimulationRunner:
    """Owns seeds and canonical metadata around an untrusted model call."""

    def run_once(
        self,
        model: SimulatorModel,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
    ) -> SimulationTrace:
        if not isinstance(seed, int):
            raise TypeError("simulation seed must be an integer")
        model_name = getattr(model, "name", None)
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model name must be a non-empty string")

        canonical = _canonical_parameters(parameters)
        rng = random.Random(seed)
        result = model.simulate(scenario, dict(canonical), rng)
        if not isinstance(result, ModelRun):
            raise TypeError("model simulate() must return ModelRun")

        return SimulationTrace(
            model_name=model_name,
            scenario_id=scenario.id,
            parameters=canonical,
            seed=seed,
            events=tuple(result.events),
            outcome=dict(result.outcome),
        )

    def run_batch(
        self,
        model: SimulatorModel,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seeds: Iterable[int],
    ) -> tuple[SimulationTrace, ...]:
        ordered_seeds = tuple(seeds)
        if not ordered_seeds:
            raise ValueError("simulation batch must contain at least one seed")
        return tuple(
            self.run_once(
                model,
                scenario,
                parameters,
                seed=seed,
            )
            for seed in ordered_seeds
        )
