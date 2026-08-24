from __future__ import annotations

from collections.abc import Mapping
import math

from narrative_dynamics.contracts import SimulationTrace


_EXPECTED_ACTIONS = ("search_box", "search_drawer")


def story_choice_metrics(trace: SimulationTrace) -> dict[str, float]:
    """Return the validated V1 search policy as stable choice coordinates."""

    policy = trace.outcome.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("story trace outcome must contain a policy mapping")
    if set(policy) != set(_EXPECTED_ACTIONS):
        raise ValueError("story policy must contain exactly the V1 search actions")

    values: dict[str, float] = {}
    for action in _EXPECTED_ACTIONS:
        raw = policy[action]
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
        ):
            raise ValueError("story policy probabilities must be numeric")
        probability = float(raw)
        if not math.isfinite(probability):
            raise ValueError("story policy probabilities must be finite")
        if probability < 0.0:
            raise ValueError("story policy probabilities must be non-negative")
        values[action] = probability

    if not math.isclose(
        sum(values.values()),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("story policy probabilities must be normalized")

    return {
        f"choice.{action}": values[action]
        for action in _EXPECTED_ACTIONS
    }
