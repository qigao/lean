from __future__ import annotations

from collections.abc import Mapping
import math

from narrative_dynamics.contracts import SimulationTrace


_ACTIONS = ("scout", "escape", "submit")


def prison_initial_action_metrics(trace: SimulationTrace) -> dict[str, float]:
    policy = trace.outcome.get("initial_policy")
    if not isinstance(policy, Mapping) or set(policy) != set(_ACTIONS):
        raise ValueError(
            "prison trace must contain exactly the three initial-policy actions"
        )
    result: dict[str, float] = {}
    for action in _ACTIONS:
        raw = policy[action]
        if isinstance(raw, bool):
            raise ValueError("prison initial-policy coordinates must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "prison initial-policy coordinates must be numeric"
            ) from error
        if not math.isfinite(value):
            raise ValueError("prison initial-policy coordinates must be finite")
        result[f"initial.{action}"] = value
    return result


prison_initial_action_metrics.version = "1.0.0"


__all__ = ["prison_initial_action_metrics"]
