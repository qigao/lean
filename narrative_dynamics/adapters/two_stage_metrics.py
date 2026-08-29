from __future__ import annotations

from collections.abc import Mapping
import math

from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalLogLoss,
    CategoricalMetricGroup,
)
from narrative_dynamics.observations import AdequacyThresholds, CategoricalTargetSpec


FIRST_STAGE_ACTIONS = ("action_0", "action_1")
FIRST_STAGE_METRIC_KEYS = (
    "first_stage.action_0",
    "first_stage.action_1",
)
FIRST_STAGE_PROBABILITY_FLOOR = 1e-12
_PROBABILITY_TOLERANCE = 1e-12
_FIRST_STAGE_GROUP = CategoricalMetricGroup(
    "first_stage",
    FIRST_STAGE_METRIC_KEYS,
)


def _raw_first_stage_policy(trace: object) -> tuple[float, float]:
    outcome = getattr(trace, "outcome", None)
    if not isinstance(outcome, Mapping):
        raise ValueError("two-stage trace outcome must be a mapping")
    policy = outcome.get("first_stage_policy")
    if not isinstance(policy, Mapping) or set(policy) != set(FIRST_STAGE_ACTIONS):
        raise ValueError(
            "two-stage trace must contain exactly action_0/action_1 first-stage policy"
        )

    values: list[float] = []
    for action in FIRST_STAGE_ACTIONS:
        raw = policy[action]
        if isinstance(raw, bool):
            raise ValueError("two-stage first-stage policy coordinates must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "two-stage first-stage policy coordinates must be numeric"
            ) from error
        if not math.isfinite(value):
            raise ValueError("two-stage first-stage policy coordinates must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError("two-stage first-stage policy coordinates must be in [0, 1]")
        values.append(value)

    if not math.isclose(
        math.fsum(values),
        1.0,
        rel_tol=_PROBABILITY_TOLERANCE,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("two-stage first-stage policy must sum to one")
    return values[0], values[1]


def two_stage_first_stage_policy_metrics(trace: object) -> dict[str, float]:
    raw_action_0, raw_action_1 = _raw_first_stage_policy(trace)
    floored = (
        max(raw_action_0, FIRST_STAGE_PROBABILITY_FLOOR),
        max(raw_action_1, FIRST_STAGE_PROBABILITY_FLOOR),
    )
    total = math.fsum(floored)
    normalized = tuple(value / total for value in floored)
    return {
        metric_key: probability
        for metric_key, probability in zip(
            FIRST_STAGE_METRIC_KEYS,
            normalized,
            strict=True,
        )
    }


two_stage_first_stage_policy_metrics.version = "1.0.0"


def two_stage_target_spec() -> CategoricalTargetSpec:
    return CategoricalTargetSpec(
        name="two-stage-first-stage-choice",
        version="1",
        categories=FIRST_STAGE_ACTIONS,
        metric_prefix="first_stage",
    )


def two_stage_brier_loss() -> CategoricalBrierLoss:
    return CategoricalBrierLoss((_FIRST_STAGE_GROUP,))


def two_stage_log_loss() -> CategoricalLogLoss:
    return CategoricalLogLoss((_FIRST_STAGE_GROUP,))


def two_stage_brier_thresholds() -> AdequacyThresholds:
    return AdequacyThresholds(
        max_mean_loss=math.nextafter(0.5, -math.inf),
        max_worst_loss=2.0,
    )


def two_stage_log_thresholds() -> AdequacyThresholds:
    normalized_floor = FIRST_STAGE_PROBABILITY_FLOOR / (
        1.0 + FIRST_STAGE_PROBABILITY_FLOOR
    )
    return AdequacyThresholds(
        max_mean_loss=math.nextafter(math.log(2.0), -math.inf),
        max_worst_loss=-math.log(normalized_floor),
    )


__all__ = [
    "FIRST_STAGE_ACTIONS",
    "FIRST_STAGE_METRIC_KEYS",
    "FIRST_STAGE_PROBABILITY_FLOOR",
    "two_stage_brier_loss",
    "two_stage_brier_thresholds",
    "two_stage_first_stage_policy_metrics",
    "two_stage_log_loss",
    "two_stage_log_thresholds",
    "two_stage_target_spec",
]
