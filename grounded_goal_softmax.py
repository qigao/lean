from __future__ import annotations

from collections.abc import Mapping
import math

from grounded_epistemic_goal import grounded_epistemic_goal_score
from grounded_goal_score_covariance import grounded_prior_epistemic_goal_score
from grounded_hypothesis_space import GroundedHypothesisSpace


def finite_softmax(
    scores: Mapping[str, float],
    *,
    beta: float,
) -> dict[str, float]:
    """Numerically stable finite softmax over arbitrary semantic goal labels."""
    if not scores:
        raise ValueError("softmax goal space must be non-empty")
    if not math.isfinite(beta):
        raise ValueError("softmax beta must be finite")
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("softmax scores must be finite")

    logits = {goal: beta * score for goal, score in scores.items()}
    max_logit = max(logits.values())
    weights = {
        goal: math.exp(logit - max_logit)
        for goal, logit in logits.items()
    }
    mass = sum(weights.values())
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("softmax normalizer must be positive and finite")
    return {goal: weight / mass for goal, weight in weights.items()}


def _validate_goal_maps(
    instrumentality: Mapping[str, Mapping[str, float]],
    costs: Mapping[str, float] | None,
    risks: Mapping[str, float] | None,
) -> tuple[Mapping[str, float], Mapping[str, float]]:
    if not instrumentality:
        raise ValueError("grounded goal space must be non-empty")
    goals = set(instrumentality)
    cost_map: Mapping[str, float] = costs or {}
    risk_map: Mapping[str, float] = risks or {}
    if not set(cost_map).issubset(goals):
        raise ValueError("costs contain an unknown goal")
    if not set(risk_map).issubset(goals):
        raise ValueError("risks contain an unknown goal")
    return cost_map, risk_map


def grounded_prior_goal_scores(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, Mapping[str, float]],
    *,
    pressure: float,
    costs: Mapping[str, float] | None = None,
    risks: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Finite goal scores before the admitted evidence reweights hypotheses."""
    cost_map, risk_map = _validate_goal_maps(instrumentality, costs, risks)
    return {
        goal: grounded_prior_epistemic_goal_score(
            space,
            predictions,
            pressure=pressure,
            cost=cost_map.get(goal, 0.0),
            risk=risk_map.get(goal, 0.0),
        )
        for goal, predictions in instrumentality.items()
    }


def grounded_goal_scores(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, Mapping[str, float]],
    *,
    pressure: float,
    costs: Mapping[str, float] | None = None,
    risks: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Finite goal scores after the common grounded observation."""
    cost_map, risk_map = _validate_goal_maps(instrumentality, costs, risks)
    return {
        goal: grounded_epistemic_goal_score(
            space,
            predictions,
            pressure=pressure,
            cost=cost_map.get(goal, 0.0),
            risk=risk_map.get(goal, 0.0),
        )
        for goal, predictions in instrumentality.items()
    }


def grounded_prior_goal_choice_probabilities(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, Mapping[str, float]],
    *,
    pressure: float,
    beta: float,
    costs: Mapping[str, float] | None = None,
    risks: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Softmax choice distribution before epistemic reweighting."""
    scores = grounded_prior_goal_scores(
        space,
        instrumentality,
        pressure=pressure,
        costs=costs,
        risks=risks,
    )
    return finite_softmax(scores, beta=beta)


def grounded_goal_choice_probabilities(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, Mapping[str, float]],
    *,
    pressure: float,
    beta: float,
    costs: Mapping[str, float] | None = None,
    risks: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Softmax choice distribution after finite grounded epistemic scoring."""
    scores = grounded_goal_scores(
        space,
        instrumentality,
        pressure=pressure,
        costs=costs,
        risks=risks,
    )
    return finite_softmax(scores, beta=beta)
