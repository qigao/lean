from __future__ import annotations

from collections.abc import Mapping

from grounded_epistemic_goal import grounded_epistemic_goal_score
from grounded_goal_covariance import grounded_prior_expected_instrumentality
from grounded_hypothesis_space import GroundedHypothesisSpace


def grounded_prior_epistemic_goal_score(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
    *,
    pressure: float,
    cost: float = 0.0,
    risk: float = 0.0,
) -> float:
    """Score a goal before the admitted evidence reweights interpretations."""
    expected = grounded_prior_expected_instrumentality(space, instrumentality)
    return pressure * expected - cost - risk


def grounded_goal_score_change(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
    *,
    pressure: float,
    cost: float = 0.0,
    risk: float = 0.0,
) -> float:
    """Posterior-minus-prior goal-score change under one grounded observation."""
    posterior = grounded_epistemic_goal_score(
        space,
        instrumentality,
        pressure=pressure,
        cost=cost,
        risk=risk,
    )
    prior = grounded_prior_epistemic_goal_score(
        space,
        instrumentality,
        pressure=pressure,
        cost=cost,
        risk=risk,
    )
    return posterior - prior
