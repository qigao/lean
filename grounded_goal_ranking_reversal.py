from __future__ import annotations

from collections.abc import Mapping

from grounded_goal_covariance import (
    grounded_evidence_mass,
    grounded_likelihood_instrumentality_covariance,
)
from grounded_goal_score_covariance import grounded_prior_epistemic_goal_score
from grounded_hypothesis_space import GroundedHypothesisSpace


def grounded_goal_ranking_reversal_margin(
    space: GroundedHypothesisSpace,
    instrumentality_a: Mapping[str, float],
    instrumentality_b: Mapping[str, float],
    *,
    pressure: float,
    cost_a: float = 0.0,
    risk_a: float = 0.0,
    cost_b: float = 0.0,
    risk_b: float = 0.0,
) -> float:
    """Covariance-driven score advantage left after paying the prior deficit.

    A positive result is the explicit sufficient margin for goal A to overtake
    goal B after the common grounded observation.
    """
    prior_a = grounded_prior_epistemic_goal_score(
        space,
        instrumentality_a,
        pressure=pressure,
        cost=cost_a,
        risk=risk_a,
    )
    prior_b = grounded_prior_epistemic_goal_score(
        space,
        instrumentality_b,
        pressure=pressure,
        cost=cost_b,
        risk=risk_b,
    )
    covariance_a = grounded_likelihood_instrumentality_covariance(
        space,
        instrumentality_a,
    )
    covariance_b = grounded_likelihood_instrumentality_covariance(
        space,
        instrumentality_b,
    )
    evidence_mass = grounded_evidence_mass(space)

    covariance_shift_advantage = (
        pressure * (covariance_a - covariance_b) / evidence_mass
    )
    prior_deficit = prior_b - prior_a
    return covariance_shift_advantage - prior_deficit


def grounded_goal_ranking_reversal_condition(
    space: GroundedHypothesisSpace,
    instrumentality_a: Mapping[str, float],
    instrumentality_b: Mapping[str, float],
    *,
    pressure: float,
    cost_a: float = 0.0,
    risk_a: float = 0.0,
    cost_b: float = 0.0,
    risk_b: float = 0.0,
) -> bool:
    """Whether A starts below B and has enough covariance advantage to pass it."""
    prior_a = grounded_prior_epistemic_goal_score(
        space,
        instrumentality_a,
        pressure=pressure,
        cost=cost_a,
        risk=risk_a,
    )
    prior_b = grounded_prior_epistemic_goal_score(
        space,
        instrumentality_b,
        pressure=pressure,
        cost=cost_b,
        risk=risk_b,
    )
    margin = grounded_goal_ranking_reversal_margin(
        space,
        instrumentality_a,
        instrumentality_b,
        pressure=pressure,
        cost_a=cost_a,
        risk_a=risk_a,
        cost_b=cost_b,
        risk_b=risk_b,
    )
    return pressure > 0.0 and prior_a < prior_b and margin > 0.0
