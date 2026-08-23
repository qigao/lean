from __future__ import annotations

from epistemic_goal import epistemic_goal_score
from motivation import softmax_probability


def epistemic_goal_choice_probability(
    *,
    beta: float,
    pressure: float,
    belief: float,
    own_instrumentality_h: float,
    own_instrumentality_not_h: float,
    own_cost: float,
    own_risk: float,
    rival_instrumentality_h: float,
    rival_instrumentality_not_h: float,
    rival_cost: float,
    rival_risk: float,
) -> float:
    """Binary-softmax choice after belief-conditioned scoring of two goals."""
    own_score = epistemic_goal_score(
        pressure=pressure,
        belief=belief,
        instrumentality_h=own_instrumentality_h,
        instrumentality_not_h=own_instrumentality_not_h,
        cost=own_cost,
        risk=own_risk,
    )
    rival_score = epistemic_goal_score(
        pressure=pressure,
        belief=belief,
        instrumentality_h=rival_instrumentality_h,
        instrumentality_not_h=rival_instrumentality_not_h,
        cost=rival_cost,
        risk=rival_risk,
    )
    return softmax_probability(beta, own_score, rival_score)
