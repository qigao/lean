from __future__ import annotations


def expected_instrumentality(
    belief: float,
    instrumentality_h: float,
    instrumentality_not_h: float,
) -> float:
    """Expected goal usefulness under a binary subjective world model."""
    return (
        belief * instrumentality_h
        + (1.0 - belief) * instrumentality_not_h
    )


def epistemic_goal_score(
    *,
    pressure: float,
    belief: float,
    instrumentality_h: float,
    instrumentality_not_h: float,
    cost: float = 0.0,
    risk: float = 0.0,
) -> float:
    """Score a goal through belief-conditioned learned instrumentality.

    Belief has no direct reward term. It changes the score only by changing the
    expected instrumentality of the goal under the agent's subjective model.
    """
    return (
        pressure
        * expected_instrumentality(
            belief,
            instrumentality_h,
            instrumentality_not_h,
        )
        - cost
        - risk
    )
