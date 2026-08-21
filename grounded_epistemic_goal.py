from __future__ import annotations

from collections.abc import Mapping

from grounded_hypothesis_space import (
    GroundedHypothesisSpace,
    grounded_hypothesis_space_admissible,
    grounded_posteriors,
)


def grounded_goal_evaluation_admissible(
    space: GroundedHypothesisSpace,
) -> bool:
    """Whether all finite interpretations remain grounded and supported."""
    return grounded_hypothesis_space_admissible(space)


def grounded_expected_instrumentality(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
) -> float:
    """Posterior-weighted learned goal instrumentality over grounded worlds."""
    if not grounded_goal_evaluation_admissible(space):
        raise ValueError("grounded goal evaluation is not admissible")

    hypotheses = set(space.states)
    if set(instrumentality) != hypotheses:
        raise ValueError("instrumentality must define exactly one value per hypothesis")

    posterior = grounded_posteriors(space)
    return sum(
        posterior[hypothesis] * instrumentality[hypothesis]
        for hypothesis in space.states
    )


def grounded_epistemic_goal_score(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
    *,
    pressure: float,
    cost: float = 0.0,
    risk: float = 0.0,
) -> float:
    """Score a goal through grounded uncertainty, never a direct belief reward."""
    expected = grounded_expected_instrumentality(space, instrumentality)
    return pressure * expected - cost - risk
