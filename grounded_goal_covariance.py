from __future__ import annotations

from collections.abc import Mapping

from grounded_epistemic_goal import (
    grounded_expected_instrumentality,
    grounded_goal_evaluation_admissible,
)
from grounded_hypothesis_space import GroundedHypothesisSpace


def _validate_instrumentality(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
) -> None:
    if not grounded_goal_evaluation_admissible(space):
        raise ValueError("grounded goal evaluation is not admissible")
    if set(instrumentality) != set(space.states):
        raise ValueError("instrumentality must define exactly one value per hypothesis")


def grounded_evidence_mass(space: GroundedHypothesisSpace) -> float:
    """Prior probability mass assigned to the admitted common evidence."""
    if not grounded_goal_evaluation_admissible(space):
        raise ValueError("grounded goal evaluation is not admissible")

    mass = sum(
        state.confidence * space.candidates[hypothesis].likelihood_h
        for hypothesis, state in space.states.items()
    )
    if mass <= 0.0:
        raise ValueError("observed evidence must have positive total mass")
    return mass


def grounded_prior_expected_instrumentality(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
) -> float:
    """Prior expectation of learned hypothesis-conditioned instrumentality."""
    _validate_instrumentality(space, instrumentality)
    return sum(
        state.confidence * instrumentality[hypothesis]
        for hypothesis, state in space.states.items()
    )


def grounded_likelihood_instrumentality_covariance(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
) -> float:
    """Prior covariance numerator of evidence likelihood and goal usefulness.

    With normalized priors this is the ordinary covariance. Dividing it by the
    evidence mass gives the observation-induced change in expected
    instrumentality.
    """
    _validate_instrumentality(space, instrumentality)
    mass = grounded_evidence_mass(space)
    prior_expected = grounded_prior_expected_instrumentality(
        space,
        instrumentality,
    )
    mixed_moment = sum(
        state.confidence
        * space.candidates[hypothesis].likelihood_h
        * instrumentality[hypothesis]
        for hypothesis, state in space.states.items()
    )
    return mixed_moment - mass * prior_expected


def grounded_posterior_instrumentality_change(
    space: GroundedHypothesisSpace,
    instrumentality: Mapping[str, float],
) -> float:
    """Actual posterior-minus-prior expected instrumentality change."""
    return grounded_expected_instrumentality(space, instrumentality) - (
        grounded_prior_expected_instrumentality(space, instrumentality)
    )
