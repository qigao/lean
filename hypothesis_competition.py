from __future__ import annotations

from collections.abc import Mapping


def posterior_distribution(
    priors: Mapping[str, float],
    likelihoods: Mapping[str, float],
) -> dict[str, float]:
    """Normalize finite Bayesian hypothesis weights for one shared observation."""
    if set(priors) != set(likelihoods):
        raise ValueError("priors and likelihoods must describe the same hypotheses")
    if not priors:
        raise ValueError("hypothesis space must be non-empty")

    weights: dict[str, float] = {}
    for hypothesis, prior in priors.items():
        likelihood = likelihoods[hypothesis]
        if prior < 0.0 or likelihood < 0.0:
            raise ValueError("priors and likelihoods must be non-negative")
        weights[hypothesis] = prior * likelihood

    mass = sum(weights.values())
    if mass <= 0.0:
        raise ValueError("observed evidence must have positive total mass")

    return {hypothesis: weight / mass for hypothesis, weight in weights.items()}
