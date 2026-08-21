from __future__ import annotations

from dataclasses import replace

from epistemic_belief import EpistemicBeliefState
from grounded_hypothesis_space import GroundedHypothesisSpace, grounded_posteriors


def posterior_belief_states(
    space: GroundedHypothesisSpace,
) -> dict[str, EpistemicBeliefState]:
    """Write normalized posterior confidence into each grounded hypothesis state.

    Symbolic KB/provenance coordinates and hypothesis identities are preserved;
    only the numerical confidence coordinate changes.
    """
    posterior = grounded_posteriors(space)
    return {
        key: replace(state, confidence=posterior[key])
        for key, state in space.states.items()
    }
