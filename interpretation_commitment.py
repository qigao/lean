from __future__ import annotations

from grounded_hypothesis_space import (
    GroundedHypothesisSpace,
    grounded_hypothesis_space_admissible,
    grounded_posteriors,
)
from truth_maintenance import proof_valid


def interpretation_committed(
    space: GroundedHypothesisSpace,
    hypothesis: str,
    threshold: float,
) -> bool:
    """Whether one particular grounded attribution is actively committed.

    This is interpretation-specific: the candidate's own attribution proof
    must remain valid, and its normalized posterior must exceed `threshold`.
    Proposition-level belief may have other independent proof paths.
    """
    if hypothesis not in space.states or hypothesis not in space.candidates:
        return False
    if not grounded_hypothesis_space_admissible(space):
        return False

    state = space.states[hypothesis]
    candidate = space.candidates[hypothesis]
    if not proof_valid(
        state.kb.graph,
        state.kb.active,
        candidate.attribution_proof,
    ):
        return False

    posterior = grounded_posteriors(space)
    return threshold < posterior[hypothesis]
