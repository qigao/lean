from __future__ import annotations

from grounded_hypothesis_space import GroundedHypothesisSpace
from interpretation_commitment import interpretation_committed


def committed_interpretations(
    space: GroundedHypothesisSpace,
    threshold: float,
) -> set[str]:
    """Return every grounded interpretation committed above `threshold`.

    A normalized posterior guarantees at most one result when the threshold is
    at least one half. Lower thresholds intentionally permit multiple tentative
    commitments.
    """
    return {
        hypothesis
        for hypothesis in space.states
        if interpretation_committed(space, hypothesis, threshold)
    }
