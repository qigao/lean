from __future__ import annotations

from epistemic_belief import EpistemicBeliefState
from interpretation import (
    InterpretationCandidate,
    apply_interpretation,
    interpretation_admissible,
)
from observation_admission import observation_evidence_admissible
from world_graph import WorldState


def perceptual_interpretation_admissible(
    world: WorldState,
    state: EpistemicBeliefState,
    event: str,
    candidate: InterpretationCandidate,
) -> bool:
    """Require both a legal world observation and a grounded private interpretation."""
    return observation_evidence_admissible(
        world,
        state.kb,
        event,
        candidate.evidence_proof,
    ) and interpretation_admissible(state, candidate)


def apply_perceptual_interpretation(
    world: WorldState,
    state: EpistemicBeliefState,
    event: str,
    candidate: InterpretationCandidate,
) -> EpistemicBeliefState:
    """Apply Bayesian interpretation only after the world-to-cognition gate passes."""
    if not perceptual_interpretation_admissible(world, state, event, candidate):
        raise ValueError("perceptual interpretation is not admissible")
    return apply_interpretation(state, candidate)
