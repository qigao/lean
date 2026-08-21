from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

from typed_graph import TypedNode
from epistemic_belief import EpistemicBeliefState, revoke_belief_evidence
from interpretation import InterpretationCandidate
from cognitive_pipeline import perceptual_interpretation_admissible
from hypothesis_competition import posterior_distribution
from world_graph import WorldState


@dataclass(frozen=True)
class GroundedHypothesisSpace:
    """Competing causal interpretations of one admitted observation.

    Each hypothesis has its own prior/confidence and attribution proof, while
    every candidate must share one raw observation fact and evidence proof.
    """

    world: WorldState
    event: str
    states: Mapping[str, EpistemicBeliefState]
    candidates: Mapping[str, InterpretationCandidate]
    common_evidence: TypedNode
    common_evidence_proof: str


def grounded_hypothesis_space_admissible(
    space: GroundedHypothesisSpace,
) -> bool:
    """Validate one agent's finite, provenance-grounded interpretation space."""
    keys = set(space.states)
    if not keys or keys != set(space.candidates):
        return False

    owners = {state.kb.owner for state in space.states.values()}
    if len(owners) != 1:
        return False

    # The competing coordinates must be views over the same private evidence
    # base, not unrelated KBs that happen to mention the same event.
    first_state = next(iter(space.states.values()))
    for state in space.states.values():
        if state.kb != first_state.kb:
            return False

    for key in keys:
        state = space.states[key]
        candidate = space.candidates[key]
        if candidate.hypothesis != state.hypothesis:
            return False
        if candidate.evidence != space.common_evidence:
            return False
        if candidate.evidence_proof != space.common_evidence_proof:
            return False
        if not perceptual_interpretation_admissible(
            space.world,
            state,
            space.event,
            candidate,
        ):
            return False
    return True


def grounded_posteriors(space: GroundedHypothesisSpace) -> dict[str, float]:
    """Normalize priors by each grounded candidate's evidence likelihood."""
    if not grounded_hypothesis_space_admissible(space):
        raise ValueError("hypothesis space is not grounded in one admitted observation")

    priors = {key: state.confidence for key, state in space.states.items()}
    likelihoods = {
        key: space.candidates[key].likelihood_h
        for key in space.states
    }
    return posterior_distribution(priors, likelihoods)


def revoke_common_evidence(
    space: GroundedHypothesisSpace,
) -> GroundedHypothesisSpace:
    """Retract the common observation proof from every hypothesis coordinate."""
    revised_states = {
        key: revoke_belief_evidence(state, space.common_evidence_proof)
        for key, state in space.states.items()
    }
    return replace(space, states=revised_states)
