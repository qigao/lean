from __future__ import annotations

from dataclasses import dataclass, replace

from typed_graph import TypedNode
from belief_support import BeliefKB, belief_supported, revoke_proof
from motivation import bayes_posterior


@dataclass(frozen=True)
class EpistemicBeliefState:
    """One hypothesis with private symbolic support and Bayesian confidence."""

    kb: BeliefKB
    hypothesis: TypedNode
    confidence: float


def belief_held(state: EpistemicBeliefState, threshold: float) -> bool:
    """A belief is held only when it is both supported and confident enough."""
    return belief_supported(state.kb, state.hypothesis) and threshold < state.confidence


def bayes_update_belief(
    state: EpistemicBeliefState,
    likelihood_h: float,
    likelihood_not_h: float,
) -> EpistemicBeliefState:
    """Update only Bayesian confidence; symbolic provenance remains unchanged."""
    return replace(
        state,
        confidence=bayes_posterior(
            state.confidence,
            likelihood_h=likelihood_h,
            likelihood_not_h=likelihood_not_h,
        ),
    )


def revoke_belief_evidence(
    state: EpistemicBeliefState,
    proof_id: str,
) -> EpistemicBeliefState:
    """Revoke one symbolic proof while leaving the stored probability intact."""
    return replace(state, kb=revoke_proof(state.kb, proof_id))
