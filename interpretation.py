from __future__ import annotations

from dataclasses import dataclass

from typed_graph import TypedNode
from epistemic_belief import EpistemicBeliefState, bayes_update_belief
from provenance import support_reachable
from truth_maintenance import proof_valid


@dataclass(frozen=True)
class InterpretationCandidate:
    """Evidence-grounded causal interpretation for one hypothesis."""

    evidence: TypedNode
    hypothesis: TypedNode
    evidence_proof: str
    attribution_proof: str
    likelihood_h: float
    likelihood_not_h: float


def interpretation_admissible(
    state: EpistemicBeliefState,
    candidate: InterpretationCandidate,
) -> bool:
    """Require valid evidence, a valid attribution proof, and a support path."""
    graph = state.kb.graph
    active = state.kb.active

    if candidate.hypothesis != state.hypothesis:
        return False
    if candidate.evidence_proof not in graph or candidate.attribution_proof not in graph:
        return False
    if not proof_valid(graph, active, candidate.evidence_proof):
        return False
    if graph[candidate.evidence_proof].fact != candidate.evidence:
        return False
    if not proof_valid(graph, active, candidate.attribution_proof):
        return False
    if graph[candidate.attribution_proof].fact != candidate.hypothesis:
        return False
    if not support_reachable(graph, candidate.evidence_proof, candidate.attribution_proof):
        return False
    return True


def apply_interpretation(
    state: EpistemicBeliefState,
    candidate: InterpretationCandidate,
) -> EpistemicBeliefState:
    """Apply Bayesian update only for a provenance-grounded interpretation."""
    if not interpretation_admissible(state, candidate):
        raise ValueError("interpretation is not grounded in valid evidence")

    return bayes_update_belief(
        state,
        likelihood_h=candidate.likelihood_h,
        likelihood_not_h=candidate.likelihood_not_h,
    )
