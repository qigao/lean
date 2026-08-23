from __future__ import annotations

from dataclasses import dataclass, replace

from typed_graph import TypedNode
from provenance import ProofRecord
from truth_maintenance import deactivate_proof, fact_supported


@dataclass(frozen=True)
class BeliefKB:
    """Private symbolic belief KB for one agent.

    `graph` is the agent's subjective provenance DAG; `active` identifies the
    evidence/proof nodes currently accepted by that agent.
    """

    owner: str
    graph: dict[str, ProofRecord]
    active: set[str]


def belief_supported(kb: BeliefKB, fact: TypedNode) -> bool:
    """A belief is held while at least one valid active proof supports it."""
    return fact_supported(kb.graph, kb.active, fact)


def revoke_proof(kb: BeliefKB, proof_id: str) -> BeliefKB:
    """Return a revised private KB after one proof/evidence node is revoked."""
    return replace(kb, active=deactivate_proof(kb.active, proof_id))
