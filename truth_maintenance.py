from __future__ import annotations

from typed_graph import TypedNode
from provenance import ProofRecord


def deactivate_proof(active: set[str], revoked: str) -> set[str]:
    """Deactivate one proof identifier without mutating the caller's set."""
    result = set(active)
    result.discard(revoked)
    return result


def proof_valid(
    graph: dict[str, ProofRecord],
    active: set[str],
    proof_id: str,
) -> bool:
    """A proof is valid iff it and all provenance ancestors are active."""
    if proof_id not in graph or proof_id not in active:
        return False

    seen: set[str] = set()
    frontier = list(graph[proof_id].parents)
    while frontier:
        parent_id = frontier.pop()
        if parent_id in seen:
            continue
        seen.add(parent_id)

        parent = graph.get(parent_id)
        if parent is None or parent_id not in active:
            return False
        frontier.extend(parent.parents)

    return True


def fact_supported(
    graph: dict[str, ProofRecord],
    active: set[str],
    fact: TypedNode,
) -> bool:
    """A fact remains true while at least one valid proof concludes it."""
    return any(
        record.fact == fact and proof_valid(graph, active, proof_id)
        for proof_id, record in graph.items()
    )
