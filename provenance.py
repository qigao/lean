from __future__ import annotations

from dataclasses import dataclass

from typed_graph import TypedNode
from typed_hypergraph import TypedHyperedge


@dataclass(frozen=True)
class ProofRecord:
    fact: TypedNode
    origin: TypedHyperedge | None
    parents: tuple[str, ...]
    rank: int


def validate_provenance(graph: dict[str, ProofRecord]) -> bool:
    """Validate provenance roots, rule agreement, support facts, and rank DAG order."""
    for proof_id, record in graph.items():
        if record.rank < 0:
            return False

        if record.origin is None:
            if record.parents:
                return False
            continue

        if record.fact != record.origin.conclusion:
            return False
        if len(record.parents) != len(record.origin.premises):
            return False

        parent_facts: list[TypedNode] = []
        for parent_id in record.parents:
            parent = graph.get(parent_id)
            if parent is None:
                return False
            if not parent.rank < record.rank:
                return False
            parent_facts.append(parent.fact)

        if tuple(parent_facts) != record.origin.premises:
            return False

    return True


def support_reachable(
    graph: dict[str, ProofRecord],
    source: str,
    target: str,
) -> bool:
    """Return whether a nonempty support path runs from source proof to target proof."""
    if source not in graph or target not in graph:
        return False

    children: dict[str, list[str]] = {}
    for child_id, record in graph.items():
        for parent_id in record.parents:
            children.setdefault(parent_id, []).append(child_id)

    seen = {source}
    frontier = [source]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, ()):
            if child == target:
                return True
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return False


def proofs_for_fact(
    graph: dict[str, ProofRecord],
    fact: TypedNode,
) -> set[str]:
    """Return every proof identifier that concludes the requested fact."""
    return {proof_id for proof_id, record in graph.items() if record.fact == fact}
