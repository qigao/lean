from __future__ import annotations

from typed_graph import TypedNode
from typed_hypergraph import TypedHyperedge


def premises_satisfied(facts: set[TypedNode], edge: TypedHyperedge) -> bool:
    """All typed premises must already be true before the hyperedge can fire."""
    return all(premise in facts for premise in edge.premises)


def fire_hyperedge(facts: set[TypedNode], edge: TypedHyperedge) -> set[TypedNode]:
    """Monotone one-step inference: preserve facts and add the conclusion iff enabled."""
    result = set(facts)
    if premises_satisfied(facts, edge):
        result.add(edge.conclusion)
    return result
