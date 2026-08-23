from __future__ import annotations

from typed_graph import TypedNode
from typed_hypergraph import TypedHyperedge
from typed_inference import fire_hyperedge


def forward_chain_once(
    facts: set[TypedNode],
    rules: list[TypedHyperedge],
) -> set[TypedNode]:
    """Apply rules sequentially once, preserving all previously known facts."""
    state = set(facts)
    for rule in rules:
        state = fire_hyperedge(state, rule)
    return state


def forward_chain(
    facts: set[TypedNode],
    rules: list[TypedHyperedge],
) -> set[TypedNode]:
    """Iterate a finite monotone rule set until no new fact is derived."""
    state = set(facts)
    while True:
        next_state = forward_chain_once(state, rules)
        if next_state == state:
            return state
        state = next_state
