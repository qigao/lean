from __future__ import annotations

from collections.abc import Callable, Mapping, Iterable
from typing import TypeVar

Node = TypeVar("Node")
Observation = TypeVar("Observation")
Belief = TypeVar("Belief")


def info_reachable(edges: Mapping[Node, Iterable[Node]], source: Node, target: Node) -> bool:
    """Return whether target is reachable from source through information edges."""
    if source == target:
        return True

    seen = {source}
    frontier = [source]
    while frontier:
        node = frontier.pop()
        for nxt in edges.get(node, ()):  # type: ignore[arg-type]
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def delivered_observation(
    edges: Mapping[Node, Iterable[Node]],
    source: Node,
    target: Node,
    payload: Observation,
) -> Observation | None:
    """Deliver a payload iff an information path exists."""
    return payload if info_reachable(edges, source, target) else None


def apply_observation(
    prior: Belief,
    observation: Observation | None,
    revise: Callable[[Belief, Observation], Belief],
) -> Belief:
    """No delivered observation means no direct belief revision."""
    return prior if observation is None else revise(prior, observation)
