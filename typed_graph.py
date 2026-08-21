from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeKind(Enum):
    AGENT = "agent"
    EVENT = "event"
    OBJECT = "object"
    LOCATION = "location"
    INSTITUTION = "institution"
    CONCEPT = "concept"


class EdgeKind(Enum):
    CAUSAL = "causal"
    OBSERVES = "observes"
    LOCATED_AT = "located_at"
    OWNS = "owns"
    MEMBER_OF = "member_of"


@dataclass(frozen=True)
class TypedNode:
    kind: NodeKind
    id: str


@dataclass(frozen=True)
class TypedEdge:
    kind: EdgeKind
    source: TypedNode
    target: TypedNode


_ALLOWED_ENDPOINTS: dict[EdgeKind, tuple[NodeKind, NodeKind]] = {
    EdgeKind.CAUSAL: (NodeKind.EVENT, NodeKind.EVENT),
    EdgeKind.OBSERVES: (NodeKind.EVENT, NodeKind.AGENT),
    EdgeKind.LOCATED_AT: (NodeKind.AGENT, NodeKind.LOCATION),
    EdgeKind.OWNS: (NodeKind.AGENT, NodeKind.OBJECT),
    EdgeKind.MEMBER_OF: (NodeKind.AGENT, NodeKind.INSTITUTION),
}


def well_typed_edge(edge: TypedEdge) -> bool:
    expected = _ALLOWED_ENDPOINTS[edge.kind]
    return (edge.source.kind, edge.target.kind) == expected


def make_edge(kind: EdgeKind, source: TypedNode, target: TypedNode) -> TypedEdge:
    """Runtime mirror of Lean's constructor-level endpoint typing."""
    edge = TypedEdge(kind, source, target)
    if not well_typed_edge(edge):
        expected_source, expected_target = _ALLOWED_ENDPOINTS[kind]
        raise ValueError(
            f"{kind.value} requires {expected_source.value}->{expected_target.value}, "
            f"got {source.kind.value}->{target.kind.value}"
        )
    return edge
