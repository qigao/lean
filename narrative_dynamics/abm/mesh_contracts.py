"""Immutable V24 society-mesh contracts over one authoritative V19 network snapshot.

A mesh is a read-only projection, not a second social-memory, world, or
delivery runtime. Membership and bridge declarations are topology inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkSnapshot


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _relation(value: object) -> "MeshRelation":
    if not isinstance(value, MeshRelation):
        raise ValueError("mesh relation must be a MeshRelation")
    return value


class MeshRelation(str, Enum):
    RELATIONSHIP = "relationship"
    DIRECT_INTERACTION = "direct_interaction"
    TRANSMISSION = "transmission"


@dataclass(frozen=True)
class MeshSociety:
    society_id: str
    member_agent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _id(self.society_id, "society id")
        if not isinstance(self.member_agent_ids, tuple):
            raise TypeError("society members must be a tuple")
        for member in self.member_agent_ids:
            _id(member, "society member id")
        if len(set(self.member_agent_ids)) != len(self.member_agent_ids):
            raise ValueError("duplicate society member")
        object.__setattr__(self, "member_agent_ids", tuple(sorted(self.member_agent_ids)))

    def to_dict(self) -> dict[str, object]:
        return {"society_id": self.society_id, "member_agent_ids": list(self.member_agent_ids)}


@dataclass(frozen=True)
class MeshBridge:
    relation: MeshRelation
    source_society_id: str
    target_society_id: str
    source_agent_id: str
    target_agent_id: str

    def __post_init__(self) -> None:
        _relation(self.relation)
        for label in ("source_society_id", "target_society_id", "source_agent_id", "target_agent_id"):
            _id(getattr(self, label), label)
        if self.source_society_id == self.target_society_id:
            raise ValueError("mesh bridge must connect distinct societies")
        if self.source_agent_id == self.target_agent_id:
            raise ValueError("mesh bridge endpoints must differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation.value,
            "source_society_id": self.source_society_id,
            "target_society_id": self.target_society_id,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
        }


def source_relation_edges(
    source: SituatedNetworkSnapshot, relation: MeshRelation
) -> tuple[tuple[str, str], ...]:
    """The only source of edges; transmission records are not a temporal walk."""
    if not isinstance(source, SituatedNetworkSnapshot):
        raise TypeError("mesh source must be SituatedNetworkSnapshot")
    _relation(relation)
    if relation is MeshRelation.RELATIONSHIP:
        records = (e for e in source.relationship_edges if e.active)
    elif relation is MeshRelation.DIRECT_INTERACTION:
        records = (e for e in source.access_edges if e.direct_interaction)
    else:
        records = iter(source.transmissions)
    return tuple(sorted({(e.source_agent_id, e.observer_agent_id) for e in records}))


@dataclass(frozen=True)
class MeshSnapshot:
    source_snapshot: SituatedNetworkSnapshot
    societies: tuple[MeshSociety, ...]
    bridges: tuple[MeshBridge, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot, SituatedNetworkSnapshot):
            raise TypeError("mesh requires an authoritative SituatedNetworkSnapshot")
        if not isinstance(self.societies, tuple) or any(
            not isinstance(s, MeshSociety) for s in self.societies
        ):
            raise TypeError("societies must be a tuple of MeshSociety")
        if not isinstance(self.bridges, tuple) or any(
            not isinstance(b, MeshBridge) for b in self.bridges
        ):
            raise TypeError("bridges must be a tuple of MeshBridge")
        societies = {s.society_id: s for s in self.societies}
        if len(societies) != len(self.societies):
            raise ValueError("duplicate society id")
        nodes = {node.agent_id for node in self.source_snapshot.nodes}
        for society in self.societies:
            if not set(society.member_agent_ids) <= nodes:
                raise ValueError("unknown society member")
        # Bridge validity is checked eagerly, even if no query is ever run.
        backed = {r: set(source_relation_edges(self.source_snapshot, r)) for r in MeshRelation}
        keys: set[tuple[str, str, str, str, str]] = set()
        for bridge in self.bridges:
            source_society = societies.get(bridge.source_society_id)
            target_society = societies.get(bridge.target_society_id)
            if source_society is None or target_society is None:
                raise ValueError("unknown mesh bridge society")
            if (bridge.source_agent_id not in source_society.member_agent_ids
                    or bridge.target_agent_id not in target_society.member_agent_ids):
                raise ValueError("bridge endpoint outside declared society")
            if (bridge.source_agent_id, bridge.target_agent_id) not in backed[bridge.relation]:
                raise ValueError("bridge lacks a directed source relation")
            key = (
                bridge.relation.value, bridge.source_society_id, bridge.target_society_id,
                bridge.source_agent_id, bridge.target_agent_id,
            )
            if key in keys:
                raise ValueError("duplicate mesh bridge declaration")
            keys.add(key)
        object.__setattr__(self, "societies", tuple(sorted(self.societies, key=lambda s: s.society_id)))
        object.__setattr__(self, "bridges", tuple(sorted(
            self.bridges,
            key=lambda b: (b.relation.value, b.source_society_id, b.target_society_id,
                           b.source_agent_id, b.target_agent_id),
        )))

    @property
    def content_hash(self) -> str:
        return stable_content_hash({
            "schema": "situated-mesh-v24",
            "source_snapshot_hash": self.source_snapshot.content_hash,
            "societies": [s.to_dict() for s in self.societies],
            "bridges": [b.to_dict() for b in self.bridges],
        })


@dataclass(frozen=True)
class MeshPath:
    mesh_hash: str
    relation: MeshRelation
    society_id: str | None
    agent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mesh_hash, str) or _HASH.fullmatch(self.mesh_hash) is None:
            raise ValueError("mesh path requires content hash")
        _relation(self.relation)
        if self.society_id is not None:
            _id(self.society_id, "path society id")
        if not isinstance(self.agent_ids, tuple) or not self.agent_ids:
            raise ValueError("path must contain at least one agent")
        for agent_id in self.agent_ids:
            _id(agent_id, "path agent id")

    @property
    def hop_count(self) -> int:
        return len(self.agent_ids) - 1
