"""V24 projections over a single immutable situated runtime network snapshot."""
from __future__ import annotations

from narrative_dynamics.abm.mesh_contracts import (
    MeshRelation, MeshSnapshot, MeshSociety, _relation, source_relation_edges,
)


def _society(snapshot: MeshSnapshot, society_id: str) -> MeshSociety:
    if not isinstance(society_id, str) or not society_id.strip():
        raise ValueError("society id must be a non-empty string")
    for society in snapshot.societies:
        if society.society_id == society_id:
            return society
    raise ValueError("unknown mesh society")


def mesh_edges(
    snapshot: MeshSnapshot, relation: MeshRelation, *, society_id: str | None = None
) -> tuple[tuple[str, str], ...]:
    """Return directed, relation-selected, source-backed effective edges."""
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("mesh_edges requires MeshSnapshot")
    _relation(relation)
    scope = None if society_id is None else set(_society(snapshot, society_id).member_agent_ids)
    memberships = {s.society_id: set(s.member_agent_ids) for s in snapshot.societies}
    bridges = {
        (b.source_agent_id, b.target_agent_id)
        for b in snapshot.bridges if b.relation is relation
    }
    effective = set()
    for source, target in source_relation_edges(snapshot.source_snapshot, relation):
        internal = any(source in members and target in members for members in memberships.values())
        if internal or (source, target) in bridges:
            if scope is None or (source in scope and target in scope):
                effective.add((source, target))
    return tuple(sorted(effective))


def society_gateway_edges(
    snapshot: MeshSnapshot, relation: MeshRelation,
) -> tuple[tuple[str, str, str, str], ...]:
    """Actual directed inter-society declarations; not new runtime edges.

    Returns (source society, target society, source agent, target agent).
    Overlap is represented separately: it is not a fabricated transmission.
    """
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("society_gateway_edges requires MeshSnapshot")
    _relation(relation)
    return tuple(sorted((
        b.source_society_id, b.target_society_id, b.source_agent_id, b.target_agent_id
    ) for b in snapshot.bridges if b.relation is relation))


def overlapping_memberships(
    snapshot: MeshSnapshot,
) -> tuple[tuple[str, str, str], ...]:
    """Identity gateways (society A, society B, same agent), zero source hops."""
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("overlapping_memberships requires MeshSnapshot")
    result = []
    for left in snapshot.societies:
        for right in snapshot.societies:
            if left.society_id != right.society_id:
                for member in set(left.member_agent_ids) & set(right.member_agent_ids):
                    result.append((left.society_id, right.society_id, member))
    return tuple(sorted(result))
