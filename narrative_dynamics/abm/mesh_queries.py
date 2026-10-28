"""Deterministic bounded, provenance-bound queries on immutable V24 topology."""
from __future__ import annotations

from collections import deque

from narrative_dynamics.abm.mesh_contracts import MeshPath, MeshRelation, MeshSnapshot, _relation
from narrative_dynamics.abm.situated_mesh import _society, mesh_edges


def _query(
    snapshot: MeshSnapshot, source: str, target: str,
    relation: MeshRelation, max_hops: int, society_id: str | None,
) -> set[str]:
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("mesh query requires MeshSnapshot")
    _relation(relation)
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops < 0:
        raise ValueError("max_hops must be a non-negative integer")
    if society_id is not None:
        inside = set(_society(snapshot, society_id).member_agent_ids)
    else:
        inside = {node.agent_id for node in snapshot.source_snapshot.nodes}
    all_nodes = {node.agent_id for node in snapshot.source_snapshot.nodes}
    for endpoint in (source, target):
        if not isinstance(endpoint, str) or endpoint not in all_nodes:
            raise ValueError("unknown mesh path endpoint")
        if endpoint not in inside:
            raise ValueError("mesh path endpoint outside society")
    return inside


def find_mesh_path(
    snapshot: MeshSnapshot, source: str, target: str, *,
    relation: MeshRelation, max_hops: int, society_id: str | None = None,
) -> MeshPath | None:
    """Shortest inclusive-budget witness; lexicographic tie-breaking.

    The witness is topological, not a temporal delivery claim or a signature.
    """
    _query(snapshot, source, target, relation, max_hops, society_id)
    if source == target:
        return MeshPath(snapshot.content_hash, relation, society_id, (source,))
    neighbors: dict[str, list[str]] = {}
    for a, b in mesh_edges(snapshot, relation, society_id=society_id):
        neighbors.setdefault(a, []).append(b)
    for values in neighbors.values():
        values.sort()
    pending: deque[tuple[str, ...]] = deque([(source,)])
    seen = {source}
    while pending:
        current = pending.popleft()
        if len(current) - 1 >= max_hops:
            continue
        for neighbor in neighbors.get(current[-1], ()):
            if neighbor in seen:
                continue
            extended = (*current, neighbor)
            if neighbor == target:
                return MeshPath(snapshot.content_hash, relation, society_id, extended)
            seen.add(neighbor)
            pending.append(extended)
    return None


def mesh_path_is_valid(
    snapshot: MeshSnapshot, path: MeshPath, source: str, target: str, *,
    relation: MeshRelation, max_hops: int, society_id: str | None = None,
) -> bool:
    """Validate caller's query, not just witness metadata; no shortestness claim."""
    inside = _query(snapshot, source, target, relation, max_hops, society_id)
    if not isinstance(path, MeshPath):
        return False
    if (path.mesh_hash != snapshot.content_hash
            or path.relation is not relation or path.society_id != society_id
            or path.agent_ids[0] != source or path.agent_ids[-1] != target
            or path.hop_count > max_hops or any(a not in inside for a in path.agent_ids)):
        return False
    edges = set(mesh_edges(snapshot, relation, society_id=society_id))
    return all((a, b) in edges for a, b in zip(path.agent_ids, path.agent_ids[1:]))


def is_closed_society(
    snapshot: MeshSnapshot, society_id: str, *, relation: MeshRelation,
) -> bool:
    """No outgoing effective edge, independent of incoming edges."""
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("is_closed_society requires MeshSnapshot")
    _relation(relation)
    members = set(_society(snapshot, society_id).member_agent_ids)
    return all(a not in members or b in members for a, b in mesh_edges(snapshot, relation))
