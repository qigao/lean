"""Executable finite checks for the *premises* of the conditional Lean six-hop theorem.

This module does not certify Python with the Lean kernel. It checks a concrete,
immutable, source-backed topology against the same structural obligations as
SocietyGatewayModel + OneHopGatewayCover + GlobalHopBound societyGraph 4.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSnapshot, _relation
from narrative_dynamics.abm.situated_mesh import mesh_edges


@dataclass(frozen=True)
class MeshSixHopAssessment:
    mesh_hash: str
    relation: MeshRelation
    premises_satisfied: bool
    failures: tuple[str, ...]
    checked_agent_count: int
    checked_society_count: int


def assess_six_hop_prerequisites(
    snapshot: MeshSnapshot, *, relation: MeshRelation,
    gateways: Mapping[str, str], assigned_societies: Mapping[str, str],
) -> MeshSixHopAssessment:
    """Fail closed on incomplete gateway mappings and non-source-backed edges.

    Society-level edges are realized *only* by existing agent edges between
    designated gateways. Every ordered society pair needs <=4 such edges.
    Each agent needs <=1 hop *each direction* to its assigned gateway.
    """
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("six-hop assessment requires MeshSnapshot")
    _relation(relation)
    if not isinstance(gateways, Mapping) or not isinstance(assigned_societies, Mapping):
        raise TypeError("gateway and assignment inputs must be mappings")
    societies = {s.society_id: frozenset(s.member_agent_ids) for s in snapshot.societies}
    agents = frozenset(node.agent_id for node in snapshot.source_snapshot.nodes)
    if set(gateways) != set(societies):
        raise ValueError("gateways must name each and only each society")
    if set(assigned_societies) != agents:
        raise ValueError("assignments must name each and only each source agent")
    if not societies or not agents:
        return MeshSixHopAssessment(snapshot.content_hash, relation, False,
                                    ("nonempty proof domain required for this executable gate",),
                                    len(agents), len(societies))
    failures = []
    for society_id, gateway in sorted(gateways.items()):
        if gateway not in societies[society_id]:
            failures.append(f"gateway {society_id}:{gateway} is not a member")
    for agent_id, society_id in sorted(assigned_societies.items()):
        if society_id not in societies or agent_id not in societies[society_id]:
            failures.append(f"agent {agent_id} is not a member of assigned society {society_id}")
    if failures:
        return MeshSixHopAssessment(snapshot.content_hash, relation, False,
                                    tuple(failures), len(agents), len(societies))

    edges = set(mesh_edges(snapshot, relation))
    for agent_id, society_id in sorted(assigned_societies.items()):
        gate = gateways[society_id]
        if agent_id != gate:
            if (agent_id, gate) not in edges:
                failures.append(f"missing one-hop to gateway: {agent_id}->{gate}")
            if (gate, agent_id) not in edges:
                failures.append(f"missing one-hop from gateway: {gate}->{agent_id}")

    # The entire society relation is materialized from actual source-backed
    # gateway-to-gateway edges. No generic 'bridge' or overlap fabricates one.
    links: dict[str, list[str]] = {}
    for source_society, source_gate in sorted(gateways.items()):
        links[source_society] = sorted(
            target_society
            for target_society, target_gate in gateways.items()
            if source_society != target_society and (source_gate, target_gate) in edges
        )
    for origin in sorted(societies):
        reached = {origin}
        pending = deque([(origin, 0)])
        while pending:
            current, distance = pending.popleft()
            if distance == 4:
                continue
            for neighbor in links[current]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append((neighbor, distance + 1))
        for target in sorted(set(societies) - reached):
            failures.append(f"no <=4-hop gateway-society path: {origin}->{target}")
    return MeshSixHopAssessment(
        snapshot.content_hash, relation, not failures, tuple(failures),
        len(agents), len(societies),
    )
