"""Executable finite checks for the *premises* of the conditional Lean six-hop theorem.

This module does not certify Python with the Lean kernel. It checks a concrete,
immutable, source-backed topology against the same structural obligations as
SocietyGatewayModel + OneHopGatewayCover + GlobalHopBound societyGraph 4.
Each assessment binds the complete caller configuration and carries finite
society-path witnesses suitable for independent replay.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSnapshot, _relation
from narrative_dynamics.abm.situated_mesh import mesh_edges


@dataclass(frozen=True)
class MeshSixHopAssessment:
    mesh_hash: str
    configuration_hash: str
    relation: MeshRelation
    premises_satisfied: bool
    failures: tuple[str, ...]
    checked_agent_count: int
    checked_society_count: int
    society_paths: tuple[tuple[str, str, tuple[str, ...]], ...]


def assess_six_hop_prerequisites(
    snapshot: MeshSnapshot, *, relation: MeshRelation,
    gateways: Mapping[str, str], assigned_societies: Mapping[str, str],
) -> MeshSixHopAssessment:
    """Fail closed on an incomplete gateway model and report actual witnesses.

    Society-level edges are realized *only* by existing agent edges between
    designated gateways. Every ordered society pair needs <=4 such edges.
    Each agent needs <=1 hop *each direction* to its assigned gateway.
    """
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("six-hop assessment requires MeshSnapshot")
    _relation(relation)
    if not isinstance(gateways, Mapping) or not isinstance(assigned_societies, Mapping):
        raise TypeError("gateway and assignment inputs must be mappings")

    # Capture both caller-owned mappings exactly once. Validation, identity
    # hashing and graph assessment must use the same bounded configuration
    # even when a Mapping changes its returned values between reads.
    frozen_gateways = dict(gateways.items())
    frozen_assignments = dict(assigned_societies.items())
    societies = {s.society_id: frozenset(s.member_agent_ids) for s in snapshot.societies}
    agents = frozenset(node.agent_id for node in snapshot.source_snapshot.nodes)
    if set(frozen_gateways) != set(societies):
        raise ValueError("gateways must name each and only each society")
    if set(frozen_assignments) != agents:
        raise ValueError("assignments must name each and only each source agent")
    if any(not isinstance(g, str) or g not in agents for g in frozen_gateways.values()):
        raise ValueError("gateways must name known source agents")
    if any(not isinstance(s, str) or s not in societies for s in frozen_assignments.values()):
        raise ValueError("assignments must name known societies")

    # Immutable evidence identity includes *all* selected gateway/assignment
    # inputs, not only the authoritative runtime snapshot's content hash.
    configuration_hash = stable_content_hash({
        "schema": "mesh-six-hop-premises-v24",
        "mesh_hash": snapshot.content_hash,
        "relation": relation.value,
        "gateways": sorted(frozen_gateways.items()),
        "assigned_societies": sorted(frozen_assignments.items()),
    })
    if not societies or not agents:
        return MeshSixHopAssessment(
            snapshot.content_hash, configuration_hash, relation, False,
            ("nonempty proof domain required for this executable gate",),
            len(agents), len(societies), (),
        )

    failures = []
    for society_id, gateway in sorted(frozen_gateways.items()):
        if gateway not in societies[society_id]:
            failures.append(f"gateway {society_id}:{gateway} is not a member")
    for agent_id, society_id in sorted(frozen_assignments.items()):
        if agent_id not in societies[society_id]:
            failures.append(f"agent {agent_id} is not a member of assigned society {society_id}")
    if failures:
        return MeshSixHopAssessment(
            snapshot.content_hash, configuration_hash, relation, False,
            tuple(failures), len(agents), len(societies), (),
        )

    edges = set(mesh_edges(snapshot, relation))
    for agent_id, society_id in sorted(frozen_assignments.items()):
        gate = frozen_gateways[society_id]
        if agent_id != gate:
            if (agent_id, gate) not in edges:
                failures.append(f"missing one-hop to gateway: {agent_id}->{gate}")
            if (gate, agent_id) not in edges:
                failures.append(f"missing one-hop from gateway: {gate}->{agent_id}")

    # The society graph is materialized from the *projected* directed edges
    # between designated gateways. Overlap never creates a synthetic edge.
    links: dict[str, list[str]] = {}
    for source_society, source_gate in sorted(frozen_gateways.items()):
        links[source_society] = sorted(
            target_society
            for target_society, target_gate in frozen_gateways.items()
            if source_society != target_society and (source_gate, target_gate) in edges
        )
    routes: list[tuple[str, str, tuple[str, ...]]] = []
    for origin in sorted(societies):
        paths: dict[str, tuple[str, ...]] = {origin: (origin,)}
        pending = deque([origin])
        while pending:
            current = pending.popleft()
            if len(paths[current]) - 1 >= 4:
                continue
            for neighbor in links[current]:
                if neighbor not in paths:
                    paths[neighbor] = (*paths[current], neighbor)
                    pending.append(neighbor)
        for target in sorted(societies):
            if target not in paths:
                failures.append(f"no <=4-hop gateway-society path: {origin}->{target}")
            else:
                routes.append((origin, target, paths[target]))

    return MeshSixHopAssessment(
        snapshot.content_hash, configuration_hash, relation, not failures,
        tuple(failures), len(agents), len(societies), tuple(routes),
    )
