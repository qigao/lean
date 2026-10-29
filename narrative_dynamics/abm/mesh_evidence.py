"""Bounded V24 finite proof inputs derived from one authoritative V19 snapshot.

This is a canonical, recomputable *observation*, never an alternate state
owner, authenticated origin, runtime cache, or claim that Lean executed Python.
The eventual Lean finite checker consumes the numeric graph portion.
"""
from __future__ import annotations

from collections.abc import Mapping

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_contracts import (
    MeshRelation, MeshSnapshot, _relation, source_relation_edges,
)
from narrative_dynamics.abm.mesh_queries import find_mesh_path, mesh_path_is_valid
from narrative_dynamics.abm.mesh_six_hop import assess_six_hop_prerequisites
from narrative_dynamics.abm.situated_mesh import mesh_edges


_SCHEMA = "situated-mesh-proof-input-v24"
_MAX_NODES = 24
_MAX_SOCIETIES = 12


def export_mesh_proof_input(
    snapshot: MeshSnapshot, *, relation: MeshRelation, max_hops: int = 6,
    gateways: Mapping[str, str] | None = None,
    assigned_societies: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Snapshot finite source topology, selected projection and checked paths.

    All edge and path endpoints are canonical indices into the source roster.
    Hash strings establish only application-level source/config identities.
    Caller-owned gateway mappings are sampled once, including on error paths.
    """
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("proof input requires an immutable MeshSnapshot")
    _relation(relation)
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops < 0:
        raise ValueError("proof input max_hops must be a nonnegative integer")
    source = snapshot.source_snapshot
    if len(source.nodes) > _MAX_NODES or len(snapshot.societies) > _MAX_SOCIETIES:
        raise ValueError("proof input finite acceptance bounds exceeded")
    if (gateways is None) != (assigned_societies is None):
        raise ValueError("gateways and assignments must be provided together")
    nodes = tuple(node.agent_id for node in source.nodes)
    node_index = {agent: index for index, agent in enumerate(nodes)}
    society_ids = tuple(s.society_id for s in snapshot.societies)
    society_index = {society: index for index, society in enumerate(society_ids)}

    def edge_list(values: tuple[tuple[str, str], ...]) -> list[list[int]]:
        return [[node_index[a], node_index[b]] for a, b in values]

    source_edges = edge_list(source_relation_edges(source, relation))
    effective_edges = edge_list(mesh_edges(snapshot, relation))
    membership = [
        [node_index[agent], society_index[society.society_id]]
        for society in snapshot.societies for agent in society.member_agent_ids
    ]
    membership.sort()
    bridges = sorted([
        [
            society_index[bridge.source_society_id],
            society_index[bridge.target_society_id],
            node_index[bridge.source_agent_id],
            node_index[bridge.target_agent_id],
        ]
        for bridge in snapshot.bridges if bridge.relation is relation
    ])

    # The original runtime source remains authoritative. No Python graph
    # algorithm is copied: all paths use the production V24 query/checker.
    paths: list[list[object]] = []
    for a in nodes:
        for b in nodes:
            witness = find_mesh_path(
                snapshot, a, b, relation=relation, max_hops=max_hops
            )
            if witness is not None:
                if not mesh_path_is_valid(
                    snapshot, witness, a, b, relation=relation,
                    max_hops=max_hops,
                ):
                    raise AssertionError("production mesh produced invalid witness")
                paths.append([
                    node_index[a], node_index[b],
                    [node_index[agent] for agent in witness.agent_ids],
                ])

    cover: dict[str, object] | None = None
    if gateways is not None and assigned_societies is not None:
        if not isinstance(gateways, Mapping) or not isinstance(
            assigned_societies, Mapping
        ):
            raise TypeError("gateway configuration must be mappings")
        frozen_gateways = dict(gateways.items())
        frozen_assignments = dict(assigned_societies.items())
        assessment = assess_six_hop_prerequisites(
            snapshot, relation=relation,
            gateways=frozen_gateways,
            assigned_societies=frozen_assignments,
        )
        cover = {
            "configuration_hash": assessment.configuration_hash,
            "gateways": [
                [society_index[s], node_index[a]]
                for s, a in sorted(frozen_gateways.items())
            ],
            "assignments": [
                [node_index[a], society_index[s]]
                for a, s in sorted(frozen_assignments.items())
            ],
            "premises_satisfied": assessment.premises_satisfied,
            "failures": list(assessment.failures),
            "society_paths": [
                [
                    society_index[a], society_index[b],
                    [society_index[s] for s in path],
                ] for a, b, path in assessment.society_paths
            ],
        }
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "source_lineage": {
            "source_snapshot_hash": source.content_hash,
            "model_id": source.model_id,
            "model_hash": source.model_hash,
            "round_index": source.round_index,
            "story_hash": source.story_hash,
            "cognitive_state_hash": source.cognitive_state_hash,
            "social_state_hash": source.social_state_hash,
        },
        "mesh_hash": snapshot.content_hash,
        "relation": relation.value,
        "max_hops": max_hops,
        "agents": list(nodes),
        "places": [node.place_id for node in source.nodes],
        "societies": list(society_ids),
        "membership": membership,
        "source_edges": source_edges,
        "projected_edges": effective_edges,
        "bridges": bridges,
        "paths": paths,
        "gateway_cover": cover,
    }
    payload["evidence_hash"] = stable_content_hash(payload)
    return payload


def validate_mesh_proof_input(
    snapshot: MeshSnapshot, evidence: object, *, relation: MeshRelation,
    max_hops: int = 6, gateways: Mapping[str, str] | None = None,
    assigned_societies: Mapping[str, str] | None = None,
) -> bool:
    """Check *every* field against a fresh authoritative source projection.

    The hash alone is not trusted; changing a source edge, round, identity,
    membership, bridge or witness must fail unless the caller explicitly
    requests a new, independently recomputed evidence input.
    """
    if not isinstance(evidence, dict):
        return False
    return evidence == export_mesh_proof_input(
        snapshot, relation=relation, max_hops=max_hops,
        gateways=gateways, assigned_societies=assigned_societies,
    )
