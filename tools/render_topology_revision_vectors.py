"""Canonical, source-derived R1 topology transition and replay vectors.

This uses the same V19/V24 fixture and real production revision APIs as the
test suite. It is an auditable sample, not a second graph algorithm or a
process-authentication certificate.
"""
from __future__ import annotations

import json

from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSnapshot
from narrative_dynamics.abm.mesh_queries import find_mesh_path, mesh_path_is_valid
from narrative_dynamics.abm.mesh_evidence import export_mesh_proof_input, validate_mesh_proof_input
from narrative_dynamics.abm.mesh_topology_events import (
    TopologyOperation, apply_topology_change, observe_topology,
    replay_topology_changes,
)
from narrative_dynamics.abm.situated_mesh import mesh_edges
from tests.test_mesh_projection import fixture, two_societies
from tests.test_mesh_topology_revisions import change, full_mesh, reverse_edge

SCHEMA = "v19-topology-replay-vectors-v1"


def build_vector_manifest() -> dict[str, object]:
    source = fixture(3)
    before = full_mesh(source)
    old_path = find_mesh_path(before, "a0", "a2",
                              relation=MeshRelation.RELATIONSHIP, max_hops=2)
    if old_path is None:
        raise AssertionError("R1 source fixture lost its directed path")
    old_proof = export_mesh_proof_input(before, relation=MeshRelation.RELATIONSHIP)
    first = change(source, "deactivate-01", TopologyOperation.DEACTIVATE,
                   "a0", "a1")
    middle = apply_topology_change(source, first).source_snapshot
    second = change(middle, "add-10", TopologyOperation.ADD, "a1", "a0",
                    edge=reverse_edge())
    replay = replay_topology_changes(source, (first, second))
    final = full_mesh(replay.source_snapshot)
    if mesh_path_is_valid(final, old_path, "a0", "a2",
                          relation=MeshRelation.RELATIONSHIP, max_hops=2):
        raise AssertionError("stale V19 source witness unexpectedly accepted")
    if validate_mesh_proof_input(final, old_proof, relation=MeshRelation.RELATIONSHIP):
        raise AssertionError("stale source-derived proof unexpectedly accepted")
    unchanged = {}
    for relation in (MeshRelation.DIRECT_INTERACTION, MeshRelation.TRANSMISSION):
        current = [list(edge) for edge in mesh_edges(final, relation)]
        if current != [list(edge) for edge in mesh_edges(before, relation)]:
            raise AssertionError("unrelated selected relation changed")
        unchanged[relation.value] = current

    source7 = fixture()
    old_mesh7 = two_societies(source7)
    retire_bridge = change(source7, "deactivate-23", TopologyOperation.DEACTIVATE,
                           "a2", "a3")
    new_source7 = apply_topology_change(source7, retire_bridge).source_snapshot
    failed_bridge = None
    try:
        MeshSnapshot(new_source7, old_mesh7.societies, old_mesh7.bridges)
    except ValueError as exc:
        failed_bridge = str(exc)
    if failed_bridge != "bridge lacks a directed source relation":
        raise AssertionError("invalidated bridge was not rejected by V24 source check")

    return {
        "schema": SCHEMA,
        "notice": "Finite directed topology-edit/replay evidence, not general six-hop proof or authentication",
        "relationship_revision": {
            "initial_source_hash": source.content_hash,
            "initial_round_index": source.round_index,
            "initial_selected_edges": [list(edge) for edge in mesh_edges(before, MeshRelation.RELATIONSHIP)],
            "initial_observation": observe_topology(before, relation=MeshRelation.RELATIONSHIP),
            "old_path": {
                "agent_ids": list(old_path.agent_ids),
                "hop_count": old_path.hop_count,
                "mesh_hash": old_path.mesh_hash,
            },
            "replay": replay.to_dict(),
            "final_source_hash": replay.source_snapshot.content_hash,
            "final_selected_edges": [list(edge) for edge in mesh_edges(final, MeshRelation.RELATIONSHIP)],
            "final_observation": observe_topology(final, relation=MeshRelation.RELATIONSHIP),
            "stale_old_path_valid": False,
            "stale_old_proof_accepted": False,
            "unchanged_other_relations": unchanged,
        },
        "bridge_rejection": {
            "prior_source_hash": source7.content_hash,
            "new_source_hash": new_source7.content_hash,
            "removed_directed_source_edge": ["a2", "a3"],
            "original_directed_bridge": [
                item.to_dict() for item in old_mesh7.bridges
            ],
            "rebuild_with_stale_bridge_rejected": failed_bridge,
            "original_source_untouched": source7.content_hash == old_mesh7.source_snapshot.content_hash,
        },
    }


def render_vector_corpus() -> str:
    return json.dumps(
        build_vector_manifest(), ensure_ascii=False, sort_keys=True, indent=2,
    ) + "\n"


def main() -> None:
    print(render_vector_corpus(), end="")


if __name__ == "__main__":
    main()
