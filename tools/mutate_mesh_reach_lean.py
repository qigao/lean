"""Real Lean RED for a stale V19 path or a stale source-snapshot identity.

The generator builds the positive graph from the *new authoritative snapshot*
and injects one obsolete claim from the prior revision. The Python validator
independently rejects stale evidence. Lean must fail the false theorem; a
Python exception or missing Lean compiler is not accepted as evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_contracts import MeshRelation
from narrative_dynamics.abm.mesh_evidence import (
    export_mesh_proof_input, validate_mesh_proof_input,
)
from tests.test_mesh_projection import fixture, two_societies
from tools.render_mesh_evidence_lean import covered_case
from tools.render_mesh_reach_lean import render_path_case


def _add_false_claim(name: str, generated: str, proof: str) -> str:
    suffix = f"\nend {name}\n"
    if not generated.endswith(suffix):
        raise AssertionError("finite Lean namespace suffix changed")
    return (
        "import NarrativeDynamics.Core.MeshProjection\n"
        f"-- An obsolete V19 observation must NOT become a valid V24 proof ({name}).\n\n"
        + generated[:-len(suffix)] + proof + suffix
    )


def render_false_reach_obligation(kind: str) -> str:
    if kind == "stale-path":
        old_mesh = two_societies(fixture())
        earlier = export_mesh_proof_input(
            old_mesh, relation=MeshRelation.RELATIONSHIP,
        )
        if [0, 1, [0, 1]] not in earlier["paths"]:
            raise AssertionError("expected original one-hop path missing")
        source = old_mesh.source_snapshot
        current_edges = tuple(
            edge for edge in source.relationship_edges
            if (edge.source_agent_id, edge.observer_agent_id) != ("a0", "a1")
        )
        if len(current_edges) != len(source.relationship_edges) - 1:
            raise AssertionError("failed to remove exactly one source-backed path edge")
        current_mesh = replace(
            old_mesh, source_snapshot=replace(source, relationship_edges=current_edges),
        )
        current = export_mesh_proof_input(
            current_mesh, relation=MeshRelation.RELATIONSHIP,
        )
        if validate_mesh_proof_input(
            current_mesh, earlier, relation=MeshRelation.RELATIONSHIP,
        ):
            raise AssertionError("previous revision's path evidence was accepted")
        if [0, 1] in current["projected_edges"] or [0, 1, [0, 1]] in current["paths"]:
            raise AssertionError("stale direct edge remains in current projection")
        if not validate_mesh_proof_input(
            current_mesh, current, relation=MeshRelation.RELATIONSHIP,
        ):
            raise AssertionError("fresh authoritative path evidence rejected")
        name = "StaleSourcePath"
        generated = render_path_case(
            name, current, authoritative_source_hash=current_mesh.source_snapshot.content_hash,
        )
        # The old one-hop witness is explicitly false on the new graph. The
        # Lean elaborator must check a missing REAL source-backed projected edge.
        false_claim = (
            "\n-- Rejected claim carried forward from the old source revision.\n"
            "example : ReachWithin projected 1 (0 : Fin 7) (1 : Fin 7) := by\n"
            "  exact ⟨1, by decide, MeshWalk.single (by decide_cbv)⟩\n"
        )
        return _add_false_claim(name, generated, false_claim)

    if kind == "identity":
        old_mesh, gateways, assignments = covered_case()
        prior = export_mesh_proof_input(
            old_mesh, relation=MeshRelation.RELATIONSHIP,
            gateways=gateways, assigned_societies=assignments,
        )
        current_mesh = replace(
            old_mesh,
            source_snapshot=replace(
                old_mesh.source_snapshot,
                round_index=old_mesh.source_snapshot.round_index + 1,
            ),
        )
        expected = export_mesh_proof_input(
            current_mesh, relation=MeshRelation.RELATIONSHIP,
            gateways=gateways, assigned_societies=assignments,
        )
        claimed = deepcopy(expected)
        claimed["source_lineage"]["source_snapshot_hash"] = (
            prior["source_lineage"]["source_snapshot_hash"]
        )
        if (
            claimed["source_lineage"]["source_snapshot_hash"]
            == current_mesh.source_snapshot.content_hash
        ):
            raise AssertionError("old and new snapshot hashes unexpectedly identical")
        # Rehash the forged candidate too: checking only its self-consistency
        # must never be enough to claim this is the current authoritative V19.
        claimed["evidence_hash"] = stable_content_hash({
            key: value for key, value in claimed.items()
            if key != "evidence_hash"
        })
        if validate_mesh_proof_input(
            current_mesh, claimed, relation=MeshRelation.RELATIONSHIP,
            gateways=gateways, assigned_societies=assignments,
        ):
            raise AssertionError("rehashed stale source identity was accepted")
        name = "StaleSourceIdentity"
        generated = render_path_case(
            name, claimed,
            authoritative_source_hash=current_mesh.source_snapshot.content_hash,
        )
        if "example : reportedSourceHash = authoritativeSourceHash := by decide" not in generated:
            raise AssertionError("identity mismatch did not reach Lean")
        return (
            "import NarrativeDynamics.Core.MeshProjection\n"
            "-- Lean checks equality of supplied finite digests; NOT external authentication.\n\n"
            + generated
        )

    raise ValueError("mutation kind must be stale-path or identity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("stale-path", "identity"), required=True)
    args = parser.parse_args()
    print(render_false_reach_obligation(args.kind), end="")


if __name__ == "__main__":
    main()
