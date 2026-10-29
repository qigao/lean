"""Versioned, reviewable finite V19/V24 proof input vectors.

Reuses the *same* runtime fixture cases as the kernel-checked Lean scripts and
recomputes every field from an authoritative MeshSnapshot. This is a text
artifact for review and CI comparison, never a second graph implementation or
a claim that Lean authenticates external processes.
"""
from __future__ import annotations

import json

from narrative_dynamics.abm.mesh_contracts import MeshRelation
from narrative_dynamics.abm.mesh_evidence import (
    export_mesh_proof_input, validate_mesh_proof_input,
)
from tools.render_mesh_evidence_lean import real_runtime_case
from tools.render_mesh_reach_lean import runtime_proof_cases

SCHEMA = "situated-mesh-runtime-proof-vectors-v1"


def build_vector_manifest() -> dict[str, object]:
    office = real_runtime_case()
    cases = [
        ("OfficeRelationship", office, MeshRelation.RELATIONSHIP, None, None, "edge"),
        ("OfficeInteraction", office, MeshRelation.DIRECT_INTERACTION, None, None, "edge"),
        ("OfficeTransmission", office, MeshRelation.TRANSMISSION, None, None, "edge"),
    ]
    for name, mesh, relation, gateways, assignments, prove_gateway, outsider in runtime_proof_cases():
        kind = ("gateway" if prove_gateway else
                "outsider" if outsider is not None else "path")
        cases.append((name, mesh, relation, gateways, assignments, kind))
    seen = set()
    vectors = []
    for name, mesh, relation, gateways, assignments, kind in cases:
        if name in seen:
            raise AssertionError("duplicate finite proof vector name")
        seen.add(name)
        evidence = export_mesh_proof_input(
            mesh, relation=relation, gateways=gateways,
            assigned_societies=assignments,
        )
        if not validate_mesh_proof_input(
            mesh, evidence, relation=relation, gateways=gateways,
            assigned_societies=assignments,
        ):
            raise AssertionError("unreproducible authoritative vector")
        if evidence["source_lineage"]["source_snapshot_hash"] != mesh.source_snapshot.content_hash:
            raise AssertionError("vector does not match its authoritative V19 snapshot")
        vectors.append({
            "name": name,
            "proof_scope": kind,
            "source_snapshot_hash": mesh.source_snapshot.content_hash,
            "evidence": evidence,
        })
    return {
        "schema": SCHEMA,
        "notice": (
            "Finite V19 evidence for V24 Lean conformance only; not authenticated "
            "external source identity, asynchronous delivery, or a universal six-hop bound"
        ),
        "cases": vectors,
    }


def render_vector_corpus() -> str:
    return json.dumps(
        build_vector_manifest(), ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"


def main() -> None:
    print(render_vector_corpus(), end="")


if __name__ == "__main__":
    main()
