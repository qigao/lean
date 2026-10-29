"""Generate intentionally invalid Lean inputs for REAL proof-to-RED tests.

Only test tooling calls this module. It leaves the authoritative V19 snapshot,
its V24 derived evidence, and the accepted canonical Lean corpus untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy

from narrative_dynamics.abm.mesh_contracts import MeshRelation
from narrative_dynamics.abm.mesh_evidence import (
    export_mesh_proof_input, validate_mesh_proof_input,
)
from tools.render_mesh_evidence_lean import covered_case, render_case


def render_false_obligation(kind: str) -> str:
    if kind not in ("source", "bridge"):
        raise ValueError("mutation kind must be source or bridge")
    mesh, gateways, assignments = covered_case()
    original = export_mesh_proof_input(
        mesh, relation=MeshRelation.RELATIONSHIP,
        gateways=gateways, assigned_societies=assignments,
    )
    # Mutate the Lean PREMISE, not the expected conclusion. This ensures that
    # a Python-only validator failure cannot be mistaken for a kernel RED.
    altered = deepcopy(original)
    field, fact = (
        ("source_edges", [0, 1]) if kind == "source"
        else ("bridges", [0, 1, 0, 1])
    )
    if fact not in altered[field] or [0, 1] not in altered["projected_edges"]:
        raise AssertionError("source-backed gateway fixture changed unexpectedly")
    altered[field].remove(fact)
    if validate_mesh_proof_input(
        mesh, altered, relation=MeshRelation.RELATIONSHIP,
        gateways=gateways, assigned_societies=assignments,
    ):
        raise AssertionError("tampered finite evidence unexpectedly accepted")
    rendered = render_case("MutatedGateway", altered)
    if "example : projected (0 : Fin 3) (1 : Fin 3) := by" not in rendered:
        raise AssertionError("mutation lost the original positive Lean obligation")
    return (
        "import NarrativeDynamics.Core.MeshProjection\n"
        f"-- Intentionally false {kind} premise, expected Lean RED.\n\n"
        + rendered
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("source", "bridge"), required=True)
    args = parser.parse_args()
    print(render_false_obligation(args.kind), end="")


if __name__ == "__main__":
    main()
