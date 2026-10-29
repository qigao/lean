"""Generate finite Lean projection checks from authoritative V19 runtime evidence.

Lean recomputes each projected edge using the existing V24 projectedMesh,
not a copy of the Python projection algorithm. Hashes remain Python-side
provenance metadata; generating Lean source is not process authentication.
"""
from __future__ import annotations

from dataclasses import replace

from narrative_dynamics.abm.mesh_contracts import (
    MeshBridge, MeshRelation, MeshSnapshot, MeshSociety,
)
from narrative_dynamics.abm.mesh_evidence import (
    export_mesh_proof_input, validate_mesh_proof_input,
)
from narrative_dynamics.abm.situated_network import project_situated_network_snapshot
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRelationshipEdge,
)
from tests.test_mesh_projection import fixture, two_societies
from tests.test_network_abm_situated_network import initial_runtime_case


def _list(values: object) -> str:
    """Render only lists of unsigned finite indices, never source text."""
    if not isinstance(values, list):
        raise TypeError("finite Lean inputs require canonical lists")
    if not values:
        return "[]"
    return "[" + ", ".join(
        _list(x) if isinstance(x, list) else _nat(x) for x in values
    ) + "]"


def _nat(value: object) -> str:
    if type(value) is not int or value < 0:
        raise ValueError("Lean fixture indices must be nonnegative integers")
    return str(value)


def render_case(name: str, evidence: dict[str, object]) -> str:
    """Every directed pair receives a kernel-checked true/false observation."""
    if not name.isidentifier():
        raise ValueError("Lean case name must be an identifier")
    nodes = evidence["agents"]
    societies = evidence["societies"]
    n, m = len(nodes), len(societies)
    if not 0 < n <= 8 or not 0 < m <= 4:
        raise ValueError("Lean conformance fixture exceeds bounded domain")
    edges = {tuple(edge) for edge in evidence["projected_edges"]}
    lines = [
        f"namespace {name}",
        "open NarrativeDynamics",
        f"private def sourceGraph : MeshGraph (Fin {n}) :=",
        f"  fun a b => [a.val, b.val] ∈ ({_list(evidence['source_edges'])} : List (List Nat))",
        f"private def member (a : Fin {n}) (s : Fin {m}) : Prop :=",
        f"  [a.val, s.val] ∈ ({_list(evidence['membership'])} : List (List Nat))",
        f"private def bridge (_ : Unit) (s t : Fin {m}) (a b : Fin {n}) : Prop :=",
        f"  [s.val, t.val, a.val, b.val] ∈ ({_list(evidence['bridges'])} : List (List Nat))",
        f"private def projected : MeshGraph (Fin {n}) :=",
        "  projectedMesh (fun (_ : Unit) => sourceGraph) member bridge ()",
        "",
    ]
    for i in range(n):
        for j in range(n):
            proposition = f"projected ({i} : Fin {n}) ({j} : Fin {n})"
            if (i, j) not in edges:
                proposition = "¬ " + proposition
            lines.extend([
                f"example : {proposition} := by",
                "  decide_cbv",
            ])
    lines.extend(["", f"end {name}", ""])
    return "\n".join(lines)


def real_runtime_case() -> MeshSnapshot:
    model, story, cognition, social = initial_runtime_case()
    source = project_situated_network_snapshot(model, story, cognition, social)
    return MeshSnapshot(source, (
        MeshSociety("office", tuple(n.agent_id for n in source.nodes)),
    ))


def covered_case() -> tuple[MeshSnapshot, dict[str, str], dict[str, str]]:
    source = fixture(3)
    edges = tuple(
        SituatedNetworkRelationshipEdge(
            f"a{i}", f"a{j}", 1.0, 0.0, 0, 0, True
        )
        for i in range(3) for j in range(3) if i != j
    )
    mesh = MeshSnapshot(
        replace(source, relationship_edges=edges),
        (MeshSociety("s0", ("a0",)),
         MeshSociety("s1", ("a1", "a2"))),
        (MeshBridge(MeshRelation.RELATIONSHIP, "s0", "s1", "a0", "a1"),
         MeshBridge(MeshRelation.RELATIONSHIP, "s1", "s0", "a1", "a0")),
    )
    return mesh, {"s0": "a0", "s1": "a1"}, {
        "a0": "s0", "a1": "s1", "a2": "s1",
    }


def render_corpus() -> str:
    office = real_runtime_case()
    chain = two_societies(fixture())
    covered, gates, assigned = covered_case()
    cases = [
        ("OfficeRelationship", office, MeshRelation.RELATIONSHIP, None, None),
        ("OfficeInteraction", office, MeshRelation.DIRECT_INTERACTION, None, None),
        ("OfficeTransmission", office, MeshRelation.TRANSMISSION, None, None),
        ("DirectedChain", chain, MeshRelation.RELATIONSHIP,
         {"left": "a2", "right": "a3"},
         {**{f"a{i}": "left" for i in range(3)},
          **{f"a{i}": "right" for i in range(3, 7)}}),
        ("CoveredGateways", covered, MeshRelation.RELATIONSHIP, gates, assigned),
    ]
    fragments = [
        "import NarrativeDynamics.Core.MeshProjection",
        "-- Generated from authoritative V19 snapshots; kernel checks V24 math.",
        "-- This file is ephemeral, not a proof of external process identity.",
        "",
    ]
    for name, mesh, relation, gateways, assignments in cases:
        proof_input = export_mesh_proof_input(
            mesh, relation=relation,
            gateways=gateways, assigned_societies=assignments,
        )
        if not validate_mesh_proof_input(
            mesh, proof_input, relation=relation,
            gateways=gateways, assigned_societies=assignments,
        ):
            raise AssertionError("unreproducible source-derived finite input")
        fragments.append(render_case(name, proof_input))
    return "\n".join(fragments)


def main() -> None:
    print(render_corpus(), end="")


if __name__ == "__main__":
    main()
