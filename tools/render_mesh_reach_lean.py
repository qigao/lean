"""Kernel-check concrete V19 paths and V24 gateway premises.

Every positive fact is generated from a recomputed, validated authoritative
MeshSnapshot. The Lean kernel checks actual MeshWalk edges, Gateway coverage
and the existing conditional six-hop theorem. No second graph implementation,
source owner, native_decide, sorry, axiom, or Python truth assertion is a proof.
"""
from __future__ import annotations

from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSnapshot, MeshSociety
import re
from narrative_dynamics.abm.mesh_evidence import (
    export_mesh_proof_input, validate_mesh_proof_input,
)
from tests.test_mesh_projection import fixture, two_societies
from tools.render_mesh_evidence_lean import covered_case, render_case


def _walk_term(path: list[int], domain: int, edge_prefix: str) -> str:
    if not path:
        raise ValueError("an exact walk requires its starting node")
    term = f"MeshWalk.refl ({path[-1]} : Fin {domain})"
    for source, target in reversed(list(zip(path, path[1:]))):
        term = f"MeshWalk.step {edge_prefix}_{source}_{target} ({term})"
    return term


def _reach(name: str, graph: str, domain: int, budget: int,
           source: int, target: int, path: list[int],
           edge_prefix: str = "projected_edge") -> list[str]:
    if (
        not path or path[0] != source or path[-1] != target
        or len(path) - 1 > budget
        or any(type(i) is not int or i < 0 or i >= domain for i in path)
    ):
        raise ValueError("noncanonical or out-of-budget concrete walk")
    return [
        f"private theorem {name} : ReachWithin {graph} {budget} "
        f"({source} : Fin {domain}) ({target} : Fin {domain}) := by",
        f"  exact ⟨{len(path) - 1}, by decide, {_walk_term(path, domain, edge_prefix)}⟩",
        "",
    ]


def _index_function(name: str, variable: str, domain: int,
                    codomain: int, values: list[int]) -> list[str]:
    if (
        len(values) != domain or domain < 1 or codomain < 1
        or any(type(v) is not int or v < 0 or v >= codomain for v in values)
    ):
        raise ValueError("incomplete finite gateway or society assignment")
    expression = f"({values[-1]} : Fin {codomain})"
    for i in reversed(range(domain - 1)):
        expression = (f"if {variable} = ({i} : Fin {domain}) then "
                      f"({values[i]} : Fin {codomain}) else ({expression})")
    return [
        f"private def {name} ({variable} : Fin {domain}) : Fin {codomain} :=",
        f"  {expression}",
        "",
    ]


def _gateway_proofs(evidence: dict[str, object]) -> list[str]:
    cover = evidence["gateway_cover"]
    if cover is None or cover["premises_satisfied"] is not True or cover["failures"]:
        raise ValueError("cannot assert gateway theorem without positive premises")
    n, m = len(evidence["agents"]), len(evidence["societies"])
    gates = dict(cover["gateways"])
    assignments = dict(cover["assignments"])
    if set(gates) != set(range(m)) or set(assignments) != set(range(n)):
        raise ValueError("incomplete gateway cover identity")
    lines = ["-- Concrete gateway premises, not an assumed Python boolean."]
    lines.extend(_index_function("gateway", "s", m, n, [gates[s] for s in range(m)]))
    lines.extend(_index_function("societyOf", "a", n, m,
                                 [assignments[a] for a in range(n)]))
    lines.extend([
        "private def societyGraph : MeshGraph (Fin " + str(m) + ") :=",
        "  projectedGatewaySocietyGraph (fun (_ : Unit) => sourceGraph) "
        "member bridge () gateway",
        "",
        "private instance : DecidableRel societyGraph := by",
        "  intro s t",
        "  unfold societyGraph projectedGatewaySocietyGraph "
        "projectedMesh sourceGraph member bridge gateway",
        "  infer_instance",
        "",
        "private theorem gatewayMember : ∀ s, member (gateway s) s := by",
        "  intro s",
        "  fin_cases s <;> (unfold member gateway; decide_cbv)",
        "",
        "private theorem assignedMember : ∀ a, member a (societyOf a) := by",
        "  intro a",
        "  fin_cases a <;> (unfold member societyOf; decide_cbv)",
        "",
    ])
    edges = {tuple(edge) for edge in evidence["projected_edges"]}
    for heading, direction in (("toGateway", True), ("fromGateway", False)):
        lines.extend([
            f"private theorem {heading} : ∀ a, "
            + ("ReachWithin projected 1 a (gateway (societyOf a))"
               if direction else
               "ReachWithin projected 1 (gateway (societyOf a)) a")
            + " := by",
            "  intro a",
            "  fin_cases a",
        ])
        for agent in range(n):
            gate = gates[assignments[agent]]
            source, target = ((agent, gate) if direction else (gate, agent))
            if agent == gate:
                statement = "⟨0, by decide, MeshWalk.refl _⟩"
            else:
                if (source, target) not in edges:
                    raise ValueError("missing concrete projected one-hop gateway edge")
                statement = "⟨1, by decide, MeshWalk.single (by decide_cbv)⟩"
            lines.append("  · exact " + statement)
        lines.append("")
    paths = cover["society_paths"]
    # Kernel-check each concrete directed gateway edge once. Reuse opaque
    # edge lemmas in MeshWalk terms to avoid repeated exponential reduction
    # of source/membership/bridge predicates during nested elaboration.
    society_links = {
        (s, t)
        for _origin, _target, path in paths
        for s, t in zip(path, path[1:])
    }
    for s, t in sorted(society_links):
        lines.extend([
            f"private theorem society_edge_{s}_{t} : societyGraph "
            f"({s} : Fin {m}) ({t} : Fin {m}) := by",
            "  decide_cbv",
            "",
        ])
    route_ids = set()
    for s, t, path in paths:
        if (s, t) in route_ids:
            raise ValueError("duplicate concrete society witness")
        route_ids.add((s, t))
        lines.extend(_reach(f"society_path_{s}_{t}", "societyGraph", m, 4,
                            s, t, path, edge_prefix="society_edge"))
    if route_ids != {(s, t) for s in range(m) for t in range(m)}:
        raise ValueError("missing one of the directed society witness pairs")
    lines.extend([
        "private theorem societyBound : GlobalHopBound societyGraph 4 := by",
        "  intro s t",
        "  fin_cases s <;> fin_cases t",
        "  all_goals first",
    ])
    for s in range(m):
        for t in range(m):
            lines.append(f"    | exact society_path_{s}_{t}")
    lines.extend([
        "",
        "-- Existing conditional Lean theorem is applied only to checked premises.",
        "example : GlobalHopBound projected 6 := by",
        "  exact projectedMesh_global_six_hop_bound",
        "    (fun (_ : Unit) => sourceGraph) member bridge () gateway",
        "    gatewayMember societyOf assignedMember toGateway fromGateway societyBound",
        "",
    ])
    return lines


def render_path_case(name: str, evidence: dict[str, object], *,
                     gateway_cover: bool = False,
                     outsider: tuple[int, int] | None = None,
                     authoritative_source_hash: str | None = None) -> str:
    base = render_case(name, evidence)
    suffix = f"\nend {name}\n"
    if not base.endswith(suffix):
        raise AssertionError("underlying source-backed Lean renderer changed")
    n = len(evidence["agents"])
    budget = evidence["max_hops"]
    extra = ["", "-- Every exported V19 projected path rechecked as a Lean MeshWalk."]
    if authoritative_source_hash is not None:
        # These are only supplied finite source-identity literals. Lean checks
        # equality, not the cryptographic hash algorithm or process origin.
        reported = evidence["source_lineage"]["source_snapshot_hash"]
        for value in (reported, authoritative_source_hash):
            if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError("finite source identity must be canonical sha256")
        extra.extend([
            f'private def reportedSourceHash : String := "{reported}"',
            f'private def authoritativeSourceHash : String := "{authoritative_source_hash}"',
            "example : reportedSourceHash = authoritativeSourceHash := by decide",
            "",
        ])
    # Each source-backed edge is independently checked once by Lean.
    # The named proof is reused in long paths; no Python edge claim
    # is ever inserted as an axiom or a trusted assumption.
    for source, target in evidence["projected_edges"]:
        extra.extend([
            f"private theorem projected_edge_{source}_{target} : projected "
            f"({source} : Fin {n}) ({target} : Fin {n}) := by",
            "  decide_cbv",
            "",
        ])
    seen = set()
    for source, target, path in evidence["paths"]:
        if (source, target) in seen:
            raise ValueError("duplicate witness for one ordered agent pair")
        seen.add((source, target))
        extra.extend(_reach(f"runtime_path_{source}_{target}",
                            "projected", n, budget, source, target, path))
    if gateway_cover:
        extra.extend(_gateway_proofs(evidence))
    if outsider is not None:
        society, agent = outsider
        if not (0 <= society < len(evidence["societies"]) and 0 <= agent < n):
            raise ValueError("outsider index outside finite graph")
        if [agent, society] in evidence["membership"]:
            raise ValueError("not an outsider to the requested society")
        extra.extend([
            "-- An unassigned outsider cannot use a local reflexive zero-hop walk.",
            "example : ¬ membershipQualifiedReach "
            "(fun (_ : Unit) => sourceGraph) member bridge () "
            f"({society} : Fin {len(evidence['societies'])}) 0 "
            f"({agent} : Fin {n}) ({agent} : Fin {n}) := by",
            "  apply membershipQualifiedReach_outsider",
            "  unfold member",
            "  decide_cbv",
            "",
        ])
    return base[:-len(suffix)] + "\n".join(extra) + suffix


def render_path_corpus() -> str:
    chain = two_societies(fixture())
    covered, gates, assigned = covered_case()
    overlap = MeshSnapshot(fixture(3), (
        MeshSociety("left", ("a0", "a1")),
        MeshSociety("right", ("a1", "a2")),
    ))
    all_members = MeshSnapshot(fixture(3), (
        MeshSociety("all", ("a0", "a1", "a2")),))
    cases = (
        ("RuntimeChainReach", chain, MeshRelation.RELATIONSHIP,
         {"left": "a2", "right": "a3"},
         {**{f"a{i}": "left" for i in range(3)},
          **{f"a{i}": "right" for i in range(3, 7)}}, False, None),
        ("RuntimeCoveredGateway", covered, MeshRelation.RELATIONSHIP,
         gates, assigned, True, None),
        ("OverlappingMembership", overlap, MeshRelation.RELATIONSHIP,
         None, None, False, (0, 2)),
        ("NonemptyInteraction", all_members, MeshRelation.DIRECT_INTERACTION,
         None, None, False, None),
        ("NonemptyTransmission", all_members, MeshRelation.TRANSMISSION,
         None, None, False, None),
    )
    pieces = [
        "import NarrativeDynamics.Core.MeshProjection",
        "-- Authoritative V19 runtime paths and concrete V24 gateway premises.",
        "-- Finite conformance only: no arbitrary external process authentication.",
        "",
    ]
    for name, mesh, relation, gateways, assignments, prove_gateway, outsider in cases:
        evidence = export_mesh_proof_input(
            mesh, relation=relation, gateways=gateways,
            assigned_societies=assignments,
        )
        if not validate_mesh_proof_input(
            mesh, evidence, relation=relation, gateways=gateways,
            assigned_societies=assignments,
        ):
            raise AssertionError("non-authoritative runtime path proof input")
        if name.startswith("Nonempty") and not evidence["projected_edges"]:
            raise AssertionError("expected selected relation positive fixture is empty")
        pieces.append(render_path_case(
            name, evidence, gateway_cover=prove_gateway, outsider=outsider,
            authoritative_source_hash=mesh.source_snapshot.content_hash,
        ))
    return "\n".join(pieces)


def main() -> None:
    print(render_path_corpus(), end="")


if __name__ == "__main__":
    main()
