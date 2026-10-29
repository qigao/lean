"""M0 finite evidence from actual V19 network snapshots, not a new runtime."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_contracts import (
    MeshBridge, MeshRelation, MeshSnapshot, MeshSociety,
)
from narrative_dynamics.abm.mesh_evidence import (
    export_mesh_proof_input, validate_mesh_proof_input,
)
from narrative_dynamics.abm.situated_network import (
    project_situated_network_snapshot,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRelationshipEdge,
)
from tests.test_mesh_projection import fixture, two_societies
from tests.test_network_abm_situated_network import initial_runtime_case


R = MeshRelation.RELATIONSHIP


class MeshEvidenceTests(unittest.TestCase):
    def test_real_v19_runtime_source_identity_and_three_selected_relations(self):
        model, story, cognition, social = initial_runtime_case()
        source = project_situated_network_snapshot(model, story, cognition, social)
        original = source.to_dict()
        mesh = MeshSnapshot(source, (
            MeshSociety("everyone", tuple(node.agent_id for node in source.nodes)),
        ))
        self.assertIs(mesh.source_snapshot, source)
        for relation in MeshRelation:
            with self.subTest(relation=relation):
                data = export_mesh_proof_input(
                    mesh, relation=relation, max_hops=6
                )
                self.assertEqual(data["source_lineage"]["source_snapshot_hash"],
                                 source.content_hash)
                self.assertEqual(data["source_lineage"]["round_index"],
                                 source.round_index)
                self.assertEqual(data["source_lineage"]["model_hash"],
                                 model.content_hash)
                self.assertEqual(data["mesh_hash"], mesh.content_hash)
                self.assertEqual(data["relation"], relation.value)
                self.assertEqual(data["source_edges"], data["projected_edges"])
                self.assertEqual(data["agents"], [
                    node.agent_id for node in source.nodes
                ])
                self.assertEqual(data["places"], [
                    node.place_id for node in source.nodes
                ])
                self.assertTrue(validate_mesh_proof_input(
                    mesh, data, relation=relation, max_hops=6
                ))
                self.assertEqual(
                    data,
                    export_mesh_proof_input(mesh, relation=relation, max_hops=6),
                )
        self.assertEqual(original, source.to_dict())

    def test_directed_chain_has_local_witness_not_global_cover(self):
        mesh = two_societies(fixture())
        gateways = {"left": "a2", "right": "a3"}
        assignments = {
            **{f"a{i}": "left" for i in range(3)},
            **{f"a{i}": "right" for i in range(3, 7)},
        }
        data = export_mesh_proof_input(
            mesh, relation=R, max_hops=6,
            gateways=gateways, assigned_societies=assignments,
        )
        self.assertEqual(data["agents"], [f"a{i}" for i in range(7)])
        self.assertEqual(data["source_edges"], [
            [i, i+1] for i in range(6)
        ])
        self.assertEqual(data["projected_edges"], data["source_edges"])
        self.assertEqual(data["bridges"], [[0, 1, 2, 3]])
        self.assertIn([0, 6, list(range(7))], data["paths"])
        self.assertNotIn([6, 0, list(reversed(range(7)))], data["paths"])
        self.assertFalse(data["gateway_cover"]["premises_satisfied"])
        self.assertTrue(any("right->left" in s
                            for s in data["gateway_cover"]["failures"]))
        self.assertTrue(validate_mesh_proof_input(
            mesh, data, relation=R, max_hops=6,
            gateways=gateways, assigned_societies=assignments,
        ))
        without_bridge = replace(mesh, bridges=())
        self.assertFalse(validate_mesh_proof_input(
            without_bridge, data, relation=R, max_hops=6,
            gateways=gateways, assigned_societies=assignments,
        ))

    def test_full_directed_gateway_cover_uses_actual_source_edges(self):
        source = fixture(3)
        source = replace(source, relationship_edges=tuple(
            SituatedNetworkRelationshipEdge(
                f"a{i}", f"a{j}", 1.0, 0.0, 0, 0, True
            )
            for i in range(3) for j in range(3) if i != j
        ))
        mesh = MeshSnapshot(
            source, (MeshSociety("s0", ("a0",)),
                     MeshSociety("s1", ("a1", "a2"))),
            (MeshBridge(R, "s0", "s1", "a0", "a1"),
             MeshBridge(R, "s1", "s0", "a1", "a0")),
        )
        gateways = {"s0": "a0", "s1": "a1"}
        assigned = {"a0": "s0", "a1": "s1", "a2": "s1"}
        data = export_mesh_proof_input(
            mesh, relation=R, gateways=gateways, assigned_societies=assigned
        )
        # Cross-society 0<->2 source edges are deliberately omitted:
        # the only declared cross-society bridge is between 0 and 1.
        self.assertEqual(data["source_edges"], [
            [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1],
        ])
        self.assertEqual(data["projected_edges"], [
            [0, 1], [1, 0], [1, 2], [2, 1],
        ])
        self.assertTrue(data["gateway_cover"]["premises_satisfied"])
        self.assertEqual(data["gateway_cover"]["failures"], [])
        self.assertEqual(len(data["gateway_cover"]["society_paths"]), 4)
        self.assertTrue(validate_mesh_proof_input(
            mesh, data, relation=R, gateways=gateways,
            assigned_societies=assigned,
        ))
        for mutated in (
            {"s0": "a0", "s1": "a2"},
            {"s0": "a1", "s1": "a0"},
        ):
            self.assertFalse(validate_mesh_proof_input(
                mesh, data, relation=R, gateways=mutated,
                assigned_societies=assigned,
            ))

    def test_modified_snapshot_identity_or_fields_are_rejected(self):
        mesh = two_societies(fixture())
        data = export_mesh_proof_input(mesh, relation=R)
        for key, value in (
            ("relation", MeshRelation.TRANSMISSION.value),
            ("mesh_hash", "sha256:" + "f" * 64),
            ("max_hops", 5),
            ("source_edges", []),
            ("projected_edges", []),
            ("membership", []),
            ("bridges", []),
            ("paths", []),
            ("evidence_hash", "sha256:" + "0" * 64),
        ):
            with self.subTest(key=key):
                corrupted = deepcopy(data)
                corrupted[key] = value
                self.assertFalse(validate_mesh_proof_input(
                    mesh, corrupted, relation=R
                ))
        stale = MeshSnapshot(
            replace(mesh.source_snapshot, round_index=2),
            mesh.societies, mesh.bridges,
        )
        self.assertFalse(validate_mesh_proof_input(
            stale, data, relation=R
        ))
        disconnected = MeshSnapshot(
            replace(mesh.source_snapshot, relationship_edges=()),
            mesh.societies, (),
        )
        self.assertFalse(validate_mesh_proof_input(
            disconnected, data, relation=R
        ))

    def test_equal_numeric_type_mutations_are_rejected(self):
        mesh = two_societies(fixture())
        gateways = {"left": "a2", "right": "a3"}
        assignments = {
            **{f"a{i}": "left" for i in range(3)},
            **{f"a{i}": "right" for i in range(3, 7)},
        }
        data = export_mesh_proof_input(
            mesh, relation=R, gateways=gateways,
            assigned_societies=assignments,
        )
        for path, replacement in (
            (("source_lineage", "round_index"), True),
            (("source_lineage", "round_index"), 1.0),
            (("max_hops",), 6.0),
            (("source_edges", 0, 0), False),
            (("paths", 0, 2, 0), False),
            (("gateway_cover", "premises_satisfied"), 0),
        ):
            for rehash in (False, True):
                with self.subTest(path=path, kind=type(replacement), rehash=rehash):
                    corrupted = deepcopy(data)
                    container = corrupted
                    for key in path[:-1]:
                        container = container[key]
                    container[path[-1]] = replacement
                    # Ordinary Python equality misses this typed corruption.
                    self.assertEqual(corrupted, data)
                    changed_hash = stable_content_hash({
                        key: value for key, value in corrupted.items()
                        if key != "evidence_hash"
                    })
                    self.assertNotEqual(changed_hash, data["evidence_hash"])
                    if rehash:
                        corrupted["evidence_hash"] = changed_hash
                    self.assertFalse(validate_mesh_proof_input(
                        mesh, corrupted, relation=R, gateways=gateways,
                        assigned_societies=assignments,
                    ))

    def test_canonical_json_round_trip_remains_valid(self):
        mesh = two_societies(fixture())
        data = export_mesh_proof_input(mesh, relation=R)
        decoded = json.loads(json.dumps(data, sort_keys=True))
        self.assertTrue(validate_mesh_proof_input(mesh, decoded, relation=R))

    def test_noncanonical_candidate_values_are_rejected(self):
        mesh = two_societies(fixture())
        data = export_mesh_proof_input(mesh, relation=R)

        class EqualityTrap:
            def __eq__(self, other):
                raise AssertionError("candidate equality must not be invoked")

        class IntegerSubclass(int):
            pass

        class ForgedDict(dict):
            def __eq__(self, other):
                return True

        cycle = []
        cycle.append(cycle)
        for case, corrupted in (
            ("tuple", {**data, "paths": tuple(data["paths"])}),
            ("custom_equality", {**data, "max_hops": EqualityTrap()}),
            ("integer_subclass", {**data, "max_hops": IntegerSubclass(6)}),
            ("cycle", {**data, "paths": cycle}),
            ("dict_subclass", ForgedDict(data, mesh_hash="invented")),
        ):
            with self.subTest(case=case):
                self.assertFalse(validate_mesh_proof_input(
                    mesh, corrupted, relation=R
                ))

    def test_validator_preserves_authoritative_input_errors(self):
        mesh = two_societies(fixture())
        data = export_mesh_proof_input(mesh, relation=R)
        with self.assertRaises(ValueError):
            validate_mesh_proof_input(mesh, data, relation=R, max_hops=True)
        with self.assertRaises(TypeError):
            validate_mesh_proof_input(None, data, relation=R)

    def test_errors_are_bounded_and_explicit(self):
        mesh = two_societies(fixture())
        for budget in (True, -1, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                export_mesh_proof_input(mesh, relation=R, max_hops=budget)
        with self.assertRaises(ValueError):
            export_mesh_proof_input(mesh, relation=R, gateways={})
        with self.assertRaises(ValueError):
            export_mesh_proof_input(mesh, relation=R,
                                    gateways={}, assigned_societies={})
        with self.assertRaises(ValueError):
            export_mesh_proof_input(
                MeshSnapshot(fixture(25),
                             (MeshSociety("all", tuple(
                                 f"a{i}" for i in range(25)
                             )),)), relation=R,
            )
        self.assertFalse(validate_mesh_proof_input(mesh, None, relation=R))
        self.assertFalse(validate_mesh_proof_input(
            mesh, {"evidence_hash": "invented"}, relation=R
        ))



    def test_generated_lean_obligations_use_authoritative_selected_graph(self):
        from tools.render_mesh_evidence_lean import (
            real_runtime_case, render_case, render_corpus,
        )
        mesh = real_runtime_case()
        proof_input = export_mesh_proof_input(
            mesh, relation=MeshRelation.RELATIONSHIP,
        )
        generated = render_case("OfficeRuntime", proof_input)
        self.assertIn("projectedMesh", generated)
        self.assertIn("private instance : DecidableRel projected", generated)
        self.assertEqual(
            generated.count("\nexample : "),
            len(proof_input["agents"]) ** 2,
        )
        corpus = render_corpus()
        for case_name in ("OfficeRelationship", "OfficeInteraction",
                          "OfficeTransmission", "DirectedChain", "CoveredGateways"):
            self.assertIn(f"namespace {case_name}", corpus)
        self.assertNotIn("native_decide", corpus)
        self.assertNotIn("sorry", corpus)

    def test_corrupt_lean_expected_edge_is_not_silently_repaired(self):
        from tools.render_mesh_evidence_lean import render_case
        data = export_mesh_proof_input(two_societies(fixture()), relation=R)
        first = data["projected_edges"][0]
        tampered = deepcopy(data)
        tampered["projected_edges"] = [
            edge for edge in data["projected_edges"] if edge != first
        ]
        self.assertFalse(validate_mesh_proof_input(
            two_societies(fixture()), tampered, relation=R
        ))
        output = render_case("Tampered", tampered)
        target = f"¬ projected ({first[0]} : Fin 7) ({first[1]} : Fin 7)"
        self.assertIn(target, output)


    def test_generated_lean_corpus_matches_checked_in_sha256_pin(self):
        from hashlib import sha256
        from pathlib import Path
        from tools.render_mesh_evidence_lean import render_corpus

        corpus = render_corpus().encode("utf-8")
        pinned = (Path(__file__).parent / "fixtures" /
                  "mesh_evidence_lean.sha256").read_text(encoding="ascii").strip()
        self.assertEqual(len(pinned), 64)
        self.assertEqual(sha256(corpus).hexdigest(), pinned)

    def test_real_lean_red_inputs_are_inconsistent_with_authority(self):
        from tools.mutate_mesh_evidence_lean import render_false_obligation

        for kind in ("source", "bridge"):
            with self.subTest(kind=kind):
                generated = render_false_obligation(kind)
                self.assertIn("import NarrativeDynamics.Core.MeshProjection", generated)
                self.assertIn("example : projected (0 : Fin 3) (1 : Fin 3) := by", generated)
                self.assertIn("decide_cbv", generated)


    def test_runtime_paths_and_gateway_premises_generate_finite_lean(self):
        from tools.render_mesh_reach_lean import (
            render_path_case, render_path_corpus,
        )
        from tools.render_mesh_evidence_lean import covered_case
        covered, gates, assignments = covered_case()
        evidence = export_mesh_proof_input(
            covered, relation=R, gateways=gates,
            assigned_societies=assignments,
        )
        lean = render_path_case("ConcreteGateway", evidence, gateway_cover=True)
        self.assertEqual(lean.count("private theorem runtime_path_"),
                         len(evidence["paths"]))
        for premise in ("gatewayMember", "assignedMember", "toGateway",
                        "fromGateway", "societyBound"):
            self.assertIn(f"private theorem {premise} :", lean)
        self.assertNotIn("projectedMesh_global_six_hop_bound", lean)
        self.assertNotIn("example : GlobalHopBound projected 6", lean)
        self.assertIn("MeshWalk.step projected_edge_", lean)
        corpus = render_path_corpus()
        for name in ("RuntimeChainReach", "RuntimeCoveredGateway",
                     "OverlappingMembership", "NonemptyInteraction",
                     "NonemptyTransmission"):
            self.assertIn(f"namespace {name}", corpus)
        self.assertIn("membershipQualifiedReach_outsider", corpus)
        self.assertNotIn("native_decide", corpus)
        self.assertNotIn("sorry", corpus)
        self.assertNotIn("axiom ", corpus)
        self.assertEqual(corpus, render_path_corpus())

    def test_runtime_path_generator_rejects_unbounded_witness(self):
        from tools.render_mesh_reach_lean import render_path_case
        evidence = export_mesh_proof_input(two_societies(fixture()), relation=R)
        stale = deepcopy(evidence)
        stale["paths"][0][2].append(99)
        with self.assertRaises(ValueError):
            render_path_case("Unbounded", stale)


    def test_source_identity_claims_checked_in_extended_lean_corpus(self):
        from tools.render_mesh_reach_lean import render_path_corpus
        corpus = render_path_corpus()
        self.assertEqual(corpus.count(
            "example : reportedSourceHash = authoritativeSourceHash := by decide"
        ), 5)
        self.assertEqual(corpus.count("private def authoritativeSourceHash : String"), 5)
        self.assertNotIn("native_decide", corpus)
        self.assertNotIn("sorry", corpus)

    def test_stale_path_and_rehashed_identity_generate_real_lean_counterclaims(self):
        from tools.mutate_mesh_reach_lean import render_false_reach_obligation
        old_path = render_false_reach_obligation("stale-path")
        self.assertIn("namespace StaleSourcePath", old_path)
        self.assertIn(
            "example : ReachWithin projected 1 (0 : Fin 7) (1 : Fin 7)",
            old_path,
        )
        self.assertIn("MeshWalk.single (by decide_cbv)", old_path)
        wrong_identity = render_false_reach_obligation("identity")
        self.assertIn("namespace StaleSourceIdentity", wrong_identity)
        self.assertIn(
            "example : reportedSourceHash = authoritativeSourceHash := by decide",
            wrong_identity,
        )
        self.assertNotIn("native_decide", wrong_identity)
        self.assertNotIn("sorry", wrong_identity)
        with self.assertRaises(ValueError):
            render_false_reach_obligation("other")


    def test_extended_runtime_lean_corpus_has_reviewed_byte_pin(self):
        from hashlib import sha256
        from pathlib import Path
        from tools.render_mesh_reach_lean import render_path_corpus

        corpus = render_path_corpus().encode("utf-8")
        pinned = (Path(__file__).parent / "fixtures" /
                  "mesh_reach_lean.sha256").read_text(encoding="ascii").strip()
        self.assertRegex(pinned, r"^[a-f0-9]{64}$")
        self.assertEqual(sha256(corpus).hexdigest(), pinned)

    def test_versioned_runtime_vectors_match_authoritative_source_bytes(self):
        from pathlib import Path
        from tools.render_mesh_proof_vectors import (
            SCHEMA, render_vector_corpus, build_vector_manifest,
        )
        path = Path(__file__).resolve().parents[1] / "conformance" / "mesh_runtime_vectors_v1.json"
        rendered = render_vector_corpus()
        self.assertEqual(path.read_bytes(), rendered.encode("utf-8"))
        vectors = build_vector_manifest()
        self.assertEqual(vectors["schema"], SCHEMA)
        self.assertEqual(len(vectors["cases"]), 8)
        self.assertEqual(
            len({case["name"] for case in vectors["cases"]}), 8,
        )
        for case in vectors["cases"]:
            self.assertEqual(
                case["source_snapshot_hash"],
                case["evidence"]["source_lineage"]["source_snapshot_hash"],
            )


if __name__ == "__main__":
    unittest.main()
