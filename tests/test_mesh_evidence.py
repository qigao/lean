"""M0 finite evidence from actual V19 network snapshots, not a new runtime."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

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
        self.assertEqual(data["projected_edges"], data["source_edges"])
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


if __name__ == "__main__":
    unittest.main()
