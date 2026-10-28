"""Contract regression: V24 never replaces the authoritative situated runtime."""
from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.abm.mesh_contracts import (
    MeshBridge, MeshPath, MeshRelation, MeshSnapshot, MeshSociety,
)
from narrative_dynamics.abm.mesh_queries import (
    find_mesh_path, is_closed_society, mesh_path_is_valid,
)
from narrative_dynamics.abm.situated_mesh import (
    mesh_edges, overlapping_memberships, society_gateway_edges,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkAccessEdge, SituatedNetworkAgentNode,
    SituatedNetworkRelationshipEdge, SituatedNetworkSnapshot, SituatedNetworkTransmission,
)
from narrative_dynamics.abm.situated import ObservationChannel
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity


HASH = "sha256:" + "0" * 64
R = MeshRelation.RELATIONSHIP
D = MeshRelation.DIRECT_INTERACTION
T = MeshRelation.TRANSMISSION


def fixture(count: int = 7) -> SituatedNetworkSnapshot:
    return SituatedNetworkSnapshot(
        model_id="fixture", model_hash=HASH, round_index=1,
        story_hash=HASH, cognitive_state_hash=HASH, social_state_hash=HASH,
        nodes=tuple(
            SituatedNetworkAgentNode(f"a{i}", "role", f"p{i}", 0.5, 0)
            for i in range(count)
        ),
        relationship_edges=tuple(
            SituatedNetworkRelationshipEdge(f"a{i}", f"a{i+1}", 0.75, 0.0, 0, 0, True)
            for i in range(count - 1)
        ),
        access_edges=(SituatedNetworkAccessEdge("a0", "a1", None, None, True),)
            if count > 1 else (),
        transmissions=(SituatedNetworkTransmission(
            "event-1", "a1", "a2", SituatedPerceptFidelity.IDENTIFIED,
            (ObservationChannel.AUDITORY,),
        ),) if count > 2 else (),
        latest_tell_event_count=1 if count > 2 else 0,
    )


def two_societies(source: SituatedNetworkSnapshot) -> MeshSnapshot:
    return MeshSnapshot(source, (
        MeshSociety("left", ("a0", "a1", "a2")),
        MeshSociety("right", ("a3", "a4", "a5", "a6")),
    ), (MeshBridge(R, "left", "right", "a2", "a3"),))


class MeshProjectionTests(unittest.TestCase):
    def test_six_hop_chain_existing_one_way_gateway(self) -> None:
        source = fixture()
        original = source.to_dict()
        source_hash = source.content_hash
        mesh = two_societies(source)
        self.assertIs(mesh.source_snapshot, source)
        self.assertEqual(society_gateway_edges(mesh, R),
                         (("left", "right", "a2", "a3"),))
        path = find_mesh_path(mesh, "a0", "a6", relation=R, max_hops=6)
        self.assertIsNotNone(path)
        self.assertEqual(path.agent_ids, tuple(f"a{i}" for i in range(7)))
        self.assertEqual(path.hop_count, 6)
        self.assertTrue(mesh_path_is_valid(mesh, path, "a0", "a6",
                                           relation=R, max_hops=6))
        self.assertIsNone(find_mesh_path(mesh, "a0", "a6", relation=R, max_hops=5))
        self.assertIsNone(find_mesh_path(mesh, "a6", "a0", relation=R, max_hops=6))
        self.assertIsNone(find_mesh_path(
            replace(mesh, bridges=()), "a0", "a6", relation=R, max_hops=6))
        self.assertEqual(source.to_dict(), original)
        self.assertEqual(source.content_hash, source_hash)

    def test_universal_society_preserves_each_selected_relation(self) -> None:
        source = fixture()
        mesh = MeshSnapshot(source, (MeshSociety(
            "all", tuple(node.agent_id for node in source.nodes)),))
        self.assertEqual(mesh_edges(mesh, R),
                         tuple((f"a{i}", f"a{i+1}") for i in range(6)))
        self.assertEqual(mesh_edges(mesh, D), (("a0", "a1"),))
        self.assertEqual(mesh_edges(mesh, T), (("a1", "a2"),))
        self.assertIsNone(find_mesh_path(mesh, "a0", "a2", relation=D, max_hops=2))
        self.assertIsNone(find_mesh_path(mesh, "a0", "a2", relation=T, max_hops=2))

    def test_membership_overlap_does_not_clone_agent_or_place(self) -> None:
        source = fixture()
        mesh = MeshSnapshot(source, (
            MeshSociety("A", ("a0", "a1", "a2")),
            MeshSociety("B", ("a2", "a3")),
        ))
        self.assertEqual(overlapping_memberships(mesh),
                         (("A", "B", "a2"), ("B", "A", "a2")))
        self.assertEqual(next(n.place_id for n in source.nodes if n.agent_id == "a2"), "p2")
        self.assertEqual(mesh_edges(mesh, R), (("a0", "a1"), ("a1", "a2"), ("a2", "a3")))

    def test_invalid_bridges_are_rejected_eagerly(self) -> None:
        source = fixture()
        societies = (MeshSociety("A", ("a0", "a1")), MeshSociety("B", ("a2", "a3")))
        for bridge in (
            MeshBridge(R, "A", "B", "a0", "a3"),  # No original directed edge
            MeshBridge(D, "A", "B", "a1", "a2"),  # Source selection differs
            MeshBridge(R, "B", "A", "a2", "a1"),  # Reverse not present
            MeshBridge(R, "A", "missing", "a1", "a2"),
            MeshBridge(R, "A", "B", "a0", "a2"),  # Wrong membership/edge
        ):
            with self.subTest(bridge=bridge), self.assertRaises(ValueError):
                MeshSnapshot(source, societies, (bridge,))
        good = MeshBridge(R, "A", "B", "a1", "a2")
        with self.assertRaises(ValueError):
            MeshSnapshot(source, societies, (good, good))
        with self.assertRaises(ValueError):
            MeshSnapshot(source, (MeshSociety("A", ("a8",)),))
        with self.assertRaises(ValueError):
            MeshSnapshot(source, (societies[0], societies[0]))
        with self.assertRaises(ValueError):
            MeshBridge("unsupported", "A", "B", "a1", "a2")

    def test_local_scope_rejects_even_zero_hop_outsider(self) -> None:
        mesh = two_societies(fixture())
        self.assertEqual(find_mesh_path(mesh, "a0", "a0",
                                        relation=R, max_hops=0, society_id="left").hop_count, 0)
        for source, target in (("a3", "a3"), ("a0", "a3")):
            with self.subTest(source=source), self.assertRaises(ValueError):
                find_mesh_path(mesh, source, target, relation=R,
                               max_hops=0, society_id="left")
        with self.assertRaises(ValueError):
            find_mesh_path(mesh, "a0", "a0", relation=R, max_hops=0, society_id="missing")
        self.assertTrue(is_closed_society(mesh, "right", relation=R))
        self.assertFalse(is_closed_society(mesh, "left", relation=R))
        self.assertTrue(is_closed_society(mesh, "left", relation=D))

    def test_paths_bind_snapshot_relation_scope_endpoints_and_budget(self) -> None:
        mesh = two_societies(fixture())
        witness = find_mesh_path(mesh, "a0", "a2", relation=R, max_hops=3, society_id="left")
        self.assertTrue(mesh_path_is_valid(mesh, witness, "a0", "a2",
                                           relation=R, max_hops=2, society_id="left"))
        self.assertFalse(mesh_path_is_valid(mesh, witness, "a0", "a1",
                                            relation=R, max_hops=3, society_id="left"))
        self.assertFalse(mesh_path_is_valid(mesh, witness, "a0", "a2",
                                            relation=R, max_hops=1, society_id="left"))
        self.assertFalse(mesh_path_is_valid(mesh, witness, "a0", "a2",
                                            relation=D, max_hops=3, society_id="left"))
        self.assertFalse(mesh_path_is_valid(mesh, witness, "a0", "a2",
                                            relation=R, max_hops=3))
        self.assertFalse(mesh_path_is_valid(replace(mesh, bridges=()), witness, "a0", "a2",
                                            relation=R, max_hops=3, society_id="left"))
        self.assertFalse(mesh_path_is_valid(
            MeshSnapshot(replace(mesh.source_snapshot, round_index=2), mesh.societies, mesh.bridges),
            witness, "a0", "a2", relation=R, max_hops=3, society_id="left"))

    def test_query_validation_and_zero_nodes(self) -> None:
        empty = MeshSnapshot(fixture(0), ())
        with self.assertRaises(ValueError):
            find_mesh_path(empty, "none", "none", relation=R, max_hops=0)
        singleton = MeshSnapshot(fixture(1), ())
        self.assertEqual(find_mesh_path(singleton, "a0", "a0",
                                        relation=R, max_hops=0).hop_count, 0)
        for budget in (-1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                find_mesh_path(singleton, "a0", "a0", relation=R, max_hops=budget)
        with self.assertRaises(ValueError):
            find_mesh_path(singleton, "a0", "none", relation=R, max_hops=0)

    def test_shortest_tie_break_and_alternate_witness(self) -> None:
        original = fixture(4)
        extra = (
            SituatedNetworkRelationshipEdge("a0", "a2", 0.7, 0.0, 0, 0, True),
            SituatedNetworkRelationshipEdge("a0", "a1", 0.7, 0.0, 0, 0, True),
            SituatedNetworkRelationshipEdge("a1", "a3", 0.7, 0.0, 0, 0, True),
            SituatedNetworkRelationshipEdge("a2", "a3", 0.7, 0.0, 0, 0, True),
        )
        mesh = MeshSnapshot(replace(original, relationship_edges=extra),
                            (MeshSociety("all", ("a0", "a1", "a2", "a3")),))
        found = find_mesh_path(mesh, "a0", "a3", relation=R, max_hops=2)
        self.assertEqual(found.agent_ids, ("a0", "a1", "a3"))
        alternate = MeshPath(mesh.content_hash, R, None, ("a0", "a2", "a3"))
        self.assertTrue(mesh_path_is_valid(mesh, alternate, "a0", "a3",
                                           relation=R, max_hops=2))


    def test_six_hop_premise_checker_rejects_uncovered_chain(self) -> None:
        from narrative_dynamics.abm.mesh_six_hop import assess_six_hop_prerequisites
        mesh = two_societies(fixture())
        result = assess_six_hop_prerequisites(
            mesh, relation=R,
            gateways={"left": "a2", "right": "a3"},
            assigned_societies={
                **{f"a{i}": "left" for i in range(3)},
                **{f"a{i}": "right" for i in range(3, 7)},
            },
        )
        self.assertFalse(result.premises_satisfied)
        self.assertEqual(result.mesh_hash, mesh.content_hash)
        self.assertTrue(any("missing one-hop" in reason for reason in result.failures))
        self.assertTrue(any("no <=4-hop" in reason for reason in result.failures))

    def test_six_hop_premise_checker_accepts_realized_gateway_cover(self) -> None:
        from narrative_dynamics.abm.mesh_six_hop import assess_six_hop_prerequisites
        source = fixture(3)
        complete = tuple(
            SituatedNetworkRelationshipEdge(f"a{i}", f"a{j}", 1.0, 0.0, 0, 0, True)
            for i in range(3) for j in range(3) if i != j
        )
        source = replace(source, relationship_edges=complete)
        mesh = MeshSnapshot(
            source, (MeshSociety("s0", ("a0",)),
                     MeshSociety("s1", ("a1", "a2"))),
            (MeshBridge(R, "s0", "s1", "a0", "a1"),
             MeshBridge(R, "s1", "s0", "a1", "a0")),
        )
        result = assess_six_hop_prerequisites(
            mesh, relation=R,
            gateways={"s0": "a0", "s1": "a1"},
            assigned_societies={"a0": "s0", "a1": "s1", "a2": "s1"},
        )
        self.assertTrue(result.premises_satisfied, result.failures)
        self.assertEqual(result.failures, ())
        self.assertFalse(assess_six_hop_prerequisites(
            mesh, relation=D,
            gateways={"s0": "a0", "s1": "a1"},
            assigned_societies={"a0": "s0", "a1": "s1", "a2": "s1"},
        ).premises_satisfied)
        with self.assertRaises(ValueError):
            assess_six_hop_prerequisites(
                mesh, relation=R,
                gateways={"s0": "a0"},
                assigned_societies={"a0": "s0", "a1": "s1", "a2": "s1"},
            )



    def test_real_runtime_projection_retains_source_identity(self) -> None:
        from narrative_dynamics.abm.situated_network import project_situated_network_snapshot
        from narrative_dynamics.abm.mesh_contracts import source_relation_edges
        from tests.test_network_abm_situated_network import initial_runtime_case
        model, story, cognitive_state, social_state = initial_runtime_case()
        source = project_situated_network_snapshot(model, story, cognitive_state, social_state)
        before = source.to_dict()
        original_hash = source.content_hash
        mesh = MeshSnapshot(
            source, (MeshSociety("office", tuple(node.agent_id for node in source.nodes)),)
        )
        self.assertIs(mesh.source_snapshot, source)
        for relation in MeshRelation:
            self.assertEqual(mesh_edges(mesh, relation), source_relation_edges(source, relation))
        for node in source.nodes:
            witness = find_mesh_path(mesh, node.agent_id, node.agent_id,
                                     relation=R, max_hops=0, society_id="office")
            self.assertTrue(mesh_path_is_valid(
                mesh, witness, node.agent_id, node.agent_id,
                relation=R, max_hops=0, society_id="office"))
        self.assertEqual(before, source.to_dict())
        self.assertEqual(original_hash, source.content_hash)
        self.assertEqual(mesh.source_snapshot.round_index, story.current_state.round_index)


if __name__ == "__main__":
    unittest.main()
