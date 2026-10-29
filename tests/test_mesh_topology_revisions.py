"""R1 contract: real source-bound directed topology changes and replay."""
from __future__ import annotations

from dataclasses import replace
import json
import unittest

from narrative_dynamics.abm.mesh_contracts import (
    MeshRelation, MeshSnapshot, MeshSociety,
)
from narrative_dynamics.abm.mesh_queries import find_mesh_path, mesh_path_is_valid
from narrative_dynamics.abm.mesh_evidence import (
    export_mesh_proof_input, validate_mesh_proof_input,
)
from narrative_dynamics.abm.mesh_topology_events import (
    TopologyChange, TopologyOperation, TopologyChangeError,
    TopologyDuplicateEvent, TopologyVersionConflict, TopologyUnknownAgent,
    TopologyEdgeConflict, TopologyUnsupportedRelation,
    apply_topology_change, begin_topology_replay,
    append_topology_change, replay_topology_changes, observe_topology,
)
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRelationshipEdge
from narrative_dynamics.abm.situated_mesh import mesh_edges
from tests.test_mesh_projection import fixture, two_societies

R = MeshRelation.RELATIONSHIP
D = MeshRelation.DIRECT_INTERACTION
T = MeshRelation.TRANSMISSION


def change(source, event_id, operation, a, b, *, relation=R, edge=None):
    return TopologyChange(
        event_id=event_id, expected_source_hash=source.content_hash,
        relation=relation, operation=operation, source_agent_id=a,
        target_agent_id=b, actor_id="fixture-author", reason="R1 revision test",
        edge=edge,
    )


def reverse_edge(a="a1", b="a0"):
    return SituatedNetworkRelationshipEdge(a, b, 0.6, 0.1, 1, 0, True)


def full_mesh(source):
    return MeshSnapshot(source, (
        MeshSociety("all", tuple(node.agent_id for node in source.nodes)),
    ))


class TopologyRevisionTests(unittest.TestCase):
    def test_directed_edit_stales_old_path_preserves_other_relations_and_original(self):
        old = fixture(3)
        before_bytes = json.dumps(old.to_dict(), sort_keys=True)
        before_mesh = full_mesh(old)
        witness = find_mesh_path(before_mesh, "a0", "a2", relation=R, max_hops=2)
        self.assertIsNotNone(witness)
        evidence = export_mesh_proof_input(before_mesh, relation=R)
        edit = change(old, "remove-01", TopologyOperation.DEACTIVATE, "a0", "a1")
        result = apply_topology_change(old, edit)
        new = result.source_snapshot
        after_mesh = full_mesh(new)
        self.assertNotEqual(old.content_hash, new.content_hash)
        self.assertEqual(old.round_index, new.round_index)
        self.assertEqual(old.model_hash, new.model_hash)
        self.assertEqual(old.story_hash, new.story_hash)
        self.assertEqual(old.cognitive_state_hash, new.cognitive_state_hash)
        self.assertEqual(old.social_state_hash, new.social_state_hash)
        self.assertEqual(old.nodes, new.nodes)
        self.assertEqual(old.access_edges, new.access_edges)
        self.assertEqual(old.transmissions, new.transmissions)
        self.assertEqual(before_bytes, json.dumps(old.to_dict(), sort_keys=True))
        self.assertIsNot(before_mesh.source_snapshot, after_mesh.source_snapshot)
        self.assertIsNone(find_mesh_path(after_mesh, "a0", "a2", relation=R, max_hops=2))
        self.assertFalse(mesh_path_is_valid(
            after_mesh, witness, "a0", "a2", relation=R, max_hops=2,
        ))
        self.assertFalse(validate_mesh_proof_input(after_mesh, evidence, relation=R))
        self.assertEqual(mesh_edges(before_mesh, D), mesh_edges(after_mesh, D))
        self.assertEqual(mesh_edges(before_mesh, T), mesh_edges(after_mesh, T))
        self.assertFalse(next(e for e in new.relationship_edges
                              if (e.source_agent_id, e.observer_agent_id) == ("a0", "a1")).active)
        self.assertEqual(result.receipt.source_hash_before, old.content_hash)
        self.assertEqual(result.receipt.source_hash_after, new.content_hash)

    def test_deactivation_of_source_bridge_fails_closed_without_bridge_pruning(self):
        old_mesh = two_societies(fixture())
        old_witness = find_mesh_path(old_mesh, "a0", "a6", relation=R, max_hops=6)
        self.assertIsNotNone(old_witness)
        edit = change(old_mesh.source_snapshot, "remove-bridge", TopologyOperation.DEACTIVATE,
                      "a2", "a3")
        result = apply_topology_change(old_mesh.source_snapshot, edit)
        with self.assertRaisesRegex(ValueError, "bridge lacks a directed source relation"):
            MeshSnapshot(result.source_snapshot, old_mesh.societies, old_mesh.bridges)
        new_mesh = MeshSnapshot(result.source_snapshot, old_mesh.societies, ())
        self.assertIsNone(find_mesh_path(new_mesh, "a0", "a6", relation=R, max_hops=6))
        self.assertFalse(mesh_path_is_valid(
            new_mesh, old_witness, "a0", "a6", relation=R, max_hops=6,
        ))
        self.assertEqual(len(old_mesh.bridges), 1)
        self.assertEqual(len(old_mesh.source_snapshot.relationship_edges), 6)

    def test_add_directed_edge_changes_only_one_selected_relation(self):
        old = fixture(3)
        edit = change(old, "add-back", TopologyOperation.ADD, "a1", "a0",
                      edge=reverse_edge())
        new = apply_topology_change(old, edit).source_snapshot
        before, after = full_mesh(old), full_mesh(new)
        self.assertNotIn(("a1", "a0"), mesh_edges(before, R))
        self.assertIn(("a1", "a0"), mesh_edges(after, R))
        self.assertIn(("a0", "a1"), mesh_edges(after, R))
        self.assertEqual(mesh_edges(before, D), mesh_edges(after, D))
        self.assertEqual(mesh_edges(before, T), mesh_edges(after, T))
        self.assertEqual(observe_topology(before, relation=R)["reachable_ordered_pair_count"], 3)
        self.assertEqual(observe_topology(after, relation=R)["reachable_ordered_pair_count"], 4)
        self.assertEqual(observe_topology(before, relation=R)["strong_components"],
                         [["a0"], ["a1"], ["a2"]])
        self.assertEqual(observe_topology(after, relation=R)["strong_components"],
                         [["a0", "a1"], ["a2"]])
        self.assertIsNone(observe_topology(after, relation=R)["directed_diameter"])

    def test_serial_replay_receipt_chain_and_order_independent_initial_source(self):
        original = fixture(3)
        first = change(original, "e1", TopologyOperation.ADD, "a1", "a0",
                       edge=reverse_edge())
        intermediate = apply_topology_change(original, first).source_snapshot
        second = change(intermediate, "e2", TopologyOperation.DEACTIVATE, "a0", "a1")
        events = (first, second)
        replay = replay_topology_changes(original, events)
        again = replay_topology_changes(original, events)
        step = append_topology_change(append_topology_change(
            begin_topology_replay(original), first), second)
        self.assertEqual(replay.to_dict(), again.to_dict())
        self.assertEqual(replay.to_dict(), step.to_dict())
        self.assertEqual(json.dumps(replay.to_dict(), sort_keys=True),
                         json.dumps(again.to_dict(), sort_keys=True))
        self.assertEqual(replay.receipts[0].ordinal, 0)
        self.assertEqual(replay.receipts[1].ordinal, 1)
        self.assertEqual(replay.receipts[1].prior_receipt_hash,
                         replay.receipts[0].content_hash)
        self.assertEqual(replay.receipts[0].source_hash_after,
                         replay.receipts[1].source_hash_before)
        self.assertEqual(replay.source_snapshot.content_hash,
                         step.source_snapshot.content_hash)
        self.assertEqual(original.relationship_edges[0].active, True)
        with self.assertRaises(TopologyVersionConflict):
            replay_topology_changes(original, (second, first))
        duplicate = replace(second, event_id="e1")
        with self.assertRaises(TopologyDuplicateEvent):
            replay_topology_changes(original, (first, duplicate))
        self.assertEqual(len(replay.changes), len(replay.receipts))

    def test_bad_inputs_fail_with_specific_error_without_partial_commit(self):
        source = fixture(3)
        before = source.content_hash
        with self.assertRaises(TopologyEdgeConflict):
            apply_topology_change(source, change(
                source, "e", TopologyOperation.ADD, "a0", "a1",
                edge=reverse_edge("a0", "a1"),
            ))
        with self.assertRaises(TopologyEdgeConflict):
            apply_topology_change(source, change(
                source, "e", TopologyOperation.DEACTIVATE, "a1", "a0",
            ))
        with self.assertRaises(TopologyUnknownAgent):
            apply_topology_change(source, change(
                source, "e", TopologyOperation.DEACTIVATE, "a0", "missing",
            ))
        with self.assertRaises(TopologyVersionConflict):
            apply_topology_change(source, replace(
                change(source, "e", TopologyOperation.DEACTIVATE, "a0", "a1"),
                expected_source_hash="sha256:" + "f"*64,
            ))
        with self.assertRaises(TopologyUnsupportedRelation):
            change(source, "e", TopologyOperation.DEACTIVATE, "a0", "a1", relation=D)
        for kwargs in (
            {"operation": "deactivate"},
            {"expected_source_hash": "not-a-hash"},
            {"event_id": "  "},
            {"source_agent_id": "a0", "target_agent_id": "a0"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(TopologyChangeError):
                replace(change(source, "ok", TopologyOperation.DEACTIVATE, "a0", "a1"),
                        **kwargs)
        with self.assertRaises(TopologyChangeError):
            change(source, "bad-add", TopologyOperation.ADD, "a1", "a0",
                   edge=reverse_edge("a0", "a1"))
        with self.assertRaises(TypeError):
            replay_topology_changes(source, [change(
                source, "e", TopologyOperation.DEACTIVATE, "a0", "a1",
            )])
        self.assertEqual(source.content_hash, before)

    def test_replay_records_do_not_authenticate_external_processes(self):
        source = fixture(3)
        record = change(source, "observed-event", TopologyOperation.DEACTIVATE, "a0", "a1")
        original_bytes = json.dumps(source.to_dict(), sort_keys=True)
        state = replay_topology_changes(source, (record,))
        ledger = state.to_dict()
        self.assertEqual(ledger["schema"], "v19-topology-replay-v1")
        self.assertEqual(ledger["receipts"][0]["event_hash"], record.content_hash)
        self.assertEqual(ledger["final_source_hash"], state.source_snapshot.content_hash)
        self.assertEqual(original_bytes, json.dumps(source.to_dict(), sort_keys=True))
        self.assertTrue(state.receipts[0].content_hash.startswith("sha256:"))

    def test_metrics_strong_components_and_directed_diameter(self):
        source = fixture(3)
        first = apply_topology_change(
            source, change(source, "reverse01", TopologyOperation.ADD, "a1", "a0",
                           edge=reverse_edge())).source_snapshot
        second = apply_topology_change(
            first, change(first, "back20", TopologyOperation.ADD, "a2", "a0",
                          edge=reverse_edge("a2", "a0"))).source_snapshot
        metrics = observe_topology(full_mesh(second), relation=R)
        self.assertEqual(metrics["reachable_ordered_pair_count"], 6)
        self.assertEqual(metrics["unreachable_ordered_pair_count"], 0)
        self.assertEqual(metrics["directed_diameter"], 2)
        self.assertEqual(metrics["strong_components"], [["a0", "a1", "a2"]])
        self.assertEqual(metrics["distance_sum"], 8)

    def test_bounded_event_budget_and_no_implicit_reactivation(self):
        source = fixture(3)
        e = change(source, "close", TopologyOperation.DEACTIVATE, "a0", "a1")
        result = apply_topology_change(source, e)
        with self.assertRaises(TopologyEdgeConflict):
            apply_topology_change(
                result.source_snapshot,
                change(result.source_snapshot, "again", TopologyOperation.DEACTIVATE, "a0", "a1"),
            )
        with self.assertRaises(TopologyEdgeConflict):
            apply_topology_change(
                result.source_snapshot,
                change(result.source_snapshot, "readd", TopologyOperation.ADD, "a0", "a1",
                       edge=reverse_edge("a0", "a1")),
            )
        with self.assertRaises(TopologyChangeError):
            replay_topology_changes(source, (e,) * 129)


    def test_real_v19_model_snapshot_remains_source_of_truth(self):
        from narrative_dynamics.abm.situated_network import project_situated_network_snapshot
        from tests.test_network_abm_situated_network import initial_runtime_case

        model, story, cognition, social = initial_runtime_case()
        source = project_situated_network_snapshot(model, story, cognition, social)
        expected_hash = source.content_hash
        before = full_mesh(source)
        self.assertIsNotNone(find_mesh_path(
            before, "alice", "bob", relation=R, max_hops=1,
        ))
        event = change(source, "office-retire-ab", TopologyOperation.DEACTIVATE,
                       "alice", "bob")
        after_source = apply_topology_change(source, event).source_snapshot
        after = full_mesh(after_source)
        self.assertNotIn(("alice", "bob"), mesh_edges(after, R))
        self.assertIn(("bob", "alice"), mesh_edges(after, R))
        self.assertEqual(mesh_edges(before, D), mesh_edges(after, D))
        self.assertEqual(mesh_edges(before, T), mesh_edges(after, T))
        self.assertEqual(source.model_hash, model.content_hash)
        self.assertEqual(source.round_index, story.current_state.round_index)
        self.assertEqual(after_source.model_hash, model.content_hash)
        self.assertEqual(after_source.round_index, source.round_index)
        self.assertEqual(source.content_hash, expected_hash)
        self.assertNotEqual(after_source.content_hash, expected_hash)
        self.assertEqual(
            json.dumps(source.to_dict(), sort_keys=True),
            json.dumps(project_situated_network_snapshot(
                model, story, cognition, social,
            ).to_dict(), sort_keys=True),
        )

    def test_reviewed_replay_vectors_match_exact_runtime_output(self):
        from pathlib import Path
        from tools.render_topology_revision_vectors import render_vector_corpus
        path = Path(__file__).resolve().parents[1] / "conformance" / "topology_revision_v1.json"
        self.assertEqual(
            path.read_bytes(), render_vector_corpus().encode("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
