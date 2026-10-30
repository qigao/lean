"""R1.1: source-authoritative V19 social-memory trust checkpoint admission."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from narrative_dynamics.abm.mesh_contracts import (
    MeshRelation, MeshSnapshot, MeshSociety, MeshBridge,
)
from narrative_dynamics.abm.mesh_queries import find_mesh_path, mesh_path_is_valid
from narrative_dynamics.abm.situated_mesh import mesh_edges
from narrative_dynamics.abm.situated_network import (
    initialize_situated_network_runtime, project_situated_network_snapshot,
    simulate_situated_network_round,
)
from narrative_dynamics.abm.situated_social_topology import (
    SocialTrustOperation, SocialTrustCheckpointIntervention,
    SocialTopologyBoundaryError, SocialTopologyTrustConflict,
    SocialTopologyIntegrityError, SocialTrustCheckpointResult,
    admit_social_trust_checkpoint, verify_social_trust_checkpoint,
)
from narrative_dynamics.abm.mesh_topology_events import (
    TopologyChangeError, TopologyVersionConflict, TopologyUnknownAgent,
)
from tests.test_network_abm_situated_network import initial_runtime_case

R = MeshRelation.RELATIONSHIP
D = MeshRelation.DIRECT_INTERACTION
T = MeshRelation.TRANSMISSION


def event_for(runtime, event_id, operation, source, observer, trust):
    return SocialTrustCheckpointIntervention(
        event_id, runtime.content_hash, runtime.snapshot.content_hash,
        runtime.social_state.content_hash, source, observer,
        "local-test-admin", "explicit checkpoint intervention",
        operation, trust,
    )


class SocialTopologyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "social-topology.sqlite3")
        self.model, self.story, self.cognition, self.social = initial_runtime_case()
        self.runtime = initialize_situated_network_runtime(
            self.path, self.model, self.story, self.cognition, self.social,
        )
        self.societies = (MeshSociety("office", ("alice", "bob")),)

    def tearDown(self):
        self.temp.cleanup()

    def mesh(self, runtime):
        return MeshSnapshot(runtime.snapshot, self.societies)

    def test_source_social_checkpoint_flips_only_selected_directed_edge(self):
        old = self.runtime
        original_bytes = json.dumps(old.to_dict(), sort_keys=True)
        pre = self.mesh(old)
        before_path = find_mesh_path(
            pre, "alice", "bob", relation=R, max_hops=1,
        )
        self.assertIsNotNone(before_path)
        intervention = event_for(
            old, "deactivate-ab", SocialTrustOperation.DEACTIVATE,
            "alice", "bob", 0.0,
        )
        first = admit_social_trust_checkpoint(self.path, self.model, old, intervention)
        again = admit_social_trust_checkpoint(self.path, self.model, old, intervention)
        self.assertEqual(first.to_dict(), again.to_dict())
        new = first.next_runtime_state
        self.assertTrue(new.checkpoint)
        self.assertIs(new.story, old.story)
        self.assertIs(new.cognitive_state, old.cognitive_state)
        self.assertEqual(new.round_index, old.round_index)
        self.assertEqual(new.memory_store_hash, old.memory_store_hash)
        self.assertNotEqual(new.social_state.content_hash, old.social_state.content_hash)
        self.assertNotEqual(new.snapshot.content_hash, old.snapshot.content_hash)
        self.assertEqual(new.snapshot.social_state_hash, new.social_state.content_hash)
        self.assertEqual(
            new.snapshot,
            project_situated_network_snapshot(
                self.model, new.story, new.cognitive_state, new.social_state,
            ),
        )
        self.assertEqual(
            mesh_edges(pre, D), mesh_edges(self.mesh(new), D),
        )
        self.assertEqual(
            mesh_edges(pre, T), mesh_edges(self.mesh(new), T),
        )
        self.assertEqual(new.social_state.claims, old.social_state.claims)
        self.assertEqual(new.social_state.processed_evidence_ids,
                         old.social_state.processed_evidence_ids)
        self.assertNotIn(("alice", "bob"), mesh_edges(self.mesh(new), R))
        self.assertIn(("bob", "alice"), mesh_edges(self.mesh(new), R))
        self.assertFalse(mesh_path_is_valid(
            self.mesh(new), before_path, "alice", "bob",
            relation=R, max_hops=1,
        ))
        self.assertTrue(first.receipt.previous_active)
        self.assertFalse(first.receipt.next_active)
        self.assertEqual(json.dumps(old.to_dict(), sort_keys=True), original_bytes)

        with self.assertRaises(TopologyVersionConflict):
            admit_social_trust_checkpoint(self.path, self.model, new, intervention)

        reactivate = event_for(
            new, "activate-ab", SocialTrustOperation.ACTIVATE,
            "alice", "bob", 1.0,
        )
        revived = admit_social_trust_checkpoint(self.path, self.model, new, reactivate)
        self.assertIn(
            ("alice", "bob"), mesh_edges(self.mesh(revived.next_runtime_state), R),
        )
        self.assertEqual(revived.receipt.before_social_hash, new.social_state.content_hash)
        self.assertEqual(revived.receipt.after_source_hash,
                         revived.next_runtime_state.snapshot.content_hash)

    def test_true_runtime_round_extends_rebound_social_memory(self):
        intervention = event_for(
            self.runtime, "deactivate-ab", SocialTrustOperation.DEACTIVATE,
            "alice", "bob", 0.0,
        )
        checked = admit_social_trust_checkpoint(
            self.path, self.model, self.runtime, intervention,
        )
        checkpoint = checked.next_runtime_state
        advanced = simulate_situated_network_round(
            self.path, self.model, checkpoint,
        )
        self.assertEqual(advanced.prior_state, checkpoint)
        self.assertEqual(advanced.next_state.parent_state_hash,
                         checkpoint.content_hash)
        self.assertEqual(advanced.next_state.social_state.parent_state_hash,
                         checkpoint.social_state.content_hash)
        self.assertEqual(advanced.next_state.snapshot.social_state_hash,
                         advanced.next_state.social_state.content_hash)
        self.assertEqual(advanced.next_state.snapshot,
                         project_situated_network_snapshot(
                             self.model, advanced.next_state.story,
                             advanced.next_state.cognitive_state,
                             advanced.next_state.social_state,
                         ))
        with self.assertRaises(SocialTopologyBoundaryError):
            admit_social_trust_checkpoint(
                self.path, self.model, advanced.next_state,
                event_for(
                    advanced.next_state, "illegal-mid-round",
                    SocialTrustOperation.ACTIVATE, "alice", "bob", 1.0,
                ),
            )

    def test_stale_provenance_unknown_agents_and_threshold_policy_fail_closed(self):
        old = self.runtime
        valid = event_for(
            old, "deactivate-ab", SocialTrustOperation.DEACTIVATE,
            "alice", "bob", 0.0,
        )
        for values in (
            {"expected_runtime_hash": "sha256:" + "f" * 64},
            {"expected_source_hash": "sha256:" + "f" * 64},
            {"expected_social_state_hash": "sha256:" + "f" * 64},
        ):
            with self.subTest(values=values), self.assertRaises(TopologyVersionConflict):
                admit_social_trust_checkpoint(
                    self.path, self.model, old, replace(valid, **values),
                )
        with self.assertRaises(TopologyUnknownAgent):
            admit_social_trust_checkpoint(
                self.path, self.model, old,
                replace(valid, observer_agent_id="missing"),
            )
        for operation, target in (
            (SocialTrustOperation.DEACTIVATE, 0.9),
            (SocialTrustOperation.ACTIVATE, 1.0),
            (SocialTrustOperation.DEACTIVATE, 0.5),
        ):
            with self.subTest(operation=operation, target=target), self.assertRaises(
                SocialTopologyTrustConflict
            ):
                admit_social_trust_checkpoint(
                    self.path, self.model, old,
                    replace(valid, operation=operation, target_trust=target),
                )
        for invalid in (
            {"target_trust": True},
            {"target_trust": float("nan")},
            {"target_trust": -0.1},
            {"target_trust": 1.1},
            {"event_id": " "},
            {"actor_id": "untrusted "},
            {"operation": "deactivate"},
            {"expected_social_state_hash": "wrong"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(TopologyChangeError):
                replace(valid, **invalid)
        self.assertEqual(old.social_state.content_hash, self.social.content_hash)

    def test_independent_source_checkpoint_verification_rejects_forged_effect(self):
        previous = self.runtime
        expected = event_for(
            previous, "deactivate-ab", SocialTrustOperation.DEACTIVATE,
            "alice", "bob", 0.0,
        )
        genuine = admit_social_trust_checkpoint(
            self.path, self.model, previous, expected,
        )
        self.assertTrue(
            verify_social_trust_checkpoint(
                self.path, self.model, previous, genuine,
            ),
        )
        different_event = replace(expected, target_trust=0.25)
        different = admit_social_trust_checkpoint(
            self.path, self.model, previous, different_event,
        )
        # Validly formed records with consistent source, social and runtime
        # digest bindings can still claim a different effective trust update.
        forged_receipt = replace(
            different.receipt, intervention_hash=expected.content_hash,
        )
        claimed = SocialTrustCheckpointResult(
            previous, different.next_runtime_state, expected, forged_receipt,
        )
        with self.assertRaises(SocialTopologyIntegrityError):
            verify_social_trust_checkpoint(
                self.path, self.model, previous, claimed,
            )
        with self.assertRaises(TopologyVersionConflict):
            verify_social_trust_checkpoint(
                self.path, self.model, genuine.next_runtime_state, genuine,
            )
        with self.assertRaises(TypeError):
            verify_social_trust_checkpoint(self.path, self.model, None, genuine)

    def test_constructor_rejects_forged_checkpoint_receipt_lineage(self):
        previous = self.runtime
        result = admit_social_trust_checkpoint(
            self.path, self.model, previous,
            event_for(previous, "deactivate-ab",
                      SocialTrustOperation.DEACTIVATE, "alice", "bob", 0.0),
        )
        for changes in (
            {"before_social_hash": "sha256:" + "f"*64},
            {"after_runtime_hash": "sha256:" + "f"*64},
            {"intervention_hash": "sha256:" + "f"*64},
            {"before_runtime_hash": "no-hash"},
            {"previous_active": 1},
            {"next_active": result.receipt.previous_active},
            {"threshold": True},
            {"threshold": float("inf")},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                SocialTopologyIntegrityError
            ):
                bad_receipt = replace(result.receipt, **changes)
                replace(result, receipt=bad_receipt)
        with self.assertRaises(SocialTopologyIntegrityError):
            replace(result, next_runtime_state=previous)

    def test_checked_in_social_trust_checkpoint_vectors_match_runtime(self):
        from pathlib import Path
        from tools.render_social_topology_checkpoint_vectors import (
            SCHEMA, build_vector_manifest, render_vector_corpus,
        )
        manifest = build_vector_manifest()
        self.assertEqual(manifest["schema"], SCHEMA)
        self.assertTrue(manifest["deactivate"]["old_path_invalid"])
        self.assertEqual(
            manifest["deactivate"]["receipt"]["before_source_hash"],
            manifest["original"]["source_hash"],
        )
        self.assertEqual(
            manifest["activate"]["receipt"]["before_source_hash"],
            manifest["deactivate"]["next_source_hash"],
        )
        reviewed = Path(__file__).resolve().parents[1] / "conformance" / "social_topology_checkpoint_v1.json"
        self.assertEqual(reviewed.read_bytes(), render_vector_corpus().encode("utf-8"))

    def test_original_authoritative_source_rejects_tampered_snapshot(self):
        old = self.runtime
        forged_snapshot = replace(
            old.snapshot,
            relationship_edges=tuple(
                replace(edge, active=False)
                if edge.source_agent_id == "alice" and edge.observer_agent_id == "bob"
                else edge
                for edge in old.snapshot.relationship_edges
            ),
        )
        forged_metrics = replace(old.metrics, snapshot_hash=forged_snapshot.content_hash)
        forged_state = replace(old, snapshot=forged_snapshot, metrics=forged_metrics)
        with self.assertRaisesRegex(ValueError, "authoritative snapshot"):
            admit_social_trust_checkpoint(
                self.path, self.model, forged_state,
                event_for(
                    forged_state, "deactivate-ab",
                    SocialTrustOperation.DEACTIVATE, "alice", "bob", 0.0,
                ),
            )
        self.assertEqual(old.snapshot, self.runtime.snapshot)

    def test_source_backed_bridge_requires_actual_trust_derived_edge(self):
        old = self.runtime
        societies = (
            MeshSociety("s0", ("alice",)), MeshSociety("s1", ("bob",)),
        )
        bridge = MeshBridge(R, "s0", "s1", "alice", "bob")
        old_mesh = MeshSnapshot(old.snapshot, societies, (bridge,))
        original = find_mesh_path(
            old_mesh, "alice", "bob", relation=R, max_hops=1,
        )
        self.assertIsNotNone(original)
        result = admit_social_trust_checkpoint(
            self.path, self.model, old,
            event_for(old, "deactivate-ab", SocialTrustOperation.DEACTIVATE,
                      "alice", "bob", 0.0),
        )
        with self.assertRaisesRegex(ValueError, "bridge lacks a directed source relation"):
            MeshSnapshot(result.next_runtime_state.snapshot, societies, (bridge,))
        new_mesh = MeshSnapshot(result.next_runtime_state.snapshot, societies)
        self.assertFalse(mesh_path_is_valid(
            new_mesh, original, "alice", "bob", relation=R, max_hops=1,
        ))


if __name__ == "__main__":
    unittest.main()
