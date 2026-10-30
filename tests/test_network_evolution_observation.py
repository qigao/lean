"""R2.M0 regression: source-derived boundary, environment, and content deltas."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from narrative_dynamics.abm.mesh_contracts import MeshSnapshot, MeshSociety
from narrative_dynamics.abm.mesh_evolution_observation import (
    NetworkLayerView, NetworkChangeReport, compare_network_layers, observe_network_layers,
)
from narrative_dynamics.abm.situated_network import initialize_situated_network_runtime
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import initialize_situated_story
from narrative_dynamics.abm.situated_social_topology import (
    SocialTrustCheckpointIntervention, SocialTrustOperation,
    admit_social_trust_checkpoint,
)
from tests.test_network_abm_situated_network import (
    initial_runtime_case, tell_case, SECRET,
)


def _view(model, story, cognition, social, societies=None):
    if societies is None:
        societies = (MeshSociety("office", ("alice", "bob")),)
    return observe_network_layers(model, story, cognition, social, societies)


def _world_variant(model, original_story, *, close_passage=False, move_agent=False):
    """Two alternative initial conditions, never a claimed runtime event."""
    initial = original_story.initial_state
    if close_passage:
        target = next(p.passage_id for p in initial.passages if p.open)
        initial = replace(initial, passages=tuple(
            replace(p, open=False) if p.passage_id == target else p
            for p in initial.passages
        ))
    if move_agent:
        first = next(a for a in initial.agents if a.agent_id == "alice")
        other_place = next(p.place_id for p in model.percept_memory_model.perception_model.world_model.places
                           if p.place_id != first.place_id)
        initial = replace(initial, agents=tuple(
            replace(a, place_id=other_place) if a.agent_id == "alice" else a
            for a in initial.agents
        ))
    next_story = initialize_situated_story(
        model.percept_memory_model.cognitive_model.world_model,
        initial, perception_model=model.percept_memory_model.perception_model,
    )
    cognition = initialize_situated_percept_memory_cognition(
        model.percept_memory_model, next_story,
    )
    social = initialize_situated_social_memory(model.social_memory_model, cognition)
    return _view(model, next_story, cognition, social)


def build_review_scenarios():
    """Share the same concrete canonical cases with the JSON vector generator."""
    model, story, cognition, social = initial_runtime_case()
    before = _view(model, story, cognition, social)
    scenarios = [
        ("physical_boundary_alternative", before,
         _world_variant(model, story, close_passage=True)),
        ("environment_alternative", before,
         _world_variant(model, story, move_agent=True)),
        ("society_boundary_change", before,
         _view(model, story, cognition, social, (
             MeshSociety("alice-society", ("alice",)),
             MeshSociety("bob-society", ("bob",)),
         ))),
    ]
    next_model, next_story, next_cognition, next_social = tell_case(door_open=True)
    if next_model.content_hash != model.content_hash:
        raise AssertionError("content fixture must preserve the exact V19 model")
    scenarios.append((
        "real_tell_content_extension", before,
        _view(next_model, next_story, next_cognition, next_social),
    ))
    with TemporaryDirectory(prefix="r2-network-observation-") as folder:
        path = str(Path(folder) / "memory.sqlite3")
        old = initialize_situated_network_runtime(
            path, model, story, cognition, social,
        )
        event = SocialTrustCheckpointIntervention(
            "threshold-cross", old.content_hash, old.snapshot.content_hash,
            old.social_state.content_hash, "alice", "bob",
            "review-test", "explicit, non-authenticated checkpoint edit",
            SocialTrustOperation.DEACTIVATE, 0.0,
        )
        changed = admit_social_trust_checkpoint(path, model, old, event)
        scenarios.append((
            "explicit_social_trust_checkpoint", before,
            _view(model, changed.next_runtime_state.story,
                  changed.next_runtime_state.cognitive_state,
                  changed.next_runtime_state.social_state),
        ))
    return tuple(scenarios)


class NetworkEvolutionObservationTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = dict((name, (before, after))
                              for name, before, after in build_review_scenarios())

    def test_physical_boundary_not_inferred_as_message_or_trust(self):
        before, after = self.scenarios["physical_boundary_alternative"]
        report = compare_network_layers(before, after).to_dict()
        self.assertEqual(report["story_lineage"], "DIFFERENT_INITIAL_CONDITIONS")
        self.assertTrue(report["observed_axes"]["physical_boundary"])
        self.assertFalse(report["observed_axes"]["social_trust"])
        self.assertFalse(report["observed_axes"]["multi_agent_content"])
        self.assertTrue(report["boundary"]["physical_passages"])
        self.assertEqual(report["interaction_content"]["new_successful_tell_event_ids"], [])
        self.assertEqual(report["causal_attribution"], "UNATTRIBUTED")
        self.assertEqual(report["selected_relation_graphs"]["relationship"]["source_edges"],
                         {"added": [], "removed": []})

    def test_environment_change_does_not_create_transmission(self):
        before, after = self.scenarios["environment_alternative"]
        report = compare_network_layers(before, after).to_dict()
        self.assertTrue(report["observed_axes"]["environment"])
        self.assertFalse(report["observed_axes"]["physical_boundary"])
        self.assertFalse(report["observed_axes"]["multi_agent_content"])
        self.assertTrue(report["environment"]["agent_movements"])
        self.assertEqual(report["selected_relation_graphs"]["transmission"]["source_edges"],
                         {"added": [], "removed": []})

    def test_society_membership_scope_can_change_without_source_edges(self):
        before, after = self.scenarios["society_boundary_change"]
        report = compare_network_layers(before, after).to_dict()
        self.assertTrue(report["observed_axes"]["society_boundary"])
        self.assertEqual(report["story_lineage"], "SAME_STORY")
        self.assertTrue(report["boundary"]["society_memberships"])
        self.assertFalse(report["observed_axes"]["social_trust"])
        self.assertFalse(report["observed_axes"]["multi_agent_content"])
        rel = report["selected_relation_graphs"]["relationship"]
        self.assertEqual(rel["source_edges"], {"added": [], "removed": []})
        self.assertGreater(len(rel["projected_edges"]["removed"]), 0)
        self.assertEqual(before.mesh.source_snapshot.content_hash,
                         after.mesh.source_snapshot.content_hash)

    def test_actual_tell_changes_recorded_content_not_automatically_trust(self):
        before, after = self.scenarios["real_tell_content_extension"]
        report = compare_network_layers(before, after).to_dict()
        self.assertEqual(report["story_lineage"], "STORY_EXTENSION")
        self.assertTrue(report["observed_axes"]["multi_agent_content"])
        self.assertFalse(report["observed_axes"]["physical_boundary"])
        self.assertFalse(report["observed_axes"]["environment"])
        self.assertTrue(report["interaction_content"]["new_story_events"])
        self.assertTrue(report["interaction_content"]["new_successful_tell_event_ids"])
        self.assertEqual(report["interaction_content"]["social_trust_changes"], [])
        self.assertEqual(report["causal_attribution"], "UNATTRIBUTED")
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn('"message":', serialized)
        self.assertNotIn('"observation_payload"', serialized)

    def test_trust_changes_are_observed_not_assumed_from_agent_content(self):
        before, after = self.scenarios["explicit_social_trust_checkpoint"]
        report = compare_network_layers(before, after).to_dict()
        self.assertEqual(report["story_lineage"], "SAME_STORY")
        self.assertTrue(report["observed_axes"]["social_trust"])
        self.assertFalse(report["observed_axes"]["multi_agent_content"])
        self.assertFalse(report["observed_axes"]["physical_boundary"])
        trust = report["interaction_content"]["social_trust_changes"]
        self.assertEqual(len(trust), 1)
        self.assertEqual((trust[0]["source_agent_id"], trust[0]["observer_agent_id"]),
                         ("alice", "bob"))
        self.assertTrue(trust[0]["effective_before"])
        self.assertFalse(trust[0]["effective_after"])
        self.assertEqual(report["selected_relation_graphs"]["relationship"]
                         ["source_edges"]["removed"], [["alice", "bob"]])
        self.assertEqual(report["interaction_content"]["new_successful_tell_event_ids"], [])
        self.assertEqual(report["causal_attribution"], "UNATTRIBUTED")

    def test_authority_mismatch_and_illegal_backward_time_fail_closed(self):
        before, after = self.scenarios["real_tell_content_extension"]
        with self.assertRaisesRegex(ValueError, "must not reverse"):
            compare_network_layers(after, before)
        with self.assertRaisesRegex(ValueError, "authoritative V19"):
            replace(before, mesh=MeshSnapshot(
                replace(before.mesh.source_snapshot, round_index=5),
                before.mesh.societies,
            ))
        with self.assertRaises(TypeError):
            compare_network_layers(before, object())
        changed_model = replace(before.model, relationship_trust_threshold=0.6)
        different = _view(changed_model, before.story, before.cognition, before.social)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            compare_network_layers(before, different)

    def test_report_content_is_immutable_and_never_claims_causes(self):
        before, after = self.scenarios["real_tell_content_extension"]
        report = compare_network_layers(before, after)
        reviewed = report.to_dict()
        unchanged_hash = report.content_hash
        reviewed["causal_attribution"] = "PROVED"
        reviewed["interaction_content"]["new_story_events"].append(
            {"event_id": "forged", "success": True, "message": SECRET},
        )
        self.assertEqual(report.to_dict()["causal_attribution"], "UNATTRIBUTED")
        self.assertNotIn("forged", json.dumps(report.to_dict(), sort_keys=True))
        self.assertNotIn(SECRET, json.dumps(report.to_dict(), sort_keys=True))
        self.assertEqual(report.content_hash, unchanged_hash)
        with self.assertRaises(FrozenInstanceError):
            report._canonical_json = "{}"
        with self.assertRaisesRegex(ValueError, "exact source and attribution"):
            replace(report, _canonical_json=json.dumps(reviewed, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "exact source and attribution"):
            replace(report, _canonical_json=json.dumps(
                report.to_dict() | {"after": {"source_snapshot_hash": "sha256:" + "f"*64}},
                sort_keys=True,
            ))
        with self.assertRaises(TypeError):
            NetworkChangeReport(report.before_source_hash,
                                report.after_source_hash, report.story_lineage, {})

    def test_canonical_reviewed_vectors_match_all_real_source_revisions(self):
        from tools.render_network_evolution_vectors import (
            SCHEMA, build_vector_manifest, render_vector_corpus,
        )
        expected = build_vector_manifest()
        self.assertEqual(expected["schema"], SCHEMA)
        self.assertEqual(len(expected["cases"]), 5)
        data = render_vector_corpus().encode("utf-8")
        path = Path(__file__).resolve().parents[1] / "conformance" / "network_evolution_observation_v1.json"
        self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
