"""R2.M4a: frozen same-origin V19 boundary/movement/content experiment.

No new policy, causal inference engine or graph owner. Two fully
reexecuted V19 story branches have the SAME initial world/model and
identical MOVE/TELL intents. The only intervention is OPEN_PASSAGE vs
WAIT in round one. Outcomes are observed through existing cognitive,
social, source-graph and V24 projection APIs.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_contracts import (
    MeshRelation, MeshSociety, source_relation_edges,
)
from narrative_dynamics.abm.mesh_evolution_observation import (
    compare_network_layers, observe_network_layers,
)
from narrative_dynamics.abm.situated import (
    ObservationChannel, SituatedActionIntent, SituatedActionKind,
)
from narrative_dynamics.abm.situated_network import (
    project_situated_network_snapshot,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRuntimeModel,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import (
    initialize_situated_story, replay_situated_story,
)
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryRecallCue,
)
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_agent_content_evidence import _run
from tests.test_network_abm_situated_percept_cognition import (
    SECRET, initial_state, perception_model,
)
from tests.test_network_abm_situated_percept_memory_cognition import (
    recall_model,
)
from tests.test_network_abm_situated_percept_social_cognition import (
    bound_social_model,
)

SCHEMA = "v19-controlled-coupled-evolution-v1"


def _tell_from(story, action_id):
    return next(
        event for event in story.rounds[-1].events
        if event.action_id == action_id
    )


def _move_from(story, action_id):
    return next(
        event for event in story.rounds[-2].events
        if event.action_id == action_id
    )


def build_paired_coupled_case() -> dict[str, object]:
    perception = perception_model()
    cognition = cognitive_office_model()
    world = cognition.world_model
    initial = initial_state(perception, door_open=False)
    root = initialize_situated_story(
        world, initial, perception_model=perception,
    )
    cue = SituatedMemoryRecallCue(
        "paired-tell", SECRET,
        event_kinds=(SituatedActionKind.TELL,),
        channels=(ObservationChannel.AUDITORY,),
    )
    memory_model = recall_model(
        {"bob": (cue,)}, cognition=cognition, perception=perception,
    )
    social_model = bound_social_model(memory_model)
    network_model = SituatedNetworkRuntimeModel(
        "r2-coupled-frozen-control", "1",
        memory_model, social_model, "approved", 0.7, 0.5,
    )
    root_cog = initialize_situated_percept_memory_cognition(
        memory_model, root,
    )
    root_social = initialize_situated_social_memory(
        social_model, root_cog,
    )
    root_source = project_situated_network_snapshot(
        network_model, root, root_cog, root_social,
    )
    # Exactly the same second and third-round authored action for both
    # branches. A TELL event ID may repeat across *different* branches;
    # its payload/world event hash and enclosing story hash must differ.
    move = SituatedActionIntent(
        "alice-common-move", "alice",
        SituatedActionKind.MOVE, target_id="records-open",
    )
    tell = SituatedActionIntent(
        "alice-common-tell", "alice",
        SituatedActionKind.TELL, message=SECRET,
    )
    start_control = SituatedActionIntent(
        "alice-control-wait", "alice", SituatedActionKind.WAIT,
    )
    start_open = SituatedActionIntent(
        "alice-open-gate", "alice",
        SituatedActionKind.OPEN_PASSAGE, target_id="records-open",
    )
    schedules = {
        "control_no_gate_change": ((start_control,), (move,), (tell,)),
        "intervention_open_gate": ((start_open,), (move,), (tell,)),
    }
    arms: dict[str, dict[str, object]] = {}
    views = {}
    origins = {}
    for name, schedule in schedules.items():
        story = replay_situated_story(
            world, initial, schedule, perception_model=perception,
        )
        mem, social, story, prior_cog, prior_social, outcome, trace = _run(
            memory_model, social_model, story, name,
        )
        if mem != memory_model or social != social_model:
            raise AssertionError("controlled branches unexpectedly changed V19 policy")
        if story.initial_state != root.initial_state:
            raise AssertionError("controlled branch changed initial world")
        source = project_situated_network_snapshot(
            network_model, story, prior_cog, prior_social,
        )
        learned = project_situated_network_snapshot(
            network_model, outcome.next_story,
            outcome.next_cognitive_state, outcome.next_social_state,
        )
        views[name] = observe_network_layers(
            network_model, story, prior_cog, prior_social,
            (MeshSociety("office", tuple(n.agent_id for n in source.nodes)),),
        )
        movement = _move_from(story, move.action_id)
        message = _tell_from(story, tell.action_id)
        if message.kind is not SituatedActionKind.TELL or not message.success:
            raise AssertionError("both branches require a real successful TELL")
        origins[name] = (message, story)
        claims = [
            {"claim_hash": c.content_hash, "status": c.status.value,
             "observer": c.observer_agent_id, "source": c.source_agent_id}
            for c in outcome.next_social_state.claims
        ]
        arm = {
            "story_hash": story.content_hash,
            "social_checkpoint_hash": prior_social.content_hash,
            "source_after_tell_hash": source.content_hash,
            "source_after_social_round_hash": learned.content_hash,
            "world_round_hashes": [r.content_hash for r in story.rounds],
            "gate_open_at_tell": next(p.open for p in story.current_state.passages
                                      if p.passage_id == "records-open"),
            "alice_place_at_tell": next(x.place_id for x in story.current_state.agents
                                        if x.agent_id == "alice"),
            "move": {
                "event_id": movement.event_id,
                "event_hash": movement.content_hash,
                "success": movement.success, "outcome": movement.outcome,
            },
            "tell": {
                "event_id": message.event_id,
                "event_hash": message.content_hash,
                "success": message.success,
            },
            "social_result_hash": outcome.content_hash,
            "content_trace_hash": trace.content_hash,
            "content_counts": trace.to_dict()["counts"],
            "other_agent_fidelity": (
                trace.to_dict()["source_tell_events"][0]
                ["other_agent_perception_by_fidelity"]
            ),
            "claims_after_learning": claims,
            "relationship_edges_at_tell": [
                list(edge) for edge in source_relation_edges(
                    source, MeshRelation.RELATIONSHIP,
                )
            ],
            "direct_interaction_edges_at_tell": [
                list(edge) for edge in source_relation_edges(
                    source, MeshRelation.DIRECT_INTERACTION,
                )
            ],
            "realized_transmission_edges_at_tell": [
                list(edge) for edge in source_relation_edges(
                    source, MeshRelation.TRANSMISSION,
                )
            ],
            "relationship_edges_after_learning": [
                list(edge) for edge in source_relation_edges(
                    learned, MeshRelation.RELATIONSHIP,
                )
            ],
        }
        arms[name] = arm
    control_message, control_story = origins["control_no_gate_change"]
    opened_message, opened_story = origins["intervention_open_gate"]
    if (
        control_message.event_id != opened_message.event_id
        or control_message.content_hash == opened_message.content_hash
        or control_story.content_hash == opened_story.content_hash
    ):
        raise AssertionError("counterfactual same event ID must bind distinct source/story hashes")
    comparison = compare_network_layers(
        views["control_no_gate_change"], views["intervention_open_gate"],
    ).to_dict()
    if (
        comparison["causal_attribution"] != "UNATTRIBUTED"
        or comparison["story_lineage"] != "DIVERGENT_STORY_BRANCH"
    ):
        raise AssertionError("divergent intervention branches must not be presented as real event history")
    return {
        "schema": SCHEMA,
        "scope": "deterministic V19 paired intervention; not statistical/empirical causal proof",
        "initial_world_hash": initial.content_hash,
        "initial_source_hash": root_source.content_hash,
        "model_hash": network_model.content_hash,
        "shared_move_action_id": move.action_id,
        "shared_tell_action_id": tell.action_id,
        "arms": arms,
        "cross_branch": {
            "story_lineage": comparison["story_lineage"],
            "causal_attribution": comparison["causal_attribution"],
            "observed_axes": comparison["observed_axes"],
            "relation_deltas": comparison["selected_relation_graphs"],
        },
    }


class CoupledNetworkEvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = build_paired_coupled_case()

    def test_identical_origin_except_first_real_passage_action(self):
        data = self.case
        control = data["arms"]["control_no_gate_change"]
        intervention = data["arms"]["intervention_open_gate"]
        self.assertEqual(data["cross_branch"]["story_lineage"],
                         "DIVERGENT_STORY_BRANCH")
        self.assertEqual(data["cross_branch"]["causal_attribution"], "UNATTRIBUTED")
        self.assertFalse(control["gate_open_at_tell"])
        self.assertTrue(intervention["gate_open_at_tell"])
        self.assertEqual(control["alice_place_at_tell"], "records")
        self.assertEqual(intervention["alice_place_at_tell"], "open")
        self.assertFalse(control["move"]["success"])
        self.assertEqual(control["move"]["outcome"], "passage_closed")
        self.assertTrue(intervention["move"]["success"])
        self.assertEqual(intervention["move"]["outcome"], "moved")
        self.assertEqual(control["tell"]["event_id"], intervention["tell"]["event_id"])
        self.assertNotEqual(control["tell"]["event_hash"], intervention["tell"]["event_hash"])

    def test_boundary_and_environment_change_opportunity_not_past_content(self):
        x = self.case["arms"]
        c, i = x["control_no_gate_change"], x["intervention_open_gate"]
        control_edges = {tuple(edge) for edge in c["direct_interaction_edges_at_tell"]}
        treated_edges = {tuple(edge) for edge in i["direct_interaction_edges_at_tell"]}
        # Bob and Carol are co-located *in both branches*; their existing
        # opportunity must not be erased to exaggerate the intervention.
        self.assertIn(("bob", "carol"), control_edges)
        self.assertIn(("carol", "bob"), control_edges)
        self.assertNotIn(("alice", "bob"), control_edges)
        self.assertNotIn(("bob", "alice"), control_edges)
        self.assertEqual(
            treated_edges - control_edges,
            {("alice", "bob"), ("bob", "alice"),
             ("alice", "carol"), ("carol", "alice")},
        )
        self.assertEqual(control_edges - treated_edges, set())
        self.assertEqual(c["realized_transmission_edges_at_tell"], [])
        self.assertIn(["alice", "bob"], i["realized_transmission_edges_at_tell"])
        self.assertFalse(self.case["cross_branch"]["observed_axes"]["social_trust"])
        self.assertTrue(self.case["cross_branch"]["observed_axes"]["physical_boundary"])
        self.assertTrue(self.case["cross_branch"]["observed_axes"]["environment"])

    def test_exact_perception_yields_testimony_but_no_auto_verification_or_trust(self):
        x = self.case["arms"]
        c, i = x["control_no_gate_change"], x["intervention_open_gate"]
        self.assertEqual(c["other_agent_fidelity"], {"detected": 2})
        self.assertEqual(i["other_agent_fidelity"], {"exact": 2})
        self.assertEqual(c["content_counts"]["actual_tell_events"], 1)
        self.assertEqual(i["content_counts"]["actual_tell_events"], 1)
        self.assertEqual(c["content_counts"]["admitted_testimony"], 0)
        self.assertEqual(i["content_counts"]["admitted_testimony"], 1)
        self.assertEqual(c["content_counts"]["admitted_verification"], 0)
        self.assertEqual(i["content_counts"]["admitted_verification"], 0)
        self.assertEqual(c["claims_after_learning"], [])
        self.assertEqual(len(i["claims_after_learning"]), 1)
        self.assertEqual(i["claims_after_learning"][0]["status"], "active")
        self.assertEqual(c["relationship_edges_at_tell"],
                         i["relationship_edges_at_tell"])
        self.assertEqual(c["relationship_edges_after_learning"],
                         i["relationship_edges_after_learning"])

    def test_canonical_paired_vector_matches_real_v19_model(self):
        from tools.render_network_coupled_evolution_vectors import render_vector_corpus
        content=render_vector_corpus()
        self.assertNotIn(SECRET,content)
        self.assertNotIn('"message":',content)
        self.assertEqual(
            json.loads(content)["schema"], SCHEMA,
        )
        pin=Path(__file__).resolve().parents[1]/"conformance/network_coupled_evolution_v1.json"
        # Mandatory reviewed V19 source vector, never created in CI.
        self.assertEqual(pin.read_bytes(),content.encode("utf-8"))


if __name__=="__main__":
    unittest.main()
