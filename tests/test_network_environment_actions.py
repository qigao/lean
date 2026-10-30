"""R2.M2: real V19 MOVE/TAKE/DROP and subsequent content/interaction evidence."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSociety
from narrative_dynamics.abm.mesh_environment_observation import (
    EnvironmentActionTrace, observe_environment_action_trace,
)
from narrative_dynamics.abm.mesh_evolution_observation import observe_network_layers
from narrative_dynamics.abm.situated import (
    ObservationChannel, SituatedActionIntent, SituatedActionKind,
)
from narrative_dynamics.abm.situated_story import (
    advance_situated_story, initialize_situated_story, replay_situated_story,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from tests.test_network_abm_situated_network import initial_runtime_case
from tests.test_network_evolution_observation import build_review_scenarios

PRIVATE_MESSAGE = "R2 M2 private collaboration content - not for shared graph metadata"
SOCIETIES = (MeshSociety("office", ("alice", "bob")),)


def _act(action_id, agent_id, kind, target_id=None, *, message=None):
    return SituatedActionIntent(action_id, agent_id, kind, target_id, message)


def _view(model, story, cognition=None, social=None):
    if cognition is None:
        cognition = initialize_situated_percept_memory_cognition(
            model.percept_memory_model, story,
        )
    if social is None:
        social = initialize_situated_social_memory(
            model.social_memory_model, cognition,
        )
    return observe_network_layers(model, story, cognition, social, SOCIETIES)


def build_environment_scenarios():
    """Real accepted V19 story actions; cognition/social are exact checkpoints.

    Reinitializing social cognition for the final source is a checkpoint
    construction, NOT a claim that these explicit actions were produced by
    the autonomous cognitive decision engine.
    """
    model, original, cognition, social = initial_runtime_case()
    prior = _view(model, original, cognition, social)
    world = model.percept_memory_model.cognitive_model.world_model

    schedule = (
        (_act("alice-take-memo", "alice", SituatedActionKind.TAKE, "memo"),),
        (_act("alice-move-open", "alice", SituatedActionKind.MOVE, "records-open"),),
        (_act("alice-drop-memo", "alice", SituatedActionKind.DROP, "memo"),),
        (_act("alice-tell-bob", "alice", SituatedActionKind.TELL, message=PRIVATE_MESSAGE),),
    )
    current = original
    for intents in schedule:
        current = advance_situated_story(world, current, intents)
    replayed_story = replay_situated_story(
        world, original.initial_state, schedule,
        perception_model=model.percept_memory_model.perception_model,
    )
    if current != replayed_story:
        raise AssertionError("environment story did not reproduce from original V19 intents")
    accepted = _view(model, current)

    failures = (
        _act("alice-cannot-take", "alice", SituatedActionKind.TAKE, "badge"),
        _act("bob-cannot-move", "bob", SituatedActionKind.MOVE, "open-manager"),
    )
    failed_story = advance_situated_story(world, original, failures)
    rejected = _view(model, failed_story)
    return (
        ("take_move_drop_then_tell", prior, accepted),
        ("failed_environment_actions", prior, rejected),
    )


class EnvironmentActionTraceTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = {name:(before,after)
                          for name,before,after in build_environment_scenarios()}

    def test_real_take_move_drop_and_then_tell_are_separate_ordered_changes(self):
        before, after = self.scenarios["take_move_drop_then_tell"]
        report = observe_environment_action_trace(before, after).to_dict()
        self.assertEqual(report["story_lineage"], "STORY_EXTENSION")
        self.assertEqual(report["causal_attribution"], "UNATTRIBUTED")
        rounds = report["actual_source_rounds"]
        self.assertEqual(len(rounds), 4)
        self.assertEqual([r["round_index"] for r in rounds], [1, 2, 3, 4])
        kinds = [[effect["kind"] for effect in item["sourced_effects"]] for item in rounds]
        self.assertEqual(kinds, [["take"], ["move"], ["drop"], []])
        take = rounds[0]["sourced_effects"][0]
        self.assertEqual(take["object_id"], "memo")
        self.assertEqual(take["before"], {"place_id": "records", "holder_agent_id": None})
        self.assertEqual(take["after"], {"place_id": None, "holder_agent_id": "alice"})
        movement = rounds[1]["sourced_effects"][0]
        self.assertEqual((movement["actor_agent_id"], movement["before_place"],
                          movement["after_place"], movement["passage_id"]),
                         ("alice", "records", "open", "records-open"))
        drop = rounds[2]["sourced_effects"][0]
        self.assertEqual(drop["before"]["holder_agent_id"], "alice")
        self.assertEqual(drop["after"], {"place_id": "open", "holder_agent_id": None})
        self.assertEqual(rounds[0]["direct_interaction_before"], [])
        self.assertEqual(rounds[0]["direct_interaction_after"], [])
        self.assertEqual(rounds[1]["direct_interaction_before"], [])
        self.assertEqual(rounds[1]["direct_interaction_after"],
                         [["alice","bob"],["bob","alice"]])
        self.assertEqual(rounds[2]["direct_interaction_after"],
                         [["alice","bob"],["bob","alice"]])
        self.assertEqual(len(rounds[3]["successful_tell_event_ids"]),1)
        self.assertEqual([r["successful_tell_event_ids"] for r in rounds[:3]],
                         [[],[],[]])
        self.assertEqual(report["physical_boundary_changes"], [])
        self.assertEqual(report["multi_agent_content"]["changed_trust_pair_count"],0)
        rel = report["selected_relation_graphs"]
        self.assertEqual(rel["direct_interaction"]["source_edges"]["added"],
                         [["alice","bob"],["bob","alice"]])
        self.assertEqual(rel["transmission"]["source_edges"]["added"],
                         [["alice","bob"]])
        self.assertEqual(rel["relationship"]["source_edges"],
                         {"added":[],"removed":[]})
        self.assertEqual(before.story.current_state.round_index, 0)
        self.assertEqual(after.story.current_state.round_index, 4)
        self.assertEqual(before.social.relationships, after.social.relationships)

    def test_drop_is_observable_after_move_but_prior_take_is_local(self):
        _, after = self.scenarios["take_move_drop_then_tell"]
        take, move, drop, tell = after.story.rounds
        take_event = next(e for e in take.events if e.action_id == "alice-take-memo")
        drop_event = next(e for e in drop.events if e.action_id == "alice-drop-memo")
        take_observers = {o.agent_id for o in take.observations if o.event_id == take_event.event_id}
        drop_channels = {
            o.agent_id:o.channel for o in drop.observations if o.event_id == drop_event.event_id
        }
        self.assertEqual(take_observers, {"alice"})
        self.assertIs(drop_channels["bob"], ObservationChannel.VISUAL)
        self.assertEqual(move.next_state.parent_state_hash,take.next_state.content_hash)
        self.assertEqual(drop.next_state.parent_state_hash,move.next_state.content_hash)
        self.assertEqual(tell.next_state.parent_state_hash,drop.next_state.content_hash)

    def test_failed_actions_cannot_be_admitted_as_environment_effects(self):
        before, after = self.scenarios["failed_environment_actions"]
        payload = observe_environment_action_trace(before, after).to_dict()
        rounds = payload["actual_source_rounds"]
        self.assertEqual(len(rounds),1)
        self.assertEqual(rounds[0]["sourced_effects"],[])
        self.assertEqual({x["outcome"] for x in rounds[0]["failed_environment_actions"]},
                         {"object_not_co_located","passage_closed"})
        self.assertEqual(rounds[0]["direct_interaction_before"],
                         rounds[0]["direct_interaction_after"])
        self.assertEqual(payload["selected_relation_graphs"]["transmission"]["source_edges"],
                         {"added":[],"removed":[]})
        self.assertEqual(before.mesh.source_snapshot.access_edges,
                         after.mesh.source_snapshot.access_edges)

    def test_forged_world_object_effect_is_not_a_real_story_event(self):
        before, valid = self.scenarios["take_move_drop_then_tell"]
        model=before.model
        taken = valid.story.rounds[0]
        # A round with a legitimate TAKE outcome but a falsified next-object
        # state may be structurally well-typed. The observer must re-execute
        # the actual V19 transition rather than believe this record.
        forged_next=replace(taken.next_state, objects=taken.prior_state.objects)
        forged_round=replace(taken, next_state=forged_next)
        forged_story=replace(before.story, rounds=(forged_round,))
        forged_view=_view(model, forged_story)
        with self.assertRaisesRegex(ValueError,"differ from authoritative V19 execution"):
            observe_environment_action_trace(before,forged_view)

    def test_forged_move_destination_and_unrelated_history_rejected(self):
        before, valid = self.scenarios["take_move_drop_then_tell"]
        model=before.model
        taken,moved=valid.story.rounds[:2]
        wrong_agent=tuple(
            replace(body,place_id="lobby") if body.agent_id=="alice" else body
            for body in moved.next_state.agents
        )
        forged=replace(moved, next_state=replace(moved.next_state,agents=wrong_agent))
        forged_story=replace(before.story,rounds=(taken,forged))
        forged_view=_view(model,forged_story)
        with self.assertRaisesRegex(ValueError,"differ from authoritative V19 execution"):
            observe_environment_action_trace(before,forged_view)
        with self.assertRaisesRegex(ValueError,"requires an exact real V19 story extension"):
            observe_environment_action_trace(before,before)
        counterfactual=next(after for name,_,after in build_review_scenarios()
                            if name=="environment_alternative")
        with self.assertRaisesRegex(ValueError,"requires an exact real V19 story extension"):
            observe_environment_action_trace(before,counterfactual)

    def test_source_trace_privacy_and_immutability(self):
        before,after=self.scenarios["take_move_drop_then_tell"]
        trace=observe_environment_action_trace(before,after)
        text=json.dumps(trace.to_dict(),sort_keys=True)
        self.assertNotIn(PRIVATE_MESSAGE,text)
        self.assertNotIn('"message":',text)
        old_hash=trace.content_hash
        payload=trace.to_dict()
        payload["causal_attribution"]="PROVED"
        payload["actual_source_rounds"][0]["sourced_effects"][0]["event_id"]="forged"
        self.assertEqual(trace.content_hash,old_hash)
        self.assertEqual(trace.to_dict()["causal_attribution"],"UNATTRIBUTED")
        with self.assertRaises(FrozenInstanceError):
            trace._canonical_json="{}"
        with self.assertRaisesRegex(ValueError,"exact sources and event scope"):
            replace(trace,_canonical_json=json.dumps(payload))
        with self.assertRaises(TypeError):
            EnvironmentActionTrace(trace.prior_source_hash,trace.next_source_hash,{})

    def test_checked_in_environment_action_vectors_are_exact_source_bytes(self):
        from tools.render_network_environment_vectors import (
            build_vector_manifest,render_vector_corpus,
        )
        manifest=build_vector_manifest()
        self.assertEqual(manifest["schema"],"v19-environment-action-trace-v1")
        self.assertEqual(len(manifest["cases"]),2)
        text=render_vector_corpus()
        self.assertNotIn(PRIVATE_MESSAGE,text)
        pin=Path(__file__).resolve().parents[1]/"conformance/network_environment_action_v1.json"
        if not pin.exists():
            # One-time fail-closed actual CI handoff; remove after source review.
            raise AssertionError("R2_ENV_VECTOR_BEGIN\n"+text+"R2_ENV_VECTOR_END")
        self.assertEqual(pin.read_bytes(),text.encode("utf-8"))


if __name__=="__main__":
    unittest.main()
