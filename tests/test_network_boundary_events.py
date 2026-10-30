"""R2.M1: real V19 physical passage events and source/mesh projections."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSociety
from narrative_dynamics.abm.mesh_evolution_observation import (
    compare_network_layers, observe_network_layers,
)
from narrative_dynamics.abm.situated import (
    SituatedActionIntent, SituatedActionKind, resolve_situated_round,
)
from narrative_dynamics.abm.situated_contracts import (
    PassageState, initialize_situated_world,
)
from narrative_dynamics.abm.situated_perception import (
    can_situated_agents_interact, derive_situated_perception_reach,
    percepts_for_agent, project_situated_percepts,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedEdgeActivation, SituatedEventSignalProfile,
    SituatedPerceptionEdge, SituatedPerceptionLayer,
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
from tests.test_network_abm_situated_network import runtime_model
from tests.test_network_abm_situated_percept_cognition import initial_state
from tests.test_network_abm_situated_perception import (
    _projection_model, _projection_state,
)

R = MeshRelation.RELATIONSHIP
D = MeshRelation.DIRECT_INTERACTION
T = MeshRelation.TRANSMISSION


def act(action_id, agent_id, kind, target_id):
    return SituatedActionIntent(action_id, agent_id, kind, target_id=target_id)


def boundary_fixture():
    """Source-only directed gate, two operators and one remote observer."""
    perception = _projection_model()
    interaction = SituatedPerceptionEdge(
        "corridor-meeting-interaction", SituatedPerceptionLayer.INTERACTION,
        "corridor", "meeting", 0.0,
        SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door",
    )
    perception = replace(
        perception,
        edges=perception.edges + (interaction,),
        signal_profiles=perception.signal_profiles + (
            SituatedEventSignalProfile(SituatedActionKind.OPEN_PASSAGE, True),
            SituatedEventSignalProfile(SituatedActionKind.CLOSE_PASSAGE, True),
        ),
    )
    source = _projection_state(perception, open_passage=False)
    source = replace(source, agents=tuple(
        replace(body, place_id="corridor") if body.agent_id == "carol" else body
        for body in source.agents
    ))
    return perception, source


def build_real_boundary_case():
    """Real V19 social/cognitive runtime with a gated interaction edge."""
    model = runtime_model()
    prior_perception = model.percept_memory_model.perception_model
    interaction = SituatedPerceptionEdge(
        "records-open-interaction", SituatedPerceptionLayer.INTERACTION,
        "records", "open", 0.0,
        SituatedEdgeActivation.PASSAGE_OPEN, "records-open",
    )
    perception = replace(
        prior_perception, edges=prior_perception.edges + (interaction,),
    )
    model = replace(model, percept_memory_model=replace(
        model.percept_memory_model, perception_model=perception,
    ))
    world = model.percept_memory_model.cognitive_model.world_model
    story = initialize_situated_story(
        world, initial_state(perception, door_open=False),
        perception_model=perception,
    )
    cognition = initialize_situated_percept_memory_cognition(
        model.percept_memory_model, story,
    )
    social = initialize_situated_social_memory(
        model.social_memory_model, cognition,
    )
    societies = (MeshSociety("office", ("alice", "bob")),)
    before = observe_network_layers(
        model, story, cognition, social, societies,
    )
    event = act(
        "alice-opens-records", "alice", SituatedActionKind.OPEN_PASSAGE,
        "records-open",
    )
    new_story = advance_situated_story(world, story, (event,))
    new_cognition = initialize_situated_percept_memory_cognition(
        model.percept_memory_model, new_story,
    )
    new_social = initialize_situated_social_memory(
        model.social_memory_model, new_cognition,
    )
    after = observe_network_layers(
        model, new_story, new_cognition, new_social, societies,
    )
    report = compare_network_layers(before, after)
    return model, before, after, report


class BoundaryEventTests(unittest.TestCase):
    def test_open_changes_next_round_perception_not_pre_round_observation(self):
        perception, before = boundary_fixture()
        world = perception.world_model
        self.assertFalse(can_situated_agents_interact(perception, before, "alice", "bob"))
        prior_reach = derive_situated_perception_reach(
            perception, before, source_place_id="corridor",
        )
        self.assertNotIn("meeting", prior_reach.visual_costs)
        self.assertEqual(prior_reach.auditory_losses["meeting"], 25.0)

        opened = resolve_situated_round(world, before, (
            act("open-door", "alice", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
            act("carol-moves", "carol", SituatedActionKind.MOVE, "meeting-door"),
        ))
        opened_event = next(e for e in opened.events if e.action_id == "open-door")
        move_event = next(e for e in opened.events if e.action_id == "carol-moves")
        self.assertTrue(opened_event.success)
        self.assertEqual(opened_event.outcome, "passage_opened")
        self.assertFalse(move_event.success)
        self.assertEqual(move_event.outcome, "passage_closed")
        self.assertTrue(opened.next_state.passages[0].open)
        self.assertFalse(before.passages[0].open)
        self.assertTrue(can_situated_agents_interact(
            perception, opened.next_state, "alice", "bob",
        ))
        self.assertFalse(can_situated_agents_interact(
            perception, opened.next_state, "bob", "alice",
        ))
        next_reach = derive_situated_perception_reach(
            perception, opened.next_state, source_place_id="corridor",
        )
        self.assertEqual(next_reach.visual_costs["meeting"], 1.0)
        self.assertEqual(next_reach.auditory_losses["meeting"], 5.0)

        observed = project_situated_percepts(perception, opened)
        carol = percepts_for_agent(observed, "carol")
        bob = percepts_for_agent(observed, "bob")
        self.assertTrue(any(p.source_event_id == opened_event.event_id for p in carol))
        self.assertFalse(any(p.source_event_id == opened_event.event_id for p in bob))

        # MOVE uses the OPEN prior snapshot, even as CLOSE succeeds this round.
        next_round = resolve_situated_round(world, opened.next_state, (
            act("alice-moves", "alice", SituatedActionKind.MOVE, "meeting-door"),
            act("carol-closes", "carol", SituatedActionKind.CLOSE_PASSAGE, "meeting-door"),
        ))
        self.assertTrue(next(e for e in next_round.events
                             if e.action_id == "alice-moves").success)
        self.assertTrue(next(e for e in next_round.events
                             if e.action_id == "carol-closes").success)
        self.assertEqual(next(a.place_id for a in next_round.next_state.agents
                              if a.agent_id == "alice"), "meeting")
        self.assertFalse(next_round.next_state.passages[0].open)
        blocked = resolve_situated_round(world, next_round.next_state, (
            act("carol-move-later", "carol", SituatedActionKind.MOVE, "meeting-door"),
        ))
        self.assertEqual(next(e for e in blocked.events
                              if e.action_id == "carol-move-later").outcome,
                         "passage_closed")

    def test_concurrent_passage_edits_fail_closed_order_independently(self):
        model, before = boundary_fixture()
        intents = (
            act("a-open", "alice", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
            act("c-open", "carol", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
        )
        first = resolve_situated_round(model.world_model, before, intents)
        reverse = resolve_situated_round(model.world_model, before, intents[::-1])
        self.assertEqual(first, reverse)
        self.assertEqual(first.next_state.passages, before.passages)
        self.assertEqual({x.outcome for x in first.events if x.kind is
                          SituatedActionKind.OPEN_PASSAGE}, {"passage_conflict"})
        combined = resolve_situated_round(model.world_model, before, (
            act("a-close", "alice", SituatedActionKind.CLOSE_PASSAGE, "meeting-door"),
            act("c-open", "carol", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
        ))
        self.assertEqual(combined.next_state.passages, before.passages)
        self.assertEqual({e.outcome for e in combined.events
                          if e.target_id == "meeting-door"}, {"passage_conflict"})

    def test_wrong_side_unknown_noop_and_untyped_target_fail(self):
        model, before = boundary_fixture()
        for action_id, agent_id, kind, target_id, expected in (
            ("wrong", "bob", SituatedActionKind.OPEN_PASSAGE, "meeting-door",
             "not_at_passage_source"),
            ("missing", "alice", SituatedActionKind.OPEN_PASSAGE, "missing",
             "unknown_passage"),
            ("noop", "alice", SituatedActionKind.CLOSE_PASSAGE, "meeting-door",
             "already_closed"),
        ):
            with self.subTest(action_id=action_id):
                result = resolve_situated_round(model.world_model, before, (
                    act(action_id, agent_id, kind, target_id),
                ))
                self.assertEqual(next(x.outcome for x in result.events
                                      if x.action_id == action_id), expected)
                self.assertEqual(result.next_state.passages, before.passages)
        with self.assertRaisesRegex(ValueError, "requires a target"):
            SituatedActionIntent("missing-target", "alice", SituatedActionKind.OPEN_PASSAGE)
        with self.assertRaisesRegex(ValueError, "cannot have a target"):
            SituatedActionIntent("bad-tell", "alice", SituatedActionKind.TELL,
                                 "meeting-door", message="not allowed")
        opened = resolve_situated_round(model.world_model, before, (
            act("open", "alice", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
        ))
        noop = resolve_situated_round(model.world_model, opened.next_state, (
            act("open-again", "alice", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
        ))
        self.assertEqual(next(e.outcome for e in noop.events
                              if e.action_id == "open-again"), "already_open")

    def test_story_exact_parent_replay_and_forged_boundary_state_rejection(self):
        model, state = boundary_fixture()
        schedule = (
            (
                act("open", "alice", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
                act("carol-wants-move", "carol", SituatedActionKind.MOVE, "meeting-door"),
            ),
            (
                act("close", "carol", SituatedActionKind.CLOSE_PASSAGE, "meeting-door"),
                act("alice-move", "alice", SituatedActionKind.MOVE, "meeting-door"),
            ),
        )
        story = initialize_situated_story(
            model.world_model, state, perception_model=model,
        )
        for actions in schedule:
            story = advance_situated_story(model.world_model, story, actions)
        replay = replay_situated_story(
            model.world_model, state, schedule, perception_model=model,
        )
        self.assertEqual(story, replay)
        self.assertEqual(story.content_hash, replay.content_hash)
        self.assertEqual(story.current_state.round_index, 2)
        self.assertFalse(story.current_state.passages[0].open)
        with self.assertRaisesRegex(ValueError, "passage change must match"):
            replace(story.rounds[0], next_state=replace(
                story.rounds[0].next_state, passages=(
                    PassageState("meeting-door", False),
                ),
            ))
        plain = resolve_situated_round(model.world_model, state, ())
        with self.assertRaisesRegex(ValueError, "passage change must match"):
            replace(plain, next_state=replace(
                plain.next_state, passages=(PassageState("meeting-door", True),),
            ))
        opening = story.rounds[0]
        wrong_authored = tuple(
            replace(intent, kind=SituatedActionKind.MOVE)
            if intent.kind is SituatedActionKind.OPEN_PASSAGE else intent
            for intent in opening.intents
        )
        with self.assertRaisesRegex(ValueError, "boundary event must match"):
            replace(opening, intents=wrong_authored)

    def test_same_round_opening_cannot_retroactively_disclose_tell_content(self):
        from narrative_dynamics.abm.situated_perception_contracts import (
            SituatedPerceptFidelity,
        )
        perception, before = boundary_fixture()
        secret = "private-review-message"
        earlier = resolve_situated_round(perception.world_model, before, (
            act("open", "alice", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
            SituatedActionIntent("tell-before-open", "carol",
                                 SituatedActionKind.TELL, message=secret),
        ))
        first_tell = next(e for e in earlier.events
                          if e.action_id == "tell-before-open")
        remote = tuple(
            p for p in percepts_for_agent(
                project_situated_percepts(perception, earlier), "bob",
            ) if p.source_event_id == first_tell.event_id
        )
        self.assertEqual(len(remote), 1)
        self.assertIs(remote[0].fidelity, SituatedPerceptFidelity.DETECTED)
        self.assertIsNone(remote[0].actor_agent_id)
        self.assertIsNone(remote[0].kind)
        self.assertEqual(remote[0].details, ())
        self.assertNotIn(secret, str(remote[0].to_dict()))

        later = resolve_situated_round(perception.world_model, earlier.next_state, (
            SituatedActionIntent("tell-after-open", "carol",
                                 SituatedActionKind.TELL, message=secret),
        ))
        second_tell = next(e for e in later.events
                           if e.action_id == "tell-after-open")
        now = tuple(
            p for p in percepts_for_agent(
                project_situated_percepts(perception, later), "bob",
            ) if p.source_event_id == second_tell.event_id
        )
        self.assertEqual(len(now), 1)
        self.assertIs(now[0].fidelity, SituatedPerceptFidelity.EXACT)
        self.assertIs(now[0].kind, SituatedActionKind.TELL)
        self.assertTrue(any(d.name == "message" and d.value == secret
                            for d in now[0].details))
        # Actual TELL is a different historical event; opening the gate
        # did not produce or pre-deliver this content.

    def test_real_v19_source_reports_actual_boundary_event_and_no_fake_tell(self):
        model, before, after, report_obj = build_real_boundary_case()
        report = report_obj.to_dict()
        self.assertEqual(report["story_lineage"], "STORY_EXTENSION")
        self.assertEqual(report["causal_attribution"], "UNATTRIBUTED")
        self.assertTrue(report["observed_axes"]["physical_boundary"])
        self.assertFalse(report["observed_axes"]["multi_agent_content"])
        self.assertFalse(report["observed_axes"]["social_trust"])
        self.assertFalse(report["observed_axes"]["environment"])
        self.assertTrue(report["boundary"]["physical_passages"])
        self.assertEqual(report["interaction_content"]["new_successful_tell_event_ids"], [])
        self.assertTrue(any(e["kind"] == "open_passage" and e["success"]
                            for e in report["interaction_content"]["new_story_events"]))
        self.assertEqual(report["selected_relation_graphs"]["direct_interaction"]
                         ["source_edges"]["added"], [["alice", "bob"]])
        for relation in ("relationship", "transmission"):
            self.assertEqual(report["selected_relation_graphs"][relation]
                             ["source_edges"], {"added": [], "removed": []})
        self.assertNotEqual(before.story.content_hash, after.story.content_hash)
        self.assertEqual(before.social.relationships, after.social.relationships)
        old_source = before.mesh.source_snapshot
        self.assertEqual(old_source.relationship_edges,
                         after.mesh.source_snapshot.relationship_edges)
        self.assertEqual(old_source.transmissions,
                         after.mesh.source_snapshot.transmissions)
        self.assertEqual(after.story.current_state.passages[0].open, True)
        self.assertTrue(model.percept_memory_model.perception_model.edges)

    def test_versioned_real_passage_vectors_match_source_bytes(self):
        from tools.render_network_boundary_events import (
            build_vector_manifest, render_vector_corpus,
        )
        from pathlib import Path
        corpus = build_vector_manifest()
        self.assertEqual(corpus["schema"], "v19-physical-boundary-events-v1")
        self.assertEqual(len(corpus["cases"]), 2)
        vector = render_vector_corpus().encode("utf-8")
        pinned = Path(__file__).resolve().parents[1] / "conformance" / "network_boundary_events_v1.json"
        self.assertEqual(pinned.read_bytes(), vector)


if __name__ == "__main__":
    unittest.main()
