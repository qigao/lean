from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
    SituatedObservation,
)
from narrative_dynamics.abm.situated_story import (
    SituatedStory,
    advance_situated_story,
    causal_ancestors,
    direct_causes,
    explain_situated_event,
    information_chain,
    initialize_situated_story,
    objective_timeline,
    perspective_timeline,
    replay_situated_story,
)
from tests.situated_fixtures import office_state


def act(action_id, agent_id, kind, target_id=None, *, message=None, source_event_ids=()):
    return SituatedActionIntent(action_id, agent_id, kind, target_id, message, source_event_ids)


def office_story():
    model, state = office_state()
    story = initialize_situated_story(model, state)
    story = advance_situated_story(model, story, (
        act("01-inspect", "alice", SituatedActionKind.INSPECT, "memo"),
    ))
    inspect_id = next(event.event_id for event in story.rounds[-1].events if event.actor_agent_id == "alice")
    story = advance_situated_story(model, story, (
        act("02-move", "alice", SituatedActionKind.MOVE, "records-open"),
    ))
    story = advance_situated_story(model, story, (
        act("03-tell", "alice", SituatedActionKind.TELL, message="Restructuring is approved.", source_event_ids=(inspect_id,)),
    ))
    alice_tell_id = next(event.event_id for event in story.rounds[-1].events if event.actor_agent_id == "alice")
    story = advance_situated_story(model, story, (
        act("04-retell", "bob", SituatedActionKind.TELL, message="Alice found an approved restructuring memo.", source_event_ids=(alice_tell_id,)),
    ))
    bob_tell_id = next(event.event_id for event in story.rounds[-1].events if event.actor_agent_id == "bob")
    return model, state, story, inspect_id, alice_tell_id, bob_tell_id


class SituatedStoryTests(unittest.TestCase):
    def test_story_rejects_forged_private_inspection_observer(self):
        model, state = office_state()
        story = advance_situated_story(
            model,
            initialize_situated_story(model, state),
            (act("inspect", "alice", SituatedActionKind.INSPECT, "memo"),),
        )
        result = story.rounds[0]
        inspection = next(
            item for item in result.events if item.actor_agent_id == "alice"
        )
        forged = SituatedObservation(
            "forged-bob-inspection",
            inspection.round_index,
            "bob",
            inspection.event_id,
            inspection.content_hash,
            ObservationChannel.INSPECTION,
        )

        with self.assertRaisesRegex(ValueError, "observation projection"):
            forged_round = replace(
                result,
                observations=result.observations + (forged,),
            )
            SituatedStory(
                story.model_id,
                story.model_hash,
                story.initial_state,
                (forged_round,),
            )

    def test_round_rejects_duplicate_observation_id_and_agent_event_pair(self):
        model, state = office_state()
        story = advance_situated_story(
            model,
            initialize_situated_story(model, state),
            (),
        )
        result = story.rounds[0]
        first, second = result.observations[:2]

        with self.assertRaisesRegex(ValueError, "observation ids must be unique"):
            replace(
                result,
                observations=(
                    first,
                    replace(second, observation_id=first.observation_id),
                ) + result.observations[2:],
            )
        with self.assertRaisesRegex(ValueError, "agent/event pairs must be unique"):
            replace(
                result,
                observations=result.observations + (
                    replace(first, observation_id="duplicate-agent-event"),
                ),
            )

    def test_office_story_preserves_objective_and_private_timelines(self):
        _, _, story, inspect_id, alice_tell_id, bob_tell_id = office_story()
        objective_ids = {item.event_id for item in objective_timeline(story)}
        self.assertTrue({inspect_id, alice_tell_id, bob_tell_id}.issubset(objective_ids))

        alice_ids = {item.event.event_id for item in perspective_timeline(story, "alice")}
        bob_ids = {item.event.event_id for item in perspective_timeline(story, "bob")}
        carol_ids = {item.event.event_id for item in perspective_timeline(story, "carol")}
        dana_ids = {item.event.event_id for item in perspective_timeline(story, "dana")}
        self.assertIn(inspect_id, alice_ids)
        self.assertNotIn(inspect_id, bob_ids | carol_ids | dana_ids)
        self.assertIn(alice_tell_id, bob_ids)
        self.assertIn(alice_tell_id, carol_ids)
        self.assertNotIn(alice_tell_id, dana_ids)
        self.assertIn(bob_tell_id, carol_ids)
        self.assertNotIn(bob_tell_id, dana_ids)

    def test_information_chain_reconstructs_grounded_propagation(self):
        _, _, story, inspect_id, alice_tell_id, bob_tell_id = office_story()
        self.assertEqual(
            tuple(item.event_id for item in information_chain(story, bob_tell_id)),
            (inspect_id, alice_tell_id, bob_tell_id),
        )
        self.assertEqual(tuple(item.event_id for item in direct_causes(story, bob_tell_id)), (alice_tell_id,))
        self.assertEqual(tuple(item.event_id for item in causal_ancestors(story, bob_tell_id)), (inspect_id, alice_tell_id))

    def test_story_rejects_sources_the_speaker_did_not_observe(self):
        model, state = office_state()
        story = initialize_situated_story(model, state)
        story = advance_situated_story(model, story, (
            act("inspect", "alice", SituatedActionKind.INSPECT, "memo"),
        ))
        inspect_id = next(event.event_id for event in story.rounds[-1].events if event.actor_agent_id == "alice")
        with self.assertRaisesRegex(ValueError, "speaker previously observed"):
            advance_situated_story(model, story, (
                act("fabricated-source", "bob", SituatedActionKind.TELL, message="I saw the memo.", source_event_ids=(inspect_id,)),
            ))

    def test_story_rejects_unknown_causal_source(self):
        model, state = office_state()
        story = initialize_situated_story(model, state)
        with self.assertRaisesRegex(ValueError, "causal reference must identify"):
            advance_situated_story(model, story, (
                act("tell", "alice", SituatedActionKind.TELL, message="Unsupported.", source_event_ids=("missing",)),
            ))

    def test_grounded_explanation_reports_rule_and_direct_causes(self):
        _, _, story, _, alice_tell_id, bob_tell_id = office_story()
        explained = explain_situated_event(story, bob_tell_id)
        self.assertTrue(explained.success)
        self.assertEqual(explained.outcome, "told")
        self.assertEqual(explained.rule, "a local audible claim was emitted")
        self.assertEqual(explained.direct_cause_event_ids, (alice_tell_id,))

    def test_conflict_explanation_identifies_the_winning_action(self):
        model, state = office_state()
        story = initialize_situated_story(model, state)
        story = advance_situated_story(model, story, (
            act("move-bob", "bob", SituatedActionKind.MOVE, "open-records"),
        ))
        story = advance_situated_story(model, story, (
            act("z-bob", "bob", SituatedActionKind.TAKE, "memo"),
            act("a-alice", "alice", SituatedActionKind.TAKE, "memo"),
        ))
        bob_event = next(event for event in story.rounds[-1].events if event.actor_agent_id == "bob")
        alice_event = next(event for event in story.rounds[-1].events if event.actor_agent_id == "alice")
        explained = explain_situated_event(story, bob_event.event_id)
        self.assertFalse(explained.success)
        self.assertEqual(explained.rule, "another eligible action won the canonical object conflict")
        self.assertEqual(explained.direct_cause_event_ids, (alice_event.event_id,))

    def test_replay_reproduces_exact_story_and_hash(self):
        model, initial, story, inspect_id, alice_tell_id, _ = office_story()
        schedule = (
            (act("01-inspect", "alice", SituatedActionKind.INSPECT, "memo"),),
            (act("02-move", "alice", SituatedActionKind.MOVE, "records-open"),),
            (act("03-tell", "alice", SituatedActionKind.TELL, message="Restructuring is approved.", source_event_ids=(inspect_id,)),),
            (act("04-retell", "bob", SituatedActionKind.TELL, message="Alice found an approved restructuring memo.", source_event_ids=(alice_tell_id,)),),
        )
        replayed = replay_situated_story(model, initial, schedule)
        self.assertEqual(replayed, story)
        self.assertEqual(replayed.content_hash, story.content_hash)


if __name__ == "__main__":
    unittest.main()
