import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
    resolve_situated_round,
)
from tests.situated_fixtures import office_state


def action(action_id, agent_id, kind, target_id=None, *, message=None, source_event_ids=()):
    return SituatedActionIntent(
        action_id=action_id,
        agent_id=agent_id,
        kind=kind,
        target_id=target_id,
        message=message,
        source_event_ids=source_event_ids,
    )


def event_for(result, agent_id):
    return next(item for item in result.events if item.actor_agent_id == agent_id)


def observed_agents(result, event_id):
    return {item.agent_id for item in result.observations if item.event_id == event_id}


class SituatedRoundTests(unittest.TestCase):
    def test_move_uses_open_directed_passage_and_is_seen_at_both_ends(self):
        model, state = office_state()
        result = resolve_situated_round(model, state, (
            action("alice-move", "alice", SituatedActionKind.MOVE, "records-open"),
        ))
        alice = next(item for item in result.next_state.agents if item.agent_id == "alice")
        event = event_for(result, "alice")
        self.assertEqual(alice.place_id, "open")
        self.assertTrue(event.success)
        self.assertEqual(event.outcome, "moved")
        self.assertEqual(observed_agents(result, event.event_id), {"alice", "bob", "carol"})

    def test_closed_or_wrong_source_passage_fails_without_movement(self):
        model, state = office_state()
        closed = resolve_situated_round(model, state, (
            action("bob-move", "bob", SituatedActionKind.MOVE, "open-manager"),
        ))
        wrong = resolve_situated_round(model, state, (
            action("alice-move", "alice", SituatedActionKind.MOVE, "lobby-open"),
        ))
        self.assertEqual(event_for(closed, "bob").outcome, "passage_closed")
        self.assertEqual(event_for(wrong, "alice").outcome, "not_at_passage_source")
        self.assertEqual(next(item for item in closed.next_state.agents if item.agent_id == "bob").place_id, "open")

    def test_look_and_inspection_are_private_and_reveal_grounded_details(self):
        model, state = office_state()
        looked = resolve_situated_round(model, state, (
            action("bob-look", "bob", SituatedActionKind.LOOK),
        ))
        look_event = event_for(looked, "bob")
        self.assertEqual(observed_agents(looked, look_event.event_id), {"bob"})
        self.assertIn(("agent:carol", "designer"), tuple((item.name, item.value) for item in look_event.details))

        inspected = resolve_situated_round(model, state, (
            action("alice-inspect", "alice", SituatedActionKind.INSPECT, "memo"),
        ))
        inspect_event = event_for(inspected, "alice")
        self.assertEqual(observed_agents(inspected, inspect_event.event_id), {"alice"})
        self.assertIn(("restructuring", "approved"), tuple((item.name, item.value) for item in inspect_event.details))
        observation = next(item for item in inspected.observations if item.event_id == inspect_event.event_id)
        self.assertEqual(observation.channel, ObservationChannel.INSPECTION)

    def test_take_and_drop_transfer_object_atomically(self):
        model, state = office_state()
        taken = resolve_situated_round(model, state, (
            action("alice-take", "alice", SituatedActionKind.TAKE, "memo"),
        ))
        memo = next(item for item in taken.next_state.objects if item.object_id == "memo")
        self.assertEqual((memo.place_id, memo.holder_agent_id), (None, "alice"))
        dropped = resolve_situated_round(model, taken.next_state, (
            action("alice-drop", "alice", SituatedActionKind.DROP, "memo"),
        ))
        memo = next(item for item in dropped.next_state.objects if item.object_id == "memo")
        self.assertEqual((memo.place_id, memo.holder_agent_id), ("records", None))
        self.assertEqual(dropped.next_state.parent_state_hash, taken.next_state.content_hash)

    def test_take_enforces_portability_location_and_capacity(self):
        model, state = office_state(memo_portable=False)
        result = resolve_situated_round(model, state, (
            action("alice-take", "alice", SituatedActionKind.TAKE, "memo"),
            action("bob-take", "bob", SituatedActionKind.TAKE, "badge"),
        ))
        self.assertEqual(event_for(result, "alice").outcome, "object_not_portable")
        self.assertEqual(event_for(result, "bob").outcome, "object_not_co_located")

        zero_model, zero_state = office_state(alice_capacity=0)
        full = resolve_situated_round(zero_model, zero_state, (
            action("alice-take", "alice", SituatedActionKind.TAKE, "memo"),
        ))
        self.assertEqual(event_for(full, "alice").outcome, "inventory_full")

    def test_tell_is_local_auditory_and_carries_source_references(self):
        model, state = office_state()
        result = resolve_situated_round(model, state, (
            action(
                "bob-tell", "bob", SituatedActionKind.TELL,
                message="The restructuring is approved.",
                source_event_ids=("r0000:e0001",),
            ),
        ))
        event = event_for(result, "bob")
        self.assertEqual(event.cause_event_ids, ("r0000:e0001",))
        self.assertEqual(observed_agents(result, event.event_id), {"bob", "carol"})
        channels = {item.agent_id: item.channel for item in result.observations if item.event_id == event.event_id}
        self.assertEqual(channels["bob"], ObservationChannel.SELF)
        self.assertEqual(channels["carol"], ObservationChannel.AUDITORY)

    def test_missing_intents_become_deterministic_private_waits(self):
        model, state = office_state()
        result = resolve_situated_round(model, state, ())
        self.assertEqual(len(result.intents), 4)
        self.assertTrue(all(item.kind is SituatedActionKind.WAIT for item in result.intents))
        self.assertEqual(len(result.events), 4)
        self.assertEqual(len(result.observations), 4)
        for event in result.events:
            self.assertEqual(observed_agents(result, event.event_id), {event.actor_agent_id})

    def test_take_conflict_has_one_canonical_winner_and_causal_loser(self):
        model, state = office_state()
        moved = resolve_situated_round(model, state, (
            action("bob-move", "bob", SituatedActionKind.MOVE, "open-records"),
        ))
        intents = (
            action("z-bob", "bob", SituatedActionKind.TAKE, "memo"),
            action("a-alice", "alice", SituatedActionKind.TAKE, "memo"),
        )
        left = resolve_situated_round(model, moved.next_state, intents)
        right = resolve_situated_round(model, moved.next_state, tuple(reversed(intents)))
        self.assertEqual(left, right)
        memo = next(item for item in left.next_state.objects if item.object_id == "memo")
        self.assertEqual(memo.holder_agent_id, "alice")
        bob_event = event_for(left, "bob")
        alice_event = event_for(left, "alice")
        self.assertEqual(bob_event.outcome, "take_conflict_lost")
        self.assertEqual(bob_event.cause_event_ids, (alice_event.event_id,))

    def test_rejects_duplicate_agent_actions_and_action_ids(self):
        model, state = office_state()
        with self.assertRaisesRegex(ValueError, "one action per agent"):
            resolve_situated_round(model, state, (
                action("a", "alice", SituatedActionKind.WAIT),
                action("b", "alice", SituatedActionKind.WAIT),
            ))
        with self.assertRaisesRegex(ValueError, "action ids must be unique"):
            resolve_situated_round(model, state, (
                action("same", "alice", SituatedActionKind.WAIT),
                action("same", "bob", SituatedActionKind.WAIT),
            ))


if __name__ == "__main__":
    unittest.main()
