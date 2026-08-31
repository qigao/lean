import unittest
from dataclasses import replace

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_cognition import (
    admit_situated_observations,
    decide_situated_action,
    simulate_situated_cognitive_round,
)
from narrative_dynamics.abm.situated_cognition_contracts import initialize_situated_cognition
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_story import (
    advance_situated_story,
    initialize_situated_story,
    perspective_timeline,
)
from tests.situated_cognition_fixtures import agent_model, cognitive_office_model


def initial_case():
    model = cognitive_office_model()
    story = initialize_situated_story(model.world_model, initialize_situated_world(model.world_model))
    state = initialize_situated_cognition(model, story)
    return model, story, state


def mind(state, agent_id):
    return next(item for item in state.minds if item.agent_id == agent_id)


class SituatedCognitionRuntimeTests(unittest.TestCase):
    def test_private_inspection_updates_only_observer_and_is_idempotent(self):
        model, story, state = initial_case()
        story = advance_situated_story(model.world_model, story, (
            SituatedActionIntent("inspect", "alice", SituatedActionKind.INSPECT, "memo"),
        ))
        alice_model = next(item for item in model.agents if item.agent_id == "alice")
        alice = admit_situated_observations(alice_model, mind(state, "alice"), perspective_timeline(story, "alice"))
        bob = admit_situated_observations(next(item for item in model.agents if item.agent_id == "bob"), mind(state, "bob"), perspective_timeline(story, "bob"))
        self.assertAlmostEqual(alice.next_mind.belief.probabilities["approved"], 0.9)
        self.assertAlmostEqual(alice.next_mind.belief.probabilities["denied"], 0.1)
        self.assertEqual(tuple(item.symbol_id for item in alice.admissions), ("approved",))
        self.assertEqual(bob.next_mind.belief, mind(state, "bob").belief)
        repeated = admit_situated_observations(alice_model, alice.next_mind, perspective_timeline(story, "alice"))
        self.assertEqual(repeated.next_mind, alice.next_mind)
        self.assertEqual(repeated.admissions, ())

    def test_self_telling_is_remembered_but_not_reused_as_external_evidence(self):
        model, story, state = initial_case()
        story = advance_situated_story(model.world_model, story, (
            SituatedActionIntent(
                "self-tell", "alice", SituatedActionKind.TELL,
                message="The restructuring is approved.",
            ),
        ))
        alice_model = next(item for item in model.agents if item.agent_id == "alice")
        result = admit_situated_observations(
            alice_model,
            mind(state, "alice"),
            perspective_timeline(story, "alice"),
        )
        tell_event_id = next(
            item.event_id for item in story.rounds[-1].events
            if item.actor_agent_id == "alice"
        )
        self.assertEqual(result.admissions, ())
        self.assertEqual(result.next_mind.belief, mind(state, "alice").belief)
        self.assertIn(tell_event_id, result.next_mind.observed_event_ids)

    def test_planner_values_information_and_uses_lexical_map_ties(self):
        model, story, state = initial_case()
        alice_model = next(item for item in model.agents if item.agent_id == "alice")
        decision = decide_situated_action(alice_model, mind(state, "alice"), (), round_index=1)
        self.assertEqual(decision.selected_action_id, "inspect")
        self.assertGreater(decision.action_values["inspect"], decision.action_values["move"])
        tied_rewards = tuple(replace(item, value=0.0) for item in alice_model.rewards)
        tied = decide_situated_action(replace(alice_model, rewards=tied_rewards, beta=1.0), mind(state, "alice"), (), round_index=1)
        self.assertEqual(tied.selected_action_id, "inspect")
        self.assertAlmostEqual(tied.action_policy["inspect"], 1 / 3)

    def test_root_feasibility_uses_private_place_history_and_source_only(self):
        model, story, state = initial_case()
        alice_model = next(item for item in model.agents if item.agent_id == "alice")
        alice = mind(state, "alice")
        first = decide_situated_action(alice_model, alice, (), round_index=1)
        self.assertEqual(set(first.feasible_action_ids), {"inspect", "move", "wait"})
        after_inspect = replace(alice, selected_action_ids=("inspect",), decision_count=1)
        second = decide_situated_action(alice_model, after_inspect, (), round_index=2)
        self.assertEqual(set(second.feasible_action_ids), {"move", "wait"})
        at_open = replace(after_inspect, own_place_id="open", observed_event_ids=("r0001:e0001",))
        private_event = perspective_timeline(
            advance_situated_story(model.world_model, story, (
                SituatedActionIntent("inspect", "alice", SituatedActionKind.INSPECT, "memo"),
            )),
            "alice",
        )
        third = decide_situated_action(alice_model, at_open, private_event, round_index=3)
        self.assertEqual(set(third.feasible_action_ids), {"tell", "wait"})
        self.assertEqual(third.intent.source_event_ids, ("r0001:e0001",))

    def test_goal_contributions_sum_to_selected_immediate_value(self):
        model, _, state = initial_case()
        alice_model = next(item for item in model.agents if item.agent_id == "alice")
        decision = decide_situated_action(alice_model, mind(state, "alice"), (), round_index=1)
        self.assertEqual(set(decision.selected_goal_contributions), {"inform"})
        self.assertAlmostEqual(sum(decision.selected_goal_contributions.values()), 3.0)

    def test_cognitive_round_delays_new_observations_until_next_decision(self):
        model, story, state = initial_case()
        first = simulate_situated_cognitive_round(model, story, state)
        alice_first = next(item for item in first.decisions if item.agent_id == "alice")
        self.assertEqual(alice_first.selected_action_id, "inspect")
        self.assertEqual(alice_first.admitted_observation_ids, ())
        self.assertAlmostEqual(mind(first.next_state, "alice").belief.probabilities["approved"], 0.5)
        second = simulate_situated_cognitive_round(model, first.next_story, first.next_state)
        alice_second = next(item for item in second.decisions if item.agent_id == "alice")
        self.assertEqual(alice_second.admitted_symbol_ids, ("approved",))
        self.assertAlmostEqual(alice_second.posterior_belief.probabilities["approved"], 0.9)
        self.assertEqual(alice_second.selected_action_id, "move")

    def test_reordered_agent_models_replay_exactly(self):
        model, story, state = initial_case()
        left = simulate_situated_cognitive_round(model, story, state)
        reordered = replace(model, agents=tuple(reversed(model.agents)))
        self.assertEqual(reordered, model)
        right = simulate_situated_cognitive_round(reordered, story, state)
        self.assertEqual(left, right)

    def test_decision_and_round_reject_duplicate_or_incomplete_agent_coverage(self):
        model, story, state = initial_case()
        result = simulate_situated_cognitive_round(model, story, state)
        decision = result.decisions[0]
        with self.assertRaisesRegex(ValueError, "feasible action ids must be unique"):
            replace(
                decision,
                feasible_action_ids=decision.feasible_action_ids + (decision.feasible_action_ids[0],),
            )
        with self.assertRaisesRegex(ValueError, "exact prior mind roster"):
            replace(result, decisions=result.decisions[:-1])


if __name__ == "__main__":
    unittest.main()
