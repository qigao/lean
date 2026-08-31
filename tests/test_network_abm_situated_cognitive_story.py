import unittest
from dataclasses import replace

from narrative_dynamics.abm.situated_cognition import (
    explain_situated_decision,
    simulate_situated_cognition,
)
from narrative_dynamics.abm.situated_cognition_contracts import initialize_situated_cognition
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_story import information_chain, initialize_situated_story
from tests.situated_cognition_fixtures import cognitive_office_model


def autonomous_office_case():
    model = cognitive_office_model()
    carol = next(item for item in model.agents if item.agent_id == "carol")
    quiet_rewards = tuple(
        replace(item, value=(-10.0 if item.action_id == "tell" else item.value))
        for item in carol.rewards
    )
    model = replace(model, agents=tuple(
        replace(item, rewards=quiet_rewards) if item.agent_id == "carol" else item
        for item in model.agents
    ))
    story = initialize_situated_story(model.world_model, initialize_situated_world(model.world_model))
    state = initialize_situated_cognition(model, story)
    return model, story, state


class SituatedCognitiveStoryTests(unittest.TestCase):
    def test_four_round_office_story_is_generated_by_independent_decisions(self):
        model, story, state = autonomous_office_case()
        trajectory = simulate_situated_cognition(model, story, state, round_count=4)
        selected = {
            (result.next_state.round_index, decision.agent_id): decision.selected_action_id
            for result in trajectory.rounds
            for decision in result.decisions
        }
        self.assertEqual(selected[(1, "alice")], "inspect")
        self.assertEqual(selected[(2, "alice")], "move")
        self.assertEqual(selected[(3, "alice")], "tell")
        self.assertEqual(selected[(4, "bob")], "tell")
        self.assertEqual(selected[(4, "carol")], "wait")

        final_minds = {item.agent_id: item for item in trajectory.final_state.minds}
        self.assertAlmostEqual(final_minds["alice"].belief.probabilities["approved"], 0.9)
        self.assertAlmostEqual(final_minds["bob"].belief.probabilities["approved"], 0.9)
        self.assertAlmostEqual(final_minds["dana"].belief.probabilities["approved"], 0.5)

    def test_autonomous_story_retains_inspection_to_two_tellings_chain(self):
        model, story, state = autonomous_office_case()
        trajectory = simulate_situated_cognition(model, story, state, round_count=4)
        bob_tell = next(
            event for event in trajectory.final_story.rounds[-1].events
            if event.actor_agent_id == "bob"
        )
        chain = information_chain(trajectory.final_story, bob_tell.event_id)
        self.assertEqual(tuple(item.kind.value for item in chain), ("inspect", "tell", "tell"))
        self.assertEqual(tuple(item.actor_agent_id for item in chain), ("alice", "alice", "bob"))

    def test_decision_explanation_reports_belief_policy_goals_and_evidence(self):
        model, story, state = autonomous_office_case()
        trajectory = simulate_situated_cognition(model, story, state, round_count=4)
        explanation = explain_situated_decision(trajectory.rounds[-1], "bob")
        self.assertEqual(explanation.selected_action_id, "tell")
        self.assertEqual(explanation.most_likely_hypothesis_id, "approved")
        self.assertAlmostEqual(explanation.hypothesis_probability, 0.9)
        self.assertEqual(set(explanation.goal_contributions), {"inform"})
        self.assertEqual(len(explanation.admitted_evidence_ids), 1)
        self.assertGreater(explanation.action_probability, 0.99)

    def test_autonomous_trajectory_replays_to_exact_story_state_and_hash(self):
        model, story, state = autonomous_office_case()
        left = simulate_situated_cognition(model, story, state, round_count=4)
        right = simulate_situated_cognition(model, story, state, round_count=4)
        self.assertEqual(left, right)
        self.assertEqual(left.content_hash, right.content_hash)
        self.assertEqual(left.final_story.content_hash, right.final_story.content_hash)
        self.assertEqual(left.final_state.content_hash, right.final_state.content_hash)

    def test_trajectory_requires_positive_round_count(self):
        model, story, state = autonomous_office_case()
        with self.assertRaisesRegex(ValueError, "positive round count"):
            simulate_situated_cognition(model, story, state, round_count=0)

    def test_trajectory_rejects_forged_model_identity(self):
        model, story, state = autonomous_office_case()
        trajectory = simulate_situated_cognition(model, story, state, round_count=1)
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            replace(trajectory, model_id="forged")


if __name__ == "__main__":
    unittest.main()
