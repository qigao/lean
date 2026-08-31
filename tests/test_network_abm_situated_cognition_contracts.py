import unittest
from dataclasses import replace

from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedGoalReward,
    SituatedHypothesisTransition,
    SituatedObservationLikelihood,
    initialize_situated_cognition,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_story import initialize_situated_story
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from tests.situated_cognition_fixtures import agent_model, cognitive_office_model


class SituatedCognitionContractTests(unittest.TestCase):
    def test_model_and_initial_private_minds_are_canonical_and_bound(self):
        model = cognitive_office_model()
        story = initialize_situated_story(model.world_model, initialize_situated_world(model.world_model))
        state = initialize_situated_cognition(model, story)
        self.assertEqual(tuple(item.agent_id for item in model.agents), ("alice", "bob", "carol", "dana"))
        self.assertEqual(tuple(item.agent_id for item in state.minds), ("alice", "bob", "carol", "dana"))
        self.assertEqual(next(item for item in state.minds if item.agent_id == "alice").own_place_id, "records")
        self.assertEqual(next(item for item in state.minds if item.agent_id == "bob").own_place_id, "open")
        self.assertEqual(state.story_hash, story.content_hash)
        self.assertIsNone(state.parent_state_hash)
        self.assertTrue(model.content_hash.startswith("sha256:"))
        self.assertTrue(state.content_hash.startswith("sha256:"))
        validate_situated_cognitive_state(model, story, state)

    def test_agent_prior_must_cover_exact_hypotheses(self):
        base = agent_model("alice")
        with self.assertRaisesRegex(ValueError, "prior belief must cover exact hypotheses"):
            replace(base, prior_belief=PlanningBeliefState({"unknown": 1.0}))

    def test_likelihood_matrix_requires_exact_coverage_and_normalized_rows(self):
        base = agent_model("alice")
        with self.assertRaisesRegex(ValueError, "likelihood matrix"):
            replace(base, likelihoods=base.likelihoods[:-1])
        wrong = tuple(
            replace(item, probability=0.8)
            if item.action_id == "inspect" and item.hypothesis_id == "approved" and item.symbol_id == "approved"
            else item
            for item in base.likelihoods
        )
        with self.assertRaisesRegex(ValueError, "likelihood row"):
            replace(base, likelihoods=wrong)

    def test_rewards_require_exact_goal_hypothesis_action_coverage(self):
        base = agent_model("alice")
        with self.assertRaisesRegex(ValueError, "reward matrix"):
            replace(base, rewards=base.rewards[:-1])
        with self.assertRaisesRegex(ValueError, "reward value"):
            SituatedGoalReward("inform", "approved", "wait", float("nan"))

    def test_transition_matrix_is_optional_identity_or_exact_stochastic(self):
        base = agent_model("alice")
        self.assertEqual(base.transitions, ())
        transitions = tuple(
            SituatedHypothesisTransition(action.action_id, prior.hypothesis_id, nxt.hypothesis_id, float(prior == nxt))
            for action in base.actions
            for prior in base.hypotheses
            for nxt in base.hypotheses
        )
        explicit = replace(base, transitions=transitions)
        self.assertEqual(len(explicit.transitions), 16)
        with self.assertRaisesRegex(ValueError, "transition matrix"):
            replace(base, transitions=transitions[:-1])

    def test_cognitive_model_requires_exact_world_agent_roster(self):
        model = cognitive_office_model()
        with self.assertRaisesRegex(ValueError, "exact world agent roster"):
            replace(model, agents=model.agents[:-1])

    def test_state_rejects_story_or_mind_roster_mismatch(self):
        model = cognitive_office_model()
        story = initialize_situated_story(model.world_model, initialize_situated_world(model.world_model))
        state = initialize_situated_cognition(model, story)
        with self.assertRaisesRegex(ValueError, "mind roster"):
            validate_situated_cognitive_state(model, story, replace(state, minds=state.minds[:-1]))
        with self.assertRaisesRegex(ValueError, "exact current story"):
            validate_situated_cognitive_state(model, story, replace(state, story_hash="sha256:" + "a" * 64))


if __name__ == "__main__":
    unittest.main()
