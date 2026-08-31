from dataclasses import replace
import unittest

from narrative_dynamics.abm import (
    NetworkABMModel,
    NetworkAgentSpec,
    SocialEdge,
    SocialNetwork,
    initialize_population,
)
from narrative_dynamics.abm.simulation import (
    PopulationTrajectory,
    simulate_population,
    simulate_round,
)
from tests.network_abm_fixtures import agent, line_case


class NetworkABMSimulationTests(unittest.TestCase):
    def test_information_moves_at_most_one_hop_per_round(self):
        model, initial = line_case()

        first = simulate_round(model, initial)
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in first.transmissions),
            (("a", "b"),),
        )
        self.assertEqual(agent(first.next_state, "b").belief, 1.0)
        self.assertTrue(agent(first.next_state, "b").broadcasting)
        self.assertEqual(agent(first.next_state, "c").belief, 0.0)
        self.assertEqual(agent(first.next_state, "d"), agent(initial, "d"))

        second = simulate_round(model, first.next_state)
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in second.transmissions),
            (("a", "b"), ("b", "c")),
        )
        self.assertEqual(agent(second.next_state, "c").belief, 1.0)
        self.assertTrue(agent(second.next_state, "c").broadcasting)
        self.assertEqual(agent(second.next_state, "b").exposure_count, 2)
        self.assertEqual(agent(second.next_state, "d"), agent(initial, "d"))

    def test_weight_and_receptivity_bound_assimilation(self):
        agents = (
            NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
            NetworkAgentSpec("b", "recipient", 0.5, 0.5, 0.5),
        )
        model = NetworkABMModel(
            "weighted",
            "1",
            agents,
            SocialNetwork(("a", "b"), (SocialEdge("a", "b", "peer", 0.5),)),
        )

        result = simulate_round(model, initialize_population(model, beliefs={"a": 1.0}))

        self.assertEqual(agent(result.next_state, "b").belief, 0.25)
        self.assertFalse(agent(result.next_state, "b").broadcasting)
        self.assertEqual(agent(result.next_state, "b").exposure_count, 1)

    def test_inactive_and_zero_influence_edges_do_not_transmit(self):
        agents = (
            NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
            NetworkAgentSpec("b", "recipient", 1.0, 0.5, 0.5),
            NetworkAgentSpec("c", "recipient", 1.0, 0.5, 0.5),
        )
        model = NetworkABMModel(
            "blocked",
            "1",
            agents,
            SocialNetwork(
                ("a", "b", "c"),
                (
                    SocialEdge("a", "b", "inactive", 1.0, active=False),
                    SocialEdge("a", "c", "zero", 0.0),
                ),
            ),
        )
        initial = initialize_population(model, beliefs={"a": 1.0})

        result = simulate_round(model, initial)

        self.assertEqual(result.transmissions, ())
        self.assertEqual(result.next_state.agents, initial.agents)

    def test_trajectory_is_exact_replayable_and_input_order_invariant(self):
        model, initial = line_case()
        reversed_model, reversed_initial = line_case(reverse=True)

        first = simulate_population(model, initial, rounds=2)
        replay = simulate_population(model, initial, rounds=2)
        reordered = simulate_population(reversed_model, reversed_initial, rounds=2)

        self.assertIsInstance(first, PopulationTrajectory)
        self.assertEqual(first, replay)
        self.assertEqual(first.content_hash, replay.content_hash)
        self.assertEqual(first.content_hash, reordered.content_hash)
        self.assertEqual(first.rounds[0].next_state, first.rounds[1].prior_state)
        self.assertEqual(first.final_state.round_index, 2)

    def test_model_state_and_round_validation_fail_before_transition(self):
        model, initial = line_case()
        other = replace(model, model_id="other")
        with self.assertRaisesRegex(ValueError, "model identity"):
            simulate_round(other, initial)
        with self.assertRaisesRegex(ValueError, "broadcasting"):
            simulate_round(
                model,
                replace(
                    initial,
                    agents=(replace(initial.agents[0], broadcasting=False),) + initial.agents[1:],
                ),
            )
        for rounds in (0, -1, True, 1.5):
            with self.subTest(rounds=rounds):
                with self.assertRaises((TypeError, ValueError)):
                    simulate_population(model, initial, rounds=rounds)


if __name__ == "__main__":
    unittest.main()
