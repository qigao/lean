import unittest

from narrative_dynamics.abm import EdgeSelector
from narrative_dynamics.abm.adaptive_contracts import TruthFeedback
from narrative_dynamics.abm.learning import (
    AdaptiveTrajectory,
    population_view,
    simulate_adaptive_population,
    simulate_adaptive_round,
)
from tests.adaptive_trust_fixtures import (
    adaptive_agent,
    adaptive_case,
    edge_trust,
    truth_feedback,
)


class AdaptiveTrustLearningTests(unittest.TestCase):
    def test_feedback_changes_next_round_influence_not_current_round(self):
        model, initial = adaptive_case()

        first = simulate_adaptive_round(model, initial, feedback=truth_feedback())

        self.assertEqual(
            tuple(item.influence for item in first.transmissions),
            (0.5, 0.5),
        )
        self.assertEqual(adaptive_agent(first.next_state, "target").belief, 0.5)
        self.assertEqual(
            edge_trust(first.next_state, "accurate", "target").trust,
            0.75,
        )
        self.assertEqual(
            edge_trust(first.next_state, "inaccurate", "target").trust,
            0.25,
        )

        second = simulate_adaptive_round(model, first.next_state)

        self.assertEqual(
            tuple(item.influence for item in second.transmissions),
            (0.75, 0.25),
        )
        self.assertEqual(adaptive_agent(second.next_state, "target").belief, 0.75)
        self.assertEqual(second.trust_updates, ())

    def test_no_feedback_preserves_trust_and_feedback_counts(self):
        model, initial = adaptive_case()

        result = simulate_adaptive_round(model, initial)

        self.assertEqual(result.next_state.edge_trust, initial.edge_trust)
        self.assertEqual(result.feedback, ())
        self.assertEqual(result.trust_updates, ())

    def test_feedback_input_order_is_canonical_and_replay_stable(self):
        model, initial = adaptive_case()

        first = simulate_adaptive_round(model, initial, feedback=truth_feedback())
        reversed_input = simulate_adaptive_round(
            model,
            initial,
            feedback=truth_feedback(reverse=True),
        )
        replay = simulate_adaptive_round(model, initial, feedback=truth_feedback())

        self.assertEqual(first, reversed_input)
        self.assertEqual(first.content_hash, reversed_input.content_hash)
        self.assertEqual(first.content_hash, replay.content_hash)

    def test_feedback_requires_unique_edge_that_transmitted_this_round(self):
        model, initial = adaptive_case()
        duplicate = truth_feedback()[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            simulate_adaptive_round(
                model,
                initial,
                feedback=(duplicate, duplicate),
            )
        with self.assertRaisesRegex(ValueError, "did not transmit"):
            simulate_adaptive_round(
                model,
                initial,
                feedback=(
                    TruthFeedback(
                        EdgeSelector("target", "accurate", "report"),
                        1.0,
                    ),
                ),
            )

    def test_zero_learning_rate_preserves_fixed_trust_propagation(self):
        model, initial = adaptive_case(learning_rate=0.0)

        first = simulate_adaptive_round(model, initial, feedback=truth_feedback())
        second = simulate_adaptive_round(model, first.next_state)

        self.assertEqual(tuple(item.trust for item in first.next_state.edge_trust), (0.5, 0.5))
        self.assertEqual(adaptive_agent(second.next_state, "target").belief, 0.5)
        self.assertEqual(
            tuple(item.feedback_count for item in first.next_state.edge_trust),
            (1, 1),
        )

    def test_multi_round_feedback_schedule_forms_exact_state_chain(self):
        model, initial = adaptive_case()

        trajectory = simulate_adaptive_population(
            model,
            initial,
            feedback_schedule=(truth_feedback(), ()),
        )
        replay = simulate_adaptive_population(
            model,
            initial,
            feedback_schedule=(truth_feedback(), ()),
        )

        self.assertIsInstance(trajectory, AdaptiveTrajectory)
        self.assertEqual(trajectory, replay)
        self.assertEqual(trajectory.content_hash, replay.content_hash)
        self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
        self.assertEqual(trajectory.final_state.round_index, 2)
        self.assertEqual(adaptive_agent(trajectory.final_state, "target").belief, 0.75)

        view = population_view(model, trajectory.final_state)
        self.assertEqual(view.model_hash, model.base_model.content_hash)
        self.assertEqual(view.round_index, 2)
        self.assertEqual(view.agents, trajectory.final_state.agents)

    def test_feedback_schedule_must_be_nonempty_tuple_of_tuples(self):
        model, initial = adaptive_case()
        for schedule in ((), [], (truth_feedback()[0],)):
            with self.subTest(schedule=schedule):
                with self.assertRaises((TypeError, ValueError)):
                    simulate_adaptive_population(
                        model,
                        initial,
                        feedback_schedule=schedule,
                    )


if __name__ == "__main__":
    unittest.main()
