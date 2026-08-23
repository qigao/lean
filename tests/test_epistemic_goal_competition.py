import unittest

from motivation import bayes_posterior
from epistemic_goal_competition import epistemic_goal_choice_probability


class EpistemicGoalCompetitionTests(unittest.TestCase):
    def test_higher_own_and_lower_rival_score_strictly_raise_choice_probability(self):
        before = epistemic_goal_choice_probability(
            beta=2.0,
            pressure=1.0,
            belief=0.20,
            own_instrumentality_h=0.70,
            own_instrumentality_not_h=0.20,
            own_cost=0.10,
            own_risk=0.10,
            rival_instrumentality_h=0.30,
            rival_instrumentality_not_h=0.70,
            rival_cost=0.05,
            rival_risk=0.05,
        )
        after = epistemic_goal_choice_probability(
            beta=2.0,
            pressure=1.0,
            belief=0.80,
            own_instrumentality_h=0.70,
            own_instrumentality_not_h=0.20,
            own_cost=0.10,
            own_risk=0.10,
            rival_instrumentality_h=0.30,
            rival_instrumentality_not_h=0.70,
            rival_cost=0.05,
            rival_risk=0.05,
        )
        self.assertGreater(after, before)

    def test_positive_evidence_can_reverse_escape_vs_submit_preference(self):
        prior = 0.20
        posterior = bayes_posterior(
            prior,
            likelihood_h=0.90,
            likelihood_not_h=0.10,
        )
        before = epistemic_goal_choice_probability(
            beta=8.0,
            pressure=1.0,
            belief=prior,
            own_instrumentality_h=0.90,
            own_instrumentality_not_h=0.10,
            own_cost=0.15,
            own_risk=0.05,
            rival_instrumentality_h=0.20,
            rival_instrumentality_not_h=0.60,
            rival_cost=0.05,
            rival_risk=0.05,
        )
        after = epistemic_goal_choice_probability(
            beta=8.0,
            pressure=1.0,
            belief=posterior,
            own_instrumentality_h=0.90,
            own_instrumentality_not_h=0.10,
            own_cost=0.15,
            own_risk=0.05,
            rival_instrumentality_h=0.20,
            rival_instrumentality_not_h=0.60,
            rival_cost=0.05,
            rival_risk=0.05,
        )
        self.assertLess(before, 0.5)
        self.assertGreater(after, 0.5)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
