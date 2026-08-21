import unittest

from motivation import bayes_posterior
from epistemic_goal import epistemic_goal_score


class EpistemicGoalTests(unittest.TestCase):
    def test_positive_evidence_raises_goal_score_when_hypothesis_supports_goal(self):
        prior = 0.30
        posterior = bayes_posterior(
            prior,
            likelihood_h=0.90,
            likelihood_not_h=0.20,
        )
        before = epistemic_goal_score(
            pressure=0.80,
            belief=prior,
            instrumentality_h=0.90,
            instrumentality_not_h=0.10,
            cost=0.15,
            risk=0.10,
        )
        after = epistemic_goal_score(
            pressure=0.80,
            belief=posterior,
            instrumentality_h=0.90,
            instrumentality_not_h=0.10,
            cost=0.15,
            risk=0.10,
        )
        self.assertGreater(after, before)

    def test_hypothesis_independent_instrumentality_ignores_belief_change(self):
        before = epistemic_goal_score(
            pressure=0.80,
            belief=0.20,
            instrumentality_h=0.55,
            instrumentality_not_h=0.55,
            cost=0.15,
            risk=0.10,
        )
        after = epistemic_goal_score(
            pressure=0.80,
            belief=0.85,
            instrumentality_h=0.55,
            instrumentality_not_h=0.55,
            cost=0.15,
            risk=0.10,
        )
        self.assertAlmostEqual(after, before)

    def test_positive_evidence_lowers_goal_score_when_hypothesis_opposes_goal(self):
        prior = 0.30
        posterior = bayes_posterior(
            prior,
            likelihood_h=0.90,
            likelihood_not_h=0.20,
        )
        before = epistemic_goal_score(
            pressure=0.80,
            belief=prior,
            instrumentality_h=0.10,
            instrumentality_not_h=0.90,
            cost=0.15,
            risk=0.10,
        )
        after = epistemic_goal_score(
            pressure=0.80,
            belief=posterior,
            instrumentality_h=0.10,
            instrumentality_not_h=0.90,
            cost=0.15,
            risk=0.10,
        )
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
