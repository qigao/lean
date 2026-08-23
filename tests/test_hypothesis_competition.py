import unittest

from hypothesis_competition import posterior_distribution


class HypothesisCompetitionTests(unittest.TestCase):
    def test_posteriors_normalize_and_reward_better_explanation(self):
        priors = {
            "guilty": 0.40,
            "fear": 0.35,
            "illness": 0.25,
        }
        likelihoods = {
            "guilty": 0.90,
            "fear": 0.30,
            "illness": 0.10,
        }

        posterior = posterior_distribution(priors, likelihoods)

        self.assertAlmostEqual(sum(posterior.values()), 1.0)
        self.assertGreater(posterior["guilty"], posterior["fear"])
        self.assertGreater(posterior["fear"], posterior["illness"])

    def test_equal_priors_rank_candidates_by_likelihood(self):
        posterior = posterior_distribution(
            {"h1": 0.5, "h2": 0.5},
            {"h1": 0.8, "h2": 0.2},
        )
        self.assertGreater(posterior["h1"], posterior["h2"])

    def test_zero_evidence_mass_is_rejected(self):
        with self.assertRaises(ValueError):
            posterior_distribution(
                {"h1": 0.5, "h2": 0.5},
                {"h1": 0.0, "h2": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
