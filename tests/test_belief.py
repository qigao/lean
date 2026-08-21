import unittest

from motivation import bayes_posterior


class BayesianInterpretationTests(unittest.TestCase):
    def test_positive_evidence_raises_posterior(self):
        prior = 0.72
        posterior = bayes_posterior(prior, likelihood_h=0.90, likelihood_not_h=0.15)
        self.assertGreater(posterior, prior)

    def test_uninformative_evidence_preserves_prior(self):
        prior = 0.37
        posterior = bayes_posterior(prior, likelihood_h=0.40, likelihood_not_h=0.40)
        self.assertAlmostEqual(posterior, prior)


if __name__ == "__main__":
    unittest.main()
