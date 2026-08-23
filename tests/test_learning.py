import unittest

from motivation import learn_instrumentality


class InstrumentalityLearningTests(unittest.TestCase):
    def test_successful_outcome_moves_estimate_up_without_overshoot(self):
        old = 0.2
        observed = 0.8
        updated = learn_instrumentality(old, observed, rate=0.25)

        self.assertGreaterEqual(updated, old)
        self.assertLessEqual(updated, observed)

    def test_disappointing_outcome_moves_estimate_down(self):
        old = 0.8
        observed = 0.2
        updated = learn_instrumentality(old, observed, rate=0.25)

        self.assertLessEqual(updated, old)
        self.assertGreaterEqual(updated, observed)


if __name__ == "__main__":
    unittest.main()
