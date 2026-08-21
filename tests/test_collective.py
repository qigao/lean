import math
import unittest

from motivation import effective_pressure, softmax_probability


class CollectiveMotivationTests(unittest.TestCase):
    def test_self_boundary_expansion_increases_effective_pressure(self):
        pressures = [0.8, 0.6, 0.2]
        narrow = [1.0, 0.1, 0.0]
        expanded = [1.0, 0.7, 0.4]

        self.assertLessEqual(
            effective_pressure(narrow, pressures),
            effective_pressure(expanded, pressures),
        )

    def test_softmax_probability_increases_with_own_score(self):
        beta = 1.4
        rival = 0.75

        low = softmax_probability(beta, own_score=0.2, rival_score=rival)
        high = softmax_probability(beta, own_score=1.1, rival_score=rival)

        self.assertLess(low, high)
        self.assertTrue(math.isclose(low + softmax_probability(beta, rival, 0.2), 1.0))


if __name__ == "__main__":
    unittest.main()
