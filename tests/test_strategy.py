import unittest

from motivation import conflict_direction, potential_change, structural_conflict


class StructuralConflictTests(unittest.TestCase):
    def test_opposed_gradients_have_a_tradeoff_direction(self):
        g1 = 2.0
        g2 = -3.0
        self.assertTrue(structural_conflict(g1, g2))

        direction = conflict_direction(g1, g2)
        self.assertLess(potential_change(g1, direction), 0.0)
        self.assertGreater(potential_change(g2, direction), 0.0)

    def test_aligned_gradients_are_not_structurally_conflicted(self):
        self.assertFalse(structural_conflict(2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
