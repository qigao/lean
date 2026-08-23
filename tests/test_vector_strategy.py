import unittest

from strategy import vector_potential_change, vector_structural_conflict, vector_tradeoff_direction


class VectorStrategyTests(unittest.TestCase):
    def test_negative_inner_product_gives_tradeoff_direction(self):
        g1 = (1.0, 0.0)
        g2 = (-2.0, 0.5)
        self.assertTrue(vector_structural_conflict(g1, g2))
        delta = vector_tradeoff_direction(g1, g2)
        self.assertLess(vector_potential_change(g1, delta), 0.0)
        self.assertGreater(vector_potential_change(g2, delta), 0.0)


if __name__ == "__main__":
    unittest.main()
