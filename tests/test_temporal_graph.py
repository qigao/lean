import unittest

from world_graph import WorldState, add_causal_edge, causal_graph_acyclic, causal_reachable


class TemporalCausalGraphTests(unittest.TestCase):
    def world(self):
        return WorldState(
            alive=set(),
            can_act=set(),
            info_edges={},
            event_nodes={"a": "a", "b": "b", "c": "c"},
            agent_nodes={},
            observations=set(),
            causal_edges=set(),
            event_time={"a": 1, "b": 2, "c": 3},
        )

    def test_transitive_causal_path_is_reachable(self):
        world = add_causal_edge(self.world(), "a", "b")
        world = add_causal_edge(world, "b", "c")
        self.assertTrue(causal_reachable(world, "a", "c"))
        self.assertLess(world.event_time["a"], world.event_time["c"])

    def test_strict_time_order_makes_causal_graph_acyclic(self):
        world = add_causal_edge(self.world(), "a", "b")
        world = add_causal_edge(world, "b", "c")
        self.assertTrue(causal_graph_acyclic(world))
        self.assertFalse(causal_reachable(world, "a", "a"))
        self.assertFalse(causal_reachable(world, "c", "a"))


if __name__ == "__main__":
    unittest.main()
