import unittest

from world_graph import (
    WorldState,
    add_causal_edge,
    add_observation,
    apply_rewrites,
    invariant_holds,
    kill_agent,
)


class WorldGraphInvariantTests(unittest.TestCase):
    def base_world(self):
        return WorldState(
            alive={"spartacus", "batiatus"},
            can_act={"spartacus", "batiatus"},
            info_edges={"sura_death": {"messenger"}, "messenger": {"spartacus"}},
            event_nodes={"sura_death": "sura_death", "rumor": "rumor"},
            agent_nodes={"spartacus": "spartacus", "batiatus": "batiatus"},
            observations=set(),
            causal_edges=set(),
            event_time={"sura_death": 10, "rumor": 20},
        )

    def test_killing_agent_removes_ability_to_act_and_preserves_invariant(self):
        world = self.base_world()
        after = kill_agent(world, "spartacus")
        self.assertNotIn("spartacus", after.alive)
        self.assertNotIn("spartacus", after.can_act)
        self.assertTrue(invariant_holds(after))

    def test_observation_can_only_be_added_when_information_path_exists(self):
        world = self.base_world()
        after = add_observation(world, "sura_death", "spartacus")
        self.assertIn(("sura_death", "spartacus"), after.observations)
        self.assertTrue(invariant_holds(after))
        with self.assertRaises(ValueError):
            add_observation(world, "rumor", "batiatus")

    def test_causal_edge_requires_strictly_increasing_event_time(self):
        world = self.base_world()
        after = add_causal_edge(world, "sura_death", "rumor")
        self.assertIn(("sura_death", "rumor"), after.causal_edges)
        self.assertTrue(invariant_holds(after))
        with self.assertRaises(ValueError):
            add_causal_edge(world, "rumor", "sura_death")

    def test_sequence_of_legal_rewrites_preserves_invariant(self):
        world = self.base_world()
        after = apply_rewrites(
            world,
            [
                lambda w: add_observation(w, "sura_death", "spartacus"),
                lambda w: add_causal_edge(w, "sura_death", "rumor"),
                lambda w: kill_agent(w, "batiatus"),
            ],
        )
        self.assertTrue(invariant_holds(after))


if __name__ == "__main__":
    unittest.main()
