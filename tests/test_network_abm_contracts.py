import math
import unittest


from narrative_dynamics.abm import (
    NetworkABMModel,
    NetworkAgentSpec,
    NetworkAgentState,
    PopulationState,
    SocialEdge,
    SocialNetwork,
    initialize_population,
)


def make_model(*, reverse: bool = False) -> NetworkABMModel:
    agents = (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 0.75, 0.6, 0.7),
        NetworkAgentSpec("c", "observer", 0.5, 0.8, 0.9),
    )
    edges = (
        SocialEdge("a", "b", "peer", 1.0),
        SocialEdge("b", "c", "peer", 0.5),
    )
    if reverse:
        agents = tuple(reversed(agents))
        edges = tuple(reversed(edges))
    return NetworkABMModel(
        "line-network",
        "1",
        agents,
        SocialNetwork(tuple(reversed(("a", "b", "c"))) if reverse else ("a", "b", "c"), edges),
    )


class NetworkABMContractTests(unittest.TestCase):
    def test_model_canonicalizes_agent_edge_and_roster_order(self):
        first = make_model()
        second = make_model(reverse=True)

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(tuple(item.agent_id for item in first.agents), ("a", "b", "c"))
        self.assertEqual(first.network.agent_ids, ("a", "b", "c"))
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in first.network.edges),
            (("a", "b"), ("b", "c")),
        )

    def test_agent_and_edge_contracts_reject_invalid_values(self):
        for bad in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(agent_value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    NetworkAgentSpec("a", "source", bad, 0.5, 0.5)
            with self.subTest(edge_value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    SocialEdge("a", "b", "peer", bad)

        with self.assertRaisesRegex(ValueError, "self-edge"):
            SocialEdge("a", "a", "peer", 1.0)
        with self.assertRaisesRegex(TypeError, "active"):
            SocialEdge("a", "b", "peer", 1.0, active=1)
        with self.assertRaisesRegex(ValueError, "role"):
            NetworkAgentSpec("a", "", 1.0, 0.5, 0.5)

    def test_network_and_model_reject_duplicate_unknown_and_mismatched_agents(self):
        edge = SocialEdge("a", "b", "peer", 1.0)
        with self.assertRaisesRegex(ValueError, "unique"):
            SocialNetwork(("a", "a"), ())
        with self.assertRaisesRegex(ValueError, "endpoint"):
            SocialNetwork(("a", "b"), (SocialEdge("a", "missing", "peer"),))
        with self.assertRaisesRegex(ValueError, "unique"):
            SocialNetwork(("a", "b"), (edge, edge))

        model = make_model()
        with self.assertRaisesRegex(ValueError, "unique"):
            NetworkABMModel(
                model.model_id,
                model.version,
                (model.agents[0], model.agents[0]),
                SocialNetwork(("a",), ()),
            )
        with self.assertRaisesRegex(ValueError, "same agent ids"):
            NetworkABMModel(
                model.model_id,
                model.version,
                model.agents[:-1],
                model.network,
            )

    def test_initialize_population_binds_exact_model_and_seeded_information(self):
        model = make_model()
        state = initialize_population(model, beliefs={"a": 1.0, "b": 0.2})
        by_id = {item.agent_id: item for item in state.agents}

        self.assertEqual(state.model_id, model.model_id)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertEqual(state.round_index, 0)
        self.assertIsNone(state.parent_state_hash)
        self.assertEqual(by_id["a"], NetworkAgentState("a", 1.0, 1, True))
        self.assertEqual(by_id["b"], NetworkAgentState("b", 0.2, 1, False))
        self.assertEqual(by_id["c"], NetworkAgentState("c", 0.0, 0, False))
        self.assertEqual(
            state.content_hash,
            initialize_population(make_model(reverse=True), beliefs={"b": 0.2, "a": 1.0}).content_hash,
        )

    def test_initialize_and_population_state_reject_invalid_bindings(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            initialize_population(model, beliefs={"missing": 1.0})
        with self.assertRaises((TypeError, ValueError)):
            initialize_population(model, beliefs={"a": math.nan})
        with self.assertRaisesRegex(ValueError, "round zero"):
            PopulationState(
                model.model_id,
                model.content_hash,
                0,
                model.content_hash,
                initialize_population(model).agents,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            PopulationState(
                model.model_id,
                model.content_hash,
                0,
                None,
                (NetworkAgentState("a", 0.0, 0, False),) * 2,
            )


if __name__ == "__main__":
    unittest.main()
