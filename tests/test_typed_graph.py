import unittest

from typed_graph import EdgeKind, NodeKind, TypedNode, make_edge, well_typed_edge


class TypedGraphTests(unittest.TestCase):
    def test_causal_edge_requires_event_to_event(self):
        event_a = TypedNode(NodeKind.EVENT, "a")
        event_b = TypedNode(NodeKind.EVENT, "b")
        agent = TypedNode(NodeKind.AGENT, "hamlet")

        edge = make_edge(EdgeKind.CAUSAL, event_a, event_b)
        self.assertTrue(well_typed_edge(edge))
        with self.assertRaises(ValueError):
            make_edge(EdgeKind.CAUSAL, agent, event_b)

    def test_observation_edge_requires_event_to_agent(self):
        event = TypedNode(NodeKind.EVENT, "play")
        agent = TypedNode(NodeKind.AGENT, "hamlet")
        obj = TypedNode(NodeKind.OBJECT, "sword")

        edge = make_edge(EdgeKind.OBSERVES, event, agent)
        self.assertTrue(well_typed_edge(edge))
        with self.assertRaises(ValueError):
            make_edge(EdgeKind.OBSERVES, event, obj)

    def test_social_world_edges_use_declared_endpoint_kinds(self):
        agent = TypedNode(NodeKind.AGENT, "spartacus")
        location = TypedNode(NodeKind.LOCATION, "ludus")
        obj = TypedNode(NodeKind.OBJECT, "sword")
        institution = TypedNode(NodeKind.INSTITUTION, "rebels")

        self.assertTrue(well_typed_edge(make_edge(EdgeKind.LOCATED_AT, agent, location)))
        self.assertTrue(well_typed_edge(make_edge(EdgeKind.OWNS, agent, obj)))
        self.assertTrue(well_typed_edge(make_edge(EdgeKind.MEMBER_OF, agent, institution)))


if __name__ == "__main__":
    unittest.main()
