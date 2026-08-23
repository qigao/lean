import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from forward_chaining import forward_chain_once, forward_chain


class ForwardChainingTests(unittest.TestCase):
    def setUp(self):
        grievance = TypedNode(NodeKind.CONCEPT, "shared_grievance")
        coalition = TypedNode(NodeKind.INSTITUTION, "coalition")
        leader = TypedNode(NodeKind.AGENT, "leader")
        revolt = TypedNode(NodeKind.EVENT, "revolt")
        organized = TypedNode(NodeKind.EVENT, "organized_rebellion")

        revolt_rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.CONCEPT, NodeKind.INSTITUTION),
                conclusion_kind=NodeKind.EVENT,
            ),
            (grievance, coalition),
            revolt,
        )
        organize_rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.EVENT, NodeKind.AGENT),
                conclusion_kind=NodeKind.EVENT,
            ),
            (revolt, leader),
            organized,
        )

        self.initial = {grievance, coalition, leader}
        self.revolt = revolt
        self.organized = organized
        # Reverse dependency order so one pass cannot derive the whole chain.
        self.rules = [organize_rule, revolt_rule]

    def test_one_pass_is_extensive_but_may_not_reach_closure(self):
        once = forward_chain_once(self.initial, self.rules)
        self.assertTrue(self.initial.issubset(once))
        self.assertIn(self.revolt, once)
        self.assertNotIn(self.organized, once)

    def test_repeated_forward_chaining_reaches_fixed_point(self):
        closure = forward_chain(self.initial, self.rules)
        self.assertIn(self.revolt, closure)
        self.assertIn(self.organized, closure)
        self.assertEqual(forward_chain_once(closure, self.rules), closure)

    def test_forward_chaining_never_retracts_initial_facts(self):
        closure = forward_chain(self.initial, self.rules)
        self.assertTrue(self.initial.issubset(closure))


if __name__ == "__main__":
    unittest.main()
