import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from typed_inference import fire_hyperedge, premises_satisfied


class TypedInferenceTests(unittest.TestCase):
    def setUp(self):
        signature = HyperedgeSignature(
            premise_kinds=(
                NodeKind.CONCEPT,
                NodeKind.INSTITUTION,
                NodeKind.AGENT,
                NodeKind.EVENT,
            ),
            conclusion_kind=NodeKind.EVENT,
        )
        self.premises = (
            TypedNode(NodeKind.CONCEPT, "shared_grievance"),
            TypedNode(NodeKind.INSTITUTION, "rebel_coalition"),
            TypedNode(NodeKind.AGENT, "leader"),
            TypedNode(NodeKind.EVENT, "opportunity"),
        )
        self.conclusion = TypedNode(NodeKind.EVENT, "rebellion")
        self.edge = make_hyperedge(signature, self.premises, self.conclusion)

    def test_all_premises_fire_conclusion(self):
        facts = set(self.premises)
        self.assertTrue(premises_satisfied(facts, self.edge))
        result = fire_hyperedge(facts, self.edge)
        self.assertIn(self.conclusion, result)

    def test_existing_facts_are_preserved(self):
        witness = TypedNode(NodeKind.EVENT, "existing_fact")
        facts = set(self.premises) | {witness}
        result = fire_hyperedge(facts, self.edge)
        self.assertIn(witness, result)

    def test_missing_premise_does_not_create_conclusion(self):
        facts = set(self.premises[:-1])
        self.assertFalse(premises_satisfied(facts, self.edge))
        result = fire_hyperedge(facts, self.edge)
        self.assertNotIn(self.conclusion, result)
        self.assertEqual(result, facts)


if __name__ == "__main__":
    unittest.main()
