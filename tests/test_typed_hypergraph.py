import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge


class TypedHypergraphTests(unittest.TestCase):
    def test_accepts_multi_premise_social_cause(self):
        signature = HyperedgeSignature(
            premise_kinds=(
                NodeKind.CONCEPT,
                NodeKind.INSTITUTION,
                NodeKind.AGENT,
                NodeKind.EVENT,
            ),
            conclusion_kind=NodeKind.EVENT,
        )
        premises = (
            TypedNode(NodeKind.CONCEPT, "shared_grievance"),
            TypedNode(NodeKind.INSTITUTION, "rebel_coalition"),
            TypedNode(NodeKind.AGENT, "leader"),
            TypedNode(NodeKind.EVENT, "opportunity"),
        )
        conclusion = TypedNode(NodeKind.EVENT, "rebellion")

        edge = make_hyperedge(signature, premises, conclusion)

        self.assertEqual(tuple(node.kind for node in edge.premises), signature.premise_kinds)
        self.assertEqual(edge.conclusion.kind, signature.conclusion_kind)

    def test_rejects_wrong_premise_kind(self):
        signature = HyperedgeSignature(
            premise_kinds=(NodeKind.AGENT, NodeKind.EVENT),
            conclusion_kind=NodeKind.EVENT,
        )
        premises = (
            TypedNode(NodeKind.EVENT, "not_an_agent"),
            TypedNode(NodeKind.EVENT, "trigger"),
        )
        with self.assertRaises(ValueError):
            make_hyperedge(signature, premises, TypedNode(NodeKind.EVENT, "outcome"))

    def test_rejects_wrong_premise_arity(self):
        signature = HyperedgeSignature(
            premise_kinds=(NodeKind.AGENT, NodeKind.EVENT),
            conclusion_kind=NodeKind.EVENT,
        )
        with self.assertRaises(ValueError):
            make_hyperedge(
                signature,
                (TypedNode(NodeKind.AGENT, "leader"),),
                TypedNode(NodeKind.EVENT, "outcome"),
            )

    def test_rejects_wrong_conclusion_kind(self):
        signature = HyperedgeSignature(
            premise_kinds=(NodeKind.AGENT,),
            conclusion_kind=NodeKind.EVENT,
        )
        with self.assertRaises(ValueError):
            make_hyperedge(
                signature,
                (TypedNode(NodeKind.AGENT, "leader"),),
                TypedNode(NodeKind.CONCEPT, "wrong"),
            )


if __name__ == "__main__":
    unittest.main()
