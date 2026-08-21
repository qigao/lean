import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord, validate_provenance, support_reachable, proofs_for_fact


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.grievance = TypedNode(NodeKind.CONCEPT, "shared_grievance")
        self.coalition = TypedNode(NodeKind.INSTITUTION, "coalition")
        self.leader = TypedNode(NodeKind.AGENT, "leader")
        self.revolt = TypedNode(NodeKind.EVENT, "revolt")
        self.organized = TypedNode(NodeKind.EVENT, "organized_rebellion")

        self.revolt_rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.CONCEPT, NodeKind.INSTITUTION),
                conclusion_kind=NodeKind.EVENT,
            ),
            (self.grievance, self.coalition),
            self.revolt,
        )
        self.organize_rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.EVENT, NodeKind.AGENT),
                conclusion_kind=NodeKind.EVENT,
            ),
            (self.revolt, self.leader),
            self.organized,
        )

        self.graph = {
            "g": ProofRecord(self.grievance, None, (), 0),
            "c": ProofRecord(self.coalition, None, (), 0),
            "l": ProofRecord(self.leader, None, (), 0),
            "r": ProofRecord(self.revolt, self.revolt_rule, ("g", "c"), 1),
            "o": ProofRecord(self.organized, self.organize_rule, ("r", "l"), 2),
        }

    def test_valid_provenance_forms_acyclic_support_graph(self):
        self.assertTrue(validate_provenance(self.graph))
        self.assertTrue(support_reachable(self.graph, "g", "o"))
        self.assertFalse(support_reachable(self.graph, "o", "g"))

    def test_derived_record_must_match_rule_and_exact_parent_facts(self):
        broken = dict(self.graph)
        broken["r"] = ProofRecord(self.organized, self.revolt_rule, ("g", "c"), 1)
        self.assertFalse(validate_provenance(broken))

    def test_asserted_record_cannot_claim_support_parents(self):
        broken = dict(self.graph)
        broken["g"] = ProofRecord(self.grievance, None, ("c",), 1)
        self.assertFalse(validate_provenance(broken))

    def test_same_fact_can_have_multiple_proofs(self):
        graph = dict(self.graph)
        graph["r_asserted"] = ProofRecord(self.revolt, None, (), 0)
        self.assertTrue(validate_provenance(graph))
        self.assertEqual(proofs_for_fact(graph, self.revolt), {"r", "r_asserted"})


if __name__ == "__main__":
    unittest.main()
