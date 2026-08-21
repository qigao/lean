import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from truth_maintenance import deactivate_proof, proof_valid, fact_supported


class TruthMaintenanceTests(unittest.TestCase):
    def setUp(self):
        grievance = TypedNode(NodeKind.CONCEPT, "shared_grievance")
        coalition = TypedNode(NodeKind.INSTITUTION, "coalition")
        leader = TypedNode(NodeKind.AGENT, "leader")
        revolt = TypedNode(NodeKind.EVENT, "revolt")
        organized = TypedNode(NodeKind.EVENT, "organized_rebellion")
        unrelated = TypedNode(NodeKind.CONCEPT, "unrelated_fact")

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

        self.revolt = revolt
        self.organized = organized
        self.unrelated = unrelated
        self.graph = {
            "g": ProofRecord(grievance, None, (), 0),
            "c": ProofRecord(coalition, None, (), 0),
            "l": ProofRecord(leader, None, (), 0),
            "r": ProofRecord(revolt, revolt_rule, ("g", "c"), 1),
            "o": ProofRecord(organized, organize_rule, ("r", "l"), 2),
            "u": ProofRecord(unrelated, None, (), 0),
            "r_alt": ProofRecord(revolt, None, (), 0),
        }
        self.active = set(self.graph)

    def test_deactivating_root_invalidates_all_dependent_proofs(self):
        active = deactivate_proof(self.active, "g")
        self.assertFalse(proof_valid(self.graph, active, "r"))
        self.assertFalse(proof_valid(self.graph, active, "o"))
        self.assertTrue(proof_valid(self.graph, active, "u"))

    def test_alternative_independent_proof_keeps_fact_supported(self):
        active = deactivate_proof(self.active, "g")
        self.assertFalse(proof_valid(self.graph, active, "r"))
        self.assertTrue(proof_valid(self.graph, active, "r_alt"))
        self.assertTrue(fact_supported(self.graph, active, self.revolt))

    def test_fact_retracts_when_all_its_proofs_depend_on_revoked_support(self):
        active_without_alt = self.active - {"r_alt"}
        active = deactivate_proof(active_without_alt, "g")
        self.assertFalse(fact_supported(self.graph, active, self.revolt))
        self.assertFalse(fact_supported(self.graph, active, self.organized))


if __name__ == "__main__":
    unittest.main()
