import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from belief_support import BeliefKB, belief_supported, revoke_proof


class BeliefSupportTests(unittest.TestCase):
    def make_kb(self, with_alternative: bool = False):
        evidence = TypedNode(NodeKind.EVENT, "observed_panic")
        hypothesis = TypedNode(NodeKind.CONCEPT, "claudius_guilty")
        rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.EVENT,),
                conclusion_kind=NodeKind.CONCEPT,
            ),
            (evidence,),
            hypothesis,
        )
        graph = {
            "evidence": ProofRecord(evidence, None, (), 0),
            "belief": ProofRecord(hypothesis, rule, ("evidence",), 1),
        }
        if with_alternative:
            graph["alternative"] = ProofRecord(hypothesis, None, (), 0)
        return BeliefKB(owner="hamlet", graph=graph, active=set(graph)), hypothesis

    def test_valid_proof_supports_belief(self):
        kb, hypothesis = self.make_kb()
        self.assertTrue(belief_supported(kb, hypothesis))

    def test_revoking_unique_evidence_retracts_belief(self):
        kb, hypothesis = self.make_kb()
        revised = revoke_proof(kb, "evidence")
        self.assertFalse(belief_supported(revised, hypothesis))

    def test_independent_alternative_proof_preserves_belief(self):
        kb, hypothesis = self.make_kb(with_alternative=True)
        revised = revoke_proof(kb, "evidence")
        self.assertTrue(belief_supported(revised, hypothesis))


if __name__ == "__main__":
    unittest.main()
