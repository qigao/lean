import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from belief_support import BeliefKB
from epistemic_belief import (
    EpistemicBeliefState,
    belief_held,
    bayes_update_belief,
    revoke_belief_evidence,
)


class EpistemicBeliefTests(unittest.TestCase):
    def make_state(self, with_alternative: bool = False):
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
        kb = BeliefKB(owner="hamlet", graph=graph, active=set(graph))
        return EpistemicBeliefState(kb=kb, hypothesis=hypothesis, confidence=0.72)

    def test_positive_evidence_raises_confidence_and_preserves_held_belief(self):
        state = self.make_state()
        self.assertTrue(belief_held(state, threshold=0.60))
        updated = bayes_update_belief(state, likelihood_h=0.90, likelihood_not_h=0.15)
        self.assertGreater(updated.confidence, state.confidence)
        self.assertTrue(belief_held(updated, threshold=0.60))

    def test_revoking_unique_support_retracts_belief_even_if_confidence_stays_high(self):
        state = self.make_state()
        revised = revoke_belief_evidence(state, "evidence")
        self.assertEqual(revised.confidence, state.confidence)
        self.assertFalse(belief_held(revised, threshold=0.60))

    def test_independent_alternative_support_preserves_held_belief(self):
        state = self.make_state(with_alternative=True)
        revised = revoke_belief_evidence(state, "evidence")
        self.assertTrue(belief_held(revised, threshold=0.60))


if __name__ == "__main__":
    unittest.main()
