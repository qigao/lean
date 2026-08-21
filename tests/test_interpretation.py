import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState, belief_held, revoke_belief_evidence
from interpretation import InterpretationCandidate, interpretation_admissible, apply_interpretation


class InterpretationTests(unittest.TestCase):
    def setUp(self):
        evidence = TypedNode(NodeKind.EVENT, "claudius_panics")
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
            "attribution": ProofRecord(hypothesis, rule, ("evidence",), 1),
        }
        kb = BeliefKB(owner="hamlet", graph=graph, active={"evidence", "attribution"})
        self.state = EpistemicBeliefState(kb=kb, hypothesis=hypothesis, confidence=0.45)
        self.candidate = InterpretationCandidate(
            evidence=evidence,
            hypothesis=hypothesis,
            evidence_proof="evidence",
            attribution_proof="attribution",
            likelihood_h=0.90,
            likelihood_not_h=0.20,
        )

    def test_supported_grounded_interpretation_raises_confidence(self):
        self.assertTrue(interpretation_admissible(self.state, self.candidate))
        updated = apply_interpretation(self.state, self.candidate)
        self.assertGreater(updated.confidence, self.state.confidence)

    def test_unsupported_evidence_cannot_drive_interpretation(self):
        broken = revoke_belief_evidence(self.state, "evidence")
        self.assertFalse(interpretation_admissible(broken, self.candidate))
        with self.assertRaises(ValueError):
            apply_interpretation(broken, self.candidate)

    def test_revoking_evidence_invalidates_unique_attribution_belief(self):
        self.assertTrue(belief_held(self.state, threshold=0.40))
        revised = revoke_belief_evidence(self.state, "evidence")
        self.assertFalse(belief_held(revised, threshold=0.40))
        self.assertFalse(interpretation_admissible(revised, self.candidate))


if __name__ == "__main__":
    unittest.main()
