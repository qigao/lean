import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from interpretation import InterpretationCandidate
from world_graph import WorldState
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from interpretation_selection import committed_interpretations


class InterpretationSelectionTests(unittest.TestCase):
    def setUp(self):
        event = TypedNode(NodeKind.EVENT, "panic")
        guilty = TypedNode(NodeKind.CONCEPT, "guilty")
        afraid = TypedNode(NodeKind.CONCEPT, "afraid")
        signature = HyperedgeSignature(
            premise_kinds=(NodeKind.EVENT,),
            conclusion_kind=NodeKind.CONCEPT,
        )
        guilty_rule = make_hyperedge(signature, (event,), guilty)
        afraid_rule = make_hyperedge(signature, (event,), afraid)
        graph = {
            "obs": ProofRecord(event, None, (), 0),
            "attr:guilty": ProofRecord(guilty, guilty_rule, ("obs",), 1),
            "attr:afraid": ProofRecord(afraid, afraid_rule, ("obs",), 1),
        }
        kb = BeliefKB(
            owner="hamlet",
            graph=graph,
            active={"obs", "attr:guilty", "attr:afraid"},
        )
        states = {
            "guilty": EpistemicBeliefState(kb=kb, hypothesis=guilty, confidence=0.5),
            "afraid": EpistemicBeliefState(kb=kb, hypothesis=afraid, confidence=0.5),
        }
        candidates = {
            "guilty": InterpretationCandidate(
                evidence=event,
                hypothesis=guilty,
                evidence_proof="obs",
                attribution_proof="attr:guilty",
                likelihood_h=0.9,
                likelihood_not_h=0.1,
            ),
            "afraid": InterpretationCandidate(
                evidence=event,
                hypothesis=afraid,
                evidence_proof="obs",
                attribution_proof="attr:afraid",
                likelihood_h=0.3,
                likelihood_not_h=0.7,
            ),
        }
        world = WorldState(
            alive={"hamlet"},
            can_act={"hamlet"},
            info_edges={"event:panic": {"agent:hamlet"}},
            event_nodes={"panic": "event:panic"},
            agent_nodes={"hamlet": "agent:hamlet"},
            observations={("panic", "hamlet")},
            causal_edges=set(),
            event_time={"panic": 1},
        )
        self.space = GroundedHypothesisSpace(
            world=world,
            event="panic",
            states=states,
            candidates=candidates,
            common_evidence=event,
            common_evidence_proof="obs",
        )

    def test_majority_threshold_selects_at_most_one_interpretation(self):
        self.assertEqual(
            committed_interpretations(self.space, threshold=0.5),
            {"guilty"},
        )

    def test_low_threshold_can_commit_multiple_interpretations(self):
        self.assertEqual(
            committed_interpretations(self.space, threshold=0.2),
            {"guilty", "afraid"},
        )

    def test_revoked_common_evidence_selects_none(self):
        revoked = revoke_common_evidence(self.space)
        self.assertEqual(
            committed_interpretations(revoked, threshold=0.2),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
