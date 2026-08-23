import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from interpretation import InterpretationCandidate
from world_graph import WorldState
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from grounded_belief_update import posterior_belief_states


class GroundedBeliefUpdateTests(unittest.TestCase):
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

    def test_posterior_states_normalize_and_preserve_symbolic_coordinates(self):
        updated = posterior_belief_states(self.space)
        self.assertAlmostEqual(sum(state.confidence for state in updated.values()), 1.0)
        for key, state in updated.items():
            self.assertEqual(state.kb, self.space.states[key].kb)
            self.assertEqual(state.hypothesis, self.space.states[key].hypothesis)
        self.assertGreater(updated["guilty"].confidence, updated["afraid"].confidence)

    def test_ungrounded_space_cannot_receive_posterior_belief_update(self):
        revoked = revoke_common_evidence(self.space)
        with self.assertRaises(ValueError):
            posterior_belief_states(revoked)


if __name__ == "__main__":
    unittest.main()
