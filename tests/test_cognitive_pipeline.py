import unittest

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState, belief_held, revoke_belief_evidence
from interpretation import InterpretationCandidate
from world_graph import WorldState
from cognitive_pipeline import perceptual_interpretation_admissible, apply_perceptual_interpretation


class CognitivePipelineTests(unittest.TestCase):
    def setUp(self):
        event = TypedNode(NodeKind.EVENT, "panic")
        hypothesis = TypedNode(NodeKind.CONCEPT, "guilty")
        rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.EVENT,),
                conclusion_kind=NodeKind.CONCEPT,
            ),
            (event,),
            hypothesis,
        )
        graph = {
            "obs": ProofRecord(event, None, (), 0),
            "attr": ProofRecord(hypothesis, rule, ("obs",), 1),
        }
        kb = BeliefKB(owner="hamlet", graph=graph, active={"obs", "attr"})
        self.state = EpistemicBeliefState(kb=kb, hypothesis=hypothesis, confidence=0.45)
        self.candidate = InterpretationCandidate(
            evidence=event,
            hypothesis=hypothesis,
            evidence_proof="obs",
            attribution_proof="attr",
            likelihood_h=0.9,
            likelihood_not_h=0.2,
        )

    def world(self, with_path: bool) -> WorldState:
        return WorldState(
            alive={"hamlet"},
            can_act={"hamlet"},
            info_edges={"event:panic": {"agent:hamlet"}} if with_path else {},
            event_nodes={"panic": "event:panic"},
            agent_nodes={"hamlet": "agent:hamlet"},
            observations={("panic", "hamlet")},
            causal_edges=set(),
            event_time={"panic": 1},
        )

    def test_valid_world_observation_can_drive_grounded_interpretation(self):
        world = self.world(with_path=True)
        self.assertTrue(
            perceptual_interpretation_admissible(
                world, self.state, "panic", self.candidate
            )
        )
        updated = apply_perceptual_interpretation(
            world, self.state, "panic", self.candidate
        )
        self.assertGreater(updated.confidence, self.state.confidence)

    def test_no_information_path_blocks_entire_cognitive_pipeline(self):
        world = self.world(with_path=False)
        self.assertFalse(
            perceptual_interpretation_admissible(
                world, self.state, "panic", self.candidate
            )
        )
        with self.assertRaises(ValueError):
            apply_perceptual_interpretation(
                world, self.state, "panic", self.candidate
            )

    def test_revoking_observation_evidence_breaks_attribution_and_belief(self):
        world = self.world(with_path=True)
        revised = revoke_belief_evidence(self.state, "obs")
        self.assertFalse(
            perceptual_interpretation_admissible(
                world, revised, "panic", self.candidate
            )
        )
        self.assertFalse(belief_held(revised, threshold=0.40))


if __name__ == "__main__":
    unittest.main()
