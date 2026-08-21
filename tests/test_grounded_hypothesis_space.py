import unittest
from dataclasses import replace

from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from provenance import ProofRecord
from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from interpretation import InterpretationCandidate
from world_graph import WorldState
from grounded_hypothesis_space import (
    GroundedHypothesisSpace,
    grounded_hypothesis_space_admissible,
    grounded_posteriors,
    revoke_common_evidence,
)


class GroundedHypothesisSpaceTests(unittest.TestCase):
    def setUp(self):
        event = TypedNode(NodeKind.EVENT, "panic")
        guilty = TypedNode(NodeKind.CONCEPT, "guilty")
        afraid = TypedNode(NodeKind.CONCEPT, "afraid")

        guilty_rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.EVENT,),
                conclusion_kind=NodeKind.CONCEPT,
            ),
            (event,),
            guilty,
        )
        afraid_rule = make_hyperedge(
            HyperedgeSignature(
                premise_kinds=(NodeKind.EVENT,),
                conclusion_kind=NodeKind.CONCEPT,
            ),
            (event,),
            afraid,
        )
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
        self.world = WorldState(
            alive={"hamlet"},
            can_act={"hamlet"},
            info_edges={"event:panic": {"agent:hamlet"}},
            event_nodes={"panic": "event:panic"},
            agent_nodes={"hamlet": "agent:hamlet"},
            observations={("panic", "hamlet")},
            causal_edges=set(),
            event_time={"panic": 1},
        )
        self.states = {
            "guilty": EpistemicBeliefState(kb=kb, hypothesis=guilty, confidence=0.5),
            "afraid": EpistemicBeliefState(kb=kb, hypothesis=afraid, confidence=0.5),
        }
        self.candidates = {
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
        self.space = GroundedHypothesisSpace(
            world=self.world,
            event="panic",
            states=self.states,
            candidates=self.candidates,
            common_evidence=event,
            common_evidence_proof="obs",
        )

    def test_all_hypotheses_compete_over_same_admitted_observation(self):
        self.assertTrue(grounded_hypothesis_space_admissible(self.space))

    def test_grounded_posteriors_normalize_and_favor_better_explanation(self):
        posterior = grounded_posteriors(self.space)
        self.assertAlmostEqual(sum(posterior.values()), 1.0)
        self.assertGreater(posterior["guilty"], posterior["afraid"])

    def test_mismatched_evidence_rejects_entire_hypothesis_space(self):
        wrong = replace(
            self.candidates["afraid"],
            evidence=TypedNode(NodeKind.EVENT, "different-event"),
        )
        bad_space = replace(
            self.space,
            candidates={**self.candidates, "afraid": wrong},
        )
        self.assertFalse(grounded_hypothesis_space_admissible(bad_space))

    def test_no_information_path_rejects_entire_hypothesis_space(self):
        disconnected = replace(self.world, info_edges={})
        self.assertFalse(
            grounded_hypothesis_space_admissible(
                replace(self.space, world=disconnected)
            )
        )

    def test_revoking_common_observation_invalidates_every_candidate(self):
        revised = revoke_common_evidence(self.space)
        self.assertFalse(grounded_hypothesis_space_admissible(revised))


if __name__ == "__main__":
    unittest.main()
