import unittest

from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from grounded_epistemic_goal import (
    grounded_epistemic_goal_score,
    grounded_expected_instrumentality,
    grounded_goal_evaluation_admissible,
)
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from interpretation import InterpretationCandidate
from provenance import ProofRecord
from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from world_graph import WorldState


class GroundedEpistemicGoalTests(unittest.TestCase):
    def setUp(self):
        event = TypedNode(NodeKind.EVENT, "panic")
        hypotheses = {
            "guilty": TypedNode(NodeKind.CONCEPT, "guilty"),
            "afraid": TypedNode(NodeKind.CONCEPT, "afraid"),
            "ill": TypedNode(NodeKind.CONCEPT, "ill"),
        }
        signature = HyperedgeSignature(
            premise_kinds=(NodeKind.EVENT,),
            conclusion_kind=NodeKind.CONCEPT,
        )
        rules = {
            key: make_hyperedge(signature, (event,), hypothesis)
            for key, hypothesis in hypotheses.items()
        }
        graph = {"obs": ProofRecord(event, None, (), 0)}
        for key, hypothesis in hypotheses.items():
            graph[f"attr:{key}"] = ProofRecord(
                hypothesis,
                rules[key],
                ("obs",),
                1,
            )
        kb = BeliefKB(
            owner="hamlet",
            graph=graph,
            active={"obs", "attr:guilty", "attr:afraid", "attr:ill"},
        )
        priors = {"guilty": 0.2, "afraid": 0.5, "ill": 0.3}
        likelihoods = {"guilty": 0.9, "afraid": 0.3, "ill": 0.1}
        states = {
            key: EpistemicBeliefState(
                kb=kb,
                hypothesis=hypothesis,
                confidence=priors[key],
            )
            for key, hypothesis in hypotheses.items()
        }
        candidates = {
            key: InterpretationCandidate(
                evidence=event,
                hypothesis=hypothesis,
                evidence_proof="obs",
                attribution_proof=f"attr:{key}",
                likelihood_h=likelihoods[key],
                likelihood_not_h=0.1,
            )
            for key, hypothesis in hypotheses.items()
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

    def test_constant_instrumentality_is_preserved_by_normalized_posterior(self):
        values = {key: 0.7 for key in self.space.states}
        self.assertAlmostEqual(
            grounded_expected_instrumentality(self.space, values),
            0.7,
        )

    def test_expected_instrumentality_lies_between_pointwise_bounds(self):
        values = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        expected = grounded_expected_instrumentality(self.space, values)
        self.assertAlmostEqual(expected, 0.65)
        self.assertGreaterEqual(expected, min(values.values()))
        self.assertLessEqual(expected, max(values.values()))
        self.assertAlmostEqual(
            grounded_epistemic_goal_score(
                self.space,
                values,
                pressure=2.0,
                cost=0.1,
                risk=0.2,
            ),
            1.0,
        )

    def test_missing_hypothesis_instrumentality_is_rejected(self):
        with self.assertRaises(ValueError):
            grounded_expected_instrumentality(
                self.space,
                {"guilty": 1.0, "afraid": 0.4},
            )

    def test_common_evidence_retraction_blocks_goal_evaluation(self):
        self.assertTrue(grounded_goal_evaluation_admissible(self.space))
        revoked = revoke_common_evidence(self.space)
        self.assertFalse(grounded_goal_evaluation_admissible(revoked))
        with self.assertRaises(ValueError):
            grounded_expected_instrumentality(
                revoked,
                {"guilty": 1.0, "afraid": 0.4, "ill": -0.2},
            )


if __name__ == "__main__":
    unittest.main()
