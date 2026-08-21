import unittest

from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from grounded_goal_softmax import (
    finite_softmax,
    grounded_goal_choice_probabilities,
    grounded_prior_goal_choice_probabilities,
)
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from interpretation import InterpretationCandidate
from provenance import ProofRecord
from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from world_graph import WorldState


class GroundedGoalSoftmaxTests(unittest.TestCase):
    def setUp(self):
        event = TypedNode(NodeKind.EVENT, "guard_reduced")
        hypotheses = {
            "weak": TypedNode(NodeKind.CONCEPT, "guard_weak"),
            "trap": TypedNode(NodeKind.CONCEPT, "guard_trap"),
            "routine": TypedNode(NodeKind.CONCEPT, "guard_routine"),
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
            owner="spartacus",
            graph=graph,
            active={"obs", "attr:weak", "attr:trap", "attr:routine"},
        )
        priors = {"weak": 0.2, "trap": 0.5, "routine": 0.3}
        likelihoods = {"weak": 0.9, "trap": 0.3, "routine": 0.1}
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
            alive={"spartacus"},
            can_act={"spartacus"},
            info_edges={"event:guard_reduced": {"agent:spartacus"}},
            event_nodes={"guard_reduced": "event:guard_reduced"},
            agent_nodes={"spartacus": "agent:spartacus"},
            observations={("guard_reduced", "spartacus")},
            causal_edges=set(),
            event_time={"guard_reduced": 1},
        )
        self.space = GroundedHypothesisSpace(
            world=world,
            event="guard_reduced",
            states=states,
            candidates=candidates,
            common_evidence=event,
            common_evidence_proof="obs",
        )
        self.goals = {
            "escape": {"weak": 1.0, "trap": 0.4, "routine": -0.2},
            "submit": {key: 0.5 for key in hypotheses},
            "wait": {key: 0.2 for key in hypotheses},
        }

    def test_finite_softmax_is_normalized_and_strictly_positive(self):
        probabilities = finite_softmax(
            {"escape": 0.6, "submit": 0.5, "wait": 0.2},
            beta=2.0,
        )
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertTrue(all(value > 0.0 for value in probabilities.values()))

    def test_positive_temperature_preserves_score_order(self):
        probabilities = finite_softmax(
            {"escape": 0.6, "submit": 0.5, "wait": 0.2},
            beta=2.0,
        )
        self.assertGreater(probabilities["escape"], probabilities["submit"])
        self.assertGreater(probabilities["submit"], probabilities["wait"])

    def test_grounded_evidence_reverses_softmax_preference(self):
        prior = grounded_prior_goal_choice_probabilities(
            self.space,
            self.goals,
            pressure=1.0,
            beta=4.0,
        )
        posterior = grounded_goal_choice_probabilities(
            self.space,
            self.goals,
            pressure=1.0,
            beta=4.0,
        )

        self.assertLess(prior["escape"], prior["submit"])
        self.assertGreater(posterior["escape"], posterior["submit"])
        self.assertAlmostEqual(sum(prior.values()), 1.0)
        self.assertAlmostEqual(sum(posterior.values()), 1.0)

    def test_common_evidence_retraction_blocks_softmax_choice(self):
        revoked = revoke_common_evidence(self.space)
        with self.assertRaises(ValueError):
            grounded_goal_choice_probabilities(
                revoked,
                self.goals,
                pressure=1.0,
                beta=4.0,
            )


if __name__ == "__main__":
    unittest.main()
