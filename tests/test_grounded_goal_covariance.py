import unittest

from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from grounded_goal_covariance import (
    grounded_evidence_mass,
    grounded_likelihood_instrumentality_covariance,
    grounded_prior_expected_instrumentality,
)
from grounded_epistemic_goal import grounded_expected_instrumentality
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from interpretation import InterpretationCandidate
from provenance import ProofRecord
from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from world_graph import WorldState


class GroundedGoalCovarianceTests(unittest.TestCase):
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

    def test_posterior_instrumentality_change_equals_covariance_over_evidence_mass(self):
        values = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        prior = grounded_prior_expected_instrumentality(self.space, values)
        posterior = grounded_expected_instrumentality(self.space, values)
        covariance = grounded_likelihood_instrumentality_covariance(
            self.space,
            values,
        )
        mass = grounded_evidence_mass(self.space)

        self.assertAlmostEqual(prior, 0.34)
        self.assertAlmostEqual(posterior, 0.65)
        self.assertAlmostEqual(covariance, 0.1116)
        self.assertAlmostEqual(posterior - prior, covariance / mass)

    def test_covariance_sign_controls_posterior_instrumentality_direction(self):
        supportive = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        opposing = {"guilty": -0.2, "afraid": 0.4, "ill": 1.0}

        supportive_covariance = grounded_likelihood_instrumentality_covariance(
            self.space,
            supportive,
        )
        opposing_covariance = grounded_likelihood_instrumentality_covariance(
            self.space,
            opposing,
        )

        self.assertGreater(supportive_covariance, 0.0)
        self.assertGreater(
            grounded_expected_instrumentality(self.space, supportive),
            grounded_prior_expected_instrumentality(self.space, supportive),
        )
        self.assertLess(opposing_covariance, 0.0)
        self.assertLess(
            grounded_expected_instrumentality(self.space, opposing),
            grounded_prior_expected_instrumentality(self.space, opposing),
        )

    def test_constant_instrumentality_has_zero_covariance(self):
        constant = {key: 0.7 for key in self.space.states}
        self.assertAlmostEqual(
            grounded_likelihood_instrumentality_covariance(self.space, constant),
            0.0,
        )

    def test_common_evidence_retraction_blocks_covariance_evaluation(self):
        revoked = revoke_common_evidence(self.space)
        values = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        with self.assertRaises(ValueError):
            grounded_likelihood_instrumentality_covariance(revoked, values)


if __name__ == "__main__":
    unittest.main()
