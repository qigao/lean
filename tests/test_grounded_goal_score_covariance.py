import unittest

from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from grounded_goal_covariance import (
    grounded_evidence_mass,
    grounded_likelihood_instrumentality_covariance,
)
from grounded_goal_score_covariance import (
    grounded_goal_score_change,
    grounded_prior_epistemic_goal_score,
)
from grounded_epistemic_goal import grounded_epistemic_goal_score
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from interpretation import InterpretationCandidate
from provenance import ProofRecord
from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from world_graph import WorldState


class GroundedGoalScoreCovarianceTests(unittest.TestCase):
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

    def test_score_change_equals_pressure_times_covariance_over_mass(self):
        values = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        pressure = 2.0
        change = grounded_goal_score_change(
            self.space,
            values,
            pressure=pressure,
            cost=0.1,
            risk=0.2,
        )
        covariance = grounded_likelihood_instrumentality_covariance(
            self.space,
            values,
        )
        mass = grounded_evidence_mass(self.space)
        self.assertAlmostEqual(change, pressure * covariance / mass)

    def test_covariance_sign_controls_goal_score_direction(self):
        supportive = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        neutral = {key: 0.34 for key in self.space.states}
        opposing = {"guilty": -0.2, "afraid": 0.4, "ill": 1.0}

        for values, relation in (
            (supportive, "raise"),
            (neutral, "same"),
            (opposing, "lower"),
        ):
            prior = grounded_prior_epistemic_goal_score(
                self.space,
                values,
                pressure=2.0,
                cost=0.1,
                risk=0.2,
            )
            posterior = grounded_epistemic_goal_score(
                self.space,
                values,
                pressure=2.0,
                cost=0.1,
                risk=0.2,
            )
            if relation == "raise":
                self.assertGreater(posterior, prior)
            elif relation == "same":
                self.assertAlmostEqual(posterior, prior)
            else:
                self.assertLess(posterior, prior)

    def test_greater_covariance_wins_when_prior_scores_are_equal(self):
        supportive = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        neutral = {key: 0.34 for key in self.space.states}
        prior_supportive = grounded_prior_epistemic_goal_score(
            self.space,
            supportive,
            pressure=2.0,
        )
        prior_neutral = grounded_prior_epistemic_goal_score(
            self.space,
            neutral,
            pressure=2.0,
        )
        self.assertAlmostEqual(prior_supportive, prior_neutral)
        self.assertGreater(
            grounded_epistemic_goal_score(
                self.space,
                supportive,
                pressure=2.0,
            ),
            grounded_epistemic_goal_score(
                self.space,
                neutral,
                pressure=2.0,
            ),
        )

    def test_common_evidence_retraction_blocks_score_change(self):
        revoked = revoke_common_evidence(self.space)
        values = {"guilty": 1.0, "afraid": 0.4, "ill": -0.2}
        with self.assertRaises(ValueError):
            grounded_goal_score_change(
                revoked,
                values,
                pressure=2.0,
            )


if __name__ == "__main__":
    unittest.main()
