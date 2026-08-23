import unittest

from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from grounded_epistemic_goal import grounded_epistemic_goal_score
from grounded_goal_ranking_reversal import (
    grounded_goal_ranking_reversal_condition,
    grounded_goal_ranking_reversal_margin,
)
from grounded_goal_score_covariance import grounded_prior_epistemic_goal_score
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from interpretation import InterpretationCandidate
from provenance import ProofRecord
from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from world_graph import WorldState


class GroundedGoalRankingReversalTests(unittest.TestCase):
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
        self.escape = {"weak": 1.0, "trap": 0.4, "routine": -0.2}

    def test_covariance_advantage_reverses_unequal_prior_goal_ranking(self):
        submit = {key: 0.5 for key in self.space.states}
        pressure = 1.0

        prior_escape = grounded_prior_epistemic_goal_score(
            self.space,
            self.escape,
            pressure=pressure,
        )
        prior_submit = grounded_prior_epistemic_goal_score(
            self.space,
            submit,
            pressure=pressure,
        )
        posterior_escape = grounded_epistemic_goal_score(
            self.space,
            self.escape,
            pressure=pressure,
        )
        posterior_submit = grounded_epistemic_goal_score(
            self.space,
            submit,
            pressure=pressure,
        )

        self.assertLess(prior_escape, prior_submit)
        self.assertGreater(
            grounded_goal_ranking_reversal_margin(
                self.space,
                self.escape,
                submit,
                pressure=pressure,
            ),
            0.0,
        )
        self.assertTrue(
            grounded_goal_ranking_reversal_condition(
                self.space,
                self.escape,
                submit,
                pressure=pressure,
            )
        )
        self.assertGreater(posterior_escape, posterior_submit)

    def test_insufficient_covariance_advantage_does_not_reverse_ranking(self):
        submit = {key: 0.7 for key in self.space.states}
        pressure = 1.0

        self.assertLessEqual(
            grounded_goal_ranking_reversal_margin(
                self.space,
                self.escape,
                submit,
                pressure=pressure,
            ),
            0.0,
        )
        self.assertFalse(
            grounded_goal_ranking_reversal_condition(
                self.space,
                self.escape,
                submit,
                pressure=pressure,
            )
        )
        self.assertLess(
            grounded_epistemic_goal_score(
                self.space,
                self.escape,
                pressure=pressure,
            ),
            grounded_epistemic_goal_score(
                self.space,
                submit,
                pressure=pressure,
            ),
        )

    def test_common_evidence_retraction_blocks_reversal_evaluation(self):
        revoked = revoke_common_evidence(self.space)
        submit = {key: 0.5 for key in self.space.states}

        with self.assertRaises(ValueError):
            grounded_goal_ranking_reversal_margin(
                revoked,
                self.escape,
                submit,
                pressure=1.0,
            )


if __name__ == "__main__":
    unittest.main()
