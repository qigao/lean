import unittest

from belief_support import BeliefKB
from epistemic_belief import EpistemicBeliefState
from grounded_hypothesis_space import GroundedHypothesisSpace, revoke_common_evidence
from interpretation import InterpretationCandidate
from narrative_dynamics.adapters.grounded_goal import (
    GroundedGoalDecisionModel,
    GroundedGoalScenario,
    selected_goal_metrics,
)
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.validation import synthetic_recovery
from provenance import ProofRecord
from typed_graph import NodeKind, TypedNode
from typed_hypergraph import HyperedgeSignature, make_hyperedge
from world_graph import WorldState


class GroundedGoalModelTests(unittest.TestCase):
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
        self.instrumentality = {
            "escape": {"weak": 1.0, "trap": 0.4, "routine": -0.2},
            "submit": {key: 0.5 for key in hypotheses},
            "wait": {key: 0.2 for key in hypotheses},
        }
        self.scenario = GroundedGoalScenario(
            id="guard-reduced",
            space=self.space,
            instrumentality=self.instrumentality,
            pressure=1.0,
        )
        self.runner = SimulationRunner()
        self.model = GroundedGoalDecisionModel()

    def test_adapter_emits_normalized_policy_and_seeded_selection(self):
        first = self.runner.run_once(
            self.model,
            self.scenario,
            {"beta": 4.0},
            seed=17,
        )
        replay = self.runner.run_once(
            self.model,
            self.scenario,
            {"beta": 4.0},
            seed=17,
        )

        self.assertEqual(first, replay)
        policy = first.outcome["policy"]
        selected = first.outcome["selected_goal"]
        self.assertAlmostEqual(sum(policy.values()), 1.0)
        self.assertTrue(all(value > 0.0 for value in policy.values()))
        self.assertIn(selected, policy)
        self.assertEqual(
            tuple(event.kind for event in first.events),
            ("policy_computed", "goal_selected"),
        )
        self.assertEqual(first.parameters, (("beta", 4.0),))

    def test_selected_goal_metrics_are_stable_one_hot_coordinates(self):
        trace = self.runner.run_once(
            self.model,
            self.scenario,
            {"beta": 2.0},
            seed=3,
        )

        metrics = selected_goal_metrics(trace)
        self.assertEqual(
            set(metrics),
            {"choice.escape", "choice.submit", "choice.wait"},
        )
        self.assertEqual(sum(metrics.values()), 1.0)
        self.assertEqual(metrics[f"choice.{trace.outcome['selected_goal']}"], 1.0)

    def test_invalid_scenario_and_beta_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            GroundedGoalScenario(
                id="empty-goals",
                space=self.space,
                instrumentality={},
                pressure=1.0,
            )
        for parameters in (
            {},
            {"beta": 0.0},
            {"beta": -1.0},
            {"beta": 2.0, "extra": 1.0},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.runner.run_once(
                        self.model,
                        self.scenario,
                        parameters,
                        seed=1,
                    )

    def test_revoked_common_evidence_blocks_model_run(self):
        revoked = GroundedGoalScenario(
            id="revoked",
            space=revoke_common_evidence(self.space),
            instrumentality=self.instrumentality,
            pressure=1.0,
        )

        with self.assertRaises(ValueError):
            self.runner.run_once(
                self.model,
                revoked,
                {"beta": 4.0},
                seed=1,
            )

    def test_choice_frequencies_recover_true_beta_on_finite_grid(self):
        report = synthetic_recovery(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            true_parameters={"beta": 4.0},
            parameter_grid={"beta": (0.5, 2.0, 4.0)},
            observation_seeds=range(500),
            calibration_seeds=range(10_000, 10_500),
            extractor=selected_goal_metrics,
        )

        self.assertTrue(report.recovered)
        self.assertEqual(report.calibration.best.parameters, (("beta", 4.0),))
        self.assertLess(
            report.calibration.ranking[0].loss,
            report.calibration.ranking[1].loss,
        )


if __name__ == "__main__":
    unittest.main()
