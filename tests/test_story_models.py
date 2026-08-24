from __future__ import annotations

from dataclasses import replace
import math
import random
import unittest

from narrative_dynamics.adapters.story_belief_search import AgentBeliefSearchModel
from narrative_dynamics.adapters.story_omniscient_search import OmniscientSearchModel
from narrative_dynamics.contracts import SimulationTrace
from narrative_dynamics.story.metrics import story_choice_metrics
from narrative_dynamics.story.scenario import project_narrative_scenario
from narrative_dynamics.story.schema import (
    NarrativeOracleV1,
    SearchActionV1,
    SearchDecisionV1,
    StoryEntitiesV1,
    load_narrative_case,
)


def _trace_with_policy(policy: dict[str, object]) -> SimulationTrace:
    return SimulationTrace(
        model_name="story-test",
        scenario_id="story-v1-test",
        parameters=(),
        seed=0,
        events=(),
        outcome={"policy": policy},
    )


class _RawTrace:
    def __init__(self, policy: dict[str, object]):
        self.outcome = {"policy": policy}


class StoryModelTests(unittest.TestCase):
    def setUp(self):
        self.false_case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        self.informed_case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )

    def test_belief_model_follows_actor_subjective_state(self):
        model = AgentBeliefSearchModel()
        self.assertEqual(model.name, "agent-belief-search")

        false_run = model.simulate(
            project_narrative_scenario(self.false_case), {}, random.Random(7)
        )
        informed_run = model.simulate(
            project_narrative_scenario(self.informed_case), {}, random.Random(7)
        )

        self.assertEqual(false_run.outcome["selected_action"], "search_drawer")
        self.assertEqual(informed_run.outcome["selected_action"], "search_box")
        self.assertEqual(
            false_run.outcome["epistemic_basis"],
            {
                "kind": "subjective",
                "target_object": "key",
                "resolved_location": "drawer",
                "supporting_event_id": "e1",
            },
        )
        self.assertEqual(
            informed_run.outcome["epistemic_basis"],
            {
                "kind": "subjective",
                "target_object": "key",
                "resolved_location": "box",
                "supporting_event_id": "e2",
            },
        )

    def test_belief_model_emits_exact_one_hot_action_contract(self):
        run = AgentBeliefSearchModel().simulate(
            project_narrative_scenario(self.false_case), {}, random.Random(19)
        )

        self.assertEqual(
            set(run.outcome),
            {"epistemic_basis", "action_scores", "policy", "selected_action"},
        )
        self.assertEqual(
            set(run.outcome["action_scores"]),
            {"search_drawer", "search_box"},
        )
        self.assertEqual(
            set(run.outcome["policy"]),
            {"search_drawer", "search_box"},
        )
        self.assertEqual(
            dict(run.outcome["action_scores"]),
            {"search_drawer": 1.0, "search_box": 0.0},
        )
        self.assertEqual(
            dict(run.outcome["policy"]),
            {"search_drawer": 1.0, "search_box": 0.0},
        )
        self.assertAlmostEqual(sum(run.outcome["policy"].values()), 1.0)

    def test_belief_model_rejects_any_parameter(self):
        with self.assertRaisesRegex(ValueError, "empty parameter"):
            AgentBeliefSearchModel().simulate(
                project_narrative_scenario(self.false_case),
                {"beta": 1.0},
                random.Random(7),
            )

    def test_belief_model_does_not_fallback_to_objective_location(self):
        entities = StoryEntitiesV1(
            agents=self.false_case.entities.agents,
            objects=self.false_case.entities.objects,
            locations=("drawer", "box", "shelf"),
        )
        decision = SearchDecisionV1(
            id=self.false_case.decision.id,
            time=self.false_case.decision.time,
            actor=self.false_case.decision.actor,
            object=self.false_case.decision.object,
            actions=(
                SearchActionV1(id="search_box", location="box"),
                SearchActionV1(id="search_shelf", location="shelf"),
            ),
        )
        no_subjective_match = replace(
            self.false_case,
            entities=entities,
            decision=decision,
            oracle=NarrativeOracleV1(
                objective_location="box",
                actor_subjective_location="drawer",
                agent_belief_ranking=("search_box", "search_shelf"),
                omniscient_ranking=("search_box", "search_shelf"),
            ),
        )

        with self.assertRaisesRegex(
            ValueError, "subjective target location.*exactly one decision action"
        ):
            AgentBeliefSearchModel().simulate(
                project_narrative_scenario(no_subjective_match),
                {},
                random.Random(7),
            )

    def test_omniscient_model_follows_objective_state_in_both_cases(self):
        model = OmniscientSearchModel()
        self.assertEqual(model.name, "omniscient-search")

        for case in (self.false_case, self.informed_case):
            with self.subTest(case=case.name):
                run = model.simulate(
                    project_narrative_scenario(case), {}, random.Random(11)
                )
                self.assertEqual(run.outcome["selected_action"], "search_box")
                self.assertEqual(
                    run.outcome["epistemic_basis"],
                    {
                        "kind": "objective",
                        "target_object": "key",
                        "resolved_location": "box",
                        "supporting_event_id": "e2",
                    },
                )
                self.assertEqual(
                    dict(run.outcome["action_scores"]),
                    {"search_drawer": 0.0, "search_box": 1.0},
                )
                self.assertEqual(
                    dict(run.outcome["policy"]),
                    {"search_drawer": 0.0, "search_box": 1.0},
                )

    def test_false_belief_separates_models_and_informed_case_reunites_them(self):
        belief = AgentBeliefSearchModel()
        omniscient = OmniscientSearchModel()
        false_scenario = project_narrative_scenario(self.false_case)
        informed_scenario = project_narrative_scenario(self.informed_case)

        self.assertNotEqual(
            belief.simulate(false_scenario, {}, random.Random(1)).outcome[
                "selected_action"
            ],
            omniscient.simulate(false_scenario, {}, random.Random(1)).outcome[
                "selected_action"
            ],
        )
        self.assertEqual(
            belief.simulate(informed_scenario, {}, random.Random(1)).outcome[
                "selected_action"
            ],
            omniscient.simulate(informed_scenario, {}, random.Random(1)).outcome[
                "selected_action"
            ],
        )

    def test_omniscient_model_rejects_parameters_and_missing_objective_action(self):
        with self.assertRaisesRegex(ValueError, "empty parameter"):
            OmniscientSearchModel().simulate(
                project_narrative_scenario(self.false_case),
                {"beta": 1.0},
                random.Random(7),
            )

        entities = StoryEntitiesV1(
            agents=self.false_case.entities.agents,
            objects=self.false_case.entities.objects,
            locations=("drawer", "box", "shelf"),
        )
        decision = SearchDecisionV1(
            id=self.false_case.decision.id,
            time=self.false_case.decision.time,
            actor=self.false_case.decision.actor,
            object=self.false_case.decision.object,
            actions=(
                SearchActionV1(id="search_drawer", location="drawer"),
                SearchActionV1(id="search_shelf", location="shelf"),
            ),
        )
        no_objective_match = replace(
            self.false_case,
            entities=entities,
            decision=decision,
            oracle=NarrativeOracleV1(
                objective_location="box",
                actor_subjective_location="drawer",
                agent_belief_ranking=("search_drawer", "search_shelf"),
                omniscient_ranking=("search_drawer", "search_shelf"),
            ),
        )

        with self.assertRaisesRegex(
            ValueError, "objective target location.*exactly one decision action"
        ):
            OmniscientSearchModel().simulate(
                project_narrative_scenario(no_objective_match),
                {},
                random.Random(7),
            )

    def test_story_choice_metrics_returns_exact_policy_coordinates(self):
        belief_run = AgentBeliefSearchModel().simulate(
            project_narrative_scenario(self.false_case), {}, random.Random(5)
        )
        trace = SimulationTrace(
            model_name="agent-belief-search",
            scenario_id="story-v1-test",
            parameters=(),
            seed=5,
            events=belief_run.events,
            outcome=belief_run.outcome,
        )

        self.assertEqual(
            story_choice_metrics(trace),
            {"choice.search_box": 0.0, "choice.search_drawer": 1.0},
        )

    def test_story_choice_metrics_rejects_action_coverage_and_normalization_errors(self):
        invalid = (
            ({"search_box": 1.0}, "exactly"),
            (
                {
                    "search_box": 0.5,
                    "search_drawer": 0.5,
                    "search_shelf": 0.0,
                },
                "exactly",
            ),
            ({"search_box": 0.6, "search_drawer": 0.6}, "normalized"),
        )
        for policy, pattern in invalid:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, pattern):
                    story_choice_metrics(_trace_with_policy(policy))

    def test_story_choice_metrics_rejects_invalid_probability_values(self):
        invalid = (
            ({"search_box": "half", "search_drawer": 0.5}, "numeric"),
            ({"search_box": True, "search_drawer": 0.0}, "numeric"),
            ({"search_box": -0.1, "search_drawer": 1.1}, "non-negative"),
        )
        for policy, pattern in invalid:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, pattern):
                    story_choice_metrics(_trace_with_policy(policy))

        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    story_choice_metrics(
                        _RawTrace(
                            {"search_box": value, "search_drawer": 1.0}
                        )
                    )


if __name__ == "__main__":
    unittest.main()
