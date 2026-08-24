from __future__ import annotations

from dataclasses import replace
import random
import unittest

from narrative_dynamics.adapters.story_belief_search import AgentBeliefSearchModel
from narrative_dynamics.story.scenario import project_narrative_scenario
from narrative_dynamics.story.schema import (
    NarrativeOracleV1,
    SearchActionV1,
    SearchDecisionV1,
    StoryEntitiesV1,
    load_narrative_case,
)


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


if __name__ == "__main__":
    unittest.main()
