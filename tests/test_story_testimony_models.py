from __future__ import annotations

import inspect
import random
import unittest

from narrative_dynamics.adapters.story_belief_search import AgentBeliefSearchModel
from narrative_dynamics.adapters.story_omniscient_search import OmniscientSearchModel
from narrative_dynamics.adapters.story_testimony_search import TestimonySearchModel
from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.story.scenario import NarrativeScenarioV1
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2, project_testimony_scenario
from narrative_dynamics.story.schema import SearchActionV1, SearchDecisionV1, StoryEntitiesV1
from narrative_dynamics.story.schema_v2 import load_narrative_case_v2


_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"


def _direct_v1_view(case) -> Scenario:
    story = NarrativeScenarioV1(
        entities=case.entities,
        events=case.events,
        observations=case.observations,
        decision=case.decision,
    )
    payload = story.to_payload()
    digest = stable_content_hash(payload).removeprefix("sha256:")
    return Scenario(id=f"story-v1-{digest}", payload=payload)


class TestimonyModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = load_narrative_case_v2(_TRUTHFUL)
        self.stale = load_narrative_case_v2(_STALE)

    def test_model_name_is_exact_and_parameters_are_empty_only(self):
        model = TestimonySearchModel()
        self.assertEqual(model.name, "testimony-search")
        with self.assertRaisesRegex(ValueError, "empty parameter"):
            model.simulate(
                project_testimony_scenario(self.truthful),
                {"beta": 1.0},
                random.Random(7),
            )

    def test_testimony_model_follows_received_report_content(self):
        model = TestimonySearchModel()
        truthful = model.simulate(
            project_testimony_scenario(self.truthful), {}, random.Random(7)
        )
        stale = model.simulate(
            project_testimony_scenario(self.stale), {}, random.Random(7)
        )
        self.assertEqual(truthful.outcome["selected_action"], "search_box")
        self.assertEqual(stale.outcome["selected_action"], "search_drawer")
        self.assertEqual(
            dict(truthful.outcome["epistemic_basis"]),
            {
                "kind": "testimony",
                "target_object": "key",
                "resolved_location": "box",
                "supporting_id": "r1",
                "source_agent": "alice",
            },
        )
        self.assertEqual(
            dict(stale.outcome["epistemic_basis"]),
            {
                "kind": "testimony",
                "target_object": "key",
                "resolved_location": "drawer",
                "supporting_id": "r1",
                "source_agent": "alice",
            },
        )

    def test_policy_and_scores_are_exact_deterministic_one_hot(self):
        model = TestimonySearchModel()
        run = model.simulate(
            project_testimony_scenario(self.truthful), {}, random.Random(11)
        )
        self.assertEqual(
            dict(run.outcome["action_scores"]),
            {"search_drawer": 0.0, "search_box": 1.0},
        )
        self.assertEqual(
            dict(run.outcome["policy"]),
            {"search_drawer": 0.0, "search_box": 1.0},
        )
        self.assertEqual(sum(run.outcome["policy"].values()), 1.0)
        replayed = model.simulate(
            project_testimony_scenario(self.truthful), {}, random.Random(999)
        )
        self.assertEqual(replayed, run)

    def test_unreceived_report_uses_direct_perception_basis(self):
        story = NarrativeScenarioV2(
            entities=self.truthful.entities,
            events=self.truthful.events,
            observations=self.truthful.observations,
            reports=self.truthful.reports,
            receptions=(),
            decision=self.truthful.decision,
        )
        payload = story.to_payload()
        digest = stable_content_hash(payload).removeprefix("sha256:")
        scenario = Scenario(id=f"story-v2-{digest}", payload=payload)
        run = TestimonySearchModel().simulate(scenario, {}, random.Random(3))
        self.assertEqual(run.outcome["selected_action"], "search_drawer")
        self.assertEqual(
            dict(run.outcome["epistemic_basis"]),
            {
                "kind": "direct_perception",
                "target_object": "key",
                "resolved_location": "drawer",
                "supporting_id": "e1",
                "source_agent": "bob",
            },
        )

    def test_private_location_without_matching_action_fails_closed(self):
        entities = StoryEntitiesV1(
            agents=self.stale.entities.agents,
            objects=self.stale.entities.objects,
            locations=("drawer", "box", "shelf"),
        )
        decision = SearchDecisionV1(
            id=self.stale.decision.id,
            time=self.stale.decision.time,
            actor=self.stale.decision.actor,
            object=self.stale.decision.object,
            actions=(
                SearchActionV1(id="search_box", location="box"),
                SearchActionV1(id="search_shelf", location="shelf"),
            ),
        )
        story = NarrativeScenarioV2(
            entities=entities,
            events=self.stale.events,
            observations=self.stale.observations,
            reports=self.stale.reports,
            receptions=self.stale.receptions,
            decision=decision,
        )
        payload = story.to_payload()
        digest = stable_content_hash(payload).removeprefix("sha256:")
        scenario = Scenario(id=f"story-v2-{digest}", payload=payload)
        with self.assertRaisesRegex(ValueError, "exactly one decision action"):
            TestimonySearchModel().simulate(scenario, {}, random.Random(5))

    def test_three_model_contrast_keeps_v1_models_direct_and_objective_only(self):
        belief = AgentBeliefSearchModel()
        omniscient = OmniscientSearchModel()
        for case in (self.truthful, self.stale):
            scenario = _direct_v1_view(case)
            belief_run = belief.simulate(scenario, {}, random.Random(13))
            omniscient_run = omniscient.simulate(scenario, {}, random.Random(13))
            self.assertEqual(belief_run.outcome["selected_action"], "search_drawer")
            self.assertEqual(omniscient_run.outcome["selected_action"], "search_box")

    def test_testimony_adapter_has_no_prison_dependency(self):
        source = inspect.getsource(__import__(
            "narrative_dynamics.adapters.story_testimony_search",
            fromlist=["TestimonySearchModel"],
        ))
        self.assertNotIn("prison", source.lower())


if __name__ == "__main__":
    unittest.main()
