from __future__ import annotations

import inspect
import unittest

import narrative_dynamics
import narrative_dynamics.adapters.story_belief_search as belief_module
import narrative_dynamics.adapters.story_omniscient_search as omniscient_module
import narrative_dynamics.story as story
from narrative_dynamics.adapters.story_belief_search import AgentBeliefSearchModel
from narrative_dynamics.adapters.story_omniscient_search import OmniscientSearchModel
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.story.metrics import story_choice_metrics
from narrative_dynamics.story.scenario import project_narrative_scenario
from narrative_dynamics.story.schema import load_narrative_case


_EXPECTED_STORY_EXPORTS = {
    "StoryEntitiesV1",
    "RelocationEventV1",
    "DirectObservationV1",
    "SearchActionV1",
    "SearchDecisionV1",
    "NarrativeOracleV1",
    "NarrativeCaseV1",
    "load_narrative_case",
    "ObjectLocationState",
    "objective_state",
    "subjective_state",
    "latest_object_location",
    "NarrativeScenarioV1",
    "project_narrative_scenario",
    "decode_narrative_scenario",
    "story_choice_metrics",
}


class StoryRuntimeTests(unittest.TestCase):
    def test_both_models_run_through_simulation_runner(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        scenario = project_narrative_scenario(case)
        runner = SimulationRunner()

        belief = runner.run_batch(
            AgentBeliefSearchModel(), scenario, {}, seeds=(101, 102)
        )
        omniscient = runner.run_batch(
            OmniscientSearchModel(), scenario, {}, seeds=(101, 102)
        )

        self.assertEqual(
            [trace.outcome["selected_action"] for trace in belief],
            ["search_drawer", "search_drawer"],
        )
        self.assertEqual(
            [trace.outcome["selected_action"] for trace in omniscient],
            ["search_box", "search_box"],
        )
        for trace in belief + omniscient:
            self.assertEqual(trace.parameters, ())
            self.assertIsNotNone(trace.manifest)
            self.assertEqual(trace.scenario_id, scenario.id)

    def test_deterministic_model_is_seed_replay_compatible(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )
        scenario = project_narrative_scenario(case)
        runner = SimulationRunner()

        a = runner.run_once(AgentBeliefSearchModel(), scenario, {}, seed=7)
        b = runner.run_once(AgentBeliefSearchModel(), scenario, {}, seed=999)

        self.assertEqual(a.outcome, b.outcome)
        self.assertNotEqual(a.seed, b.seed)

    def test_common_metrics_work_for_both_models(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        scenario = project_narrative_scenario(case)
        runner = SimulationRunner()

        belief = runner.run_once(AgentBeliefSearchModel(), scenario, {}, seed=1)
        omniscient = runner.run_once(OmniscientSearchModel(), scenario, {}, seed=1)

        self.assertEqual(
            story_choice_metrics(belief),
            {
                "choice.search_box": 0.0,
                "choice.search_drawer": 1.0,
            },
        )
        self.assertEqual(
            story_choice_metrics(omniscient),
            {
                "choice.search_box": 1.0,
                "choice.search_drawer": 0.0,
            },
        )

    def test_story_api_is_exported_only_from_story_package(self):
        self.assertEqual(set(getattr(story, "__all__", ())), _EXPECTED_STORY_EXPORTS)
        for name in _EXPECTED_STORY_EXPORTS:
            self.assertTrue(hasattr(story, name), name)
            self.assertFalse(hasattr(narrative_dynamics, name), name)

        self.assertFalse(hasattr(story, "AgentBeliefSearchModel"))
        self.assertFalse(hasattr(story, "OmniscientSearchModel"))

    def test_story_adapters_do_not_reference_prison_adapters(self):
        for module in (belief_module, omniscient_module):
            source = inspect.getsource(module)
            self.assertNotIn("prison_pomdp", source)
            self.assertNotIn("prison_reactive", source)


if __name__ == "__main__":
    unittest.main()
