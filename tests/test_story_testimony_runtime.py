from __future__ import annotations

import inspect
import unittest

import narrative_dynamics
import narrative_dynamics.adapters.story_belief_search as belief_module
import narrative_dynamics.adapters.story_omniscient_search as omniscient_module
import narrative_dynamics.adapters.story_testimony_search as testimony_module
import narrative_dynamics.story as story
from narrative_dynamics.adapters.story_testimony_search import TestimonySearchModel
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.story.metrics import story_choice_metrics
from narrative_dynamics.story.scenario_v2 import project_testimony_scenario
from narrative_dynamics.story.schema_v2 import load_narrative_case_v2


_V1_EXPORTS = {
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

_V2_EXPORTS = {
    "LocationReportV2",
    "ReportReceptionV2",
    "NarrativeOracleV2",
    "NarrativeCaseV2",
    "load_narrative_case_v2",
    "NarrativeScenarioV2",
    "project_testimony_scenario",
    "decode_testimony_scenario",
    "EpistemicLocationStateV2",
    "testimony_state",
    "latest_epistemic_location",
}


class TestimonyRuntimeTests(unittest.TestCase):
    def test_testimony_model_runs_through_simulation_runner(self):
        case = load_narrative_case_v2(
            "fixtures/stories/key_location_truthful_testimony_v2.json"
        )
        scenario = project_testimony_scenario(case)
        traces = SimulationRunner().run_batch(
            TestimonySearchModel(), scenario, {}, seeds=(101, 102)
        )

        self.assertEqual(
            [trace.outcome["selected_action"] for trace in traces],
            ["search_box", "search_box"],
        )
        for trace in traces:
            self.assertEqual(trace.parameters, ())
            self.assertIsNotNone(trace.manifest)
            self.assertEqual(trace.scenario_id, scenario.id)

    def test_common_story_metrics_accept_testimony_trace(self):
        case = load_narrative_case_v2(
            "fixtures/stories/key_location_stale_testimony_v2.json"
        )
        trace = SimulationRunner().run_once(
            TestimonySearchModel(),
            project_testimony_scenario(case),
            {},
            seed=1,
        )

        self.assertEqual(
            story_choice_metrics(trace),
            {
                "choice.search_box": 0.0,
                "choice.search_drawer": 1.0,
            },
        )

    def test_deterministic_testimony_model_replays_across_seeds(self):
        case = load_narrative_case_v2(
            "fixtures/stories/key_location_truthful_testimony_v2.json"
        )
        scenario = project_testimony_scenario(case)
        runner = SimulationRunner()

        a = runner.run_once(TestimonySearchModel(), scenario, {}, seed=7)
        b = runner.run_once(TestimonySearchModel(), scenario, {}, seed=999)

        self.assertEqual(a.outcome, b.outcome)
        self.assertNotEqual(a.seed, b.seed)

    def test_story_package_exports_exact_v1_plus_v2_canonical_surface(self):
        expected = _V1_EXPORTS | _V2_EXPORTS
        self.assertEqual(set(getattr(story, "__all__", ())), expected)
        for name in expected:
            self.assertTrue(hasattr(story, name), name)

    def test_v2_surface_and_model_do_not_leak_to_package_root(self):
        for name in _V2_EXPORTS:
            self.assertFalse(hasattr(narrative_dynamics, name), name)

        self.assertFalse(hasattr(story, "TestimonySearchModel"))
        self.assertFalse(hasattr(narrative_dynamics, "TestimonySearchModel"))

    def test_all_story_adapters_remain_prison_independent(self):
        for module in (belief_module, omniscient_module, testimony_module):
            source = inspect.getsource(module)
            self.assertNotIn("prison_pomdp", source)
            self.assertNotIn("prison_reactive", source)


if __name__ == "__main__":
    unittest.main()
