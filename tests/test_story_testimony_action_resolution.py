from __future__ import annotations

import random
import unittest

from narrative_dynamics.adapters.story_testimony_search import TestimonySearchModel
from narrative_dynamics.story.replay_v2 import resolve_testimony_action
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2, project_testimony_scenario
from narrative_dynamics.story.schema import SearchActionV1, SearchDecisionV1, StoryEntitiesV1
from narrative_dynamics.story.schema_v2 import load_narrative_case_v2

_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"


class TestimonyActionResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful_case = load_narrative_case_v2(_TRUTHFUL)
        self.stale_case = load_narrative_case_v2(_STALE)
        self.truthful = NarrativeScenarioV2.from_case(self.truthful_case)
        self.stale = NarrativeScenarioV2.from_case(self.stale_case)

    def test_shared_resolver_matches_committed_testimony_decisions(self):
        self.assertEqual(resolve_testimony_action(self.truthful), "search_box")
        self.assertEqual(resolve_testimony_action(self.stale), "search_drawer")

    def test_shared_resolver_fails_closed_without_exact_action_match(self):
        entities = StoryEntitiesV1(
            agents=self.stale.entities.agents,
            objects=self.stale.entities.objects,
            locations=("drawer", "box", "shelf"),
        )
        decision = SearchDecisionV1(
            id="d1",
            time=4,
            actor="bob",
            object="key",
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
        with self.assertRaisesRegex(ValueError, "exactly one decision action"):
            resolve_testimony_action(story)

    def test_adapter_contract_remains_exact_after_resolver_extraction(self):
        run = TestimonySearchModel().simulate(
            project_testimony_scenario(self.truthful_case), {}, random.Random(7)
        )
        self.assertEqual(
            tuple((event.tick, event.kind, dict(event.data)) for event in run.events),
            (
                (0, "epistemic_basis_resolved", {
                    "kind": "testimony",
                    "target_object": "key",
                    "resolved_location": "box",
                    "supporting_id": "r1",
                    "source_agent": "alice",
                }),
                (1, "action_policy_computed", {
                    "policy": {"search_drawer": 0.0, "search_box": 1.0}
                }),
                (2, "action_selected", {"action": "search_box"}),
            ),
        )
        self.assertEqual(
            dict(run.outcome),
            {
                "epistemic_basis": {
                    "kind": "testimony",
                    "target_object": "key",
                    "resolved_location": "box",
                    "supporting_id": "r1",
                    "source_agent": "alice",
                },
                "action_scores": {"search_drawer": 0.0, "search_box": 1.0},
                "policy": {"search_drawer": 0.0, "search_box": 1.0},
                "selected_action": "search_box",
            },
        )


if __name__ == "__main__":
    unittest.main()
