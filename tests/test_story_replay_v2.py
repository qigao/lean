from __future__ import annotations

import unittest

from narrative_dynamics.story.replay import (
    latest_object_location,
    objective_state,
    subjective_state,
)
from narrative_dynamics.story.replay_v2 import (
    EpistemicLocationStateV2,
    latest_epistemic_location,
    testimony_state,
)
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2
from narrative_dynamics.story.schema_v2 import (
    ReportReceptionV2,
    load_narrative_case_v2,
)


_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"


class TestimonyReplayV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        truthful = load_narrative_case_v2(_TRUTHFUL)
        stale = load_narrative_case_v2(_STALE)
        self.truthful = NarrativeScenarioV2.from_case(truthful)
        self.stale = NarrativeScenarioV2.from_case(stale)

    def test_truthful_received_report_updates_bob_to_box(self):
        state = latest_epistemic_location(
            testimony_state(self.truthful, "bob"), "key"
        )
        self.assertEqual(
            (
                state.location,
                state.evidence_kind,
                state.supporting_id,
                state.source_agent,
                state.logical_time,
            ),
            ("box", "testimony", "r1", "alice", 3),
        )

    def test_stale_received_report_updates_bob_to_drawer(self):
        state = latest_epistemic_location(
            testimony_state(self.stale, "bob"), "key"
        )
        self.assertEqual(
            (
                state.location,
                state.evidence_kind,
                state.supporting_id,
                state.source_agent,
                state.logical_time,
            ),
            ("drawer", "testimony", "r1", "alice", 3),
        )

    def test_v1_direct_and_objective_replay_remain_distinct(self):
        for story in (self.truthful, self.stale):
            direct = latest_object_location(
                subjective_state(story.events, story.observations, "bob"),
                "key",
            )
            objective = latest_object_location(objective_state(story.events), "key")
            self.assertEqual(direct.location, "drawer")
            self.assertEqual(direct.supporting_event_id, "e1")
            self.assertEqual(objective.location, "box")
            self.assertEqual(objective.supporting_event_id, "e2")

    def test_unreceived_report_leaves_bob_on_direct_perception(self):
        story = NarrativeScenarioV2(
            entities=self.truthful.entities,
            events=self.truthful.events,
            observations=self.truthful.observations,
            reports=self.truthful.reports,
            receptions=(),
            decision=self.truthful.decision,
        )
        state = latest_epistemic_location(testimony_state(story, "bob"), "key")
        self.assertEqual(
            (
                state.location,
                state.evidence_kind,
                state.supporting_id,
                state.source_agent,
                state.logical_time,
            ),
            ("drawer", "direct_perception", "e1", "bob", 1),
        )

    def test_time_cutoff_excludes_later_testimony_until_received_time(self):
        at_two = latest_epistemic_location(
            testimony_state(self.truthful, "bob", at_time=2), "key"
        )
        at_three = latest_epistemic_location(
            testimony_state(self.truthful, "bob", at_time=3), "key"
        )
        self.assertEqual(
            (at_two.location, at_two.evidence_kind, at_two.supporting_id),
            ("drawer", "direct_perception", "e1"),
        )
        self.assertEqual(
            (at_three.location, at_three.evidence_kind, at_three.supporting_id),
            ("box", "testimony", "r1"),
        )

    def test_report_received_by_another_agent_does_not_update_bob(self):
        story = NarrativeScenarioV2(
            entities=self.truthful.entities,
            events=self.truthful.events,
            observations=self.truthful.observations,
            reports=self.truthful.reports,
            receptions=(
                ReportReceptionV2(report="r1", recipient="alice"),
            ),
            decision=self.truthful.decision,
        )
        state = latest_epistemic_location(testimony_state(story, "bob"), "key")
        self.assertEqual(
            (state.location, state.evidence_kind, state.supporting_id),
            ("drawer", "direct_perception", "e1"),
        )

    def test_latest_epistemic_location_fails_closed_without_support(self):
        with self.assertRaisesRegex(ValueError, "no supported testimony-aware location"):
            latest_epistemic_location({}, "key")
        with self.assertRaisesRegex(TypeError, "EpistemicLocationStateV2"):
            latest_epistemic_location({"key": object()}, "key")

    def test_invalid_time_agent_and_object_ids_fail_closed(self):
        with self.assertRaisesRegex(TypeError, "at_time"):
            testimony_state(self.truthful, "bob", at_time=True)
        with self.assertRaisesRegex(ValueError, "at_time"):
            testimony_state(self.truthful, "bob", at_time=-1)
        with self.assertRaisesRegex(ValueError, "agent"):
            testimony_state(self.truthful, "", at_time=2)
        with self.assertRaisesRegex(ValueError, "object id"):
            latest_epistemic_location(testimony_state(self.truthful, "bob"), "")

    def test_replay_requires_validated_v2_scenario(self):
        with self.assertRaisesRegex(TypeError, "validated NarrativeScenarioV2"):
            testimony_state(object(), "bob")
        valid = EpistemicLocationStateV2(
            object_id="key",
            location="drawer",
            evidence_kind="direct_perception",
            supporting_id="e1",
            source_agent="bob",
            logical_time=1,
        )
        self.assertIs(latest_epistemic_location({"key": valid}, "key"), valid)


if __name__ == "__main__":
    unittest.main()
