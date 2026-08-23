from __future__ import annotations

import unittest

from narrative_dynamics.story.replay import (
    latest_object_location,
    objective_state,
    subjective_state,
)
from narrative_dynamics.story.schema import (
    RelocationEventV1,
    load_narrative_case,
)


class StoryReplayTests(unittest.TestCase):
    def setUp(self):
        self.false_case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        self.informed_case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )

    def test_false_belief_has_objective_box_and_subjective_drawer(self):
        objective = latest_object_location(
            objective_state(self.false_case.events), "key"
        )
        subjective = latest_object_location(
            subjective_state(
                self.false_case.events,
                self.false_case.observations,
                "bob",
            ),
            "key",
        )
        self.assertEqual(
            (objective.location, objective.supporting_event_id, objective.logical_time),
            ("box", "e2", 2),
        )
        self.assertEqual(
            (subjective.location, subjective.supporting_event_id, subjective.logical_time),
            ("drawer", "e1", 1),
        )

    def test_informed_observation_updates_subjective_state(self):
        subjective = latest_object_location(
            subjective_state(
                self.informed_case.events,
                self.informed_case.observations,
                "bob",
            ),
            "key",
        )
        self.assertEqual(
            (subjective.location, subjective.supporting_event_id),
            ("box", "e2"),
        )

    def test_unobserved_later_relocation_changes_objective_not_subjective(self):
        e3 = RelocationEventV1(
            id="e3",
            logical_time=4,
            actor="alice",
            object="key",
            from_location="box",
            to_location="drawer",
        )
        events = self.informed_case.events + (e3,)
        objective = latest_object_location(objective_state(events), "key")
        subjective = latest_object_location(
            subjective_state(events, self.informed_case.observations, "bob"),
            "key",
        )
        self.assertEqual((objective.location, objective.supporting_event_id), ("drawer", "e3"))
        self.assertEqual((subjective.location, subjective.supporting_event_id), ("box", "e2"))

    def test_at_time_preserves_historical_location(self):
        early = latest_object_location(
            objective_state(self.false_case.events, at_time=1), "key"
        )
        late = latest_object_location(
            objective_state(self.false_case.events, at_time=2), "key"
        )
        self.assertEqual((early.location, early.supporting_event_id), ("drawer", "e1"))
        self.assertEqual((late.location, late.supporting_event_id), ("box", "e2"))

    def test_missing_supported_location_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no supported location"):
            latest_object_location({}, "key")


if __name__ == "__main__":
    unittest.main()
