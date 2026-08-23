from __future__ import annotations

from dataclasses import replace
import json
import unittest

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.story.scenario import (
    NarrativeScenarioV1,
    decode_narrative_scenario,
    project_narrative_scenario,
)
from narrative_dynamics.story.schema import NarrativeOracleV1, load_narrative_case


def scenario_from_payload(payload: dict[str, object]) -> Scenario:
    digest = stable_content_hash(payload).removeprefix("sha256:")
    return Scenario(id=f"story-v1-{digest}", payload=payload)


class NarrativeScenarioProjectionTests(unittest.TestCase):
    def setUp(self):
        self.case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )

    def test_projection_contains_only_model_visible_semantics(self):
        scenario = project_narrative_scenario(self.case)
        self.assertEqual(
            set(scenario.payload),
            {"entities", "events", "observations", "decision"},
        )
        serialized = json.dumps(
            NarrativeScenarioV1.from_case(self.case).to_payload(),
            sort_keys=True,
        ).lower()
        for forbidden in (
            "oracle",
            "source_text",
            "false-belief",
            "population_representative",
            "agent_belief_ranking",
            "omniscient_ranking",
            "observed_choice",
        ):
            self.assertNotIn(forbidden, serialized)
            self.assertNotIn(forbidden, scenario.id.lower())

    def test_runtime_id_is_derived_only_from_visible_payload(self):
        scenario = project_narrative_scenario(self.case)
        digest = stable_content_hash(scenario.payload).removeprefix("sha256:")
        self.assertEqual(scenario.id, f"story-v1-{digest}")

    def test_nonsemantic_fixture_metadata_cannot_change_projection(self):
        alternate = replace(
            self.case,
            name="opaque-case-name",
            version="9.9.9",
            source={"kind": "alternate-authoring-source"},
            provenance={"synthetic": True, "note": "different metadata"},
            source_text="Completely different prose that is not model visible.",
            oracle=NarrativeOracleV1(
                objective_location="drawer",
                actor_subjective_location="box",
                agent_belief_ranking=("search_box", "search_drawer"),
                omniscient_ranking=("search_drawer", "search_box"),
            ),
        )
        original = project_narrative_scenario(self.case)
        changed = project_narrative_scenario(alternate)
        self.assertEqual(original.id, changed.id)
        self.assertEqual(dict(original.payload), dict(changed.payload))

    def test_decoder_roundtrips_projected_scenario(self):
        scenario = project_narrative_scenario(self.case)
        decoded = decode_narrative_scenario(scenario)
        self.assertEqual(decoded, NarrativeScenarioV1.from_case(self.case))
        self.assertEqual(decoded.to_payload(), dict(scenario.payload))

    def test_decoder_rejects_non_neutral_scenario_id(self):
        projected = project_narrative_scenario(self.case)
        disguised = Scenario(id="key-location-false-belief", payload=projected.payload)
        with self.assertRaisesRegex(ValueError, "scenario id"):
            decode_narrative_scenario(disguised)

    def test_decoder_rejects_extra_or_missing_payload_keys(self):
        payload = NarrativeScenarioV1.from_case(self.case).to_payload()
        extra = dict(payload)
        extra["oracle"] = {"selected_action": "search_drawer"}
        with self.assertRaisesRegex(ValueError, "payload keys"):
            decode_narrative_scenario(scenario_from_payload(extra))

        missing = dict(payload)
        del missing["observations"]
        with self.assertRaisesRegex(ValueError, "payload keys"):
            decode_narrative_scenario(scenario_from_payload(missing))

    def test_decoder_rejects_unsupported_event_and_observation_semantics(self):
        payload = NarrativeScenarioV1.from_case(self.case).to_payload()
        payload["events"][0]["kind"] = "teleport_object"
        with self.assertRaisesRegex(ValueError, "event kind"):
            decode_narrative_scenario(scenario_from_payload(payload))

        payload = NarrativeScenarioV1.from_case(self.case).to_payload()
        payload["observations"][0]["channel"] = "hearsay"
        with self.assertRaisesRegex(ValueError, "channel"):
            decode_narrative_scenario(scenario_from_payload(payload))

    def test_decoder_reuses_cross_field_story_validation(self):
        payload = NarrativeScenarioV1.from_case(self.case).to_payload()
        payload["events"][1]["from_location"] = "box"
        with self.assertRaisesRegex(ValueError, "from_location"):
            decode_narrative_scenario(scenario_from_payload(payload))

        payload = NarrativeScenarioV1.from_case(self.case).to_payload()
        payload["decision"]["actor"] = "carol"
        with self.assertRaisesRegex(ValueError, "decision actor"):
            decode_narrative_scenario(scenario_from_payload(payload))


if __name__ == "__main__":
    unittest.main()
