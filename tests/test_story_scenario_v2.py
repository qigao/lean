from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
import unittest

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.story.scenario import (
    decode_narrative_scenario,
    project_narrative_scenario,
)
from narrative_dynamics.story.scenario_v2 import (
    NarrativeScenarioV2,
    decode_testimony_scenario,
    project_testimony_scenario,
)
from narrative_dynamics.story.schema import load_narrative_case
from narrative_dynamics.story.schema_v2 import (
    NarrativeOracleV2,
    load_narrative_case_v2,
)


_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"
_V1_FALSE = "fixtures/stories/key_location_false_belief_v1.json"


def _mutable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


class NarrativeScenarioProjectionV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = load_narrative_case_v2(_TRUTHFUL)
        self.stale = load_narrative_case_v2(_STALE)

    def mutable_payload(self) -> dict[str, object]:
        payload = _mutable(project_testimony_scenario(self.truthful).payload)
        assert isinstance(payload, dict)
        return payload

    def scenario_from_payload(self, payload: Mapping[str, object]) -> Scenario:
        digest = stable_content_hash(payload).removeprefix("sha256:")
        return Scenario(id=f"story-v2-{digest}", payload=payload)

    def test_projection_contains_exact_v2_visible_keys(self):
        scenario = project_testimony_scenario(self.truthful)
        self.assertEqual(
            set(scenario.payload),
            {"entities", "events", "observations", "reports", "receptions", "decision"},
        )

    def test_runtime_id_is_v2_payload_digest_only(self):
        scenario = project_testimony_scenario(self.truthful)
        digest = stable_content_hash(scenario.payload).removeprefix("sha256:")
        self.assertEqual(scenario.id, f"story-v2-{digest}")

    def test_authored_metadata_and_oracle_cannot_change_projection(self):
        baseline = project_testimony_scenario(self.truthful)
        changed = replace(
            self.truthful,
            name="different-authored-name",
            version="99.0.0",
            source={"kind": "different_source", "language": "fr"},
            provenance={
                "synthetic": False,
                "empirical_human_data": True,
                "population_representative": True,
            },
            source_text="Completely different authored source text.",
            oracle=NarrativeOracleV2(
                objective_location="drawer",
                actor_direct_location="box",
                actor_testimony_location="drawer",
                testimony_ranking=("search_drawer", "search_box"),
                omniscient_ranking=("search_drawer", "search_box"),
            ),
        )
        projected = project_testimony_scenario(changed)
        self.assertEqual(projected.payload, baseline.payload)
        self.assertEqual(projected.id, baseline.id)

    def test_projection_contains_no_authored_or_gold_labels(self):
        scenario = project_testimony_scenario(self.stale)
        serialized = json.dumps(
            _mutable({"id": scenario.id, "payload": scenario.payload}),
            sort_keys=True,
        ).lower()
        for forbidden in (
            "oracle",
            "source_text",
            "truthful",
            "false",
            "stale",
            "correct",
            "gold",
            "agent_belief_ranking",
            "testimony_ranking",
            "omniscient_ranking",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    def test_report_location_is_model_visible_and_changes_runtime_identity(self):
        truthful = project_testimony_scenario(self.truthful)
        changed_case = replace(
            self.truthful,
            reports=(replace(self.truthful.reports[0], location="drawer"),),
        )
        changed = project_testimony_scenario(changed_case)
        self.assertNotEqual(changed.payload, truthful.payload)
        self.assertNotEqual(changed.id, truthful.id)
        self.assertEqual(changed.payload["reports"][0]["location"], "drawer")

    def test_decoder_roundtrips_canonical_v2_scenario(self):
        scenario = project_testimony_scenario(self.truthful)
        decoded = decode_testimony_scenario(scenario)
        self.assertEqual(decoded, NarrativeScenarioV2.from_case(self.truthful))
        self.assertEqual(decoded.to_payload(), scenario.payload)

    def test_decoder_rejects_extra_missing_keys_and_non_neutral_id(self):
        payload = self.mutable_payload()
        payload["extra"] = True
        with self.assertRaisesRegex(ValueError, "payload keys"):
            decode_testimony_scenario(self.scenario_from_payload(payload))

        payload = self.mutable_payload()
        del payload["receptions"]
        with self.assertRaisesRegex(ValueError, "payload keys"):
            decode_testimony_scenario(self.scenario_from_payload(payload))

        canonical = project_testimony_scenario(self.truthful)
        with self.assertRaisesRegex(ValueError, "visible payload digest"):
            decode_testimony_scenario(
                Scenario(id="truthful-story", payload=canonical.payload)
            )

    def test_runtime_decoder_reuses_speaker_support_validation(self):
        payload = self.mutable_payload()
        payload["observations"] = [
            item
            for item in payload["observations"]
            if not (item["event"] == "e2" and item["agent"] == "alice")
        ]
        with self.assertRaisesRegex(ValueError, "speaker must have directly observed"):
            decode_testimony_scenario(self.scenario_from_payload(payload))

    def test_runtime_decoder_reuses_reception_validation(self):
        payload = self.mutable_payload()
        payload["receptions"][0]["report"] = "r404"
        with self.assertRaisesRegex(ValueError, "declared report"):
            decode_testimony_scenario(self.scenario_from_payload(payload))

        payload = self.mutable_payload()
        payload["receptions"].append(dict(payload["receptions"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate report-recipient"):
            decode_testimony_scenario(self.scenario_from_payload(payload))

    def test_runtime_decoder_rejects_report_at_or_after_decision(self):
        payload = self.mutable_payload()
        payload["reports"][0]["logical_time"] = payload["decision"]["time"]
        with self.assertRaisesRegex(ValueError, "before the decision"):
            decode_testimony_scenario(self.scenario_from_payload(payload))

    def test_v1_and_v2_decoders_reject_each_others_payload_shapes(self):
        v2 = project_testimony_scenario(self.truthful)
        with self.assertRaisesRegex(ValueError, "payload keys"):
            decode_narrative_scenario(v2)

        v1 = project_narrative_scenario(load_narrative_case(_V1_FALSE))
        with self.assertRaisesRegex(ValueError, "payload keys"):
            decode_testimony_scenario(v1)


if __name__ == "__main__":
    unittest.main()
