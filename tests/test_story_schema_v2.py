from __future__ import annotations

from dataclasses import FrozenInstanceError
import copy
import unittest

from narrative_dynamics.story.scenario import project_narrative_scenario
from narrative_dynamics.story.schema import load_narrative_case
from narrative_dynamics.story.schema_v2 import (
    LocationReportV2,
    NarrativeCaseV2,
    NarrativeOracleV2,
    ReportReceptionV2,
    load_narrative_case_v2,
)


_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"


class NarrativeStorySchemaV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = load_narrative_case_v2(_TRUTHFUL)
        self.stale = load_narrative_case_v2(_STALE)

    def raw_truthful(self) -> dict[str, object]:
        return copy.deepcopy(self.truthful.to_dict())

    def unchecked(self, raw: dict[str, object]) -> NarrativeCaseV2:
        return NarrativeCaseV2.from_dict(raw, verify_declared_hash=False)

    def test_v1_identity_regression_is_exact(self):
        false_case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        informed_case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )
        self.assertEqual(
            false_case.content_hash,
            "sha256:761f019271bfb097d95c7c77e80811496c22ee17089f586a433a1a8c21020420",
        )
        self.assertEqual(
            informed_case.content_hash,
            "sha256:2d63d1cc5a50a059cbcf9f425ba54275474bff4abb74ed37703514e069cff862",
        )
        self.assertEqual(
            project_narrative_scenario(false_case).id,
            "story-v1-3b8036566da885c9f07136e141fe3a03f4d80844341dd6d97d109bcbdad4b38e",
        )
        self.assertEqual(
            project_narrative_scenario(informed_case).id,
            "story-v1-ecc5c96321e624e27ea4f07146875ba4a1878e1afadc09dd1337e169478f5368",
        )

    def test_committed_pair_is_matched_except_report_location(self):
        self.assertEqual(self.truthful.schema_version, 2)
        self.assertEqual(self.stale.schema_version, 2)
        self.assertEqual(self.truthful.entities, self.stale.entities)
        self.assertEqual(self.truthful.events, self.stale.events)
        self.assertEqual(self.truthful.observations, self.stale.observations)
        self.assertEqual(self.truthful.receptions, self.stale.receptions)
        self.assertEqual(self.truthful.decision, self.stale.decision)
        self.assertEqual(len(self.truthful.reports), 1)
        self.assertEqual(len(self.stale.reports), 1)
        truthful_report = self.truthful.reports[0]
        stale_report = self.stale.reports[0]
        self.assertEqual(truthful_report.location, "box")
        self.assertEqual(stale_report.location, "drawer")
        self.assertEqual(
            {k: v for k, v in truthful_report.to_dict().items() if k != "location"},
            {k: v for k, v in stale_report.to_dict().items() if k != "location"},
        )
        self.assertEqual(
            {(item.event, item.agent) for item in self.truthful.observations},
            {("e1", "bob"), ("e2", "alice")},
        )
        self.assertEqual(
            {(item.report, item.recipient) for item in self.truthful.receptions},
            {("r1", "bob")},
        )

    def test_committed_pair_metadata_declares_synthetic_non_empirical_scope(self):
        for case in (self.truthful, self.stale):
            self.assertEqual(case.source["kind"], "authored_microstory")
            self.assertEqual(case.source["language"], "en")
            self.assertIs(case.provenance["synthetic"], True)
            self.assertIs(case.provenance["empirical_human_data"], False)
            self.assertIs(case.provenance["population_representative"], False)

    def test_case_roundtrip_hash_and_frozen_values(self):
        for case in (self.truthful, self.stale):
            raw = case.to_dict()
            rebuilt = NarrativeCaseV2.from_dict(raw)
            self.assertEqual(rebuilt, case)
            self.assertEqual(raw["content_hash"], case.content_hash)
        with self.assertRaises(FrozenInstanceError):
            self.truthful.reports[0].location = "drawer"
        with self.assertRaises(TypeError):
            self.truthful.source["kind"] = "changed"

    def test_fixed_kinds_and_channels_are_strict(self):
        with self.assertRaisesRegex(ValueError, "report kind"):
            LocationReportV2(
                id="r1", logical_time=3, speaker="alice", object="key",
                location="box", support_event="e2", kind="say_location",
            )
        with self.assertRaisesRegex(ValueError, "reception channel"):
            ReportReceptionV2(report="r1", recipient="bob", channel="hearsay")

    def test_duplicate_report_id_is_rejected(self):
        raw = self.raw_truthful()
        raw["reports"].append(copy.deepcopy(raw["reports"][0]))
        with self.assertRaisesRegex(ValueError, "report ids"):
            self.unchecked(raw)

    def test_report_references_must_be_declared(self):
        for field, value, message in (
            ("speaker", "carol", "speaker"),
            ("object", "coin", "object"),
            ("location", "shelf", "location"),
        ):
            raw = self.raw_truthful()
            raw["reports"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    self.unchecked(raw)

    def test_report_support_event_must_exist_and_match_object(self):
        raw = self.raw_truthful()
        raw["reports"][0]["support_event"] = "e404"
        with self.assertRaisesRegex(ValueError, "support_event"):
            self.unchecked(raw)

        raw = self.raw_truthful()
        raw["entities"]["objects"].append("coin")
        raw["reports"][0]["object"] = "coin"
        with self.assertRaisesRegex(ValueError, "reported object"):
            self.unchecked(raw)

    def test_report_support_must_precede_report_and_be_seen_by_speaker(self):
        raw = self.raw_truthful()
        raw["reports"][0]["logical_time"] = 2
        with self.assertRaisesRegex(ValueError, "support_event must occur before"):
            self.unchecked(raw)

        raw = self.raw_truthful()
        raw["observations"] = [
            item for item in raw["observations"]
            if not (item["event"] == "e2" and item["agent"] == "alice")
        ]
        with self.assertRaisesRegex(ValueError, "speaker must have directly observed"):
            self.unchecked(raw)

    def test_report_truth_is_not_required_for_schema_validity(self):
        raw = self.raw_truthful()
        raw["reports"][0]["location"] = "drawer"
        case = self.unchecked(raw)
        self.assertEqual(case.reports[0].support_event, "e2")
        self.assertEqual(case.events[1].to_location, "box")
        self.assertEqual(case.reports[0].location, "drawer")

    def test_reception_references_uniqueness_and_recipient_are_strict(self):
        raw = self.raw_truthful()
        raw["receptions"].append(copy.deepcopy(raw["receptions"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate report-recipient"):
            self.unchecked(raw)

        raw = self.raw_truthful()
        raw["receptions"][0]["report"] = "r404"
        with self.assertRaisesRegex(ValueError, "declared report"):
            self.unchecked(raw)

        raw = self.raw_truthful()
        raw["receptions"][0]["recipient"] = "carol"
        with self.assertRaisesRegex(ValueError, "recipient"):
            self.unchecked(raw)

    def test_relocation_and_report_times_are_globally_unambiguous(self):
        raw = self.raw_truthful()
        raw["observations"].append(
            {"event": "e1", "agent": "alice", "channel": "direct_perception"}
        )
        raw["reports"][0]["support_event"] = "e1"
        raw["reports"][0]["logical_time"] = 2
        with self.assertRaisesRegex(ValueError, "logical times must be unique"):
            self.unchecked(raw)

    def test_report_must_precede_decision(self):
        raw = self.raw_truthful()
        raw["reports"][0]["logical_time"] = 4
        with self.assertRaisesRegex(ValueError, "before the decision"):
            self.unchecked(raw)

    def test_v2_exact_keys_and_schema_version_fail_closed(self):
        raw = self.raw_truthful()
        del raw["receptions"]
        with self.assertRaisesRegex(ValueError, "V2 schema"):
            self.unchecked(raw)

        raw = self.raw_truthful()
        raw["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "V2 schema"):
            self.unchecked(raw)

        raw = self.raw_truthful()
        raw["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema version"):
            self.unchecked(raw)

    def test_declared_hash_is_required_and_verified(self):
        raw = self.raw_truthful()
        del raw["content_hash"]
        with self.assertRaisesRegex(ValueError, "content hash is required"):
            NarrativeCaseV2.from_dict(raw)

        raw = self.raw_truthful()
        raw["content_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash does not match"):
            NarrativeCaseV2.from_dict(raw)

    def test_oracle_only_references_declared_locations_and_exact_actions(self):
        raw = self.raw_truthful()
        raw["oracle"]["actor_testimony_location"] = "shelf"
        with self.assertRaisesRegex(ValueError, "oracle testimony location"):
            self.unchecked(raw)

        with self.assertRaisesRegex(ValueError, "testimony ranking"):
            NarrativeOracleV2(
                objective_location="box",
                actor_direct_location="drawer",
                actor_testimony_location="box",
                testimony_ranking=("search_box", "search_box"),
                omniscient_ranking=("search_box", "search_drawer"),
            )


if __name__ == "__main__":
    unittest.main()
