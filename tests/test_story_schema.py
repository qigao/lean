from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from narrative_dynamics.story.schema import (
    DirectObservationV1,
    NarrativeCaseV1,
    NarrativeOracleV1,
    RelocationEventV1,
    SearchActionV1,
    SearchDecisionV1,
    StoryEntitiesV1,
    load_narrative_case,
)


FALSE_TEXT = (
    "Alice and Bob are in a room. While Bob is watching, Alice puts a key "
    "in the drawer. Bob leaves the room. While Bob is away, Alice moves the "
    "key from the drawer to the box. Bob returns and wants to find the key. "
    "Where will Bob search first: the drawer or the box?"
)


def make_false_belief_case() -> NarrativeCaseV1:
    return NarrativeCaseV1(
        name="key-location-false-belief",
        version="1.0.0",
        source={"kind": "authored_microstory", "language": "en"},
        provenance={
            "synthetic": True,
            "empirical_human_data": False,
            "population_representative": False,
        },
        source_text=FALSE_TEXT,
        entities=StoryEntitiesV1(
            agents=("alice", "bob"),
            objects=("key",),
            locations=("drawer", "box"),
        ),
        events=(
            RelocationEventV1(
                id="e1",
                logical_time=1,
                actor="alice",
                object="key",
                from_location=None,
                to_location="drawer",
            ),
            RelocationEventV1(
                id="e2",
                logical_time=2,
                actor="alice",
                object="key",
                from_location="drawer",
                to_location="box",
            ),
        ),
        observations=(DirectObservationV1(event="e1", agent="bob"),),
        decision=SearchDecisionV1(
            id="d1",
            time=3,
            actor="bob",
            object="key",
            actions=(
                SearchActionV1(id="search_drawer", location="drawer"),
                SearchActionV1(id="search_box", location="box"),
            ),
        ),
        oracle=NarrativeOracleV1(
            objective_location="box",
            actor_subjective_location="drawer",
            agent_belief_ranking=("search_drawer", "search_box"),
            omniscient_ranking=("search_box", "search_drawer"),
        ),
    )


class NarrativeStorySchemaTests(unittest.TestCase):
    def payload(self) -> dict[str, object]:
        return make_false_belief_case().to_dict(include_content_hash=False)

    def test_case_is_immutable_roundtrippable_and_hash_stable(self):
        case = make_false_belief_case()
        encoded = case.to_json()
        loaded = NarrativeCaseV1.from_dict(json.loads(encoded))
        self.assertEqual(loaded, case)
        self.assertEqual(loaded.content_hash, case.content_hash)
        self.assertEqual(loaded.to_dict(), json.loads(encoded))
        with self.assertRaises((FrozenInstanceError, TypeError, AttributeError)):
            case.name = "changed"
        with self.assertRaises(TypeError):
            case.source["kind"] = "changed"

    def test_declared_hash_is_verified(self):
        payload = make_false_belief_case().to_dict()
        payload["content_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash"):
            NarrativeCaseV1.from_dict(payload)

    def test_schema_version_and_exact_keys_fail_closed(self):
        payload = self.payload()
        payload["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema version"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "keys"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_duplicate_entity_and_event_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "agents.*unique"):
            StoryEntitiesV1(
                agents=("alice", "alice"),
                objects=("key",),
                locations=("drawer", "box"),
            )

        payload = self.payload()
        payload["events"][1]["id"] = "e1"
        with self.assertRaisesRegex(ValueError, "event.*unique"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_undeclared_event_references_are_rejected(self):
        for field, value, pattern in (
            ("actor", "carol", "actor"),
            ("object", "coin", "object"),
            ("to_location", "shelf", "location"),
        ):
            with self.subTest(field=field):
                payload = self.payload()
                payload["events"][0][field] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_event_time_must_be_strictly_increasing(self):
        payload = self.payload()
        payload["events"][1]["logical_time"] = 1
        with self.assertRaisesRegex(ValueError, "time"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_first_relocation_must_start_from_unknown(self):
        payload = self.payload()
        payload["events"][0]["from_location"] = "drawer"
        with self.assertRaisesRegex(ValueError, "from_location"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_objective_history_continuity_is_enforced(self):
        payload = self.payload()
        payload["events"][1]["from_location"] = "box"
        with self.assertRaisesRegex(ValueError, "from_location"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_observation_references_duplicates_and_channel_fail_closed(self):
        payload = self.payload()
        payload["observations"][0]["event"] = "missing"
        with self.assertRaisesRegex(ValueError, "observation.*event"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["observations"][0]["agent"] = "carol"
        with self.assertRaisesRegex(ValueError, "observation.*agent"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["observations"].append(dict(payload["observations"][0]))
        with self.assertRaisesRegex(ValueError, "observation.*duplicate"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["observations"][0]["channel"] = "hearsay"
        with self.assertRaisesRegex(ValueError, "channel"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_decision_must_follow_events_and_resolve_entities(self):
        payload = self.payload()
        payload["decision"]["time"] = 2
        with self.assertRaisesRegex(ValueError, "decision.*time"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        for field, value, pattern in (
            ("actor", "carol", "decision.*actor"),
            ("object", "coin", "decision.*object"),
        ):
            with self.subTest(field=field):
                payload = self.payload()
                payload["decision"][field] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_decision_actions_must_be_unique_declared_and_nontrivial(self):
        payload = self.payload()
        payload["decision"]["actions"][1]["id"] = "search_drawer"
        with self.assertRaisesRegex(ValueError, "action.*unique"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["decision"]["actions"][1]["location"] = "drawer"
        with self.assertRaisesRegex(ValueError, "action.*location.*unique"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["decision"]["actions"][1]["location"] = "shelf"
        with self.assertRaisesRegex(ValueError, "action.*location"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["decision"]["actions"] = payload["decision"]["actions"][:1]
        with self.assertRaisesRegex(ValueError, "at least two"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_decision_actor_must_have_observed_target_location(self):
        payload = self.payload()
        payload["observations"] = []
        with self.assertRaisesRegex(ValueError, "decision actor.*observed"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_oracle_must_reference_declared_locations_and_actions(self):
        payload = self.payload()
        payload["oracle"]["objective_location"] = "shelf"
        with self.assertRaisesRegex(ValueError, "oracle.*location"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

        payload = self.payload()
        payload["oracle"]["agent_belief_ranking"] = ["search_unknown", "search_box"]
        with self.assertRaisesRegex(ValueError, "oracle.*ranking"):
            NarrativeCaseV1.from_dict(payload, verify_declared_hash=False)

    def test_committed_pair_loads_and_has_identical_objective_events(self):
        false_case = load_narrative_case(
            Path("fixtures/stories/key_location_false_belief_v1.json")
        )
        informed_case = load_narrative_case(
            Path("fixtures/stories/key_location_informed_v1.json")
        )
        self.assertEqual(false_case.events, informed_case.events)
        self.assertNotEqual(false_case.observations, informed_case.observations)
        self.assertFalse(false_case.provenance["empirical_human_data"])
        self.assertFalse(false_case.provenance["population_representative"])
        self.assertEqual(false_case.oracle.actor_subjective_location, "drawer")
        self.assertEqual(informed_case.oracle.actor_subjective_location, "box")


if __name__ == "__main__":
    unittest.main()
