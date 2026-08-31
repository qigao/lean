import json
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_grounding import (
    build_situated_grounding_prompt,
    parse_situated_grounding_retrieval_plan,
)
from narrative_dynamics.abm.situated_grounding_contracts import (
    SituatedGroundingPredicate,
    SituatedGroundingProviderIdentity,
    SituatedGroundingRequest,
    SituatedSemanticGroundingModel,
)
from narrative_dynamics.abm.situated_percept_memory import (
    ingest_situated_percept_story,
    list_situated_percept_memories,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import SECRET, perception_model
from tests.test_network_abm_situated_percept_memory import private_story
from tests.test_network_abm_situated_percept_memory_cognition import recall_model
from tests.test_network_abm_situated_percept_social_cognition import bound_social_model


def grounding_model(perception) -> SituatedSemanticGroundingModel:
    cognition = cognitive_office_model()
    memory = recall_model({}, cognition=cognition, perception=perception)
    social = bound_social_model(memory)
    return SituatedSemanticGroundingModel(
        "office-grounding",
        "1",
        memory,
        social,
        ("alice", "bob", "memo"),
        (SituatedGroundingPredicate(
            "restructuring-status",
            ("approved", "denied"),
            "restructuring",
        ),),
    )


class FixtureProvider:
    identity = SituatedGroundingProviderIdentity("fixture", "1", "json-fixture")

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        return json.dumps(self.response, ensure_ascii=False)


class SituatedPrivateGroundingContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.database = f"{self.temporary.name}/grounding.sqlite3"
        self.perception = perception_model()
        self.story = private_story(self.perception, door_open=True)
        for agent_id in ("alice", "bob"):
            ingest_situated_percept_story(
                self.database, self.perception, self.story, agent_id
            )
        self.model = grounding_model(self.perception)

    def tearDown(self):
        self.temporary.cleanup()

    def _primary(self, agent_id="bob"):
        return next(
            item
            for item in list_situated_percept_memories(self.database, agent_id)
            if item.kind is SituatedActionKind.TELL
        )

    def test_private_fts_context_is_agent_scoped_and_planner_cannot_supply_rows(self):
        primary = self._primary()
        planner = FixtureProvider({
            "terms": ["restructuring approved", "approved"],
            "actor_agent_id": "alice",
            "place_id": None,
            "fidelities": ["exact"],
            "event_kinds": ["tell"],
            "channels": ["auditory"],
            "min_round": 1,
            "max_round": 9,
        })
        request = SituatedGroundingRequest(
            "ground-bob-1", "bob", primary.memory_id, "What did Alice mean?", 5
        )

        prompt = build_situated_grounding_prompt(
            self.database, self.model, request, retrieval_planner=planner
        )

        self.assertEqual(len(planner.calls), 1)
        task, planner_payload = planner.calls[0]
        self.assertEqual(task, "situated_memory_retrieval_plan_v1")
        self.assertEqual(planner_payload["query"], "What did Alice mean?")
        self.assertNotIn("evidence", planner_payload)
        self.assertNotIn("memory_ids", json.dumps(planner_payload))
        self.assertTrue(prompt.evidence)
        self.assertTrue(all(item.agent_id == "bob" for item in prompt.evidence))
        self.assertEqual(prompt.primary_evidence_id, primary.memory_id)
        alice_private_events = {
            item.source_event_id
            for item in list_situated_percept_memories(self.database, "alice")
            if item.kind is SituatedActionKind.INSPECT
        }
        self.assertFalse(alice_private_events & {item.source_event_id for item in prompt.evidence})

    def test_retrieval_plan_is_strict_bounded_and_uses_only_allowlisted_filters(self):
        request = SituatedGroundingRequest(
            "request", "bob", self._primary().memory_id, "meeting", 5
        )
        valid = json.dumps({
            "terms": ["meeting", "conversation"],
            "actor_agent_id": "alice",
            "place_id": "records",
            "fidelities": ["exact"],
            "event_kinds": ["tell"],
            "channels": ["auditory"],
            "min_round": None,
            "max_round": None,
        })
        plan = parse_situated_grounding_retrieval_plan(self.model, request, valid)
        self.assertEqual(plan.terms, ("conversation", "meeting"))
        self.assertEqual(plan.actor_agent_id, "alice")

        unknown = json.loads(valid)
        unknown["memory_ids"] = [self._primary().memory_id]
        with self.assertRaisesRegex(ValueError, "exact schema"):
            parse_situated_grounding_retrieval_plan(
                self.model, request, json.dumps(unknown)
            )
        unknown = json.loads(valid)
        unknown["actor_agent_id"] = "invented"
        with self.assertRaisesRegex(ValueError, "actor"):
            parse_situated_grounding_retrieval_plan(
                self.model, request, json.dumps(unknown)
            )
        unknown = json.loads(valid)
        unknown["terms"] = ["a", "b", "c", "d", "e"]
        with self.assertRaisesRegex(ValueError, "at most"):
            parse_situated_grounding_retrieval_plan(
                self.model, request, json.dumps(unknown)
            )

    def test_detected_memory_prompt_contains_no_secret_or_invented_detail(self):
        database = f"{self.temporary.name}/detected.sqlite3"
        perception = perception_model()
        story = private_story(perception, door_open=False)
        ingest_situated_percept_story(database, perception, story, "bob")
        model = grounding_model(perception)
        primary = next(
            item
            for item in list_situated_percept_memories(database, "bob")
            if item.fidelity is SituatedPerceptFidelity.DETECTED
        )
        prompt = build_situated_grounding_prompt(
            database,
            model,
            SituatedGroundingRequest(
                "detected", "bob", primary.memory_id, "What happened?", 3
            ),
        )
        payload = json.dumps(prompt.to_provider_payload(), ensure_ascii=False)

        self.assertNotIn(SECRET, payload)
        self.assertNotIn("alice-inspect", payload)
        disclosed = next(item for item in prompt.evidence if item.evidence_id == primary.memory_id)
        self.assertIsNone(disclosed.actor_agent_id)
        self.assertEqual(disclosed.details, ())

    def test_primary_memory_must_belong_to_requesting_agent(self):
        alice = self._primary("alice")
        with self.assertRaisesRegex(ValueError, "primary memory"):
            build_situated_grounding_prompt(
                self.database,
                self.model,
                SituatedGroundingRequest(
                    "cross-agent", "bob", alice.memory_id, "What happened?", 3
                ),
            )


if __name__ == "__main__":
    unittest.main()
