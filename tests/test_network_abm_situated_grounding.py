from dataclasses import replace
import json
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_grounding import (
    build_situated_grounding_prompt,
    compile_situated_semantic_grounding,
    grounded_claims_to_situated_social_evidence,
    parse_situated_grounding_retrieval_plan,
    replay_situated_semantic_grounding,
)
from narrative_dynamics.abm.situated_grounding_contracts import (
    SituatedGroundingModality,
    SituatedGroundingPolarity,
    SituatedGroundingPredicate,
    SituatedGroundingProviderIdentity,
    SituatedGroundingRequest,
    SituatedGroundingTemporalScope,
    SituatedSemanticGroundingModel,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_memory import (
    ingest_situated_percept_story,
    list_situated_percept_memories,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from narrative_dynamics.abm.situated_social_memory import advance_situated_social_memory
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedSocialEvidenceKind,
    initialize_situated_social_memory,
)
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
            predicate_id="restructuring-status",
            value_ids=("approved", "denied"),
            social_topic_id="restructuring",
            subject_ids=("memo",),
            social_subject_id="memo",
        ),),
    )


class FixtureProvider:
    identity = SituatedGroundingProviderIdentity("fixture", "1", "json-fixture")

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, *, task, payload):
        self.calls.append((task, payload))
        return (
            self.response
            if isinstance(self.response, str)
            else json.dumps(self.response, ensure_ascii=False)
        )


class IdentityChangingProvider(FixtureProvider):
    def complete_json(self, *, task, payload):
        response = super().complete_json(task=task, payload=payload)
        self.identity = SituatedGroundingProviderIdentity(
            "substituted", "9", "different-model"
        )
        return response


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
        self.assertEqual(prompt.retrieval_provider, planner.identity)
        self.assertTrue(prompt.retrieval_response_hash.startswith("sha256:"))
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


class SituatedSemanticGroundingTests(SituatedPrivateGroundingContextTests):
    def _prompt(self, *, agent_id="bob", memory=None):
        primary = self._primary(agent_id) if memory is None else memory
        return build_situated_grounding_prompt(
            self.database,
            self.model,
            SituatedGroundingRequest(
                f"ground-{agent_id}",
                agent_id,
                primary.memory_id,
                "What is the restructuring status?",
                5,
            ),
        )

    @staticmethod
    def _claim_payload(evidence_id, **changes):
        payload = {
            "subject_id": "memo",
            "predicate_id": "restructuring-status",
            "value_id": "approved",
            "polarity": "affirmed",
            "modality": "asserted",
            "temporal_scope": "present",
            "source_agent_id": "alice",
            "confidence": 0.82,
            "evidence_ids": [evidence_id],
        }
        payload.update(changes)
        return payload

    def test_provider_compiles_strict_json_into_replayable_artifact(self):
        prompt = self._prompt()
        provider = FixtureProvider({
            "claims": [self._claim_payload(prompt.primary_evidence_id)]
        })

        artifact = compile_situated_semantic_grounding(prompt, provider)

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][0], "situated_semantic_grounding_v1")
        self.assertEqual(artifact.prompt_hash, prompt.content_hash)
        self.assertEqual(artifact.provider, provider.identity)
        self.assertEqual(artifact.primary_evidence_id, prompt.primary_evidence_id)
        self.assertEqual(artifact.schema_hash, prompt.schema_hash)
        self.assertEqual(artifact.prompt_template_hash, prompt.prompt_template_hash)
        self.assertEqual(artifact.private_context_hash, prompt.private_context_hash)
        self.assertEqual(artifact.validation_result, "accepted")
        accepted = artifact.claims[0]
        self.assertIs(accepted.polarity, SituatedGroundingPolarity.AFFIRMED)
        self.assertIs(accepted.modality, SituatedGroundingModality.ASSERTED)
        self.assertIs(accepted.temporal_scope, SituatedGroundingTemporalScope.PRESENT)
        self.assertIs(replay_situated_semantic_grounding(self.model, artifact), artifact)
        self.assertEqual(len(provider.calls), 1, "replay must not invoke a provider")

        same = compile_situated_semantic_grounding(prompt, provider)
        self.assertEqual(same, artifact)
        self.assertEqual(same.content_hash, artifact.content_hash)

    def test_provider_identity_is_captured_before_the_nondeterministic_call(self):
        prompt = self._prompt()
        initial = SituatedGroundingProviderIdentity("fixture", "1", "json-fixture")
        provider = IdentityChangingProvider({
            "claims": [self._claim_payload(prompt.primary_evidence_id)]
        })

        artifact = compile_situated_semantic_grounding(prompt, provider)

        self.assertEqual(artifact.provider, initial)
        self.assertNotEqual(artifact.provider, provider.identity)

    def test_retrieval_planner_provenance_is_retained_by_the_artifact(self):
        primary = self._primary()
        initial = SituatedGroundingProviderIdentity("fixture", "1", "json-fixture")
        planner = IdentityChangingProvider({
            "terms": ["restructuring"],
            "actor_agent_id": "alice",
            "place_id": None,
            "fidelities": ["exact"],
            "event_kinds": ["tell"],
            "channels": ["auditory"],
            "min_round": None,
            "max_round": None,
        })
        prompt = build_situated_grounding_prompt(
            self.database,
            self.model,
            SituatedGroundingRequest(
                "planner-provenance", "bob", primary.memory_id, "What was said?", 5
            ),
            retrieval_planner=planner,
        )
        artifact = compile_situated_semantic_grounding(
            prompt,
            FixtureProvider({
                "claims": [self._claim_payload(prompt.primary_evidence_id)]
            }),
        )

        self.assertEqual(prompt.retrieval_provider, initial)
        self.assertEqual(artifact.retrieval_provider, initial)
        self.assertEqual(
            artifact.retrieval_response_hash, prompt.retrieval_response_hash
        )

    def test_claim_must_cite_the_primary_interpreted_memory(self):
        inspection = next(
            item
            for item in list_situated_percept_memories(self.database, "alice")
            if item.kind is SituatedActionKind.INSPECT
        )
        request = SituatedGroundingRequest(
            "ground-alice-primary",
            "alice",
            inspection.memory_id,
            "What is the restructuring status?",
            5,
        )
        prompt = build_situated_grounding_prompt(
            self.database,
            self.model,
            request,
            retrieval_planner=FixtureProvider({
                "terms": ["restructuring"],
                "actor_agent_id": "alice",
                "place_id": None,
                "fidelities": ["exact"],
                "event_kinds": [],
                "channels": [],
                "min_round": None,
                "max_round": None,
            }),
        )
        other = next(
            item for item in prompt.evidence
            if item.evidence_id != prompt.primary_evidence_id
            and item.fidelity is SituatedPerceptFidelity.EXACT
        )
        with self.assertRaisesRegex(ValueError, "primary evidence"):
            compile_situated_semantic_grounding(
                prompt,
                FixtureProvider({"claims": [self._claim_payload(
                    other.evidence_id, source_agent_id="alice"
                )]}),
            )

    def test_unknown_schema_vocabulary_and_unavailable_evidence_fail_closed(self):
        prompt = self._prompt()
        base = self._claim_payload(prompt.primary_evidence_id)
        cases = []
        extra_top = {"claims": [base], "explanation": "trust me"}
        cases.append((extra_top, "exact schema"))
        extra_claim = dict(base, explanation="hidden reasoning")
        cases.append(({"claims": [extra_claim]}, "exact schema"))
        cases.append(({"claims": [dict(base, subject_id="invented")]}, "subject"))
        cases.append(({"claims": [dict(base, predicate_id="invented")]}, "predicate"))
        cases.append(({"claims": [dict(base, value_id="invented")]}, "value"))
        cases.append(({"claims": [dict(base, evidence_ids=["private-to-someone-else"])]}, "evidence"))
        cases.append(({"claims": [dict(base, source_agent_id="carol")]}, "source"))
        for response, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    compile_situated_semantic_grounding(
                        prompt, FixtureProvider(response)
                    )

    def test_malformed_duplicate_or_non_object_provider_json_is_rejected(self):
        prompt = self._prompt()
        responses = (
            "not-json",
            "[]",
            '{"claims": [], "claims": []}',
            '{"claims": NaN}',
        )
        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    compile_situated_semantic_grounding(
                        prompt, FixtureProvider(response)
                    )

    def test_detected_sound_cannot_support_message_semantics(self):
        database = f"{self.temporary.name}/detected-grounding.sqlite3"
        perception = perception_model()
        story = private_story(perception, door_open=False)
        ingest_situated_percept_story(database, perception, story, "bob")
        model = grounding_model(perception)
        memory = next(
            item
            for item in list_situated_percept_memories(database, "bob")
            if item.fidelity is SituatedPerceptFidelity.DETECTED
        )
        prompt = build_situated_grounding_prompt(
            database,
            model,
            SituatedGroundingRequest(
                "detected-claim", "bob", memory.memory_id, "What was said?", 3
            ),
        )

        with self.assertRaisesRegex(ValueError, "fidelity"):
            compile_situated_semantic_grounding(
                prompt,
                FixtureProvider({
                    "claims": [self._claim_payload(prompt.primary_evidence_id)]
                }),
            )

    def test_declared_partial_predicate_can_ground_detected_sound_but_not_v14(self):
        database = f"{self.temporary.name}/partial-grounding.sqlite3"
        perception = perception_model()
        story = private_story(perception, door_open=False)
        ingest_situated_percept_story(database, perception, story, "bob")
        base = grounding_model(perception)
        model = replace(
            base,
            predicates=base.predicates + (SituatedGroundingPredicate(
                predicate_id="audible-event",
                value_ids=("occurred",),
                subject_ids=("bob",),
                minimum_evidence_fidelity=SituatedPerceptFidelity.DETECTED,
                allowed_temporal_scopes=(SituatedGroundingTemporalScope.PRESENT,),
            ),),
        )
        memory = next(
            item
            for item in list_situated_percept_memories(database, "bob")
            if item.fidelity is SituatedPerceptFidelity.DETECTED
        )
        prompt = build_situated_grounding_prompt(
            database,
            model,
            SituatedGroundingRequest(
                "partial-claim", "bob", memory.memory_id, "Did I hear something?", 3
            ),
        )
        response = self._claim_payload(
            prompt.primary_evidence_id,
            subject_id="bob",
            predicate_id="audible-event",
            value_id="occurred",
            source_agent_id=None,
            confidence=0.4,
        )

        artifact = compile_situated_semantic_grounding(
            prompt, FixtureProvider({"claims": [response]})
        )

        self.assertEqual(artifact.claims[0].predicate_id, "audible-event")
        self.assertEqual(
            grounded_claims_to_situated_social_evidence(model, artifact), ()
        )

    def test_asserted_affirmed_tell_claim_bridges_to_v14_testimony(self):
        prompt = self._prompt()
        artifact = compile_situated_semantic_grounding(
            prompt,
            FixtureProvider({
                "claims": [self._claim_payload(prompt.primary_evidence_id)]
            }),
        )

        evidence = grounded_claims_to_situated_social_evidence(self.model, artifact)

        self.assertEqual(len(evidence), 1)
        self.assertIs(evidence[0].kind, SituatedSocialEvidenceKind.TESTIMONY)
        self.assertEqual(evidence[0].observer_agent_id, "bob")
        self.assertEqual(evidence[0].source_agent_id, "alice")
        self.assertEqual(evidence[0].topic_id, "restructuring")
        self.assertEqual(evidence[0].symbol_id, "approved")
        perception_bound_story = replace(
            self.story, perception_model=self.perception
        )
        cognition = initialize_situated_percept_memory_cognition(
            self.model.percept_memory_model, perception_bound_story
        )
        social = initialize_situated_social_memory(
            self.model.social_memory_model, cognition
        )
        update = advance_situated_social_memory(
            self.model.social_memory_model,
            cognition,
            social,
            cognition,
            evidence,
        )
        self.assertEqual(update.next_state.claims[0].symbol_id, "approved")

    def test_bridge_deduplicates_same_event_and_rejects_conflicting_symbols(self):
        prompt = self._prompt()
        duplicate = compile_situated_semantic_grounding(
            prompt,
            FixtureProvider({"claims": [
                self._claim_payload(prompt.primary_evidence_id, confidence=0.6),
                self._claim_payload(prompt.primary_evidence_id, confidence=0.9),
            ]}),
        )
        duplicate_evidence = grounded_claims_to_situated_social_evidence(
            self.model, duplicate
        )
        self.assertEqual(len(duplicate_evidence), 1)

        single = compile_situated_semantic_grounding(
            prompt,
            FixtureProvider({
                "claims": [self._claim_payload(prompt.primary_evidence_id)]
            }),
        )
        self.assertEqual(
            duplicate_evidence[0].evidence_id,
            grounded_claims_to_situated_social_evidence(self.model, single)[0].evidence_id,
        )

        conflicting = compile_situated_semantic_grounding(
            prompt,
            FixtureProvider({"claims": [
                self._claim_payload(prompt.primary_evidence_id, value_id="approved"),
                self._claim_payload(prompt.primary_evidence_id, value_id="denied"),
            ]}),
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            grounded_claims_to_situated_social_evidence(self.model, conflicting)

    def test_unsupported_future_claim_does_not_enter_timeless_v14(self):
        prompt = self._prompt()
        with self.assertRaisesRegex(ValueError, "temporal"):
            compile_situated_semantic_grounding(
                prompt,
                FixtureProvider({"claims": [self._claim_payload(
                    prompt.primary_evidence_id, temporal_scope="future"
                )]}),
            )

    def test_exact_inspection_bridges_to_source_less_verification(self):
        inspection = next(
            item
            for item in list_situated_percept_memories(self.database, "alice")
            if item.kind is SituatedActionKind.INSPECT
        )
        prompt = self._prompt(agent_id="alice", memory=inspection)
        artifact = compile_situated_semantic_grounding(
            prompt,
            FixtureProvider({"claims": [self._claim_payload(
                prompt.primary_evidence_id, source_agent_id=None
            )]}),
        )

        evidence = grounded_claims_to_situated_social_evidence(self.model, artifact)

        self.assertEqual(len(evidence), 1)
        self.assertIs(evidence[0].kind, SituatedSocialEvidenceKind.VERIFICATION)
        self.assertEqual(evidence[0].observer_agent_id, "alice")
        self.assertIsNone(evidence[0].source_agent_id)

    def test_non_asserted_or_negative_claim_stays_grounded_but_does_not_enter_v14(self):
        prompt = self._prompt()
        responses = (
            self._claim_payload(prompt.primary_evidence_id, modality="uncertain"),
            self._claim_payload(prompt.primary_evidence_id, polarity="denied"),
        )
        for response in responses:
            artifact = compile_situated_semantic_grounding(
                prompt, FixtureProvider({"claims": [response]})
            )
            self.assertEqual(
                grounded_claims_to_situated_social_evidence(self.model, artifact),
                (),
            )


if __name__ == "__main__":
    unittest.main()
