from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
    SituatedWorldEvent,
)
from narrative_dynamics.abm.situated_memory import (
    ingest_situated_memory,
    ingest_situated_story,
    list_situated_memories,
)
from narrative_dynamics.abm.situated_cognition import admit_situated_observations
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryRecallCue,
    initialize_situated_memory_cognition,
)
from narrative_dynamics.abm.situated_memory_cognition import recall_situated_memories
from narrative_dynamics.abm.situated_cognition_contracts import SituatedObservationRule
from narrative_dynamics.abm.situated_social_cognition import (
    recall_situated_memories_with_social_trust,
    simulate_situated_social_cognition,
    simulate_situated_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import advance_situated_story, perspective_timeline
from tests.test_network_abm_situated_memory_cognition import (
    agent_model,
    inspection_and_move_story,
    mind,
    recall_model,
    with_telling,
)
from tests.test_network_abm_situated_social_memory_contracts import social_model


def testimony_social_case():
    cognition, story, inspection = inspection_and_move_story()
    story = with_telling(cognition, story, inspection)
    cue = SituatedMemoryRecallCue(
        "heard-restructuring",
        "The restructuring is approved.",
        event_kinds=(SituatedActionKind.TELL,),
        channels=(ObservationChannel.AUDITORY,),
    )
    memory_model = recall_model({"bob": (cue,)})
    model = replace(social_model(), memory_cognitive_model=memory_model)
    cognitive_state = initialize_situated_memory_cognition(memory_model, story)
    social_state = initialize_situated_social_memory(model, cognitive_state)
    return model, story, cognitive_state, social_state


def social_relationship(state, observer, source):
    return next(
        item for item in state.relationships
        if item.observer_agent_id == observer and item.source_agent_id == source
    )


def with_denied_tell_rule(model):
    return replace(
        model,
        observation_rules=model.observation_rules + (
            SituatedObservationRule(
                "tell-denied",
                "denied",
                "tell",
                SituatedActionKind.TELL,
                "told",
                "message",
                "The restructuring is denied.",
            ),
        ),
    )


class SituatedSocialCognitionTests(unittest.TestCase):
    def test_repeated_direct_testimony_updates_belief_only_once(self):
        cognition, story, inspection = inspection_and_move_story()
        story = with_telling(cognition, story, inspection)
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent(
                "tell-direct-again",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is approved.",
                source_event_ids=(inspection.event_id,),
            ),
        ))
        memory_model = recall_model({})
        cognitive_state = initialize_situated_memory_cognition(memory_model, story)
        bob = replace(
            mind(cognitive_state, "bob"),
            observation_floor_round=0,
        )
        direct_testimony = tuple(
            item
            for item in perspective_timeline(story, "bob")
            if item.event.kind is SituatedActionKind.TELL
        )

        result = admit_situated_observations(
            agent_model(memory_model, "bob"),
            bob,
            direct_testimony,
            _claim_topic_by_symbol={"approved": "restructuring"},
            _consolidated_claim_keys=frozenset(),
        )

        self.assertEqual(len(result.admissions), 2)
        self.assertEqual(sum(item.consolidated for item in result.admissions), 1)
        self.assertAlmostEqual(
            result.next_mind.belief.probabilities["approved"],
            0.9,
        )
        cross_round = admit_situated_observations(
            agent_model(memory_model, "bob"),
            bob,
            (direct_testimony[-1],),
            _claim_topic_by_symbol={"approved": "restructuring"},
            _consolidated_claim_keys=frozenset({(
                "alice", "restructuring", "approved",
            )}),
        )
        self.assertTrue(cross_round.admissions[0].consolidated)
        self.assertEqual(cross_round.next_mind.belief, bob.belief)

    def test_direct_testimony_reversal_opens_a_new_claim_revision(self):
        cognition, story, inspection = inspection_and_move_story()
        story = with_telling(cognition, story, inspection)
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent(
                "tell-denied",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is denied.",
                source_event_ids=(inspection.event_id,),
            ),
        ))
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent(
                "tell-approved-again",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is approved.",
                source_event_ids=(inspection.event_id,),
            ),
        ))
        memory_model = recall_model({})
        cognitive_state = initialize_situated_memory_cognition(memory_model, story)
        bob = replace(mind(cognitive_state, "bob"), observation_floor_round=0)
        direct_testimony = tuple(
            item
            for item in perspective_timeline(story, "bob")
            if item.event.kind is SituatedActionKind.TELL
        )

        bob_model = with_denied_tell_rule(agent_model(memory_model, "bob"))
        result = admit_situated_observations(
            bob_model,
            bob,
            direct_testimony,
            _claim_topic_by_symbol={
                "approved": "restructuring",
                "denied": "restructuring",
            },
            _consolidated_claim_keys=frozenset(),
        )

        self.assertEqual(
            [item.consolidated for item in result.admissions],
            [False, False, False],
        )
        self.assertAlmostEqual(
            result.next_mind.belief.probabilities["approved"],
            0.9,
        )

    def test_recalled_testimony_reversal_opens_a_new_claim_revision(self):
        cognition, story, inspection = inspection_and_move_story()
        story = with_telling(cognition, story, inspection)
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent(
                "tell-recalled-denied",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is denied.",
                source_event_ids=(inspection.event_id,),
            ),
        ))
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent(
                "tell-recalled-approved",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is approved.",
                source_event_ids=(inspection.event_id,),
            ),
        ))
        cue = SituatedMemoryRecallCue(
            "restructuring-testimony",
            "restructuring",
            event_kinds=(SituatedActionKind.TELL,),
            channels=(ObservationChannel.AUDITORY,),
        )
        memory_model = recall_model({"bob": (cue,)})
        bob_model = with_denied_tell_rule(agent_model(memory_model, "bob"))
        memory_model = replace(
            memory_model,
            cognitive_model=replace(
                memory_model.cognitive_model,
                agents=tuple(
                    bob_model if item.agent_id == "bob" else item
                    for item in memory_model.cognitive_model.agents
                ),
            ),
        )
        cognitive_state = initialize_situated_memory_cognition(memory_model, story)

        with TemporaryDirectory() as directory:
            database = f"{directory}/memory.sqlite3"
            ingest_situated_story(
                database,
                story,
                "bob",
                memory_model.memory_policy,
            )
            result = recall_situated_memories(
                database,
                memory_model,
                bob_model,
                mind(cognitive_state, "bob"),
                story=story,
                state=cognitive_state,
                _claim_topic_by_symbol={
                    "approved": "restructuring",
                    "denied": "restructuring",
                },
                _consolidated_claim_keys=frozenset(),
            )

        self.assertEqual(len(result.admissions), 3)
        self.assertEqual(
            [item.consolidated for item in result.admissions],
            [False, False, False],
        )

    def test_social_recall_ignores_memory_from_a_divergent_story_branch(self):
        model, story, cognitive_state, social_state = testimony_social_case()
        with TemporaryDirectory() as directory:
            database = f"{directory}/memory.sqlite3"
            ingest_situated_story(
                database,
                story,
                "bob",
                model.memory_cognitive_model.memory_policy,
            )
            original = next(
                item
                for item in list_situated_memories(database, "bob")
                if item.kind is SituatedActionKind.TELL
            )
            alternate_event = SituatedWorldEvent(
                "branch-only-event",
                original.round_index,
                original.sequence,
                "branch-only-tell",
                original.kind,
                original.actor_agent_id,
                original.place_id,
                original.target_id,
                original.success,
                original.outcome,
                original.details,
                original.cause_event_ids,
            )
            alternate = replace(
                original,
                memory_id="branch-only-observation",
                observation_id="branch-only-observation",
                event_id=alternate_event.event_id,
                event_hash=alternate_event.content_hash,
                action_id=alternate_event.action_id,
            )
            ingest_situated_memory(database, "bob", (alternate,))

            result = recall_situated_memories_with_social_trust(
                database,
                model,
                agent_model(model.memory_cognitive_model, "bob"),
                mind(cognitive_state, "bob"),
                story=story,
                cognitive_state=cognitive_state,
                social_state=social_state,
            )

        self.assertEqual(len(result.admissions), 1)
        self.assertNotEqual(result.admissions[0].memory_id, alternate.memory_id)

    def test_repeated_same_source_claim_updates_belief_only_once(self):
        cognition, story, inspection = inspection_and_move_story()
        story = with_telling(cognition, story, inspection)
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent(
                "tell-again",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is approved.",
                source_event_ids=(inspection.event_id,),
            ),
        ))
        cue = SituatedMemoryRecallCue(
            "heard-restructuring",
            "The restructuring is approved.",
            event_kinds=(SituatedActionKind.TELL,),
            channels=(ObservationChannel.AUDITORY,),
        )
        memory_model = recall_model({"bob": (cue,)})
        model = replace(social_model(), memory_cognitive_model=memory_model)
        cognitive_state = initialize_situated_memory_cognition(memory_model, story)
        social_state = initialize_situated_social_memory(model, cognitive_state)

        with TemporaryDirectory() as directory:
            result = simulate_situated_social_cognitive_round(
                f"{directory}/memory.sqlite3",
                model,
                story,
                cognitive_state,
                social_state,
            )

        bob_recall = next(
            item for item in result.recalls if item.prior_mind.agent_id == "bob"
        )
        self.assertEqual(len(bob_recall.admissions), 2)
        self.assertEqual(sum(item.consolidated for item in bob_recall.admissions), 1)
        self.assertAlmostEqual(
            bob_recall.next_mind.belief.probabilities["approved"],
            0.560431654676259,
        )
        self.assertEqual(result.next_social_state.claims[0].support_count, 2)

    def test_half_trusted_auditory_memory_has_literal_tempered_weight(self):
        model, story, cognitive_state, social_state = testimony_social_case()
        with TemporaryDirectory() as directory:
            database = f"{directory}/memory.sqlite3"
            ingest_situated_story(database, story, "bob", model.memory_cognitive_model.memory_policy)
            result = recall_situated_memories_with_social_trust(
                database,
                model,
                agent_model(model.memory_cognitive_model, "bob"),
                mind(cognitive_state, "bob"),
                story=story,
                cognitive_state=cognitive_state,
                social_state=social_state,
            )

        self.assertEqual(len(result.admissions), 1)
        admission = result.admissions[0]
        self.assertAlmostEqual(admission.source_trust, 0.5)
        self.assertAlmostEqual(admission.evidence_weight, 0.2625)
        self.assertAlmostEqual(
            admission.posterior_belief.probabilities["approved"],
            0.560431654676259,
        )

    def test_social_round_projects_recalled_testimony_into_one_directed_claim(self):
        model, story, cognitive_state, social_state = testimony_social_case()
        with TemporaryDirectory() as directory:
            result = simulate_situated_social_cognitive_round(
                f"{directory}/memory.sqlite3",
                model,
                story,
                cognitive_state,
                social_state,
            )

        self.assertEqual(len(result.social_update.admitted_evidence), 1)
        evidence = result.social_update.admitted_evidence[0]
        self.assertEqual(evidence.observer_agent_id, "bob")
        self.assertEqual(evidence.source_agent_id, "alice")
        self.assertEqual(evidence.topic_id, "restructuring")
        self.assertEqual(evidence.symbol_id, "approved")
        self.assertIsNotNone(evidence.memory_id)
        self.assertEqual(len(result.next_social_state.claims), 1)
        claim = result.next_social_state.claims[0]
        self.assertEqual(claim.status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(claim.observer_agent_id, "bob")
        self.assertEqual(claim.source_agent_id, "alice")
        self.assertEqual(result.next_social_state.cognitive_state_hash, result.next_cognitive_state.content_hash)
        self.assertEqual(social_relationship(result.next_social_state, "bob", "alice").trust, 0.5)

    def test_social_trajectory_is_database_path_independent(self):
        model, story, cognitive_state, social_state = testimony_social_case()
        with TemporaryDirectory() as left_directory, TemporaryDirectory() as right_directory:
            left = simulate_situated_social_cognition(
                f"{left_directory}/memory.sqlite3",
                model,
                story,
                cognitive_state,
                social_state,
                round_count=2,
            )
            right = simulate_situated_social_cognition(
                f"{right_directory}/memory.sqlite3",
                model,
                story,
                cognitive_state,
                social_state,
                round_count=2,
            )

        self.assertEqual(left, right)
        self.assertEqual(left.content_hash, right.content_hash)
        self.assertEqual(left.final_social_state.round_index, social_state.round_index + 2)


if __name__ == "__main__":
    unittest.main()
