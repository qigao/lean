from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
)
from narrative_dynamics.abm.situated_memory import (
    ingest_situated_memory,
    ingest_situated_story,
    list_situated_memories,
    set_situated_memory_active,
)
from narrative_dynamics.abm.situated_memory_cognition import recall_situated_memories
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
    SituatedMemoryCognitiveModel,
    SituatedMemoryRecallCue,
    initialize_situated_memory_cognition,
)
from narrative_dynamics.abm.situated_memory_contracts import standard_situated_memory_policy
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from tests.situated_cognition_fixtures import cognitive_office_model


OTHER_WORLD_HASH = "sha256:" + "d" * 64


def recall_model(cues_by_agent):
    cognition = cognitive_office_model()
    policies = tuple(
        SituatedAgentRecallPolicy(
            agent.agent_id,
            tuple(cues_by_agent.get(agent.agent_id, ())),
            max_memories_per_round=5,
        )
        for agent in cognition.agents
    )
    return SituatedMemoryCognitiveModel(
        "office-memory-cognition",
        "1",
        cognition,
        standard_situated_memory_policy(),
        policies,
    )


def inspection_and_move_story():
    cognition = cognitive_office_model()
    story = initialize_situated_story(
        cognition.world_model,
        initialize_situated_world(cognition.world_model),
    )
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("inspect", "alice", SituatedActionKind.INSPECT, "memo"),
    ))
    inspection = next(
        item for item in story.rounds[-1].events if item.actor_agent_id == "alice"
    )
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("move", "alice", SituatedActionKind.MOVE, "records-open"),
    ))
    return cognition, story, inspection


def with_telling(cognition, story, inspection):
    return advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent(
            "tell",
            "alice",
            SituatedActionKind.TELL,
            message="The restructuring is approved.",
            source_event_ids=(inspection.event_id,),
        ),
    ))


def mind(state, agent_id):
    return next(item for item in state.minds if item.agent_id == agent_id)


def agent_model(model, agent_id):
    return next(item for item in model.cognitive_model.agents if item.agent_id == agent_id)


class SituatedMemoryRecallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.database_path = f"{self.temporary.name}/recall.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def test_private_inspection_is_recalled_once_and_updates_belief(self):
        cognition, story, inspection = inspection_and_move_story()
        cue = SituatedMemoryRecallCue(
            "restructuring",
            "restructuring",
            required_place_ids=("open",),
            event_kinds=(SituatedActionKind.INSPECT,),
            channels=(ObservationChannel.INSPECTION,),
        )
        model = recall_model({"alice": (cue,), "bob": (cue,)})
        for agent_id in ("alice", "bob"):
            ingest_situated_story(
                self.database_path,
                story,
                agent_id,
                model.memory_policy,
            )
        state = initialize_situated_memory_cognition(model, story)

        alice = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "alice"),
            mind(state, "alice"),
            story=story,
            state=state,
        )
        bob = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "bob"),
            mind(state, "bob"),
            story=story,
            state=state,
        )

        self.assertEqual(len(alice.admissions), 1)
        admitted = alice.admissions[0]
        self.assertEqual(admitted.cue_id, "restructuring")
        self.assertEqual(admitted.event_id, inspection.event_id)
        self.assertEqual(admitted.rule_id, "inspect-approved")
        self.assertEqual(admitted.symbol_id, "approved")
        self.assertEqual(admitted.likelihood_action_id, "inspect")
        self.assertEqual(admitted.evidence_weight, 1.0)
        self.assertAlmostEqual(admitted.prior_belief.probabilities["approved"], 0.5)
        self.assertAlmostEqual(admitted.posterior_belief.probabilities["approved"], 0.9)
        self.assertAlmostEqual(alice.next_mind.belief.probabilities["approved"], 0.9)
        self.assertEqual(alice.next_mind.recalled_memory_ids, (admitted.memory_id,))
        self.assertEqual(alice.next_mind.observed_event_ids, (inspection.event_id,))
        self.assertEqual(bob.admissions, ())
        self.assertEqual(bob.next_mind, mind(state, "bob"))

        repeated = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "alice"),
            alice.next_mind,
            story=story,
            state=state,
        )
        self.assertEqual(repeated.admissions, ())
        self.assertEqual(repeated.next_mind, alice.next_mind)

    def test_directly_processed_memory_is_not_admitted_again(self):
        _, story, _ = inspection_and_move_story()
        cue = SituatedMemoryRecallCue("restructuring", "restructuring")
        model = recall_model({"alice": (cue,)})
        ingest_situated_story(self.database_path, story, "alice", model.memory_policy)
        state = initialize_situated_memory_cognition(model, story)
        alice = mind(state, "alice")
        inspection_memory = next(
            item
            for item in list_situated_memories(self.database_path, "alice")
            if item.kind is SituatedActionKind.INSPECT
        )
        already_processed = replace(
            alice,
            processed_observation_ids=(inspection_memory.observation_id,),
        )

        result = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "alice"),
            already_processed,
            story=story,
            state=state,
        )

        self.assertEqual(result.admissions, ())
        self.assertEqual(result.next_mind, already_processed)

    def test_excluded_top_hit_is_backfilled_with_next_eligible_memory(self):
        cognition = cognitive_office_model()
        story = initialize_situated_story(
            cognition.world_model,
            initialize_situated_world(cognition.world_model),
        )
        for round_number in (1, 2):
            story = advance_situated_story(cognition.world_model, story, (
                SituatedActionIntent(
                    f"inspect-{round_number}",
                    "alice",
                    SituatedActionKind.INSPECT,
                    "memo",
                ),
            ))
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent("move", "alice", SituatedActionKind.MOVE, "records-open"),
        ))
        cue = SituatedMemoryRecallCue(
            "restructuring",
            "restructuring",
            required_place_ids=("open",),
            event_kinds=(SituatedActionKind.INSPECT,),
            limit=1,
        )
        model = recall_model({"alice": (cue,)})
        ingest_situated_story(self.database_path, story, "alice", model.memory_policy)
        state = initialize_situated_memory_cognition(model, story)

        first = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "alice"),
            mind(state, "alice"),
            story=story,
            state=state,
        )
        second = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "alice"),
            first.next_mind,
            story=story,
            state=state,
        )

        self.assertEqual(len(first.admissions), 1)
        self.assertEqual(first.admissions[0].event_id, story.rounds[1].events[0].event_id)
        self.assertEqual(len(second.admissions), 1)
        self.assertEqual(second.admissions[0].event_id, story.rounds[0].events[0].event_id)

    def test_auditory_memory_uses_confidence_and_salience_tempering(self):
        cognition, story, inspection = inspection_and_move_story()
        story = with_telling(cognition, story, inspection)
        cue = SituatedMemoryRecallCue(
            "heard-restructuring",
            "The restructuring is approved.",
            event_kinds=(SituatedActionKind.TELL,),
            channels=(ObservationChannel.AUDITORY,),
        )
        model = recall_model({"bob": (cue,)})
        ingest_situated_story(self.database_path, story, "bob", model.memory_policy)
        state = initialize_situated_memory_cognition(model, story)

        result = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "bob"),
            mind(state, "bob"),
            story=story,
            state=state,
        )

        self.assertEqual(len(result.admissions), 1)
        admission = result.admissions[0]
        self.assertAlmostEqual(admission.evidence_weight, 0.525)
        self.assertAlmostEqual(
            admission.posterior_belief.probabilities["approved"],
            0.6423728813559322,
        )

    def test_unmatched_memory_restores_event_access_without_changing_belief(self):
        _, story, _ = inspection_and_move_story()
        cue = SituatedMemoryRecallCue(
            "movement",
            "destination_place_id",
            event_kinds=(SituatedActionKind.MOVE,),
            channels=(ObservationChannel.SELF,),
        )
        model = recall_model({"alice": (cue,)})
        ingest_situated_story(self.database_path, story, "alice", model.memory_policy)
        state = initialize_situated_memory_cognition(model, story)

        result = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "alice"),
            mind(state, "alice"),
            story=story,
            state=state,
        )

        self.assertEqual(len(result.admissions), 1)
        admission = result.admissions[0]
        self.assertIsNone(admission.rule_id)
        self.assertIsNone(admission.symbol_id)
        self.assertEqual(admission.prior_belief, admission.posterior_belief)
        self.assertEqual(result.next_mind.belief, mind(state, "alice").belief)
        self.assertEqual(result.next_mind.observed_event_ids, (admission.event_id,))

    def test_future_other_world_and_inactive_rows_cannot_enter_checkpoint(self):
        cognition, checkpoint_story, inspection = inspection_and_move_story()
        future_story = with_telling(cognition, checkpoint_story, inspection)
        cue = SituatedMemoryRecallCue("future-telling", "The restructuring is approved.")
        model = recall_model({"alice": (cue,)})
        ingest_situated_story(self.database_path, future_story, "alice", model.memory_policy)
        memories = list_situated_memories(self.database_path, "alice")
        future = next(item for item in memories if item.kind is SituatedActionKind.TELL)
        other_world = replace(
            future,
            memory_id="other-world",
            observation_id="other-world",
            event_id="other-world-event",
            round_index=1,
            story_model_hash=OTHER_WORLD_HASH,
        )
        inactive = replace(
            future,
            memory_id="inactive-past",
            observation_id="inactive-past",
            event_id="inactive-past-event",
            round_index=1,
        )
        ingest_situated_memory(self.database_path, "alice", (other_world, inactive))
        set_situated_memory_active(self.database_path, "alice", inactive.memory_id, False)
        state = initialize_situated_memory_cognition(model, checkpoint_story)

        result = recall_situated_memories(
            self.database_path,
            model,
            agent_model(model, "alice"),
            mind(state, "alice"),
            story=checkpoint_story,
            state=state,
        )

        self.assertEqual(result.admissions, ())
        self.assertEqual(result.next_mind, mind(state, "alice"))


if __name__ == "__main__":
    unittest.main()
