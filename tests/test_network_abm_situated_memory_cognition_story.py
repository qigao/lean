from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
)
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_cognition import explain_situated_decision
from narrative_dynamics.abm.situated_memory import ingest_situated_story
from narrative_dynamics.abm.situated_memory_cognition import (
    simulate_situated_memory_cognition,
    simulate_situated_memory_cognitive_round,
)
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
    SituatedMemoryCognitiveModel,
    SituatedMemoryRecallCue,
    initialize_situated_memory_cognition,
)
from narrative_dynamics.abm.situated_memory_contracts import standard_situated_memory_policy
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from tests.situated_cognition_fixtures import cognitive_office_model


def historical_office_case():
    cognition = cognitive_office_model()
    story = initialize_situated_story(
        cognition.world_model,
        initialize_situated_world(cognition.world_model),
    )
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("historical-inspect", "alice", SituatedActionKind.INSPECT, "memo"),
    ))
    inspection = next(item for item in story.rounds[-1].events if item.actor_agent_id == "alice")
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("historical-move", "alice", SituatedActionKind.MOVE, "records-open"),
    ))
    return cognition, story, inspection


def memory_model(cognition, *, recall_alice):
    cue = SituatedMemoryRecallCue(
        "restructuring",
        "restructuring",
        required_place_ids=("open",),
        event_kinds=(SituatedActionKind.INSPECT,),
        channels=(ObservationChannel.INSPECTION,),
    )
    return SituatedMemoryCognitiveModel(
        "office-memory-cognition",
        "1",
        cognition,
        standard_situated_memory_policy(),
        tuple(
            SituatedAgentRecallPolicy(
                item.agent_id,
                (cue,) if recall_alice and item.agent_id == "alice" else (),
                max_memories_per_round=5,
            )
            for item in cognition.agents
        ),
    )


def decision(result, agent_id):
    return next(item for item in result.decisions if item.agent_id == agent_id)


def recall(result, agent_id):
    return next(item for item in result.recalls if item.prior_mind.agent_id == agent_id)


def mind(state, agent_id):
    return next(item for item in state.minds if item.agent_id == agent_id)


class SituatedMemoryCognitiveStoryTests(unittest.TestCase):
    def test_office_checkpoint_control_and_recall_choose_different_actions(self):
        cognition, story, inspection = historical_office_case()
        control_model = memory_model(cognition, recall_alice=False)
        recall_model = memory_model(cognition, recall_alice=True)
        control_state = initialize_situated_memory_cognition(control_model, story)
        recall_state = initialize_situated_memory_cognition(recall_model, story)
        with TemporaryDirectory() as directory:
            control = simulate_situated_memory_cognitive_round(
                f"{directory}/control.sqlite3", control_model, story, control_state
            )
            recalled = simulate_situated_memory_cognitive_round(
                f"{directory}/recall.sqlite3", recall_model, story, recall_state
            )

        control_alice = decision(control, "alice")
        recalled_alice = decision(recalled, "alice")
        self.assertEqual(control_alice.selected_action_id, "wait")
        self.assertAlmostEqual(control_alice.posterior_belief.probabilities["approved"], 0.5)
        self.assertEqual(control_alice.recalled_memory_ids, ())
        self.assertEqual(recalled_alice.selected_action_id, "tell")
        self.assertAlmostEqual(recalled_alice.posterior_belief.probabilities["approved"], 0.9)
        self.assertEqual(recalled_alice.admitted_observation_ids, ())
        self.assertEqual(recalled_alice.recalled_memory_ids, (recall(recalled, "alice").admissions[0].memory_id,))
        self.assertEqual(recalled_alice.recalled_symbol_ids, ("approved",))
        self.assertEqual(recalled_alice.intent.source_event_ids, (inspection.event_id,))
        self.assertEqual(recall(recalled, "alice").admissions[0].event_id, inspection.event_id)
        explanation = explain_situated_decision(recalled.cognitive_round, "alice")
        self.assertEqual(explanation.admitted_evidence_ids, ())
        self.assertEqual(explanation.recalled_memory_ids, recalled_alice.recalled_memory_ids)
        self.assertEqual(explanation.recalled_symbol_ids, ("approved",))
        self.assertFalse(recalled.next_state.checkpoint)
        self.assertEqual(recalled.next_state.parent_state_hash, recall_state.content_hash)
        for agent_id in ("bob", "carol", "dana"):
            self.assertEqual(recall(recalled, agent_id).admissions, ())
            self.assertEqual(decision(recalled, agent_id).recalled_memory_ids, ())

    def test_recall_is_not_reinforced_and_telling_is_heard_next_round(self):
        cognition, story, _ = historical_office_case()
        model = memory_model(cognition, recall_alice=True)
        state = initialize_situated_memory_cognition(model, story)
        with TemporaryDirectory() as directory:
            first = simulate_situated_memory_cognitive_round(
                f"{directory}/memory.sqlite3", model, story, state
            )
            second = simulate_situated_memory_cognitive_round(
                f"{directory}/memory.sqlite3", model, first.next_story, first.next_state
            )

        self.assertEqual(len(recall(first, "alice").admissions), 1)
        self.assertEqual(recall(second, "alice").admissions, ())
        self.assertAlmostEqual(decision(second, "alice").posterior_belief.probabilities["approved"], 0.9)
        self.assertEqual(decision(first, "bob").admitted_symbol_ids, ())
        self.assertEqual(decision(second, "bob").admitted_symbol_ids, ("approved",))
        self.assertAlmostEqual(decision(second, "bob").posterior_belief.probabilities["approved"], 0.9)

    def test_future_database_rows_do_not_change_an_earlier_checkpoint(self):
        cognition, checkpoint_story, inspection = historical_office_case()
        future_story = advance_situated_story(cognition.world_model, checkpoint_story, (
            SituatedActionIntent(
                "future-tell",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is approved.",
                source_event_ids=(inspection.event_id,),
            ),
        ))
        model = memory_model(cognition, recall_alice=True)
        state = initialize_situated_memory_cognition(model, checkpoint_story)
        with TemporaryDirectory() as directory:
            database = f"{directory}/future.sqlite3"
            ingest_situated_story(database, future_story, "bob", model.memory_policy)
            result = simulate_situated_memory_cognitive_round(database, model, checkpoint_story, state)

        self.assertEqual(recall(result, "bob").admissions, ())
        self.assertAlmostEqual(decision(result, "bob").posterior_belief.probabilities["approved"], 0.5)

    def test_trajectory_is_database_path_independent(self):
        cognition, story, _ = historical_office_case()
        model = memory_model(cognition, recall_alice=True)
        state = initialize_situated_memory_cognition(model, story)
        with TemporaryDirectory() as left_directory, TemporaryDirectory() as right_directory:
            left = simulate_situated_memory_cognition(
                f"{left_directory}/memory.sqlite3",
                model,
                story,
                state,
                round_count=2,
            )
            right = simulate_situated_memory_cognition(
                f"{right_directory}/memory.sqlite3",
                model,
                story,
                state,
                round_count=2,
            )

        self.assertEqual(left, right)
        self.assertEqual(left.content_hash, right.content_hash)
        self.assertEqual(left.final_state.round_index, state.round_index + 2)


if __name__ == "__main__":
    unittest.main()
