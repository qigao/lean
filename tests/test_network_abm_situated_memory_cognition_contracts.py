from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    initialize_situated_cognition,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
    SituatedMemoryCognitiveModel,
    SituatedMemoryRecallCue,
    initialize_situated_memory_cognition,
)
from narrative_dynamics.abm.situated_memory_contracts import standard_situated_memory_policy
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from tests.situated_cognition_fixtures import cognitive_office_model


def memory_cognitive_model(*, policies=None):
    cognition = cognitive_office_model()
    if policies is None:
        policies = tuple(
            SituatedAgentRecallPolicy(
                agent.agent_id,
                (
                    SituatedMemoryRecallCue(
                        "restructuring",
                        "restructuring",
                        required_place_ids=("open",),
                        event_kinds=(SituatedActionKind.TELL, SituatedActionKind.INSPECT),
                        channels=(ObservationChannel.AUDITORY, ObservationChannel.INSPECTION),
                        min_confidence=0.7,
                        limit=3,
                    ),
                ) if agent.agent_id == "alice" else (),
                max_memories_per_round=4,
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


def late_office_story():
    cognition = cognitive_office_model()
    story = initialize_situated_story(
        cognition.world_model,
        initialize_situated_world(cognition.world_model),
    )
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("inspect", "alice", SituatedActionKind.INSPECT, "memo"),
    ))
    story = advance_situated_story(cognition.world_model, story, (
        SituatedActionIntent("move", "alice", SituatedActionKind.MOVE, "records-open"),
    ))
    return cognition, story


class SituatedMemoryCognitionContractTests(unittest.TestCase):
    def test_cues_and_agent_policies_are_canonical_and_bounded(self):
        cue = SituatedMemoryRecallCue(
            "restructuring",
            "restructuring",
            required_place_ids=("records", "open"),
            event_kinds=(SituatedActionKind.TELL, SituatedActionKind.INSPECT),
            channels=(ObservationChannel.INSPECTION, ObservationChannel.AUDITORY),
            min_confidence=0.7,
            limit=3,
        )
        policy = SituatedAgentRecallPolicy(
            "alice",
            (SituatedMemoryRecallCue("z-last", "last"), cue),
            max_memories_per_round=4,
        )

        self.assertEqual(cue.required_place_ids, ("open", "records"))
        self.assertEqual(tuple(item.value for item in cue.event_kinds), ("inspect", "tell"))
        self.assertEqual(tuple(item.value for item in cue.channels), ("auditory", "inspection"))
        self.assertEqual(tuple(item.cue_id for item in policy.cues), ("restructuring", "z-last"))
        with self.assertRaisesRegex(ValueError, "cue limit must be between 1 and 1000"):
            replace(cue, limit=0)
        with self.assertRaisesRegex(ValueError, "maximum memories per round must be between 1 and 1000"):
            replace(policy, max_memories_per_round=1001)

    def test_memory_cognitive_model_requires_exact_agent_roster(self):
        complete = memory_cognitive_model()
        self.assertEqual(
            tuple(item.agent_id for item in complete.agents),
            ("alice", "bob", "carol", "dana"),
        )
        self.assertTrue(complete.content_hash.startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, "cover exact cognitive agent roster"):
            replace(complete, agents=complete.agents[:-1])

    def test_late_story_checkpoint_starts_with_private_priors_and_no_free_history(self):
        cognition, story = late_office_story()
        model = memory_cognitive_model()

        state = initialize_situated_memory_cognition(model, story)
        alice = next(item for item in state.minds if item.agent_id == "alice")

        self.assertEqual(state.model_id, cognition.model_id)
        self.assertEqual(state.model_hash, cognition.content_hash)
        self.assertEqual(state.round_index, 2)
        self.assertTrue(state.checkpoint)
        self.assertIsNone(state.parent_state_hash)
        self.assertEqual(state.story_hash, story.content_hash)
        self.assertEqual(alice.own_place_id, "open")
        self.assertEqual(alice.observation_floor_round, 2)
        self.assertEqual(alice.processed_observation_ids, ())
        self.assertEqual(alice.recalled_memory_ids, ())
        self.assertEqual(alice.observed_event_ids, ())
        self.assertEqual(alice.belief.probabilities, {"approved": 0.5, "denied": 0.5})
        validate_situated_cognitive_state(cognition, story, state)

    def test_v11_initial_state_remains_a_non_checkpoint_with_zero_floor(self):
        cognition = cognitive_office_model()
        story = initialize_situated_story(
            cognition.world_model,
            initialize_situated_world(cognition.world_model),
        )

        state = initialize_situated_cognition(cognition, story)

        self.assertFalse(state.checkpoint)
        self.assertTrue(all(item.observation_floor_round == 0 for item in state.minds))
        self.assertTrue(all(item.recalled_memory_ids == () for item in state.minds))


if __name__ == "__main__":
    unittest.main()
