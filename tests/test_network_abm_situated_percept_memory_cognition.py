from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
    resolve_situated_round,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedObservationRule,
)
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
    SituatedMemoryRecallCue,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
    initialize_situated_percept_memory_cognition,
    recall_situated_percept_memories,
    simulate_situated_percept_memory_cognitive_round,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    standard_situated_percept_memory_policy,
)
from narrative_dynamics.abm.situated_story import (
    advance_situated_story,
    initialize_situated_story,
)
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import (
    SECRET,
    initial_state,
    perception_model,
)


def recall_model(cues_by_agent, *, cognition=None, perception=None):
    cognition = cognitive_office_model() if cognition is None else cognition
    perception = perception_model() if perception is None else perception
    return SituatedPerceptMemoryCognitiveModel(
        "office-percept-memory-cognition",
        "1",
        perception,
        cognition,
        standard_situated_percept_memory_policy(),
        tuple(
            SituatedAgentRecallPolicy(
                agent.agent_id,
                tuple(cues_by_agent.get(agent.agent_id, ())),
                max_memories_per_round=5,
            )
            for agent in cognition.agents
        ),
    )


def inspection_checkpoint(perception):
    cognition = cognitive_office_model()
    story = initialize_situated_story(
        cognition.world_model,
        initialize_situated_world(cognition.world_model),
        perception_model=perception,
    )
    story = advance_situated_story(
        cognition.world_model,
        story,
        (SituatedActionIntent("inspect", "alice", SituatedActionKind.INSPECT, "memo"),),
    )
    inspection = story.rounds[-1].events[0]
    story = advance_situated_story(
        cognition.world_model,
        story,
        (SituatedActionIntent("move", "alice", SituatedActionKind.MOVE, "records-open"),),
    )
    return cognition, story, inspection


def mind(state, agent_id):
    return next(item for item in state.minds if item.agent_id == agent_id)


def agent(model, agent_id):
    return next(item for item in model.cognitive_model.agents if item.agent_id == agent_id)


class SituatedPerceptMemoryCognitiveModelTests(unittest.TestCase):
    def test_model_binds_exact_perception_cognition_world_and_agent_roster(self):
        model = recall_model({})

        self.assertEqual(model.perception_model.world_model, model.cognitive_model.world_model)
        self.assertEqual(model.to_dict()["perception_model_hash"], model.perception_model.content_hash)
        self.assertEqual(
            tuple(item.agent_id for item in model.agents),
            tuple(sorted(item.agent_id for item in model.cognitive_model.agents)),
        )

        with self.assertRaisesRegex(ValueError, "world"):
            replace(model, perception_model=replace(model.perception_model, world_model=replace(
                model.perception_model.world_model,
                version="other",
            )))
        with self.assertRaisesRegex(ValueError, "roster"):
            replace(model, agents=model.agents[:1])


class SituatedPerceptMemoryRecallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.database = f"{self.temporary.name}/recall.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def test_private_exact_inspection_is_recalled_once_and_changes_later_action(self):
        perception = perception_model()
        cognition, story, inspection = inspection_checkpoint(perception)
        cue = SituatedMemoryRecallCue(
            "restructuring",
            "restructuring",
            required_place_ids=("open",),
            event_kinds=(SituatedActionKind.INSPECT,),
            channels=(ObservationChannel.INSPECTION,),
        )
        model = recall_model({"alice": (cue,), "bob": (cue,)}, cognition=cognition, perception=perception)
        state = initialize_situated_percept_memory_cognition(model, story)

        alice = recall_situated_percept_memories(
            self.database, model, agent(model, "alice"), mind(state, "alice"),
            story=story, state=state,
        )
        bob = recall_situated_percept_memories(
            self.database, model, agent(model, "bob"), mind(state, "bob"),
            story=story, state=state,
        )

        self.assertEqual(len(alice.admissions), 1)
        admitted = alice.admissions[0]
        self.assertEqual(admitted.event_id, inspection.event_id)
        self.assertEqual(admitted.rule_id, "inspect-approved")
        self.assertEqual(admitted.evidence_weight, 1.0)
        self.assertAlmostEqual(alice.next_mind.belief.probabilities["approved"], 0.9)
        self.assertEqual(bob.admissions, ())

        repeated = recall_situated_percept_memories(
            self.database, model, agent(model, "alice"), alice.next_mind,
            story=story, state=state,
        )
        self.assertEqual(repeated.admissions, ())

        round_result = simulate_situated_percept_memory_cognitive_round(
            self.database, model, story, state,
        )
        alice_decision = next(item for item in round_result.decisions if item.agent_id == "alice")
        self.assertEqual(alice_decision.selected_action_id, "tell")
        self.assertEqual(alice_decision.intent.source_event_ids, (inspection.event_id,))
        self.assertEqual(
            mind(round_result.next_state, "alice").recalled_memory_ids,
            (admitted.memory_id,),
        )

    def test_detected_memory_is_an_episode_but_cannot_change_belief(self):
        perception = perception_model()
        cognition = cognitive_office_model()
        story = initialize_situated_story(
            cognition.world_model,
            initial_state(perception, door_open=False),
            perception_model=perception,
        )
        story = advance_situated_story(
            cognition.world_model,
            story,
            (SituatedActionIntent("alice-tell", "alice", SituatedActionKind.TELL, message=SECRET),),
        )
        cue = SituatedMemoryRecallCue("sound", "detected", channels=(ObservationChannel.AUDITORY,))
        model = recall_model({"bob": (cue,)}, cognition=cognition, perception=perception)
        state = initialize_situated_percept_memory_cognition(model, story)

        recalled = recall_situated_percept_memories(
            self.database, model, agent(model, "bob"), mind(state, "bob"),
            story=story, state=state,
        )

        self.assertEqual(len(recalled.admissions), 1)
        self.assertIsNone(recalled.admissions[0].rule_id)
        self.assertEqual(recalled.next_mind.belief, mind(state, "bob").belief)
        self.assertEqual(recalled.next_mind.observed_event_ids, (story.rounds[0].events[0].event_id,))

    def test_exact_tell_recall_is_tempered_by_directed_source_trust(self):
        perception = perception_model()
        cognition = cognitive_office_model()
        story = initialize_situated_story(
            cognition.world_model,
            initial_state(perception, door_open=True),
            perception_model=perception,
        )
        story = advance_situated_story(
            cognition.world_model,
            story,
            (SituatedActionIntent("alice-tell", "alice", SituatedActionKind.TELL, message=SECRET),),
        )
        cue = SituatedMemoryRecallCue(
            "testimony", SECRET,
            event_kinds=(SituatedActionKind.TELL,),
            channels=(ObservationChannel.AUDITORY,),
        )
        model = recall_model({"bob": (cue,)}, cognition=cognition, perception=perception)
        state = initialize_situated_percept_memory_cognition(model, story)

        recalled = recall_situated_percept_memories(
            self.database, model, agent(model, "bob"), mind(state, "bob"),
            story=story, state=state, _source_trust_by_source={"alice": 0.5},
        )

        self.assertEqual(len(recalled.admissions), 1)
        self.assertEqual(recalled.admissions[0].source_trust, 0.5)
        self.assertEqual(recalled.admissions[0].evidence_weight, 0.5)
        self.assertIsNone(recalled.admissions[0].lexical_rank)
        self.assertGreater(recalled.next_mind.belief.probabilities["approved"], 0.5)
        self.assertLess(recalled.next_mind.belief.probabilities["approved"], 0.9)

    def test_integrated_round_is_restart_stable_and_uses_objective_v10_transition(self):
        perception = perception_model()
        cognition, story, _ = inspection_checkpoint(perception)
        cue = SituatedMemoryRecallCue(
            "restructuring",
            "restructuring",
            required_place_ids=("open",),
            event_kinds=(SituatedActionKind.INSPECT,),
        )
        recalled_model = recall_model(
            {"alice": (cue,)}, cognition=cognition, perception=perception
        )
        recalled_state = initialize_situated_percept_memory_cognition(recalled_model, story)

        first = simulate_situated_percept_memory_cognitive_round(
            self.database, recalled_model, story, recalled_state
        )
        reopened = simulate_situated_percept_memory_cognitive_round(
            self.database, recalled_model, story, recalled_state
        )
        expected = resolve_situated_round(
            cognition.world_model,
            story.current_state,
            tuple(item.intent for item in first.decisions),
        )

        self.assertEqual(first, reopened)
        self.assertEqual(first.content_hash, reopened.content_hash)
        self.assertEqual(first.next_story.rounds[-1], expected)

        control_model = recall_model({}, cognition=cognition, perception=perception)
        control_state = initialize_situated_percept_memory_cognition(control_model, story)
        control = simulate_situated_percept_memory_cognitive_round(
            f"{self.temporary.name}/control.sqlite3",
            control_model,
            story,
            control_state,
        )
        recalled_action = next(
            item.selected_action_id for item in first.decisions if item.agent_id == "alice"
        )
        control_action = next(
            item.selected_action_id for item in control.decisions if item.agent_id == "alice"
        )
        self.assertEqual(recalled_action, "tell")
        self.assertNotEqual(control_action, recalled_action)

    def test_identified_memory_can_match_only_a_kind_only_rule(self):
        perception = perception_model(visual=True)
        cognition = cognitive_office_model()
        bob = next(item for item in cognition.agents if item.agent_id == "bob")
        bob = replace(
            bob,
            observation_rules=(SituatedObservationRule(
                "saw-tell", "approved", "tell", SituatedActionKind.TELL
            ),),
        )
        cognition = replace(
            cognition,
            agents=tuple(bob if item.agent_id == "bob" else item for item in cognition.agents),
        )
        story = initialize_situated_story(
            cognition.world_model,
            initial_state(perception, door_open=False),
            perception_model=perception,
        )
        story = advance_situated_story(
            cognition.world_model,
            story,
            (SituatedActionIntent("alice-tell", "alice", SituatedActionKind.TELL, message=SECRET),),
        )
        cue = SituatedMemoryRecallCue(
            "visual-tell", "tell", event_kinds=(SituatedActionKind.TELL,)
        )
        model = recall_model({"bob": (cue,)}, cognition=cognition, perception=perception)
        state = initialize_situated_percept_memory_cognition(model, story)

        recalled = recall_situated_percept_memories(
            self.database, model, agent(model, "bob"), mind(state, "bob"),
            story=story, state=state,
        )

        self.assertEqual(len(recalled.admissions), 1)
        self.assertEqual(recalled.admissions[0].rule_id, "saw-tell")


if __name__ == "__main__":
    unittest.main()
