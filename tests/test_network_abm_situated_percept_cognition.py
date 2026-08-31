from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedCognitiveState,
    SituatedObservationRule,
    initialize_situated_cognition,
)
from narrative_dynamics.abm.situated_contracts import PassageState, SituatedWorldState, initialize_situated_world
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedAgentPerceptionProfile,
    SituatedEdgeActivation,
    SituatedEventSignalProfile,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
    SituatedPerceptFidelity,
)
from narrative_dynamics.abm.situated_percept_cognition import (
    admit_situated_percepts,
    match_situated_percept_rule,
    perceptual_timeline,
    project_situated_story_percepts,
    simulate_situated_percept_cognitive_round,
)
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from tests.situated_cognition_fixtures import cognitive_office_model


SECRET = "The restructuring is approved."


def perception_model(*, visual: bool = False) -> SituatedPerceptionModel:
    cognition = cognitive_office_model()
    edges = (
        SituatedPerceptionEdge(
            "records-open-audio-open",
            SituatedPerceptionLayer.AUDITORY,
            "records",
            "open",
            5.0,
            SituatedEdgeActivation.PASSAGE_OPEN,
            "records-open",
        ),
        SituatedPerceptionEdge(
            "records-open-audio-closed",
            SituatedPerceptionLayer.AUDITORY,
            "records",
            "open",
            25.0,
            SituatedEdgeActivation.PASSAGE_CLOSED,
            "records-open",
        ),
    ) + ((
        SituatedPerceptionEdge(
            "records-open-visual",
            SituatedPerceptionLayer.VISIBILITY,
            "records",
            "open",
            1.0,
        ),
    ) if visual else ())
    return SituatedPerceptionModel(
        "office-perception",
        "1",
        cognition.world_model,
        edges,
        tuple(
            SituatedAgentPerceptionProfile(agent.agent_id, 1.0, 20.0, 45.0)
            for agent in cognition.world_model.agents
        ),
        (SituatedEventSignalProfile(SituatedActionKind.TELL, visual, 60.0),),
    )


def initial_state(model: SituatedPerceptionModel, *, door_open: bool) -> SituatedWorldState:
    state = initialize_situated_world(model.world_model)
    return SituatedWorldState(
        state.model_id,
        state.model_hash,
        state.round_index,
        state.parent_state_hash,
        state.agents,
        state.objects,
        tuple(
            PassageState(item.passage_id, door_open)
            if item.passage_id == "records-open"
            else item
            for item in state.passages
        ),
    )


def told_story(model: SituatedPerceptionModel, *, door_open: bool):
    story = initialize_situated_story(model.world_model, initial_state(model, door_open=door_open))
    return advance_situated_story(
        model.world_model,
        story,
        (SituatedActionIntent("alice-tell", "alice", SituatedActionKind.TELL, message=SECRET),),
    )


def agent(cognition, agent_id: str):
    return next(item for item in cognition.agents if item.agent_id == agent_id)


def mind(state, agent_id: str):
    return next(item for item in state.minds if item.agent_id == agent_id)


def alice_tell_percept(perception: SituatedPerceptionModel, story, observer: str):
    event = next(
        item
        for item in story.rounds[-1].events
        if item.actor_agent_id == "alice" and item.kind is SituatedActionKind.TELL
    )
    return next(
        item
        for item in perceptual_timeline(perception, story, observer)
        if item.source_event_id == event.event_id
    )


def cognitive_state_for_story(cognition, story, perception):
    initial_story = initialize_situated_story(
        cognition.world_model,
        initial_state(perception, door_open=(
            next(
                item.open
                for item in story.initial_state.passages
                if item.passage_id == "records-open"
            )
        )),
    )
    initial = initialize_situated_cognition(cognition, initial_story)
    return SituatedCognitiveState(
        cognition.model_id,
        cognition.content_hash,
        story.current_state.round_index,
        initial.content_hash,
        story.content_hash,
        initial.minds,
    )
class SituatedPerceptCognitionBoundaryTests(unittest.TestCase):
    def test_detected_sound_matches_no_cognitive_rule(self) -> None:
        perception = perception_model()
        cognition = cognitive_office_model()

        story = told_story(perception, door_open=False)
        bob = alice_tell_percept(perception, story, "bob")

        self.assertIs(bob.fidelity, SituatedPerceptFidelity.DETECTED)
        self.assertIsNone(match_situated_percept_rule(agent(cognition, "bob"), bob))

    def test_identified_percept_matches_kind_only_but_not_outcome_or_detail_rule(self) -> None:
        perception = perception_model(visual=True)
        cognition = cognitive_office_model()
        bob_model = agent(cognition, "bob")
        kind_only = replace(
            bob_model,
            observation_rules=(SituatedObservationRule(
                "saw-tell",
                "approved",
                "tell",
                SituatedActionKind.TELL,
            ),),
        )

        story = told_story(perception, door_open=False)
        bob = alice_tell_percept(perception, story, "bob")

        self.assertIs(bob.fidelity, SituatedPerceptFidelity.IDENTIFIED)
        self.assertEqual(match_situated_percept_rule(kind_only, bob).rule_id, "saw-tell")
        self.assertIsNone(match_situated_percept_rule(bob_model, bob))

    def test_exact_percept_matches_disclosed_outcome_and_detail(self) -> None:
        perception = perception_model()
        cognition = cognitive_office_model()

        story = told_story(perception, door_open=True)
        bob = alice_tell_percept(perception, story, "bob")

        self.assertIs(bob.fidelity, SituatedPerceptFidelity.EXACT)
        self.assertEqual(match_situated_percept_rule(agent(cognition, "bob"), bob).symbol_id, "approved")

    def test_story_projection_rejects_unknown_agent_and_wrong_world(self) -> None:
        perception = perception_model()
        story = told_story(perception, door_open=True)

        with self.assertRaisesRegex(ValueError, "agent"):
            perceptual_timeline(perception, story, "unknown")

        wrong_world = replace(perception.world_model, version="wrong-world-version")
        other = replace(
            perception,
            model_id="other-perception",
            world_model=wrong_world,
        )
        with self.assertRaisesRegex(ValueError, "world"):
            project_situated_story_percepts(other, story)


class SituatedPerceptAdmissionTests(unittest.TestCase):
    def test_exact_private_percept_updates_belief_once(self) -> None:
        perception = perception_model()
        cognition = cognitive_office_model()
        initial_story = initialize_situated_story(
            cognition.world_model,
            initial_state(perception, door_open=True),
        )
        state = initialize_situated_cognition(cognition, initial_story)
        story = told_story(perception, door_open=True)
        bob_model = agent(cognition, "bob")
        bob_mind = mind(state, "bob")
        private = perceptual_timeline(perception, story, "bob")

        admitted = admit_situated_percepts(bob_model, bob_mind, private)

        self.assertEqual(tuple(item.symbol_id for item in admitted.admissions), ("approved",))
        self.assertAlmostEqual(admitted.next_mind.belief.probabilities["approved"], 0.9)
        tell = alice_tell_percept(perception, story, "bob")
        self.assertIn(tell.percept_id, admitted.next_mind.processed_observation_ids)
        self.assertIn(tell.source_event_id, admitted.next_mind.observed_event_ids)
        repeated = admit_situated_percepts(bob_model, admitted.next_mind, private)
        self.assertEqual(repeated.admissions, ())
        self.assertEqual(repeated.next_mind, admitted.next_mind)

    def test_detected_percept_is_processed_without_identity_or_belief_change(self) -> None:
        perception = perception_model()
        cognition = cognitive_office_model()
        initial_story = initialize_situated_story(
            cognition.world_model,
            initial_state(perception, door_open=False),
        )
        state = initialize_situated_cognition(cognition, initial_story)
        story = told_story(perception, door_open=False)
        bob_model = agent(cognition, "bob")
        bob_mind = mind(state, "bob")
        tell = alice_tell_percept(perception, story, "bob")

        admitted = admit_situated_percepts(
            bob_model,
            bob_mind,
            perceptual_timeline(perception, story, "bob"),
        )

        self.assertEqual(admitted.admissions, ())
        self.assertEqual(admitted.next_mind.belief, bob_mind.belief)
        self.assertIn(tell.percept_id, admitted.next_mind.processed_observation_ids)
        self.assertIn(tell.source_event_id, admitted.next_mind.observed_event_ids)

    def test_admission_rejects_another_agents_percept(self) -> None:
        perception = perception_model()
        cognition = cognitive_office_model()
        story = told_story(perception, door_open=True)
        state = initialize_situated_cognition(
            cognition,
            initialize_situated_story(
                cognition.world_model,
                initial_state(perception, door_open=True),
            ),
        )

        with self.assertRaisesRegex(ValueError, "private"):
            admit_situated_percepts(
                agent(cognition, "bob"),
                mind(state, "bob"),
                perceptual_timeline(perception, story, "alice"),
            )


class SituatedPerceptCognitiveRoundTests(unittest.TestCase):
    def test_detected_event_cannot_authorize_a_tell_source(self) -> None:
        perception = perception_model()
        closed = replace(
            told_story(perception, door_open=False),
            perception_model=perception,
        )
        source = next(
            item
            for item in closed.rounds[-1].events
            if item.actor_agent_id == "alice" and item.kind is SituatedActionKind.TELL
        )

        with self.assertRaisesRegex(ValueError, "previously perceived"):
            advance_situated_story(
                perception.world_model,
                closed,
                (SituatedActionIntent(
                    "bob-repeats-detected",
                    "bob",
                    SituatedActionKind.TELL,
                    message="I heard the confidential message.",
                    source_event_ids=(source.event_id,),
                ),),
            )

    def test_open_exact_tell_changes_bobs_action_while_closed_detection_does_not(self) -> None:
        cognition = cognitive_office_model()
        open_perception = perception_model()
        closed_perception = perception_model()
        open_story = told_story(open_perception, door_open=True)
        closed_story = told_story(closed_perception, door_open=False)

        opened = simulate_situated_percept_cognitive_round(
            open_perception,
            cognition,
            open_story,
            cognitive_state_for_story(cognition, open_story, open_perception),
        )
        closed = simulate_situated_percept_cognitive_round(
            closed_perception,
            cognition,
            closed_story,
            cognitive_state_for_story(cognition, closed_story, closed_perception),
        )

        bob_open = next(item for item in opened.decisions if item.agent_id == "bob")
        bob_closed = next(item for item in closed.decisions if item.agent_id == "bob")
        self.assertEqual(bob_open.admitted_symbol_ids, ("approved",))
        self.assertEqual(bob_open.selected_action_id, "tell")
        self.assertEqual(bob_closed.admitted_symbol_ids, ())
        self.assertEqual(bob_closed.selected_action_id, "wait")

    def test_round_uses_v10_transition_and_v15_observed_event_roster(self) -> None:
        cognition = cognitive_office_model()
        perception = perception_model()
        story = told_story(perception, door_open=True)
        state = cognitive_state_for_story(cognition, story, perception)

        result = simulate_situated_percept_cognitive_round(
            perception,
            cognition,
            story,
            state,
        )

        expected_story = advance_situated_story(
            cognition.world_model,
            replace(story, perception_model=perception),
            tuple(item.intent for item in result.decisions),
        )
        self.assertEqual(result.next_story, expected_story)
        self.assertEqual(
            result.prior_projection_hashes,
            tuple(item.content_hash for item in project_situated_story_percepts(perception, story)),
        )
        self.assertEqual(
            result.next_projection,
            project_situated_story_percepts(perception, result.next_story)[-1],
        )
        for next_mind in result.next_state.minds:
            expected_ids = {
                item.source_event_id
                for item in perceptual_timeline(perception, result.next_story, next_mind.agent_id)
            }
            self.assertEqual(set(next_mind.observed_event_ids), expected_ids)


if __name__ == "__main__":
    unittest.main()
