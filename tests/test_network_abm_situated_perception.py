from __future__ import annotations

import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
    resolve_situated_round,
)
from narrative_dynamics.abm.situated_contracts import (
    AgentBodyState,
    EmbodiedAgentSpec,
    PassageSpec,
    PassageState,
    PlaceSpec,
    SituatedWorldModel,
    SituatedWorldState,
    initialize_situated_world,
)
from narrative_dynamics.abm.situated_perception import (
    can_situated_agents_interact,
    derive_situated_perception_reach,
    percepts_for_agent,
    project_situated_percepts,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedAgentPerceptionProfile,
    SituatedEdgeActivation,
    SituatedEventSignalProfile,
    SituatedPerceptionEdge,
    SituatedPerceptFidelity,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
)


def _world(*, initially_open: bool = True) -> SituatedWorldModel:
    return SituatedWorldModel(
        "perception-office",
        "1",
        (
            PlaceSpec("corridor", "Corridor"),
            PlaceSpec("meeting", "Meeting room"),
            PlaceSpec("records", "Records room"),
            PlaceSpec("vestibule", "Vestibule"),
        ),
        (PassageSpec("meeting-door", "corridor", "meeting", initially_open=initially_open),),
        (
            EmbodiedAgentSpec("alice", "staff", "records"),
            EmbodiedAgentSpec("bob", "staff", "meeting"),
            EmbodiedAgentSpec("carol", "staff", "vestibule"),
        ),
    )


def _model(edges: tuple[SituatedPerceptionEdge, ...], *, initially_open: bool = True) -> SituatedPerceptionModel:
    return SituatedPerceptionModel(
        "perception",
        "1",
        _world(initially_open=initially_open),
        edges,
        tuple(SituatedAgentPerceptionProfile(agent_id, 10.0, 1.0, 2.0) for agent_id in ("alice", "bob", "carol")),
        (),
    )


def _state(model: SituatedPerceptionModel, *, open_passage: bool | None = None) -> SituatedWorldState:
    state = initialize_situated_world(model.world_model)
    if open_passage is None:
        return state
    return SituatedWorldState(
        state.model_id,
        state.model_hash,
        state.round_index,
        state.parent_state_hash,
        state.agents,
        state.objects,
        (PassageState("meeting-door", open_passage),),
    )


def _projection_world() -> SituatedWorldModel:
    return SituatedWorldModel(
        "projection-office",
        "1",
        (
            PlaceSpec("corridor", "Corridor"),
            PlaceSpec("meeting", "Meeting room"),
            PlaceSpec("glass", "Glass office"),
            PlaceSpec("distant", "Distant office"),
        ),
        (PassageSpec("meeting-door", "corridor", "meeting", initially_open=True),),
        (
            EmbodiedAgentSpec("alice", "staff", "corridor"),
            EmbodiedAgentSpec("bob", "staff", "meeting"),
            EmbodiedAgentSpec("carol", "staff", "glass"),
            EmbodiedAgentSpec("dana", "staff", "distant"),
        ),
    )


def _projection_model(*, initially_open: bool = True, edge_order_reversed: bool = False) -> SituatedPerceptionModel:
    edges = (
        SituatedPerceptionEdge("corridor-meeting-visual-open", SituatedPerceptionLayer.VISIBILITY, "corridor", "meeting", 1.0, SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door"),
        SituatedPerceptionEdge("corridor-meeting-audio-open", SituatedPerceptionLayer.AUDITORY, "corridor", "meeting", 5.0, SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door"),
        SituatedPerceptionEdge("corridor-meeting-audio-closed", SituatedPerceptionLayer.AUDITORY, "corridor", "meeting", 25.0, SituatedEdgeActivation.PASSAGE_CLOSED, "meeting-door"),
        SituatedPerceptionEdge("corridor-glass-visual", SituatedPerceptionLayer.VISIBILITY, "corridor", "glass", 1.0),
        SituatedPerceptionEdge("corridor-distant-audio", SituatedPerceptionLayer.AUDITORY, "corridor", "distant", 50.0),
    )
    return SituatedPerceptionModel(
        "projection-perception",
        "1",
        _projection_world(),
        tuple(reversed(edges)) if edge_order_reversed else edges,
        tuple(SituatedAgentPerceptionProfile(agent_id, 1.0, 20.0, 45.0) for agent_id in ("alice", "bob", "carol", "dana")),
        (
            SituatedEventSignalProfile(SituatedActionKind.MOVE, True),
            SituatedEventSignalProfile(SituatedActionKind.TAKE, True),
            SituatedEventSignalProfile(SituatedActionKind.DROP, True),
            SituatedEventSignalProfile(SituatedActionKind.TELL, True, 60.0),
        ),
    )


def _projection_state(model: SituatedPerceptionModel, *, open_passage: bool | None = None) -> SituatedWorldState:
    return _state(model, open_passage=open_passage)


def _action(action_id: str, agent_id: str, kind: SituatedActionKind, target_id: str | None = None, *, message: str | None = None) -> SituatedActionIntent:
    return SituatedActionIntent(action_id, agent_id, kind, target_id, message)


def _event_for(round_result, agent_id: str):
    return next(item for item in round_result.events if item.actor_agent_id == agent_id)


def _contains_text(value: object, text: str) -> bool:
    if isinstance(value, str):
        return value == text
    if isinstance(value, dict):
        return any(_contains_text(item, text) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_text(item, text) for item in value)
    return False


class SituatedPerceptionReachTests(unittest.TestCase):
    def test_shortest_cost_reach_is_independent_of_edge_order_and_has_zero_source_cost(self) -> None:
        edges = (
            SituatedPerceptionEdge("records-corridor", SituatedPerceptionLayer.VISIBILITY, "records", "corridor", 4.0),
            SituatedPerceptionEdge("corridor-records", SituatedPerceptionLayer.VISIBILITY, "corridor", "records", 4.0),
            SituatedPerceptionEdge("corridor-meeting", SituatedPerceptionLayer.VISIBILITY, "corridor", "meeting", 4.0),
            SituatedPerceptionEdge("meeting-corridor", SituatedPerceptionLayer.VISIBILITY, "meeting", "corridor", 4.0),
            SituatedPerceptionEdge("records-meeting-glass", SituatedPerceptionLayer.VISIBILITY, "records", "meeting", 2.0),
            SituatedPerceptionEdge("meeting-records-glass", SituatedPerceptionLayer.VISIBILITY, "meeting", "records", 2.0),
            SituatedPerceptionEdge("records-corridor-audio", SituatedPerceptionLayer.AUDITORY, "records", "corridor", 10.0),
            SituatedPerceptionEdge("corridor-records-audio", SituatedPerceptionLayer.AUDITORY, "corridor", "records", 10.0),
            SituatedPerceptionEdge("corridor-meeting-audio", SituatedPerceptionLayer.AUDITORY, "corridor", "meeting", 5.0),
            SituatedPerceptionEdge("meeting-corridor-audio", SituatedPerceptionLayer.AUDITORY, "meeting", "corridor", 5.0),
            SituatedPerceptionEdge("records-meeting-glass-audio", SituatedPerceptionLayer.AUDITORY, "records", "meeting", 30.0),
            SituatedPerceptionEdge("meeting-records-glass-audio", SituatedPerceptionLayer.AUDITORY, "meeting", "records", 30.0),
            SituatedPerceptionEdge("meeting-records-cycle", SituatedPerceptionLayer.VISIBILITY, "meeting", "records", 1.0),
        )
        model = _model(edges)
        state = _state(model)

        reach = derive_situated_perception_reach(model, state, source_place_id="records")

        self.assertEqual(reach.visual_costs["records"], 0.0)
        self.assertEqual(reach.visual_costs["meeting"], 2.0)
        self.assertEqual(reach.auditory_losses["meeting"], 15.0)
        self.assertEqual(tuple(reach.visual_costs), ("corridor", "meeting", "records"))
        self.assertEqual(tuple(reach.auditory_losses), ("corridor", "meeting", "records"))

        reversed_reach = derive_situated_perception_reach(
            _model(tuple(reversed(edges))), _state(_model(tuple(reversed(edges)))), source_place_id="records"
        )
        self.assertEqual(reach, reversed_reach)
        self.assertEqual(reach.content_hash, reversed_reach.content_hash)

    def test_closed_edges_do_not_leak_through_a_cycle_and_unknown_source_is_rejected(self) -> None:
        model = _model((
            SituatedPerceptionEdge("records-vestibule", SituatedPerceptionLayer.VISIBILITY, "records", "vestibule", 1.0),
            SituatedPerceptionEdge("vestibule-records", SituatedPerceptionLayer.VISIBILITY, "vestibule", "records", 1.0),
            SituatedPerceptionEdge("vestibule-corridor-closed", SituatedPerceptionLayer.VISIBILITY, "vestibule", "corridor", 1.0, SituatedEdgeActivation.PASSAGE_CLOSED, "meeting-door"),
            SituatedPerceptionEdge("corridor-vestibule", SituatedPerceptionLayer.VISIBILITY, "corridor", "vestibule", 1.0),
        ), initially_open=True)

        reach = derive_situated_perception_reach(model, _state(model, open_passage=True), source_place_id="records")

        self.assertNotIn("corridor", reach.visual_costs)
        with self.assertRaises(ValueError):
            derive_situated_perception_reach(model, _state(model), source_place_id="unknown")

    def test_passage_activation_selects_open_or_closed_edge(self) -> None:
        model = _model((
            SituatedPerceptionEdge("open-visual", SituatedPerceptionLayer.VISIBILITY, "corridor", "meeting", 1.0, SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door"),
            SituatedPerceptionEdge("open-audio", SituatedPerceptionLayer.AUDITORY, "corridor", "meeting", 5.0, SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door"),
            SituatedPerceptionEdge("closed-audio", SituatedPerceptionLayer.AUDITORY, "corridor", "meeting", 25.0, SituatedEdgeActivation.PASSAGE_CLOSED, "meeting-door"),
            SituatedPerceptionEdge("open-interaction", SituatedPerceptionLayer.INTERACTION, "corridor", "meeting", 0.0, SituatedEdgeActivation.PASSAGE_OPEN, "meeting-door"),
        ), initially_open=False)

        open_state = _state(model, open_passage=True)
        closed_state = _state(model, open_passage=False)
        open_reach = derive_situated_perception_reach(model, open_state, source_place_id="corridor")
        closed_reach = derive_situated_perception_reach(model, closed_state, source_place_id="corridor")

        self.assertEqual(open_reach.visual_costs["meeting"], 1.0)
        self.assertEqual(open_reach.auditory_losses["meeting"], 5.0)
        self.assertEqual(open_reach.interaction_place_ids, ("meeting",))
        self.assertNotIn("meeting", closed_reach.visual_costs)
        self.assertEqual(closed_reach.auditory_losses["meeting"], 25.0)
        self.assertEqual(closed_reach.interaction_place_ids, ())


class SituatedInteractionQueryTests(unittest.TestCase):
    def test_interaction_is_directed_non_transitive_and_co_location_is_true(self) -> None:
        model = _model((
            SituatedPerceptionEdge("records-meeting", SituatedPerceptionLayer.INTERACTION, "records", "meeting", 0.0),
            SituatedPerceptionEdge("meeting-vestibule", SituatedPerceptionLayer.INTERACTION, "meeting", "vestibule", 0.0),
        ))
        state = _state(model)

        self.assertFalse(can_situated_agents_interact(model, state, "alice", "alice"))
        self.assertTrue(can_situated_agents_interact(model, state, "alice", "bob"))
        self.assertFalse(can_situated_agents_interact(model, state, "alice", "carol"))
        self.assertFalse(can_situated_agents_interact(model, state, "bob", "alice"))
        colocated = SituatedWorldState(
            state.model_id,
            state.model_hash,
            state.round_index,
            state.parent_state_hash,
            (AgentBodyState("alice", "meeting"), AgentBodyState("bob", "meeting"), AgentBodyState("carol", "vestibule")),
            state.objects,
            state.passages,
        )
        self.assertTrue(can_situated_agents_interact(model, colocated, "alice", "bob"))

    def test_unknown_agents_and_mismatched_state_are_rejected(self) -> None:
        model = _model(())
        state = _state(model)
        with self.assertRaises(ValueError):
            can_situated_agents_interact(model, state, "unknown", "alice")

        other_model = _model(())
        mismatched = SituatedWorldState(
            other_model.model_id,
            "sha256:" + "0" * 64,
            0,
            None,
            state.agents,
            state.objects,
            state.passages,
        )
        with self.assertRaises(ValueError):
            can_situated_agents_interact(model, mismatched, "alice", "bob")
        with self.assertRaises(ValueError):
            derive_situated_perception_reach(model, mismatched, source_place_id="records")


class SituatedPerceptProjectionTests(unittest.TestCase):
    def test_inspection_is_exact_and_private_even_when_a_visual_path_exists(self) -> None:
        model = _projection_model()
        state = _projection_state(model)
        round_result = resolve_situated_round(
            model.world_model,
            state,
            (_action("alice-inspect", "alice", SituatedActionKind.INSPECT, "missing"),),
        )
        event = _event_for(round_result, "alice")

        projection = project_situated_percepts(model, round_result)

        percepts = tuple(item for item in projection.percepts if item.source_event_id == event.event_id)
        self.assertEqual(len(percepts), 1)
        percept = percepts[0]
        self.assertEqual(percept.percept_id, f"{event.event_id}:p:alice")
        self.assertEqual(percept.agent_id, "alice")
        self.assertEqual(percept.channels, (ObservationChannel.INSPECTION,))
        self.assertEqual(percept.fidelity, SituatedPerceptFidelity.EXACT)
        self.assertEqual(percept.actor_agent_id, "alice")
        self.assertEqual(percept.kind, SituatedActionKind.INSPECT)
        self.assertEqual(percept.place_id, "corridor")
        self.assertEqual(percept.outcome, "unknown_object")
        self.assertEqual(percept.details, ())
        self.assertEqual(percepts_for_agent(projection, "alice"), (percept,))
        self.assertEqual(
            tuple(item for item in percepts if item.agent_id != "alice"),
            (),
        )

    def test_open_door_merges_clear_hearing_with_vision_and_sanitizes_visual_only_percept(self) -> None:
        model = _projection_model()
        state = _projection_state(model)
        round_result = resolve_situated_round(
            model.world_model,
            state,
            (_action("alice-tell", "alice", SituatedActionKind.TELL, message="The meeting is confidential."),),
        )
        event = _event_for(round_result, "alice")

        projection = project_situated_percepts(model, round_result)

        bob = next(item for item in projection.percepts if item.source_event_id == event.event_id and item.agent_id == "bob")
        self.assertEqual(bob.percept_id, f"{event.event_id}:p:bob")
        self.assertEqual(bob.channels, (ObservationChannel.AUDITORY, ObservationChannel.VISUAL))
        self.assertEqual(bob.fidelity, SituatedPerceptFidelity.EXACT)
        self.assertEqual(bob.actor_agent_id, "alice")
        self.assertEqual(bob.kind, SituatedActionKind.TELL)
        self.assertEqual(bob.place_id, "corridor")
        self.assertEqual(bob.outcome, "told")
        self.assertEqual(tuple((item.name, item.value) for item in bob.details), (("message", "The meeting is confidential."),))

        carol = next(item for item in projection.percepts if item.source_event_id == event.event_id and item.agent_id == "carol")
        self.assertEqual(carol.channels, (ObservationChannel.VISUAL,))
        self.assertEqual(carol.fidelity, SituatedPerceptFidelity.IDENTIFIED)
        self.assertEqual((carol.actor_agent_id, carol.kind, carol.place_id), ("alice", SituatedActionKind.TELL, "corridor"))
        self.assertIsNone(carol.outcome)
        self.assertEqual(carol.details, ())

    def test_closed_door_detects_sound_without_disclosing_the_message(self) -> None:
        model = _projection_model(initially_open=False)
        state = _projection_state(model, open_passage=False)
        secret = "The meeting is confidential."
        round_result = resolve_situated_round(
            model.world_model,
            state,
            (_action("alice-tell", "alice", SituatedActionKind.TELL, message=secret),),
        )
        event = _event_for(round_result, "alice")

        projection = project_situated_percepts(model, round_result)

        bob = next(item for item in projection.percepts if item.source_event_id == event.event_id and item.agent_id == "bob")
        self.assertEqual(bob.channels, (ObservationChannel.AUDITORY,))
        self.assertEqual(bob.fidelity, SituatedPerceptFidelity.DETECTED)
        self.assertIsNone(bob.actor_agent_id)
        self.assertIsNone(bob.kind)
        self.assertIsNone(bob.place_id)
        self.assertIsNone(bob.outcome)
        self.assertEqual(bob.details, ())
        self.assertFalse(_contains_text(bob.to_dict(), secret))
        self.assertEqual(
            tuple(item for item in projection.percepts if item.source_event_id == event.event_id and item.agent_id == "dana"),
            (),
        )

    def test_declared_visual_signals_cover_move_take_and_drop_but_wait_remains_private(self) -> None:
        model = _projection_model()
        state = _projection_state(model)

        wait_round = resolve_situated_round(
            model.world_model,
            state,
            (_action("alice-wait", "alice", SituatedActionKind.WAIT),),
        )
        wait_event = _event_for(wait_round, "alice")
        wait_percepts = tuple(
            item for item in project_situated_percepts(model, wait_round).percepts
            if item.source_event_id == wait_event.event_id
        )
        self.assertEqual(tuple(item.agent_id for item in wait_percepts), ("alice",))

        for kind, target_id in (
            (SituatedActionKind.MOVE, "meeting-door"),
            (SituatedActionKind.TAKE, "missing"),
            (SituatedActionKind.DROP, "missing"),
        ):
            with self.subTest(kind=kind):
                round_result = resolve_situated_round(
                    model.world_model,
                    state,
                    (_action(f"alice-{kind.value}", "alice", kind, target_id),),
                )
                event = _event_for(round_result, "alice")
                percepts = tuple(
                    item for item in project_situated_percepts(model, round_result).percepts
                    if item.source_event_id == event.event_id and item.agent_id == "carol"
                )
                self.assertEqual(len(percepts), 1)
                percept = percepts[0]
                self.assertEqual(percept.channels, (ObservationChannel.VISUAL,))
                self.assertEqual(percept.fidelity, SituatedPerceptFidelity.IDENTIFIED)
                self.assertEqual((percept.actor_agent_id, percept.kind, percept.place_id), ("alice", kind, "corridor"))
                self.assertIsNone(percept.outcome)
                self.assertEqual(percept.details, ())

    def test_projection_replays_canonically_when_model_edges_and_state_agents_are_reordered(self) -> None:
        left_model = _projection_model()
        right_model = _projection_model(edge_order_reversed=True)
        left_state = _projection_state(left_model)
        right_initial = _projection_state(right_model)
        right_state = SituatedWorldState(
            right_initial.model_id,
            right_initial.model_hash,
            right_initial.round_index,
            right_initial.parent_state_hash,
            tuple(reversed(right_initial.agents)),
            right_initial.objects,
            right_initial.passages,
        )
        intents = (_action("alice-tell", "alice", SituatedActionKind.TELL, message="The meeting is confidential."),)

        left_round = resolve_situated_round(left_model.world_model, left_state, intents)
        right_round = resolve_situated_round(right_model.world_model, right_state, intents)
        left = project_situated_percepts(left_model, left_round)
        right = project_situated_percepts(right_model, right_round)

        self.assertEqual(left_model, right_model)
        self.assertEqual(left_state, right_state)
        self.assertEqual(left_round, right_round)
        self.assertEqual(left, right)
        self.assertEqual(left.content_hash, right.content_hash)


if __name__ == "__main__":
    unittest.main()
