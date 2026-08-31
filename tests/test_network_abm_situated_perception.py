from __future__ import annotations

import unittest

from narrative_dynamics.abm.situated import SituatedActionKind
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
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedAgentPerceptionProfile,
    SituatedEdgeActivation,
    SituatedPerceptionEdge,
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


if __name__ == "__main__":
    unittest.main()
