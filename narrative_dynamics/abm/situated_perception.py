"""Deterministic graph reach and interaction queries for situated worlds."""

from __future__ import annotations

import heapq
import math

from narrative_dynamics.abm.situated_contracts import (
    SituatedWorldState,
    validate_situated_state,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedEdgeActivation,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
    SituatedPerceptionReach,
)


def _validate_inputs(model: SituatedPerceptionModel, state: SituatedWorldState) -> None:
    if not isinstance(model, SituatedPerceptionModel):
        raise TypeError("situated perception model must be SituatedPerceptionModel")
    if not isinstance(state, SituatedWorldState):
        raise TypeError("situated perception state must be SituatedWorldState")
    validate_situated_state(model.world_model, state)


def _active(edge: SituatedPerceptionEdge, passages: dict[str, bool]) -> bool:
    if edge.activation is SituatedEdgeActivation.ALWAYS:
        return True
    is_open = passages[edge.passage_id]  # model/state validation guarantees this key.
    if edge.activation is SituatedEdgeActivation.PASSAGE_OPEN:
        return is_open
    return not is_open


def _shortest_costs(
    edges: tuple[SituatedPerceptionEdge, ...],
    *,
    source_place_id: str,
    passages: dict[str, bool],
) -> dict[str, float]:
    adjacency: dict[str, list[SituatedPerceptionEdge]] = {}
    for edge in edges:
        if _active(edge, passages):
            adjacency.setdefault(edge.source_place_id, []).append(edge)
    for candidates in adjacency.values():
        candidates.sort(key=lambda item: (item.target_place_id, item.cost, item.edge_id))

    costs = {source_place_id: 0.0}
    pending: list[tuple[float, str]] = [(0.0, source_place_id)]
    while pending:
        cost, place_id = heapq.heappop(pending)
        if cost != costs.get(place_id):
            continue
        for edge in adjacency.get(place_id, ()):
            candidate = cost + edge.cost
            if not math.isfinite(candidate):
                continue
            prior = costs.get(edge.target_place_id, math.inf)
            if candidate < prior:
                costs[edge.target_place_id] = candidate
                heapq.heappush(pending, (candidate, edge.target_place_id))
    return costs


def derive_situated_perception_reach(
    model: SituatedPerceptionModel,
    state: SituatedWorldState,
    *,
    source_place_id: str,
) -> SituatedPerceptionReach:
    """Derive deterministic visibility, auditory, and direct interaction reach."""

    _validate_inputs(model, state)
    place_ids = {place.place_id for place in model.world_model.places}
    if not isinstance(source_place_id, str) or source_place_id not in place_ids:
        raise ValueError("perception reach source place must reference a world place")
    passages = {passage.passage_id: passage.open for passage in state.passages}
    visual_costs = _shortest_costs(
        tuple(edge for edge in model.edges if edge.layer is SituatedPerceptionLayer.VISIBILITY),
        source_place_id=source_place_id,
        passages=passages,
    )
    auditory_losses = _shortest_costs(
        tuple(edge for edge in model.edges if edge.layer is SituatedPerceptionLayer.AUDITORY),
        source_place_id=source_place_id,
        passages=passages,
    )
    interaction_place_ids = tuple(sorted({
        edge.target_place_id
        for edge in model.edges
        if edge.layer is SituatedPerceptionLayer.INTERACTION
        and _active(edge, passages)
        and edge.source_place_id == source_place_id
    }))
    return SituatedPerceptionReach(
        model_hash=model.content_hash,
        state_hash=state.content_hash,
        source_place_id=source_place_id,
        visual_costs=visual_costs,
        auditory_losses=auditory_losses,
        interaction_place_ids=interaction_place_ids,
    )


def can_situated_agents_interact(
    model: SituatedPerceptionModel,
    state: SituatedWorldState,
    left_agent_id: str,
    right_agent_id: str,
) -> bool:
    """Return whether two agents can interact through co-location or one direct edge."""

    _validate_inputs(model, state)
    bodies = {agent.agent_id: agent for agent in state.agents}
    if left_agent_id not in bodies or right_agent_id not in bodies:
        raise ValueError("interaction query agents must belong to the model")
    if left_agent_id == right_agent_id:
        return False
    left_place_id = bodies[left_agent_id].place_id
    right_place_id = bodies[right_agent_id].place_id
    if left_place_id == right_place_id:
        return True
    passages = {passage.passage_id: passage.open for passage in state.passages}
    return any(
        edge.layer is SituatedPerceptionLayer.INTERACTION
        and edge.source_place_id == left_place_id
        and edge.target_place_id == right_place_id
        and _active(edge, passages)
        for edge in model.edges
    )


__all__ = (
    "derive_situated_perception_reach",
    "can_situated_agents_interact",
)
