"""Deterministic graph reach and interaction queries for situated worlds."""

from __future__ import annotations

import heapq
import math

from narrative_dynamics.abm.situated_contracts import (
    SituatedWorldState,
    validate_situated_state,
)
from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionKind,
    SituatedRoundResult,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedEdgeActivation,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
    SituatedPerceptionReach,
    SituatedPercept,
    SituatedPerceptFidelity,
    SituatedPerceptualProjection,
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


def project_situated_percepts(
    model: SituatedPerceptionModel,
    round_result: SituatedRoundResult,
) -> SituatedPerceptualProjection:
    """Project one resolved round into private, sanitized actor percepts."""

    if not isinstance(round_result, SituatedRoundResult):
        raise TypeError("situated percept projection requires SituatedRoundResult")
    _validate_inputs(model, round_result.prior_state)
    _validate_inputs(model, round_result.next_state)

    percepts = []
    bodies = {item.agent_id: item for item in round_result.prior_state.agents}
    profiles = {item.agent_id: item for item in model.agent_profiles}
    signals = {item.kind: item for item in model.signal_profiles}
    reaches = {
        place_id: derive_situated_perception_reach(
            model,
            round_result.prior_state,
            source_place_id=place_id,
        )
        for place_id in {item.place_id for item in round_result.events}
    }
    for event in round_result.events:
        channel = (
            ObservationChannel.INSPECTION
            if event.kind is SituatedActionKind.INSPECT
            else ObservationChannel.SELF
        )
        percepts.append(SituatedPercept(
            percept_id=f"{event.event_id}:p:{event.actor_agent_id}",
            round_index=event.round_index,
            agent_id=event.actor_agent_id,
            source_event_id=event.event_id,
            source_event_hash=event.content_hash,
            channels=(channel,),
            fidelity=SituatedPerceptFidelity.EXACT,
            actor_agent_id=event.actor_agent_id,
            kind=event.kind,
            place_id=event.place_id,
            outcome=event.outcome,
            details=event.details,
        ))
        if event.kind is SituatedActionKind.INSPECT:
            continue
        signal = signals.get(event.kind)
        if signal is None:
            continue
        reach = reaches[event.place_id]
        for agent_id in sorted(bodies):
            if agent_id == event.actor_agent_id:
                continue
            profile = profiles[agent_id]
            observer_place_id = bodies[agent_id].place_id
            visual_cost = reach.visual_costs.get(observer_place_id)
            visible = (
                signal.visually_observable
                and visual_cost is not None
                and visual_cost <= profile.max_visual_cost
            )
            auditory_loss = reach.auditory_losses.get(observer_place_id)
            received_sound = (
                None
                if signal.auditory_intensity is None or auditory_loss is None
                else signal.auditory_intensity - auditory_loss
            )
            clear = (
                received_sound is not None
                and received_sound >= profile.minimum_clear_sound
            )
            detected = (
                received_sound is not None
                and received_sound >= profile.minimum_detectable_sound
            )
            channels = tuple(
                channel
                for channel, accessible in (
                    (ObservationChannel.VISUAL, visible),
                    (ObservationChannel.AUDITORY, detected),
                )
                if accessible
            )
            if clear:
                fidelity = SituatedPerceptFidelity.EXACT
            elif visible:
                fidelity = SituatedPerceptFidelity.IDENTIFIED
            elif detected:
                fidelity = SituatedPerceptFidelity.DETECTED
            else:
                continue
            percepts.append(SituatedPercept(
                percept_id=f"{event.event_id}:p:{agent_id}",
                round_index=event.round_index,
                agent_id=agent_id,
                source_event_id=event.event_id,
                source_event_hash=event.content_hash,
                channels=channels,
                fidelity=fidelity,
                actor_agent_id=event.actor_agent_id if fidelity is not SituatedPerceptFidelity.DETECTED else None,
                kind=event.kind if fidelity is not SituatedPerceptFidelity.DETECTED else None,
                place_id=event.place_id if fidelity is not SituatedPerceptFidelity.DETECTED else None,
                outcome=event.outcome if fidelity is SituatedPerceptFidelity.EXACT else None,
                details=event.details if fidelity is SituatedPerceptFidelity.EXACT else (),
            ))
    return SituatedPerceptualProjection(
        model_id=model.model_id,
        model_hash=model.content_hash,
        prior_state_hash=round_result.prior_state.content_hash,
        round_result_hash=round_result.content_hash,
        percepts=tuple(percepts),
    )


def percepts_for_agent(
    projection: SituatedPerceptualProjection,
    agent_id: str,
) -> tuple[SituatedPercept, ...]:
    """Return the canonical percept sequence addressed to one agent."""

    if not isinstance(projection, SituatedPerceptualProjection):
        raise TypeError("situated percept query requires SituatedPerceptualProjection")
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("situated percept query agent id must be a non-empty string")
    return tuple(item for item in projection.percepts if item.agent_id == agent_id)


__all__ = (
    "derive_situated_perception_reach",
    "can_situated_agents_interact",
    "project_situated_percepts",
    "percepts_for_agent",
)
