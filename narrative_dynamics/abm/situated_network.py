"""Read-only V19 multiplex-network projections over situated runtime state."""

from __future__ import annotations

from pathlib import Path

from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedCognitiveState,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkAccessEdge,
    SituatedNetworkAgentNode,
    SituatedNetworkEmergenceMetrics,
    SituatedNetworkRelationshipEdge,
    SituatedNetworkRoundResult,
    SituatedNetworkRuntimeModel,
    SituatedNetworkRuntimeState,
    SituatedNetworkSnapshot,
    SituatedNetworkTransmission,
    SituatedNetworkTrajectory,
)
from narrative_dynamics.abm.situated_perception import (
    can_situated_agents_interact,
    derive_situated_perception_reach,
    project_situated_percepts,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedSocialMemoryState,
    validate_situated_social_memory_state,
)
from narrative_dynamics.abm.situated_story import SituatedStory


def _validate_projection_inputs(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> None:
    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("situated network projection requires a runtime model")
    if not isinstance(story, SituatedStory):
        raise TypeError("situated network projection requires a situated story")
    if not isinstance(cognitive_state, SituatedCognitiveState):
        raise TypeError("situated network projection requires a cognitive state")
    if not isinstance(social_state, SituatedSocialMemoryState):
        raise TypeError("situated network projection requires a social state")

    percept_memory = model.percept_memory_model
    cognitive = percept_memory.cognitive_model
    if story.perception_model != percept_memory.perception_model:
        raise ValueError("situated network story must bind the exact perception model")
    if (
        story.model_id != cognitive.world_model.model_id
        or story.model_hash != cognitive.world_model.content_hash
    ):
        raise ValueError("situated network story must bind the exact world model")
    validate_situated_cognitive_state(cognitive, story, cognitive_state)
    validate_situated_social_memory_state(
        model.social_memory_model, cognitive_state, social_state
    )


def _network_nodes(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> tuple[SituatedNetworkAgentNode, ...]:
    bodies = {item.agent_id: item for item in story.current_state.agents}
    roles = {
        item.agent_id: item.role
        for item in model.percept_memory_model.cognitive_model.world_model.agents
    }
    active_claims = {
        agent_id: sum(
            item.observer_agent_id == agent_id
            and item.status is SituatedClaimStatus.ACTIVE
            for item in social_state.claims
        )
        for agent_id in bodies
    }
    return tuple(
        SituatedNetworkAgentNode(
            agent_id,
            roles[agent_id],
            body.place_id,
            next(
                item.belief.probabilities[model.tracked_hypothesis_id]
                for item in cognitive_state.minds
                if item.agent_id == agent_id
            ),
            active_claims[agent_id],
        )
        for agent_id, body in bodies.items()
    )


def _relationship_edges(
    model: SituatedNetworkRuntimeModel,
    social_state: SituatedSocialMemoryState,
) -> tuple[SituatedNetworkRelationshipEdge, ...]:
    return tuple(
        SituatedNetworkRelationshipEdge(
            item.source_agent_id,
            item.observer_agent_id,
            item.trust,
            item.affinity,
            item.confirmation_count,
            item.contradiction_count,
            item.trust >= model.relationship_trust_threshold,
        )
        for item in social_state.relationships
    )


def _access_edges(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
) -> tuple[SituatedNetworkAccessEdge, ...]:
    state = story.current_state
    bodies = {item.agent_id: item for item in state.agents}
    perception = model.percept_memory_model.perception_model
    reaches = {
        agent_id: derive_situated_perception_reach(
            perception, state, source_place_id=body.place_id
        )
        for agent_id, body in bodies.items()
    }
    return tuple(
        SituatedNetworkAccessEdge(
            source_agent_id,
            observer_agent_id,
            reaches[source_agent_id].visual_costs.get(observer.place_id),
            reaches[source_agent_id].auditory_losses.get(observer.place_id),
            can_situated_agents_interact(
                perception, state, source_agent_id, observer_agent_id
            ),
        )
        for source_agent_id in bodies
        for observer_agent_id, observer in bodies.items()
        if source_agent_id != observer_agent_id
    )


def _latest_tell_transmissions(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
) -> tuple[SituatedNetworkTransmission, ...]:
    if not story.rounds:
        return ()
    latest_round = story.rounds[-1]
    tell_events = {
        item.event_id: item
        for item in latest_round.events
        if item.kind is SituatedActionKind.TELL and item.success
    }
    projection = project_situated_percepts(
        model.percept_memory_model.perception_model, latest_round
    )
    return tuple(
        SituatedNetworkTransmission(
            percept.source_event_id,
            tell_events[percept.source_event_id].actor_agent_id,
            percept.agent_id,
            percept.fidelity,
            percept.channels,
        )
        for percept in projection.percepts
        if (
            percept.source_event_id in tell_events
            and percept.agent_id
            != tell_events[percept.source_event_id].actor_agent_id
        )
    )


def _latest_successful_tell_event_count(story: SituatedStory) -> int:
    if not story.rounds:
        return 0
    return sum(
        item.kind is SituatedActionKind.TELL and item.success
        for item in story.rounds[-1].events
    )


def project_situated_network_snapshot(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> SituatedNetworkSnapshot:
    """Project the current V15/V14 state without exposing event payloads."""

    _validate_projection_inputs(model, story, cognitive_state, social_state)
    transmissions = _latest_tell_transmissions(model, story)
    return SituatedNetworkSnapshot(
        model.model_id,
        model.content_hash,
        story.current_state.round_index,
        story.content_hash,
        cognitive_state.content_hash,
        social_state.content_hash,
        _network_nodes(model, story, cognitive_state, social_state),
        _relationship_edges(model, social_state),
        _access_edges(model, story),
        transmissions,
        _latest_successful_tell_event_count(story),
    )


def _validate_metric_inputs(
    model: SituatedNetworkRuntimeModel,
    snapshot: SituatedNetworkSnapshot,
    social_state: SituatedSocialMemoryState,
) -> None:
    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("situated network metrics require a runtime model")
    if not isinstance(snapshot, SituatedNetworkSnapshot):
        raise TypeError("situated network metrics require a network snapshot")
    if not isinstance(social_state, SituatedSocialMemoryState):
        raise TypeError("situated network metrics require a social state")
    if snapshot.model_id != model.model_id or snapshot.model_hash != model.content_hash:
        raise ValueError("situated network snapshot must bind the exact runtime model")
    if (
        social_state.model_id != model.social_memory_model.model_id
        or social_state.model_hash != model.social_memory_model.content_hash
    ):
        raise ValueError("situated network social state must bind the exact social model")
    if (
        snapshot.round_index != social_state.round_index
        or snapshot.social_state_hash != social_state.content_hash
        or snapshot.cognitive_state_hash != social_state.cognitive_state_hash
    ):
        raise ValueError("situated network snapshot must bind the exact social state round")


def measure_situated_network_emergence(
    model: SituatedNetworkRuntimeModel,
    snapshot: SituatedNetworkSnapshot,
    social_state: SituatedSocialMemoryState,
) -> SituatedNetworkEmergenceMetrics:
    """Measure aggregate V19 emergence values from an exact network snapshot."""

    _validate_metric_inputs(model, snapshot, social_state)
    population_size = len(snapshot.nodes)
    beliefs = tuple(item.tracked_belief_probability for item in snapshot.nodes)
    belief_mean = sum(beliefs) / population_size
    adopted_count = sum(
        item.tracked_belief_probability >= model.adoption_threshold
        for item in snapshot.nodes
    )
    active_relationships = tuple(
        item for item in snapshot.relationship_edges if item.active
    )
    transmissions = snapshot.transmissions
    fidelity_counts = {
        fidelity: sum(item.fidelity is fidelity for item in transmissions)
        for fidelity in SituatedPerceptFidelity
    }
    return SituatedNetworkEmergenceMetrics(
        snapshot.content_hash,
        snapshot.round_index,
        population_size,
        len({item.place_id for item in snapshot.nodes}),
        adopted_count,
        adopted_count / population_size,
        belief_mean,
        sum((item - belief_mean) ** 2 for item in beliefs) / population_size,
        len(active_relationships),
        (
            sum(item.trust for item in snapshot.relationship_edges)
            / len(snapshot.relationship_edges)
            if snapshot.relationship_edges
            else 0.0
        ),
        sum(item.direct_interaction for item in snapshot.access_edges),
        snapshot.latest_tell_event_count,
        len(transmissions),
        len({item.observer_agent_id for item in transmissions}),
        fidelity_counts[SituatedPerceptFidelity.EXACT],
        fidelity_counts[SituatedPerceptFidelity.DETECTED],
        fidelity_counts[SituatedPerceptFidelity.IDENTIFIED],
        sum(item.status is SituatedClaimStatus.ACTIVE for item in social_state.claims),
        sum(item.status is SituatedClaimStatus.CONFIRMED for item in social_state.claims),
        sum(item.status is SituatedClaimStatus.CONTRADICTED for item in social_state.claims),
        sum(item.status is SituatedClaimStatus.SUPERSEDED for item in social_state.claims),
        sum(item.status is SituatedClaimStatus.FORGOTTEN for item in social_state.claims),
    )


def initialize_situated_network_runtime(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> SituatedNetworkRuntimeState:
    """Bind one validated V15.1/V14 checkpoint to its V19 observability view."""

    snapshot = project_situated_network_snapshot(
        model, story, cognitive_state, social_state
    )
    metrics = measure_situated_network_emergence(model, snapshot, social_state)
    return SituatedNetworkRuntimeState(
        model.model_id,
        model.content_hash,
        story.current_state.round_index,
        None,
        story,
        cognitive_state,
        social_state,
        snapshot,
        metrics,
    )


def _validate_runtime_state(
    model: SituatedNetworkRuntimeModel,
    state: SituatedNetworkRuntimeState,
) -> None:
    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("situated network simulation requires a runtime model")
    if not isinstance(state, SituatedNetworkRuntimeState):
        raise TypeError("situated network simulation requires a runtime state")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("situated network state must bind the exact runtime model")
    validate_situated_cognitive_state(
        model.percept_memory_model.cognitive_model,
        state.story,
        state.cognitive_state,
    )
    validate_situated_social_memory_state(
        model.social_memory_model,
        state.cognitive_state,
        state.social_state,
    )


def simulate_situated_network_round(
    database_path: str | Path,
    model: SituatedNetworkRuntimeModel,
    state: SituatedNetworkRuntimeState,
) -> SituatedNetworkRoundResult:
    """Advance all private state once, then bind one synchronized V19 record."""

    _validate_runtime_state(model, state)
    advanced = simulate_situated_percept_social_cognitive_round(
        database_path,
        model.percept_memory_model,
        model.social_memory_model,
        state.story,
        state.cognitive_state,
        state.social_state,
    )
    snapshot = project_situated_network_snapshot(
        model,
        advanced.next_story,
        advanced.next_cognitive_state,
        advanced.next_social_state,
    )
    metrics = measure_situated_network_emergence(
        model, snapshot, advanced.next_social_state
    )
    next_state = SituatedNetworkRuntimeState(
        model.model_id,
        model.content_hash,
        state.round_index + 1,
        state.content_hash,
        advanced.next_story,
        advanced.next_cognitive_state,
        advanced.next_social_state,
        snapshot,
        metrics,
    )
    return SituatedNetworkRoundResult(
        model.model_id, model.content_hash, state, next_state
    )


def simulate_situated_network_runtime(
    database_path: str | Path,
    model: SituatedNetworkRuntimeModel,
    initial_state: SituatedNetworkRuntimeState,
    *,
    round_count: int,
) -> SituatedNetworkTrajectory:
    """Run a positive number of exact parent-linked V19 rounds."""

    if (
        not isinstance(round_count, int)
        or isinstance(round_count, bool)
        or round_count <= 0
    ):
        raise ValueError("situated network simulation requires a positive round count")
    _validate_runtime_state(model, initial_state)
    state = initial_state
    rounds = []
    for _ in range(round_count):
        item = simulate_situated_network_round(database_path, model, state)
        rounds.append(item)
        state = item.next_state
    return SituatedNetworkTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(rounds),
        state,
    )


__all__ = (
    "project_situated_network_snapshot",
    "measure_situated_network_emergence",
    "initialize_situated_network_runtime",
    "simulate_situated_network_round",
    "simulate_situated_network_runtime",
)
