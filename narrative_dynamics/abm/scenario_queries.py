"""Pure, capability-scoped projections over one scenario runtime state."""

from __future__ import annotations

from narrative_dynamics.abm.scenario_coordinator_contracts import (
    ScenarioAgentStateView,
    ScenarioCheckpoint,
    ScenarioPublicStateView,
    ScenarioRunStatus,
    ScenarioRunView,
)
from narrative_dynamics.abm.simulation_output import filter_simulation_output
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationOutputAudience,
    SimulationOutputBatch,
    SimulationOutputView,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRuntimeState,
    SituatedNetworkSnapshot,
)


_AGENT_STATE_UNAUTHORIZED = "scenario agent state is not authorized"
_NETWORK_STATE_UNAUTHORIZED = "scenario network state is not authorized"


def _require_runtime_state(state: object) -> SituatedNetworkRuntimeState:
    if not isinstance(state, SituatedNetworkRuntimeState):
        raise TypeError("scenario query state must be SituatedNetworkRuntimeState")
    return state


def _require_capability(capability: object) -> SimulationAudienceCapability:
    if not isinstance(capability, SimulationAudienceCapability):
        raise TypeError(
            "scenario query capability must be SimulationAudienceCapability"
        )
    return capability


def project_scenario_run_view(
    *,
    run_id: str,
    stream_id: str,
    scenario_hash: str,
    coordinator_epoch: int,
    status: ScenarioRunStatus,
    state: SituatedNetworkRuntimeState,
    next_sequence: int,
    output_batches: tuple[SimulationOutputBatch, ...],
    checkpoints: tuple[ScenarioCheckpoint, ...],
    parent_checkpoint_hash: str | None = None,
) -> ScenarioRunView:
    """Return public run metadata without embedding retained state or output."""

    exact_state = _require_runtime_state(state)
    if not isinstance(output_batches, tuple) or any(
        not isinstance(batch, SimulationOutputBatch) for batch in output_batches
    ):
        raise TypeError(
            "scenario run output batches must be a tuple of SimulationOutputBatch values"
        )
    if not isinstance(checkpoints, tuple) or any(
        not isinstance(checkpoint, ScenarioCheckpoint) for checkpoint in checkpoints
    ):
        raise TypeError(
            "scenario run checkpoints must be a tuple of ScenarioCheckpoint values"
        )
    return ScenarioRunView(
        run_id=run_id,
        stream_id=stream_id,
        scenario_hash=scenario_hash,
        coordinator_epoch=coordinator_epoch,
        status=status,
        round_index=exact_state.round_index,
        state_hash=exact_state.content_hash,
        next_sequence=next_sequence,
        output_batch_hashes=tuple(batch.content_hash for batch in output_batches),
        checkpoint_hashes=tuple(
            checkpoint.content_hash for checkpoint in checkpoints
        ),
        parent_checkpoint_hash=parent_checkpoint_hash,
    )


def project_scenario_public_state(
    run_id: str,
    scenario_hash: str,
    state: SituatedNetworkRuntimeState,
) -> ScenarioPublicStateView:
    """Project exact current physical state and aggregate metrics."""

    exact_state = _require_runtime_state(state)
    world = exact_state.story.current_state
    return ScenarioPublicStateView(
        run_id=run_id,
        scenario_hash=scenario_hash,
        round_index=exact_state.round_index,
        state_hash=exact_state.content_hash,
        agent_places=tuple(
            (agent.agent_id, agent.place_id)
            for agent in sorted(world.agents, key=lambda item: item.agent_id)
        ),
        passage_states=tuple(
            (passage.passage_id, passage.open)
            for passage in sorted(world.passages, key=lambda item: item.passage_id)
        ),
        object_placements=tuple(
            (world_object.object_id, world_object.place_id, world_object.holder_agent_id)
            for world_object in sorted(
                world.objects,
                key=lambda item: item.object_id,
            )
        ),
        metrics=exact_state.metrics,
    )


def project_scenario_agent_state(
    run_id: str,
    scenario_hash: str,
    state: SituatedNetworkRuntimeState,
    agent_id: str,
    capability: SimulationAudienceCapability,
) -> ScenarioAgentStateView:
    """Project one Agent's exact private state after authorization."""

    exact_state = _require_runtime_state(state)
    exact_capability = _require_capability(capability)
    authorized = exact_capability.audience is SimulationOutputAudience.INTERNAL or (
        exact_capability.audience is SimulationOutputAudience.AGENT
        and exact_capability.owner_agent_id == agent_id
    )
    if not authorized:
        raise PermissionError(_AGENT_STATE_UNAUTHORIZED)

    mind = next(
        (
            candidate
            for candidate in exact_state.cognitive_state.minds
            if candidate.agent_id == agent_id
        ),
        None,
    )
    body = next(
        (
            candidate
            for candidate in exact_state.story.current_state.agents
            if candidate.agent_id == agent_id
        ),
        None,
    )
    if mind is None or body is None:
        raise ValueError("scenario agent does not exist")

    return ScenarioAgentStateView(
        run_id=run_id,
        scenario_hash=scenario_hash,
        round_index=exact_state.round_index,
        state_hash=exact_state.content_hash,
        agent_id=agent_id,
        place_id=body.place_id,
        mind=mind,
        relationships=tuple(
            relationship
            for relationship in exact_state.social_state.relationships
            if relationship.observer_agent_id == agent_id
        ),
        claims=tuple(
            claim
            for claim in exact_state.social_state.claims
            if claim.observer_agent_id == agent_id
        ),
    )


def project_scenario_network_state(
    state: SituatedNetworkRuntimeState,
    capability: SimulationAudienceCapability,
) -> SituatedNetworkSnapshot:
    """Return the exact network snapshot only to aggregate-state audiences."""

    exact_state = _require_runtime_state(state)
    exact_capability = _require_capability(capability)
    if exact_capability.audience not in (
        SimulationOutputAudience.OBJECTIVE,
        SimulationOutputAudience.ANALYST,
        SimulationOutputAudience.INTERNAL,
    ):
        raise PermissionError(_NETWORK_STATE_UNAUTHORIZED)
    return exact_state.snapshot


def project_scenario_output_view(
    batch: SimulationOutputBatch,
    capability: SimulationAudienceCapability,
) -> SimulationOutputView:
    """Delegate retained output projection to the V21.2 capability boundary."""

    return filter_simulation_output(batch, capability)


__all__ = (
    "project_scenario_run_view",
    "project_scenario_public_state",
    "project_scenario_agent_state",
    "project_scenario_network_state",
    "project_scenario_output_view",
)
