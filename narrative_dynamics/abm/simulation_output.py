"""Pure deterministic projection of accepted V19 rounds into V21 output batches."""

from __future__ import annotations

from dataclasses import dataclass, replace

from narrative_dynamics.abm.scenario_package_contracts import (
    CompiledSituatedScenario,
)
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAgentDecisionPayload,
    SimulationAudienceCapability,
    SimulationBlenderDeltaPayload,
    SimulationMemoryUpdatePayload,
    SimulationNetworkMetricsPayload,
    SimulationObjectiveEventPayload,
    SimulationOutputAudience,
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputPayload,
    SimulationOutputRecord,
    SimulationOutputView,
    SimulationPrivatePerceptPayload,
    SimulationSocialUpdatePayload,
    SimulationStateDeltaPayload,
    output_record_sort_key,
)
from narrative_dynamics.abm.situated_contracts import SituatedWorldState
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRoundResult,
    SituatedNetworkRuntimeState,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    SituatedPerceptSocialCognitiveRoundResult,
)


@dataclass(frozen=True)
class _OutputCandidate:
    kind: SimulationOutputKind
    audience: SimulationOutputAudience
    owner_agent_id: str | None
    source_artifact_hashes: tuple[str, ...]
    payload: SimulationOutputPayload


def _world_state(state: SituatedNetworkRuntimeState) -> SituatedWorldState:
    return state.story.current_state


def _values_by_id(values: tuple[object, ...], attribute: str) -> dict[str, object]:
    return {getattr(value, attribute): value for value in values}


def _changed_ids(
    prior_values: tuple[object, ...],
    next_values: tuple[object, ...],
    attribute: str,
) -> tuple[str, ...]:
    prior_by_id = _values_by_id(prior_values, attribute)
    next_by_id = _values_by_id(next_values, attribute)
    return tuple(
        identity
        for identity in sorted(set(prior_by_id) | set(next_by_id))
        if prior_by_id.get(identity) != next_by_id.get(identity)
    )


def _objective_event_payloads(
    round_result: SituatedNetworkRoundResult,
) -> tuple[SimulationObjectiveEventPayload, ...]:
    world_round = round_result.next_state.story.rounds[-1]
    return tuple(
        SimulationObjectiveEventPayload(
            event.event_id,
            event.content_hash,
            event.action_id,
            event.kind.value,
            event.actor_agent_id,
            event.place_id,
            event.target_id,
            event.success,
            event.cause_event_ids,
        )
        for event in world_round.events
    )


def _state_delta_payload(
    prior_state: SituatedNetworkRuntimeState,
    next_state: SituatedNetworkRuntimeState,
) -> SimulationStateDeltaPayload:
    prior_world = _world_state(prior_state)
    next_world = _world_state(next_state)
    return SimulationStateDeltaPayload(
        prior_world.content_hash,
        next_world.content_hash,
        _changed_ids(prior_world.agents, next_world.agents, "agent_id"),
        _changed_ids(prior_world.passages, next_world.passages, "passage_id"),
        _changed_ids(prior_world.objects, next_world.objects, "object_id"),
    )


def _private_percept_payloads(
    transition: SituatedPerceptSocialCognitiveRoundResult,
) -> tuple[SimulationPrivatePerceptPayload, ...]:
    projection = transition.cognitive_round.percept_cognitive_round.next_projection
    return tuple(
        SimulationPrivatePerceptPayload(percept)
        for percept in projection.percepts
    )


def _decision_payloads(
    transition: SituatedPerceptSocialCognitiveRoundResult,
) -> tuple[SimulationAgentDecisionPayload, ...]:
    return tuple(
        SimulationAgentDecisionPayload(decision)
        for decision in transition.decisions
    )


def _memory_payloads(
    transition: SituatedPerceptSocialCognitiveRoundResult,
) -> tuple[SimulationMemoryUpdatePayload, ...]:
    cognitive_round = transition.cognitive_round.cognitive_round
    prior_minds = {
        mind.agent_id: mind
        for mind in cognitive_round.prior_state.minds
    }
    next_minds = {
        mind.agent_id: mind
        for mind in transition.next_cognitive_state.minds
    }
    decisions = {decision.agent_id: decision for decision in transition.decisions}
    recalls = {
        recall.prior_mind.agent_id: recall
        for recall in transition.recalls
    }
    return tuple(
        SimulationMemoryUpdatePayload(
            agent_id,
            prior_minds[agent_id].content_hash,
            next_minds[agent_id].content_hash,
            tuple(
                admission.memory_id
                for admission in recalls[agent_id].admissions
            ),
            decisions[agent_id].admitted_observation_ids,
        )
        for agent_id in sorted(prior_minds)
    )


def _social_payloads(
    transition: SituatedPerceptSocialCognitiveRoundResult,
) -> tuple[SimulationSocialUpdatePayload, ...]:
    update = transition.social_update
    observer_ids = tuple(
        mind.agent_id for mind in transition.next_cognitive_state.minds
    )
    return tuple(
        SimulationSocialUpdatePayload(
            observer_id,
            tuple(
                claim.content_hash
                for claim in update.prior_state.claims
                if claim.observer_agent_id == observer_id
            ),
            tuple(
                claim.content_hash
                for claim in update.next_state.claims
                if claim.observer_agent_id == observer_id
            ),
            tuple(
                relationship.content_hash
                for relationship in update.prior_state.relationships
                if relationship.observer_agent_id == observer_id
            ),
            tuple(
                relationship.content_hash
                for relationship in update.next_state.relationships
                if relationship.observer_agent_id == observer_id
            ),
            tuple(
                evidence.evidence_id
                for evidence in update.admitted_evidence
                if evidence.observer_agent_id == observer_id
            ),
        )
        for observer_id in observer_ids
    )


def _network_metrics_payload(
    next_state: SituatedNetworkRuntimeState,
) -> SimulationNetworkMetricsPayload:
    return SimulationNetworkMetricsPayload(next_state.metrics)


def _blender_delta_payload(
    prior_state: SituatedNetworkRuntimeState,
    next_state: SituatedNetworkRuntimeState,
) -> SimulationBlenderDeltaPayload:
    prior_world = _world_state(prior_state)
    next_world = _world_state(next_state)
    changed_agent_ids = set(
        _changed_ids(prior_world.agents, next_world.agents, "agent_id")
    )
    changed_passage_ids = set(
        _changed_ids(prior_world.passages, next_world.passages, "passage_id")
    )
    changed_object_ids = set(
        _changed_ids(prior_world.objects, next_world.objects, "object_id")
    )
    return SimulationBlenderDeltaPayload(
        tuple(
            (agent.agent_id, agent.place_id)
            for agent in next_world.agents
            if agent.agent_id in changed_agent_ids
        ),
        tuple(
            (passage.passage_id, passage.open)
            for passage in next_world.passages
            if passage.passage_id in changed_passage_ids
        ),
        tuple(
            (obj.object_id, obj.place_id, obj.holder_agent_id)
            for obj in next_world.objects
            if obj.object_id in changed_object_ids
        ),
    )


def _validate_projection_binding(
    scenario: CompiledSituatedScenario,
    round_result: SituatedNetworkRoundResult,
) -> SituatedPerceptSocialCognitiveRoundResult:
    if not isinstance(scenario, CompiledSituatedScenario):
        raise TypeError("simulation output projection requires CompiledSituatedScenario")
    if not isinstance(round_result, SituatedNetworkRoundResult):
        raise TypeError("simulation output projection requires SituatedNetworkRoundResult")
    runtime_model = scenario.runtime_model
    if (
        round_result.model_id != runtime_model.model_id
        or round_result.model_hash != runtime_model.content_hash
    ):
        raise ValueError("simulation output round must bind the exact runtime model")
    transition = round_result.transition
    if transition is None:
        raise ValueError("transition evidence")
    if (
        transition.percept_memory_model_id
        != runtime_model.percept_memory_model.model_id
        or transition.percept_memory_model_hash
        != runtime_model.percept_memory_model.content_hash
        or transition.social_model_id != runtime_model.social_memory_model.model_id
        or transition.social_model_hash
        != runtime_model.social_memory_model.content_hash
    ):
        raise ValueError("simulation output transition must bind the exact runtime model")
    return transition


def _projection_candidates(
    round_result: SituatedNetworkRoundResult,
    transition: SituatedPerceptSocialCognitiveRoundResult,
) -> tuple[_OutputCandidate, ...]:
    prior_state = round_result.prior_state
    next_state = round_result.next_state
    prior_world = _world_state(prior_state)
    next_world = _world_state(next_state)
    candidates: list[_OutputCandidate] = []

    for payload in _objective_event_payloads(round_result):
        candidates.append(_OutputCandidate(
            SimulationOutputKind.EVENT_OBJECTIVE,
            SimulationOutputAudience.PUBLIC,
            None,
            (payload.event_hash,),
            payload,
        ))
    candidates.append(_OutputCandidate(
        SimulationOutputKind.STATE_DELTA,
        SimulationOutputAudience.PUBLIC,
        None,
        (prior_world.content_hash, next_world.content_hash),
        _state_delta_payload(prior_state, next_state),
    ))

    for payload in _private_percept_payloads(transition):
        candidates.append(_OutputCandidate(
            SimulationOutputKind.PERCEPT_PRIVATE,
            SimulationOutputAudience.AGENT,
            payload.percept.agent_id,
            (payload.percept.content_hash,),
            payload,
        ))
    for payload in _decision_payloads(transition):
        candidates.append(_OutputCandidate(
            SimulationOutputKind.AGENT_DECISION,
            SimulationOutputAudience.AGENT,
            payload.decision.agent_id,
            (payload.decision.content_hash,),
            payload,
        ))

    prior_minds = {
        mind.agent_id: mind
        for mind in transition.cognitive_round.cognitive_round.prior_state.minds
    }
    next_minds = {
        mind.agent_id: mind
        for mind in transition.next_cognitive_state.minds
    }
    decisions = {decision.agent_id: decision for decision in transition.decisions}
    recalls = {
        recall.prior_mind.agent_id: recall
        for recall in transition.recalls
    }
    for payload in _memory_payloads(transition):
        agent_id = payload.agent_id
        candidates.append(_OutputCandidate(
            SimulationOutputKind.MEMORY_UPDATE,
            SimulationOutputAudience.AGENT,
            agent_id,
            (
                prior_minds[agent_id].content_hash,
                next_minds[agent_id].content_hash,
                decisions[agent_id].content_hash,
                recalls[agent_id].content_hash,
            ),
            payload,
        ))

    for payload in _social_payloads(transition):
        candidates.append(_OutputCandidate(
            SimulationOutputKind.SOCIAL_UPDATE,
            SimulationOutputAudience.AGENT,
            payload.observer_agent_id,
            (transition.social_update.content_hash,),
            payload,
        ))

    metrics_payload = _network_metrics_payload(next_state)
    candidates.append(_OutputCandidate(
        SimulationOutputKind.NETWORK_METRICS,
        SimulationOutputAudience.PUBLIC,
        None,
        (
            next_state.metrics.content_hash,
            next_state.metrics.snapshot_hash,
        ),
        metrics_payload,
    ))
    candidates.append(_OutputCandidate(
        SimulationOutputKind.BLENDER_DELTA,
        SimulationOutputAudience.PUBLIC,
        None,
        (prior_world.content_hash, next_world.content_hash),
        _blender_delta_payload(prior_state, next_state),
    ))
    return tuple(candidates)


def _candidate_record(
    candidate: _OutputCandidate,
    *,
    scenario_hash: str,
    stream_id: str,
    sequence: int,
    round_index: int,
    state_hash: str,
) -> SimulationOutputRecord:
    return SimulationOutputRecord(
        stream_id,
        scenario_hash,
        sequence,
        round_index,
        state_hash,
        candidate.kind,
        candidate.audience,
        candidate.owner_agent_id,
        candidate.source_artifact_hashes,
        candidate.payload,
    )


def project_simulation_output(
    scenario: CompiledSituatedScenario,
    round_result: SituatedNetworkRoundResult,
    *,
    stream_id: str,
    first_sequence: int = 1,
) -> SimulationOutputBatch:
    """Project one accepted evidence-bearing round without external side effects."""

    transition = _validate_projection_binding(scenario, round_result)
    allowed_kinds = set(scenario.run_policy.allowed_output_kinds)
    candidates = tuple(
        candidate
        for candidate in _projection_candidates(round_result, transition)
        if candidate.kind.value in allowed_kinds
    )
    provisional = tuple(
        _candidate_record(
            candidate,
            scenario_hash=scenario.content_hash,
            stream_id=stream_id,
            sequence=first_sequence + index,
            round_index=round_result.next_state.round_index,
            state_hash=round_result.next_state.content_hash,
        )
        for index, candidate in enumerate(candidates)
    )
    ordered = tuple(sorted(provisional, key=output_record_sort_key))
    records = tuple(
        replace(record, sequence=first_sequence + index)
        for index, record in enumerate(ordered)
    )
    return SimulationOutputBatch(
        stream_id,
        scenario.content_hash,
        round_result.prior_state.content_hash,
        round_result.next_state.content_hash,
        round_result.content_hash,
        first_sequence,
        first_sequence + len(records) - 1,
        records,
        checkpoint=round_result.next_state.checkpoint,
    )


def filter_simulation_output(
    batch: SimulationOutputBatch,
    capability: SimulationAudienceCapability,
) -> SimulationOutputView:
    """Return a typed audience view without serialization."""

    return SimulationOutputView.from_batch(batch, capability)


__all__ = (
    "project_simulation_output",
    "filter_simulation_output",
)
