from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from narrative_dynamics.abm.adaptive_contracts import EdgeTrustState
from narrative_dynamics.abm.autonomy_contracts import (
    AgentActionIntent,
    AutonomousAgentState,
    AutonomousNetworkModel,
    AutonomousPopulationState,
    SharingDecision,
    TruthObservation,
)
from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentState,
    PopulationState,
    SocialNetwork,
    _hash,
    _nonnegative_integer,
    _text,
)
from narrative_dynamics.abm.evolving import (
    _apply_events,
    _canonical_events,
    _validate_evolving_model_state,
)
from narrative_dynamics.abm.evolving_contracts import EvolvingPopulationState
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.learning import EdgeTrustUpdate
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleMemberState,
    LifecycleStatus,
    PopulationLifecycleEvent,
)
from narrative_dynamics.abm.rewiring import EdgeRewiringUpdate
from narrative_dynamics.abm.rewiring_contracts import EdgeTopologyState
from narrative_dynamics.abm.simulation import InformationTransmission, simulate_round
from narrative_dynamics.contracts import stable_content_hash


def _transmission_identity(
    value: InformationTransmission,
) -> tuple[str, str, str]:
    return (
        value.source_agent_id,
        value.target_agent_id,
        value.relation_type,
    )


def _edge_value_identity(value) -> tuple[str, str, str]:
    return value.edge.identity


def _event_key(value: PopulationLifecycleEvent) -> str:
    return value.agent_id


def _intent_key(value: AgentActionIntent) -> str:
    return value.agent_id


def _validate_autonomous_model_state(
    model: AutonomousNetworkModel,
    state: AutonomousPopulationState,
) -> None:
    if not isinstance(model, AutonomousNetworkModel):
        raise TypeError("autonomy simulation requires AutonomousNetworkModel")
    if not isinstance(state, AutonomousPopulationState):
        raise TypeError("autonomy simulation requires AutonomousPopulationState")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("autonomy state does not bind exact model identity")
    if state.round_index != state.evolving_state.round_index:
        raise ValueError("autonomy and evolving round index must match")
    _validate_evolving_model_state(model.evolving_model, state.evolving_state)
    resources = {item.agent_id: item for item in state.agents}
    members = {item.agent_id: item for item in state.evolving_state.members}
    catalog_ids = set(model.evolving_model.base_model.network.agent_ids)
    if set(resources) != catalog_ids:
        raise ValueError("autonomy state must contain the exact agent resource catalog")
    profiles = {
        item.agent_id: item for item in model.evolving_model.base_model.agents
    }
    policies = {item.role: item for item in model.role_policies}
    for agent_id, resource in resources.items():
        policy = policies[profiles[agent_id].role]
        if (
            resource.remaining_verification_budget + resource.verification_count
            != policy.initial_verification_budget
        ):
            raise ValueError("autonomy verification budget history is inconsistent")
        if members[agent_id].status is not LifecycleStatus.ACTIVE and resource.sharing:
            raise ValueError("inactive or dead autonomous agent cannot share")
    if state.round_index == 0:
        for agent_id, resource in resources.items():
            member = members[agent_id]
            profile = profiles[agent_id]
            policy = policies[profile.role]
            expected = (
                member.status is LifecycleStatus.ACTIVE
                and policy.sharing_enabled
                and member.belief >= policy.share_belief_threshold
                and member.belief >= profile.broadcast_threshold
            )
            if resource.sharing is not expected:
                raise ValueError("round-zero autonomy sharing is inconsistent with policy")
            if resource.decision_count != 0:
                raise ValueError("round-zero autonomy state cannot contain decisions")


def _canonical_role_assignment(
    model: AutonomousNetworkModel,
    role_by_agent: Mapping[str, str] | None,
) -> dict[str, str]:
    profiles = {
        item.agent_id: item for item in model.evolving_model.base_model.agents
    }
    if role_by_agent is None:
        return {
            agent_id: profiles[agent_id].role
            for agent_id in sorted(profiles)
        }
    if not isinstance(role_by_agent, Mapping):
        raise TypeError("autonomy role assignment must be a mapping")
    if set(role_by_agent) != set(profiles):
        raise ValueError("autonomy role assignment must cover exact agent catalog")
    configured_roles = {item.role for item in model.role_policies}
    if any(
        not isinstance(role, str) or role not in configured_roles
        for role in role_by_agent.values()
    ):
        raise ValueError("autonomy role assignment must use configured roles")
    return {
        agent_id: role_by_agent[agent_id]
        for agent_id in sorted(profiles)
    }


def _prepare_resources_after_environment(
    model: AutonomousNetworkModel,
    state: AutonomousPopulationState,
    members: tuple[LifecycleMemberState, ...],
    events: tuple[PopulationLifecycleEvent, ...],
    role_by_agent: Mapping[str, str],
) -> tuple[AutonomousAgentState, ...]:
    event_by_id = {item.agent_id: item for item in events}
    member_by_id = {item.agent_id: item for item in members}
    profiles = {
        item.agent_id: item for item in model.evolving_model.base_model.agents
    }
    policies = {item.role: item for item in model.role_policies}
    result: list[AutonomousAgentState] = []
    for prior in state.agents:
        member = member_by_id[prior.agent_id]
        item = event_by_id.get(prior.agent_id)
        sharing = prior.sharing
        if member.status is not LifecycleStatus.ACTIVE:
            sharing = False
        elif item is not None and item.kind is LifecycleEventKind.ENTER:
            profile = profiles[prior.agent_id]
            policy = policies[role_by_agent[prior.agent_id]]
            sharing = (
                policy.sharing_enabled
                and member.belief >= policy.share_belief_threshold
                and member.belief >= profile.broadcast_threshold
            )
        result.append(replace(prior, sharing=sharing))
    return tuple(result)


def _active_projection(
    model: AutonomousNetworkModel,
    state: AutonomousPopulationState,
    members: tuple[LifecycleMemberState, ...],
    resources: tuple[AutonomousAgentState, ...],
) -> tuple[NetworkABMModel, PopulationState] | None:
    active_ids = tuple(
        item.agent_id for item in members if item.status is LifecycleStatus.ACTIVE
    )
    if not active_ids:
        return None
    active = set(active_ids)
    sharing = {item.agent_id: item.sharing for item in resources}
    evolving = state.evolving_state
    trust = {item.edge.identity: item.trust for item in evolving.edge_trust}
    topology = {item.edge.identity: item.active for item in evolving.edge_topology}
    base = model.evolving_model.base_model
    edges = tuple(
        replace(
            edge,
            influence=edge.influence * trust[edge.identity],
            active=topology[edge.identity] and sharing[edge.source_agent_id],
        )
        for edge in base.network.edges
        if edge.source_agent_id in active and edge.target_agent_id in active
    )
    effective_model = NetworkABMModel(
        f"{model.model_id}:active:{state.content_hash}",
        model.version,
        tuple(item for item in base.agents if item.agent_id in active),
        SocialNetwork(active_ids, edges),
    )
    effective_state = PopulationState(
        effective_model.model_id,
        effective_model.content_hash,
        state.round_index,
        state.parent_state_hash if state.round_index > 0 else None,
        tuple(
            NetworkAgentState(
                item.agent_id,
                item.belief,
                item.exposure_count,
                item.broadcasting,
            )
            for item in members
            if item.status is LifecycleStatus.ACTIVE
        ),
    )
    return effective_model, effective_state


def _canonical_truth_observations(
    values: tuple[TruthObservation, ...],
) -> tuple[TruthObservation, ...]:
    if not isinstance(values, tuple):
        raise TypeError("autonomy truth observations must be a tuple")
    if any(not isinstance(item, TruthObservation) for item in values):
        raise TypeError("autonomy truth observations must contain TruthObservation values")
    identities = tuple(item.edge.identity for item in values)
    if len(set(identities)) != len(identities):
        raise ValueError("autonomy truth observation edge identities must be unique")
    return tuple(sorted(values, key=_edge_value_identity))


def _decide(
    model: AutonomousNetworkModel,
    state: AutonomousPopulationState,
    members: tuple[LifecycleMemberState, ...],
    resources: tuple[AutonomousAgentState, ...],
    transmissions: tuple[InformationTransmission, ...],
    role_by_agent: Mapping[str, str],
) -> tuple[AgentActionIntent, ...]:
    profiles = {
        item.agent_id: item for item in model.evolving_model.base_model.agents
    }
    policies = {item.role: item for item in model.role_policies}
    resource_by_id = {item.agent_id: item for item in resources}
    trust = {
        item.edge.identity: item.trust for item in state.evolving_state.edge_trust
    }
    incoming: dict[str, list[InformationTransmission]] = {
        item.agent_id: [] for item in members
    }
    for item in transmissions:
        incoming[item.target_agent_id].append(item)
    intents: list[AgentActionIntent] = []
    for member in members:
        if member.status is not LifecycleStatus.ACTIVE:
            continue
        profile = profiles[member.agent_id]
        current_role = role_by_agent[member.agent_id]
        policy = policies[current_role]
        resource = resource_by_id[member.agent_id]
        received = tuple(sorted(incoming[member.agent_id], key=_transmission_identity))
        candidate = min(
            received,
            key=lambda item: (
                trust[_transmission_identity(item)],
                _transmission_identity(item),
            ),
            default=None,
        )
        verification_edge = None
        if (
            candidate is not None
            and resource.remaining_verification_budget > 0
            and trust[_transmission_identity(candidate)]
            <= policy.verify_trust_threshold
        ):
            verification_edge = EdgeSelector(*_transmission_identity(candidate))
        exit_requested = (
            policy.exit_belief_threshold is not None
            and member.belief < policy.exit_belief_threshold
        )
        sharing = (
            not exit_requested
            and policy.sharing_enabled
            and member.belief >= policy.share_belief_threshold
            and member.belief >= profile.broadcast_threshold
        )
        incoming_trust = tuple(trust[_transmission_identity(item)] for item in received)
        intents.append(
            AgentActionIntent(
                state.round_index + 1,
                member.agent_id,
                current_role,
                member.belief,
                len(received),
                (
                    sum(incoming_trust) / len(incoming_trust)
                    if incoming_trust
                    else None
                ),
                SharingDecision.SHARE if sharing else SharingDecision.SILENT,
                verification_edge,
                exit_requested,
                resource.remaining_verification_budget,
                resource.remaining_verification_budget
                - int(verification_edge is not None),
            )
        )
    return tuple(sorted(intents, key=_intent_key))


@dataclass(frozen=True)
class AutonomousRoundResult:
    """One environment, propagation, decision, learning, and rewiring round."""

    model_id: str
    model_hash: str
    round_index: int
    prior_state: AutonomousPopulationState
    environment_events: tuple[PopulationLifecycleEvent, ...]
    transmissions: tuple[InformationTransmission, ...]
    intents: tuple[AgentActionIntent, ...]
    truth_observations: tuple[TruthObservation, ...]
    trust_updates: tuple[EdgeTrustUpdate, ...]
    autonomous_exits: tuple[PopulationLifecycleEvent, ...]
    rewiring_updates: tuple[EdgeRewiringUpdate, ...]
    next_state: AutonomousPopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="autonomous round model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="autonomous round model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="autonomous round index",
        )
        if round_index == 0:
            raise ValueError("autonomous round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        if not isinstance(self.prior_state, AutonomousPopulationState):
            raise TypeError("autonomous round prior state must be AutonomousPopulationState")
        if not isinstance(self.next_state, AutonomousPopulationState):
            raise TypeError("autonomous round next state must be AutonomousPopulationState")
        specifications = (
            ("environment_events", self.environment_events, PopulationLifecycleEvent, _event_key),
            ("transmissions", self.transmissions, InformationTransmission, _transmission_identity),
            ("intents", self.intents, AgentActionIntent, _intent_key),
            ("truth_observations", self.truth_observations, TruthObservation, _edge_value_identity),
            ("trust_updates", self.trust_updates, EdgeTrustUpdate, _edge_value_identity),
            ("autonomous_exits", self.autonomous_exits, PopulationLifecycleEvent, _event_key),
            ("rewiring_updates", self.rewiring_updates, EdgeRewiringUpdate, _edge_value_identity),
        )
        for field_name, values, value_type, key in specifications:
            if not isinstance(values, tuple):
                raise TypeError(f"autonomous round {field_name} must be a tuple")
            if any(not isinstance(item, value_type) for item in values):
                raise TypeError(
                    f"autonomous round {field_name} must contain {value_type.__name__} values"
                )
            identities = tuple(key(item) for item in values)
            if len(set(identities)) != len(identities):
                raise ValueError(f"autonomous round {field_name} identities must be unique")
            object.__setattr__(self, field_name, tuple(sorted(values, key=key)))
        if any(item.round_index != round_index for item in self.transmissions):
            raise ValueError("autonomous transmissions must bind exact round index")
        selected = {
            item.verification_edge.identity
            for item in self.intents
            if item.verification_edge is not None
        }
        observations = {item.edge.identity for item in self.truth_observations}
        updates = {item.edge.identity for item in self.trust_updates}
        if selected != observations or observations != updates:
            raise ValueError("autonomous truth and trust updates must cover selected edges")
        if any(item.kind is not LifecycleEventKind.EXIT for item in self.autonomous_exits):
            raise ValueError("autonomous exits must contain only EXIT events")
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("autonomous round states must bind exact model identity")
        if round_index != self.prior_state.round_index + 1:
            raise ValueError("autonomous round must immediately follow prior state")
        if self.next_state.round_index != round_index:
            raise ValueError("autonomous next state must bind exact round index")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("autonomous next state must bind exact prior state hash")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "prior_state_hash": self.prior_state.content_hash,
            "environment_events": [item.to_dict() for item in self.environment_events],
            "transmissions": [item.to_dict() for item in self.transmissions],
            "intents": [item.to_dict() for item in self.intents],
            "truth_observations": [item.to_dict() for item in self.truth_observations],
            "trust_updates": [item.to_dict() for item in self.trust_updates],
            "autonomous_exits": [item.to_dict() for item in self.autonomous_exits],
            "rewiring_updates": [item.to_dict() for item in self.rewiring_updates],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AutonomousTrajectory:
    """Validated positive chain of autonomous rounds."""

    model_id: str
    model_hash: str
    initial_state: AutonomousPopulationState
    rounds: tuple[AutonomousRoundResult, ...]
    final_state: AutonomousPopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="autonomous trajectory model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="autonomous trajectory model hash"),
        )
        if not isinstance(self.initial_state, AutonomousPopulationState):
            raise TypeError(
                "autonomous trajectory initial state must be AutonomousPopulationState"
            )
        if not isinstance(self.final_state, AutonomousPopulationState):
            raise TypeError(
                "autonomous trajectory final state must be AutonomousPopulationState"
            )
        if not isinstance(self.rounds, tuple):
            raise TypeError("autonomous trajectory rounds must be a tuple")
        if not self.rounds:
            raise ValueError("autonomous trajectory requires at least one round")
        if any(not isinstance(item, AutonomousRoundResult) for item in self.rounds):
            raise TypeError(
                "autonomous trajectory rounds must contain AutonomousRoundResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("autonomous trajectory states must bind exact model identity")
        current = self.initial_state
        for result in self.rounds:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError("autonomous trajectory rounds must bind exact model identity")
            if result.prior_state != current:
                raise ValueError("autonomous trajectory state chain is discontinuous")
            current = result.next_state
        if current != self.final_state:
            raise ValueError("autonomous trajectory final state must equal chain tail")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "initial_state": self.initial_state.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "final_state": self.final_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _simulate_autonomous_round_with_roles(
    model: AutonomousNetworkModel,
    prior_state: AutonomousPopulationState,
    *,
    environment_events: tuple[PopulationLifecycleEvent, ...] = (),
    truth_observations: tuple[TruthObservation, ...] = (),
    role_by_agent: Mapping[str, str] | None = None,
) -> AutonomousRoundResult:
    """Internal V6 round with an optional exact role assignment for V7."""

    _validate_autonomous_model_state(model, prior_state)
    canonical_roles = _canonical_role_assignment(model, role_by_agent)
    if any(
        isinstance(item, PopulationLifecycleEvent)
        and item.kind is LifecycleEventKind.EXIT
        for item in environment_events
    ):
        raise ValueError("environment EXIT is forbidden in autonomous simulation")
    canonical_events = _canonical_events(model.evolving_model, environment_events)
    canonical_truth = _canonical_truth_observations(truth_observations)
    post_environment_members = _apply_events(
        model.evolving_model,
        prior_state.evolving_state,
        canonical_events,
    )
    prepared_resources = _prepare_resources_after_environment(
        model,
        prior_state,
        post_environment_members,
        canonical_events,
        canonical_roles,
    )
    projection = _active_projection(
        model,
        prior_state,
        post_environment_members,
        prepared_resources,
    )
    transmissions: tuple[InformationTransmission, ...] = ()
    propagated: dict[str, NetworkAgentState] = {}
    if projection is not None:
        effective_model, effective_state = projection
        base_result = simulate_round(effective_model, effective_state)
        transmissions = base_result.transmissions
        propagated = {item.agent_id: item for item in base_result.next_state.agents}
    post_propagation_members = tuple(
        LifecycleMemberState(
            item.agent_id,
            propagated[item.agent_id].belief,
            propagated[item.agent_id].exposure_count,
            propagated[item.agent_id].broadcasting,
            item.status,
            item.entry_count,
            item.exit_count,
        )
        if item.agent_id in propagated
        else item
        for item in post_environment_members
    )
    intents = _decide(
        model,
        prior_state,
        post_propagation_members,
        prepared_resources,
        transmissions,
        canonical_roles,
    )
    selected = {
        item.verification_edge.identity
        for item in intents
        if item.verification_edge is not None
    }
    observations = {item.edge.identity for item in canonical_truth}
    if selected != observations:
        raise ValueError("truth observations must cover exact selected verification edges")
    observation_by_id = {item.edge.identity: item for item in canonical_truth}
    transmission_by_id = {
        _transmission_identity(item): item for item in transmissions
    }
    next_trust: list[EdgeTrustState] = []
    trust_updates: list[EdgeTrustUpdate] = []
    for prior in prior_state.evolving_state.edge_trust:
        observation = observation_by_id.get(prior.edge.identity)
        if observation is None:
            next_trust.append(prior)
            continue
        transmission = transmission_by_id[prior.edge.identity]
        accuracy = 1.0 - abs(transmission.signal - observation.observed_truth)
        learned = prior.trust + model.evolving_model.learning_rate * (
            accuracy - prior.trust
        )
        learned = min(1.0, max(0.0, learned))
        count = prior.feedback_count + 1
        next_trust.append(EdgeTrustState(prior.edge, learned, count))
        trust_updates.append(
            EdgeTrustUpdate(
                prior.edge,
                prior.trust,
                transmission.signal,
                observation.observed_truth,
                accuracy,
                learned,
                count,
            )
        )
    intent_by_id = {item.agent_id: item for item in intents}
    autonomous_exits = tuple(
        PopulationLifecycleEvent(item.agent_id, LifecycleEventKind.EXIT)
        for item in intents
        if item.exit_requested
    )
    exit_ids = {item.agent_id for item in autonomous_exits}
    final_members = tuple(
        LifecycleMemberState(
            item.agent_id,
            item.belief,
            item.exposure_count,
            False,
            LifecycleStatus.INACTIVE,
            item.entry_count,
            item.exit_count + 1,
        )
        if item.agent_id in exit_ids
        else item
        for item in post_propagation_members
    )
    next_resources: list[AutonomousAgentState] = []
    for prior in prepared_resources:
        intent = intent_by_id.get(prior.agent_id)
        if intent is None:
            next_resources.append(replace(prior, sharing=False))
            continue
        silent = intent.sharing_decision is SharingDecision.SILENT
        next_resources.append(
            AutonomousAgentState(
                prior.agent_id,
                intent.sharing_decision is SharingDecision.SHARE,
                intent.verification_budget_after,
                prior.verification_count + int(intent.verification_edge is not None),
                prior.decision_count + 1,
                prior.silent_decision_count + int(silent),
            )
        )
    final_member_by_id = {item.agent_id: item for item in final_members}
    active_ids = {
        item.agent_id
        for item in final_members
        if item.status is LifecycleStatus.ACTIVE
    }
    next_topology: list[EdgeTopologyState] = []
    rewiring_updates: list[EdgeRewiringUpdate] = []
    for prior in prior_state.evolving_state.edge_topology:
        endpoints_active = (
            prior.edge.source_agent_id in active_ids
            and prior.edge.target_agent_id in active_ids
        )
        if not endpoints_active:
            next_topology.append(prior)
            continue
        similarity = 1.0 - abs(
            final_member_by_id[prior.edge.source_agent_id].belief
            - final_member_by_id[prior.edge.target_agent_id].belief
        )
        if similarity >= model.evolving_model.formation_similarity:
            next_active = True
        elif similarity <= model.evolving_model.dissolution_similarity:
            next_active = False
        else:
            next_active = prior.active
        changed = prior.active != next_active
        next_topology.append(
            EdgeTopologyState(
                prior.edge,
                next_active,
                similarity,
                prior.rewiring_count + int(changed),
            )
        )
        rewiring_updates.append(
            EdgeRewiringUpdate(
                prior.edge,
                prior.active,
                similarity,
                next_active,
                changed,
            )
        )
    next_evolving_state = EvolvingPopulationState(
        model.evolving_model.model_id,
        model.evolving_model.content_hash,
        prior_state.round_index + 1,
        prior_state.evolving_state.content_hash,
        final_members,
        tuple(next_trust),
        tuple(next_topology),
    )
    next_state = AutonomousPopulationState(
        model.model_id,
        model.content_hash,
        prior_state.round_index + 1,
        prior_state.content_hash,
        next_evolving_state,
        tuple(next_resources),
    )
    return AutonomousRoundResult(
        model.model_id,
        model.content_hash,
        next_state.round_index,
        prior_state,
        canonical_events,
        transmissions,
        intents,
        canonical_truth,
        tuple(trust_updates),
        autonomous_exits,
        tuple(rewiring_updates),
        next_state,
    )


def simulate_autonomous_round(
    model: AutonomousNetworkModel,
    prior_state: AutonomousPopulationState,
    *,
    environment_events: tuple[PopulationLifecycleEvent, ...] = (),
    truth_observations: tuple[TruthObservation, ...] = (),
) -> AutonomousRoundResult:
    """Run one local autonomous decision round over the evolving network."""

    return _simulate_autonomous_round_with_roles(
        model,
        prior_state,
        environment_events=environment_events,
        truth_observations=truth_observations,
    )


def simulate_autonomous_population(
    model: AutonomousNetworkModel,
    initial_state: AutonomousPopulationState,
    *,
    environment_event_schedule: tuple[tuple[PopulationLifecycleEvent, ...], ...],
    truth_observation_schedule: tuple[tuple[TruthObservation, ...], ...],
) -> AutonomousTrajectory:
    """Run paired environment-event and selected-truth schedules."""

    _validate_autonomous_model_state(model, initial_state)
    if not isinstance(environment_event_schedule, tuple) or not isinstance(
        truth_observation_schedule,
        tuple,
    ):
        raise TypeError("autonomy schedules must be tuples")
    if not environment_event_schedule or not truth_observation_schedule:
        raise ValueError("autonomy schedules must be nonempty")
    if len(environment_event_schedule) != len(truth_observation_schedule):
        raise ValueError("autonomy schedules must have equal lengths")
    if any(not isinstance(item, tuple) for item in environment_event_schedule):
        raise TypeError("autonomy environment schedule entries must be tuples")
    if any(not isinstance(item, tuple) for item in truth_observation_schedule):
        raise TypeError("autonomy truth schedule entries must be tuples")
    current = initial_state
    results: list[AutonomousRoundResult] = []
    for events, observations in zip(
        environment_event_schedule,
        truth_observation_schedule,
        strict=True,
    ):
        result = simulate_autonomous_round(
            model,
            current,
            environment_events=events,
            truth_observations=observations,
        )
        results.append(result)
        current = result.next_state
    return AutonomousTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(results),
        current,
    )


__all__ = (
    "AutonomousRoundResult",
    "AutonomousTrajectory",
    "simulate_autonomous_round",
    "simulate_autonomous_population",
)
