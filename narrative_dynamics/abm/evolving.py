from __future__ import annotations

from dataclasses import dataclass, replace
import math

from narrative_dynamics.abm.adaptive_contracts import EdgeTrustState, TruthFeedback
from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentState,
    PopulationState,
    SocialNetwork,
    _hash,
    _nonnegative_integer,
    _text,
)
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel,
    EvolvingPopulationState,
)
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


def _edge_identity(value) -> tuple[str, str, str]:
    if isinstance(value, InformationTransmission):
        return (
            value.source_agent_id,
            value.target_agent_id,
            value.relation_type,
        )
    return value.edge.identity


def _event_key(value: PopulationLifecycleEvent) -> str:
    return value.agent_id


def _validate_evolving_model_state(
    model: EvolvingNetworkModel,
    state: EvolvingPopulationState,
) -> None:
    if not isinstance(model, EvolvingNetworkModel):
        raise TypeError("evolving simulation requires EvolvingNetworkModel")
    if not isinstance(state, EvolvingPopulationState):
        raise TypeError("evolving simulation requires EvolvingPopulationState")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("evolving state does not bind exact model identity")
    catalog_ids = set(model.base_model.network.agent_ids)
    members = {item.agent_id: item for item in state.members}
    if set(members) != catalog_ids:
        raise ValueError("evolving state must contain the exact member catalog")
    profiles = {item.agent_id: item for item in model.base_model.agents}
    for item in state.members:
        if item.status is LifecycleStatus.ACTIVE:
            expected = item.belief >= profiles[item.agent_id].broadcast_threshold
            if item.broadcasting is not expected:
                raise ValueError(
                    "active evolving member broadcasting is inconsistent with threshold"
                )
            if item.entry_count == 0:
                raise ValueError("active evolving member must have entered the population")
        elif item.broadcasting:
            raise ValueError("inactive or dead evolving member cannot broadcast")
    catalog_edges = {
        edge.identity: edge for edge in model.base_model.network.edges
    }
    trust = {item.edge.identity: item for item in state.edge_trust}
    topology = {item.edge.identity: item for item in state.edge_topology}
    if set(trust) != set(catalog_edges):
        raise ValueError("evolving state must contain the exact trust edge catalog")
    if set(topology) != set(catalog_edges):
        raise ValueError("evolving state must contain the exact topology edge catalog")
    for identity, item in topology.items():
        endpoints_active = all(
            members[agent_id].status is LifecycleStatus.ACTIVE
            for agent_id in identity[:2]
        )
        if state.round_index == 0 or endpoints_active:
            expected_similarity = 1.0 - abs(
                members[identity[0]].belief - members[identity[1]].belief
            )
            if not math.isclose(
                item.last_similarity,
                expected_similarity,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "evolving edge similarity is inconsistent with active endpoint beliefs"
                )
        if state.round_index == 0:
            if item.active is not catalog_edges[identity].active:
                raise ValueError("round-zero evolving topology must match base activity")
            if item.rewiring_count != 0:
                raise ValueError("round-zero evolving topology cannot contain rewiring history")
            if trust[identity].trust != model.initial_trust:
                raise ValueError("round-zero evolving trust must match initial trust")
            if trust[identity].feedback_count != 0:
                raise ValueError("round-zero evolving trust cannot contain feedback history")
    if state.round_index == 0:
        active_ids = {
            item.agent_id
            for item in state.members
            if item.status is LifecycleStatus.ACTIVE
        }
        if active_ids != set(model.initial_active_agent_ids):
            raise ValueError("round-zero evolving membership must match initial active ids")


def _canonical_events(
    model: EvolvingNetworkModel,
    events: tuple[PopulationLifecycleEvent, ...],
) -> tuple[PopulationLifecycleEvent, ...]:
    if not isinstance(events, tuple):
        raise TypeError("evolving round events must be a tuple")
    if any(not isinstance(item, PopulationLifecycleEvent) for item in events):
        raise TypeError("evolving events must contain PopulationLifecycleEvent values")
    agent_ids = tuple(item.agent_id for item in events)
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("evolving round allows at most one event per agent")
    if any(item not in model.base_model.network.agent_ids for item in agent_ids):
        raise ValueError("evolving event targets an unknown catalog agent")
    return tuple(sorted(events, key=_event_key))


def _canonical_feedback(
    feedback: tuple[TruthFeedback, ...],
) -> tuple[TruthFeedback, ...]:
    if not isinstance(feedback, tuple):
        raise TypeError("evolving round feedback must be a tuple")
    if any(not isinstance(item, TruthFeedback) for item in feedback):
        raise TypeError("evolving feedback must contain TruthFeedback values")
    identities = tuple(item.edge.identity for item in feedback)
    if len(set(identities)) != len(identities):
        raise ValueError("evolving feedback edge identities must be unique")
    return tuple(sorted(feedback, key=_edge_identity))


def _apply_events(
    model: EvolvingNetworkModel,
    prior_state: EvolvingPopulationState,
    events: tuple[PopulationLifecycleEvent, ...],
) -> tuple[LifecycleMemberState, ...]:
    profiles = {item.agent_id: item for item in model.base_model.agents}
    event_by_id = {item.agent_id: item for item in events}
    next_members: list[LifecycleMemberState] = []
    for prior in prior_state.members:
        item = event_by_id.get(prior.agent_id)
        if item is None:
            next_members.append(prior)
            continue
        if item.kind is LifecycleEventKind.ENTER:
            if prior.status is LifecycleStatus.DEAD:
                raise ValueError("dead agent cannot enter the evolving population")
            if prior.status is not LifecycleStatus.INACTIVE:
                raise ValueError("only inactive agents can enter the evolving population")
            belief = prior.belief if item.belief is None else item.belief
            next_members.append(
                LifecycleMemberState(
                    prior.agent_id,
                    belief,
                    prior.exposure_count + int(item.belief is not None),
                    belief >= profiles[prior.agent_id].broadcast_threshold,
                    LifecycleStatus.ACTIVE,
                    prior.entry_count + 1,
                    prior.exit_count,
                )
            )
        elif item.kind is LifecycleEventKind.EXIT:
            if prior.status is not LifecycleStatus.ACTIVE:
                raise ValueError("only active agents can exit the evolving population")
            next_members.append(
                LifecycleMemberState(
                    prior.agent_id,
                    prior.belief,
                    prior.exposure_count,
                    False,
                    LifecycleStatus.INACTIVE,
                    prior.entry_count,
                    prior.exit_count + 1,
                )
            )
        else:
            if prior.status is LifecycleStatus.DEAD:
                raise ValueError("dead agent cannot die again")
            next_members.append(
                LifecycleMemberState(
                    prior.agent_id,
                    prior.belief,
                    prior.exposure_count,
                    False,
                    LifecycleStatus.DEAD,
                    prior.entry_count,
                    prior.exit_count,
                )
            )
    return tuple(next_members)


def _active_projection(
    model: EvolvingNetworkModel,
    state: EvolvingPopulationState,
    members: tuple[LifecycleMemberState, ...],
) -> tuple[NetworkABMModel, PopulationState] | None:
    active_ids = tuple(
        item.agent_id for item in members if item.status is LifecycleStatus.ACTIVE
    )
    if not active_ids:
        return None
    active = set(active_ids)
    trust = {item.edge.identity: item.trust for item in state.edge_trust}
    topology = {item.edge.identity: item.active for item in state.edge_topology}
    base = model.base_model
    effective_edges = tuple(
        replace(
            edge,
            influence=edge.influence * trust[edge.identity],
            active=topology[edge.identity],
        )
        for edge in base.network.edges
        if edge.source_agent_id in active and edge.target_agent_id in active
    )
    effective_model = NetworkABMModel(
        f"{model.model_id}:active:{state.content_hash}",
        model.version,
        tuple(item for item in base.agents if item.agent_id in active),
        SocialNetwork(active_ids, effective_edges),
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


def evolving_active_population_view(
    model: EvolvingNetworkModel,
    state: EvolvingPopulationState,
) -> PopulationState | None:
    """Project active members with current trust and topology to V1 state."""

    _validate_evolving_model_state(model, state)
    projection = _active_projection(model, state, state.members)
    return None if projection is None else projection[1]


@dataclass(frozen=True)
class EvolvingRoundResult:
    """One atomic lifecycle, propagation, learning, and rewiring transition."""

    model_id: str
    model_hash: str
    round_index: int
    prior_state: EvolvingPopulationState
    events: tuple[PopulationLifecycleEvent, ...]
    transmissions: tuple[InformationTransmission, ...]
    feedback: tuple[TruthFeedback, ...]
    trust_updates: tuple[EdgeTrustUpdate, ...]
    rewiring_updates: tuple[EdgeRewiringUpdate, ...]
    next_state: EvolvingPopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="evolving round model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="evolving round model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="evolving round index",
        )
        if round_index == 0:
            raise ValueError("evolving round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        if not isinstance(self.prior_state, EvolvingPopulationState):
            raise TypeError("evolving round prior state must be EvolvingPopulationState")
        if not isinstance(self.next_state, EvolvingPopulationState):
            raise TypeError("evolving round next state must be EvolvingPopulationState")
        specifications = (
            ("events", self.events, PopulationLifecycleEvent, _event_key),
            ("transmissions", self.transmissions, InformationTransmission, _edge_identity),
            ("feedback", self.feedback, TruthFeedback, _edge_identity),
            ("trust_updates", self.trust_updates, EdgeTrustUpdate, _edge_identity),
            (
                "rewiring_updates",
                self.rewiring_updates,
                EdgeRewiringUpdate,
                _edge_identity,
            ),
        )
        for field_name, values, value_type, key in specifications:
            if not isinstance(values, tuple):
                raise TypeError(f"evolving round {field_name} must be a tuple")
            if any(not isinstance(item, value_type) for item in values):
                raise TypeError(
                    f"evolving round {field_name} must contain {value_type.__name__} values"
                )
            identities = tuple(key(item) for item in values)
            if len(set(identities)) != len(identities):
                raise ValueError(f"evolving round {field_name} identities must be unique")
            object.__setattr__(self, field_name, tuple(sorted(values, key=key)))
        if any(item.round_index != round_index for item in self.transmissions):
            raise ValueError("evolving transmissions must bind exact round index")
        if {_edge_identity(item) for item in self.feedback} != {
            _edge_identity(item) for item in self.trust_updates
        }:
            raise ValueError("evolving trust updates must cover exact feedback identities")
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("evolving round states must bind exact model identity")
        if round_index != self.prior_state.round_index + 1:
            raise ValueError("evolving round must immediately follow prior state")
        if self.next_state.round_index != round_index:
            raise ValueError("evolving next state must bind exact round index")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("evolving next state must bind exact prior state hash")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "prior_state_hash": self.prior_state.content_hash,
            "events": [item.to_dict() for item in self.events],
            "transmissions": [item.to_dict() for item in self.transmissions],
            "feedback": [item.to_dict() for item in self.feedback],
            "trust_updates": [item.to_dict() for item in self.trust_updates],
            "rewiring_updates": [item.to_dict() for item in self.rewiring_updates],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class EvolvingTrajectory:
    """Validated positive chain of unified evolving rounds."""

    model_id: str
    model_hash: str
    initial_state: EvolvingPopulationState
    rounds: tuple[EvolvingRoundResult, ...]
    final_state: EvolvingPopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="evolving trajectory model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="evolving trajectory model hash"),
        )
        if not isinstance(self.initial_state, EvolvingPopulationState):
            raise TypeError(
                "evolving trajectory initial state must be EvolvingPopulationState"
            )
        if not isinstance(self.final_state, EvolvingPopulationState):
            raise TypeError(
                "evolving trajectory final state must be EvolvingPopulationState"
            )
        if not isinstance(self.rounds, tuple):
            raise TypeError("evolving trajectory rounds must be a tuple")
        if not self.rounds:
            raise ValueError("evolving trajectory requires at least one round")
        if any(not isinstance(item, EvolvingRoundResult) for item in self.rounds):
            raise TypeError(
                "evolving trajectory rounds must contain EvolvingRoundResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("evolving trajectory states must bind exact model identity")
        current = self.initial_state
        for result in self.rounds:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError("evolving trajectory rounds must bind exact model identity")
            if result.prior_state != current:
                raise ValueError("evolving trajectory state chain is discontinuous")
            current = result.next_state
        if current != self.final_state:
            raise ValueError("evolving trajectory final state must equal chain tail")

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


def simulate_evolving_round(
    model: EvolvingNetworkModel,
    prior_state: EvolvingPopulationState,
    *,
    events: tuple[PopulationLifecycleEvent, ...] = (),
    feedback: tuple[TruthFeedback, ...] = (),
) -> EvolvingRoundResult:
    """Advance every evolving dimension through one ordered atomic round."""

    _validate_evolving_model_state(model, prior_state)
    canonical_events = _canonical_events(model, events)
    canonical_feedback = _canonical_feedback(feedback)
    post_event_members = _apply_events(model, prior_state, canonical_events)
    projection = _active_projection(model, prior_state, post_event_members)
    transmissions: tuple[InformationTransmission, ...] = ()
    propagated: dict[str, NetworkAgentState] = {}
    if projection is not None:
        effective_model, effective_state = projection
        base_result = simulate_round(effective_model, effective_state)
        transmissions = base_result.transmissions
        propagated = {item.agent_id: item for item in base_result.next_state.agents}
    next_members = tuple(
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
        for item in post_event_members
    )
    transmitted = {_edge_identity(item): item for item in transmissions}
    feedback_ids = {item.edge.identity for item in canonical_feedback}
    if not feedback_ids.issubset(transmitted):
        raise ValueError("evolving truth feedback targets an edge that did not transmit")
    feedback_by_id = {item.edge.identity: item for item in canonical_feedback}
    next_trust: list[EdgeTrustState] = []
    trust_updates: list[EdgeTrustUpdate] = []
    for prior in prior_state.edge_trust:
        item = feedback_by_id.get(prior.edge.identity)
        if item is None:
            next_trust.append(prior)
            continue
        transmission = transmitted[prior.edge.identity]
        accuracy = 1.0 - abs(transmission.signal - item.observed_truth)
        learned = prior.trust + model.learning_rate * (accuracy - prior.trust)
        learned = min(1.0, max(0.0, learned))
        count = prior.feedback_count + 1
        next_trust.append(EdgeTrustState(prior.edge, learned, count))
        trust_updates.append(
            EdgeTrustUpdate(
                prior.edge,
                prior.trust,
                transmission.signal,
                item.observed_truth,
                accuracy,
                learned,
                count,
            )
        )
    next_member_by_id = {item.agent_id: item for item in next_members}
    active_ids = {
        item.agent_id
        for item in next_members
        if item.status is LifecycleStatus.ACTIVE
    }
    next_topology: list[EdgeTopologyState] = []
    rewiring_updates: list[EdgeRewiringUpdate] = []
    for prior in prior_state.edge_topology:
        endpoints_active = (
            prior.edge.source_agent_id in active_ids
            and prior.edge.target_agent_id in active_ids
        )
        if not endpoints_active:
            next_topology.append(prior)
            continue
        similarity = 1.0 - abs(
            next_member_by_id[prior.edge.source_agent_id].belief
            - next_member_by_id[prior.edge.target_agent_id].belief
        )
        if similarity >= model.formation_similarity:
            next_active = True
        elif similarity <= model.dissolution_similarity:
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
    next_state = EvolvingPopulationState(
        model.model_id,
        model.content_hash,
        prior_state.round_index + 1,
        prior_state.content_hash,
        next_members,
        tuple(next_trust),
        tuple(next_topology),
    )
    return EvolvingRoundResult(
        model.model_id,
        model.content_hash,
        next_state.round_index,
        prior_state,
        canonical_events,
        transmissions,
        canonical_feedback,
        tuple(trust_updates),
        tuple(rewiring_updates),
        next_state,
    )


def simulate_evolving_population(
    model: EvolvingNetworkModel,
    initial_state: EvolvingPopulationState,
    *,
    event_schedule: tuple[tuple[PopulationLifecycleEvent, ...], ...],
    feedback_schedule: tuple[tuple[TruthFeedback, ...], ...],
) -> EvolvingTrajectory:
    """Run paired lifecycle-event and truth-feedback schedules."""

    _validate_evolving_model_state(model, initial_state)
    if not isinstance(event_schedule, tuple) or not isinstance(feedback_schedule, tuple):
        raise TypeError("evolving schedules must be tuples")
    if not event_schedule or not feedback_schedule:
        raise ValueError("evolving schedules must be nonempty")
    if len(event_schedule) != len(feedback_schedule):
        raise ValueError("evolving schedules must have equal lengths")
    if any(not isinstance(item, tuple) for item in event_schedule):
        raise TypeError("evolving event schedule entries must be tuples")
    if any(not isinstance(item, tuple) for item in feedback_schedule):
        raise TypeError("evolving feedback schedule entries must be tuples")
    current = initial_state
    results: list[EvolvingRoundResult] = []
    for events, feedback in zip(event_schedule, feedback_schedule, strict=True):
        result = simulate_evolving_round(
            model,
            current,
            events=events,
            feedback=feedback,
        )
        results.append(result)
        current = result.next_state
    return EvolvingTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(results),
        current,
    )


__all__ = (
    "EvolvingRoundResult",
    "EvolvingTrajectory",
    "evolving_active_population_view",
    "simulate_evolving_round",
    "simulate_evolving_population",
)
