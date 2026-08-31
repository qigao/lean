from __future__ import annotations

from dataclasses import dataclass

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentState,
    PopulationState,
    SocialNetwork,
    _hash,
    _nonnegative_integer,
    _text,
)
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleMemberState,
    LifecycleStatus,
    PopulationLifecycleEvent,
    PopulationLifecycleModel,
    PopulationLifecycleState,
)
from narrative_dynamics.abm.simulation import InformationTransmission, simulate_round
from narrative_dynamics.contracts import stable_content_hash


def _event_key(value: PopulationLifecycleEvent) -> str:
    return value.agent_id


def _transmission_key(value: InformationTransmission) -> tuple[str, str, str]:
    return (
        value.source_agent_id,
        value.target_agent_id,
        value.relation_type,
    )


def _validate_model_state(
    model: PopulationLifecycleModel,
    state: PopulationLifecycleState,
) -> None:
    if not isinstance(model, PopulationLifecycleModel):
        raise TypeError("lifecycle simulation requires PopulationLifecycleModel")
    if not isinstance(state, PopulationLifecycleState):
        raise TypeError("lifecycle simulation requires PopulationLifecycleState")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("lifecycle state does not bind exact model identity")
    catalog_ids = set(model.catalog_model.network.agent_ids)
    if {item.agent_id for item in state.members} != catalog_ids:
        raise ValueError("lifecycle state must cover the exact catalog roster")
    profiles = {item.agent_id: item for item in model.catalog_model.agents}
    for item in state.members:
        if item.status is LifecycleStatus.ACTIVE:
            expected = item.belief >= profiles[item.agent_id].broadcast_threshold
            if item.broadcasting is not expected:
                raise ValueError(
                    "active lifecycle member broadcasting is inconsistent with threshold"
                )
            if item.entry_count == 0:
                raise ValueError("active lifecycle member must have entered the population")
        elif item.broadcasting:
            raise ValueError("inactive or dead lifecycle member cannot broadcast")


def _canonical_events(
    model: PopulationLifecycleModel,
    events: tuple[PopulationLifecycleEvent, ...],
) -> tuple[PopulationLifecycleEvent, ...]:
    if not isinstance(events, tuple):
        raise TypeError("lifecycle round events must be a tuple")
    if any(not isinstance(item, PopulationLifecycleEvent) for item in events):
        raise TypeError(
            "lifecycle round events must contain PopulationLifecycleEvent values"
        )
    agent_ids = tuple(item.agent_id for item in events)
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("lifecycle round allows at most one event per agent")
    catalog_ids = set(model.catalog_model.network.agent_ids)
    if any(item not in catalog_ids for item in agent_ids):
        raise ValueError("lifecycle event targets an unknown catalog agent")
    return tuple(sorted(events, key=_event_key))


def _apply_events(
    model: PopulationLifecycleModel,
    prior_state: PopulationLifecycleState,
    events: tuple[PopulationLifecycleEvent, ...],
) -> tuple[LifecycleMemberState, ...]:
    profiles = {item.agent_id: item for item in model.catalog_model.agents}
    event_by_id = {item.agent_id: item for item in events}
    members: list[LifecycleMemberState] = []
    for prior in prior_state.members:
        item = event_by_id.get(prior.agent_id)
        if item is None:
            members.append(prior)
            continue
        if item.kind is LifecycleEventKind.ENTER:
            if prior.status is LifecycleStatus.DEAD:
                raise ValueError("dead agent cannot enter the population")
            if prior.status is not LifecycleStatus.INACTIVE:
                raise ValueError("only inactive agents can enter the population")
            belief = prior.belief if item.belief is None else item.belief
            exposure_count = prior.exposure_count + (item.belief is not None)
            members.append(
                LifecycleMemberState(
                    prior.agent_id,
                    belief,
                    exposure_count,
                    belief >= profiles[prior.agent_id].broadcast_threshold,
                    LifecycleStatus.ACTIVE,
                    prior.entry_count + 1,
                    prior.exit_count,
                )
            )
        elif item.kind is LifecycleEventKind.EXIT:
            if prior.status is not LifecycleStatus.ACTIVE:
                raise ValueError("only active agents can exit the population")
            members.append(
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
            members.append(
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
    return tuple(members)


def _active_projection(
    model: PopulationLifecycleModel,
    members: tuple[LifecycleMemberState, ...],
    *,
    round_index: int,
    parent_state_hash: str | None,
) -> tuple[NetworkABMModel, PopulationState] | None:
    active_ids = tuple(
        item.agent_id for item in members if item.status is LifecycleStatus.ACTIVE
    )
    if not active_ids:
        return None
    active = set(active_ids)
    catalog = model.catalog_model
    induced_model = NetworkABMModel(
        f"{model.model_id}:active",
        model.version,
        tuple(item for item in catalog.agents if item.agent_id in active),
        SocialNetwork(
            active_ids,
            tuple(
                edge
                for edge in catalog.network.edges
                if edge.source_agent_id in active and edge.target_agent_id in active
            ),
        ),
    )
    projected = PopulationState(
        induced_model.model_id,
        induced_model.content_hash,
        round_index,
        parent_state_hash if round_index > 0 else None,
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
    return induced_model, projected


def active_population_view(
    model: PopulationLifecycleModel,
    state: PopulationLifecycleState,
) -> PopulationState | None:
    """Project the current effective population to a V1-compatible snapshot."""

    _validate_model_state(model, state)
    projection = _active_projection(
        model,
        state.members,
        round_index=state.round_index,
        parent_state_hash=state.parent_state_hash,
    )
    return None if projection is None else projection[1]


@dataclass(frozen=True)
class LifecycleRoundResult:
    """One boundary-event and active-subgraph propagation transition."""

    model_id: str
    model_hash: str
    round_index: int
    prior_state: PopulationLifecycleState
    events: tuple[PopulationLifecycleEvent, ...]
    transmissions: tuple[InformationTransmission, ...]
    next_state: PopulationLifecycleState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="lifecycle round model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="lifecycle round model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="lifecycle round index",
        )
        if round_index == 0:
            raise ValueError("lifecycle round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        if not isinstance(self.prior_state, PopulationLifecycleState):
            raise TypeError("lifecycle round prior state must be PopulationLifecycleState")
        if not isinstance(self.next_state, PopulationLifecycleState):
            raise TypeError("lifecycle round next state must be PopulationLifecycleState")
        if not isinstance(self.events, tuple):
            raise TypeError("lifecycle round events must be a tuple")
        if any(not isinstance(item, PopulationLifecycleEvent) for item in self.events):
            raise TypeError(
                "lifecycle round events must contain PopulationLifecycleEvent values"
            )
        event_ids = tuple(item.agent_id for item in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("lifecycle round allows at most one event per agent")
        if not isinstance(self.transmissions, tuple):
            raise TypeError("lifecycle round transmissions must be a tuple")
        if any(not isinstance(item, InformationTransmission) for item in self.transmissions):
            raise TypeError(
                "lifecycle round transmissions must contain InformationTransmission values"
            )
        transmission_ids = tuple(_transmission_key(item) for item in self.transmissions)
        if len(set(transmission_ids)) != len(transmission_ids):
            raise ValueError("lifecycle round transmission identities must be unique")
        if any(item.round_index != round_index for item in self.transmissions):
            raise ValueError("lifecycle transmissions must bind exact round index")
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("lifecycle round states must bind exact model identity")
        if round_index != self.prior_state.round_index + 1:
            raise ValueError("lifecycle round must immediately follow prior state")
        if self.next_state.round_index != round_index:
            raise ValueError("lifecycle next state must bind exact round index")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("lifecycle next state must bind exact prior state hash")
        if {item.agent_id for item in self.prior_state.members} != {
            item.agent_id for item in self.next_state.members
        }:
            raise ValueError("lifecycle round cannot change the identity catalog")
        object.__setattr__(self, "events", tuple(sorted(self.events, key=_event_key)))
        object.__setattr__(
            self,
            "transmissions",
            tuple(sorted(self.transmissions, key=_transmission_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "prior_state_hash": self.prior_state.content_hash,
            "events": [item.to_dict() for item in self.events],
            "transmissions": [item.to_dict() for item in self.transmissions],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class LifecycleTrajectory:
    """Validated multi-round population lifecycle chain."""

    model_id: str
    model_hash: str
    initial_state: PopulationLifecycleState
    rounds: tuple[LifecycleRoundResult, ...]
    final_state: PopulationLifecycleState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="lifecycle trajectory model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="lifecycle trajectory model hash"),
        )
        if not isinstance(self.initial_state, PopulationLifecycleState):
            raise TypeError(
                "lifecycle trajectory initial state must be PopulationLifecycleState"
            )
        if not isinstance(self.final_state, PopulationLifecycleState):
            raise TypeError(
                "lifecycle trajectory final state must be PopulationLifecycleState"
            )
        if not isinstance(self.rounds, tuple):
            raise TypeError("lifecycle trajectory rounds must be a tuple")
        if not self.rounds:
            raise ValueError("lifecycle trajectory requires at least one round")
        if any(not isinstance(item, LifecycleRoundResult) for item in self.rounds):
            raise TypeError(
                "lifecycle trajectory rounds must contain LifecycleRoundResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("lifecycle trajectory states must bind exact model identity")
        current = self.initial_state
        for result in self.rounds:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError("lifecycle trajectory rounds must bind exact model identity")
            if result.prior_state != current:
                raise ValueError("lifecycle trajectory state chain is discontinuous")
            current = result.next_state
        if current != self.final_state:
            raise ValueError("lifecycle trajectory final state must equal chain tail")

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


def simulate_lifecycle_round(
    model: PopulationLifecycleModel,
    prior_state: PopulationLifecycleState,
    *,
    events: tuple[PopulationLifecycleEvent, ...] = (),
) -> LifecycleRoundResult:
    """Apply boundary lifecycle events, then propagate over active agents."""

    _validate_model_state(model, prior_state)
    canonical_events = _canonical_events(model, events)
    post_event_members = _apply_events(model, prior_state, canonical_events)
    next_round = prior_state.round_index + 1
    projection = _active_projection(
        model,
        post_event_members,
        round_index=prior_state.round_index,
        parent_state_hash=prior_state.parent_state_hash,
    )
    transmissions: tuple[InformationTransmission, ...] = ()
    propagated: dict[str, NetworkAgentState] = {}
    if projection is not None:
        induced_model, induced_state = projection
        base_result = simulate_round(induced_model, induced_state)
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
    next_state = PopulationLifecycleState(
        model.model_id,
        model.content_hash,
        next_round,
        prior_state.content_hash,
        next_members,
    )
    return LifecycleRoundResult(
        model.model_id,
        model.content_hash,
        next_round,
        prior_state,
        canonical_events,
        transmissions,
        next_state,
    )


def simulate_lifecycle_population(
    model: PopulationLifecycleModel,
    initial_state: PopulationLifecycleState,
    *,
    event_schedule: tuple[tuple[PopulationLifecycleEvent, ...], ...],
) -> LifecycleTrajectory:
    """Run one lifecycle round for each event tuple in a positive schedule."""

    _validate_model_state(model, initial_state)
    if not isinstance(event_schedule, tuple):
        raise TypeError("lifecycle event schedule must be a tuple")
    if not event_schedule:
        raise ValueError("lifecycle event schedule requires at least one round")
    if any(not isinstance(item, tuple) for item in event_schedule):
        raise TypeError("each lifecycle event schedule round must be a tuple")
    current = initial_state
    results: list[LifecycleRoundResult] = []
    for events in event_schedule:
        result = simulate_lifecycle_round(model, current, events=events)
        results.append(result)
        current = result.next_state
    return LifecycleTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(results),
        current,
    )


__all__ = (
    "LifecycleRoundResult",
    "LifecycleTrajectory",
    "active_population_view",
    "simulate_lifecycle_round",
    "simulate_lifecycle_population",
)
