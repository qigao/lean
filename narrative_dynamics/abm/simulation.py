from __future__ import annotations

from dataclasses import dataclass

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentState,
    PopulationState,
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.contracts import stable_content_hash


def _transmission_key(
    value: InformationTransmission,
) -> tuple[str, str, str]:
    return (
        value.source_agent_id,
        value.target_agent_id,
        value.relation_type,
    )


@dataclass(frozen=True)
class InformationTransmission:
    """One edge-local delivery computed from a prior population snapshot."""

    round_index: int
    source_agent_id: str
    target_agent_id: str
    relation_type: str
    influence: float
    signal: float

    def __post_init__(self) -> None:
        round_index = _nonnegative_integer(
            self.round_index,
            label="transmission round index",
        )
        if round_index == 0:
            raise ValueError("transmission round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        object.__setattr__(
            self,
            "source_agent_id",
            _text(self.source_agent_id, label="transmission source agent id"),
        )
        object.__setattr__(
            self,
            "target_agent_id",
            _text(self.target_agent_id, label="transmission target agent id"),
        )
        object.__setattr__(
            self,
            "relation_type",
            _text(self.relation_type, label="transmission relation type"),
        )
        object.__setattr__(
            self,
            "influence",
            _probability(self.influence, label="transmission influence"),
        )
        object.__setattr__(
            self,
            "signal",
            _probability(self.signal, label="transmission signal"),
        )
        if self.source_agent_id == self.target_agent_id:
            raise ValueError("information transmission cannot be a self-edge")

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "relation_type": self.relation_type,
            "influence": self.influence,
            "signal": self.signal,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NetworkRoundResult:
    """Atomic synchronous transition from one population snapshot to the next."""

    model_id: str
    model_hash: str
    round_index: int
    prior_state: PopulationState
    transmissions: tuple[InformationTransmission, ...]
    next_state: PopulationState

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="round model id"))
        round_index = _nonnegative_integer(self.round_index, label="round index")
        if round_index == 0:
            raise ValueError("network round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        if not isinstance(self.prior_state, PopulationState):
            raise TypeError("network round prior state must be PopulationState")
        if not isinstance(self.next_state, PopulationState):
            raise TypeError("network round next state must be PopulationState")
        if not isinstance(self.transmissions, tuple):
            raise TypeError("network round transmissions must be a tuple")
        transmissions = tuple(self.transmissions)
        if any(not isinstance(item, InformationTransmission) for item in transmissions):
            raise TypeError(
                "network round transmissions must contain InformationTransmission values"
            )
        identities = tuple(_transmission_key(item) for item in transmissions)
        if len(set(identities)) != len(identities):
            raise ValueError("network round transmission identities must be unique")
        if any(item.round_index != round_index for item in transmissions):
            raise ValueError("network round transmissions must bind exact round index")
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("network round states must bind exact model identity")
        if round_index != self.prior_state.round_index + 1:
            raise ValueError("network round must immediately follow prior state")
        if self.next_state.round_index != round_index:
            raise ValueError("network next state must bind exact round index")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("network next state must bind exact prior state hash")
        if {item.agent_id for item in self.prior_state.agents} != {
            item.agent_id for item in self.next_state.agents
        }:
            raise ValueError("network round cannot change the population roster")
        object.__setattr__(
            self,
            "transmissions",
            tuple(sorted(transmissions, key=_transmission_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "prior_state_hash": self.prior_state.content_hash,
            "transmissions": [item.to_dict() for item in self.transmissions],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PopulationTrajectory:
    """Validated multi-round population state chain."""

    model_id: str
    model_hash: str
    initial_state: PopulationState
    rounds: tuple[NetworkRoundResult, ...]
    final_state: PopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="trajectory model id"),
        )
        if not isinstance(self.initial_state, PopulationState):
            raise TypeError("population trajectory initial state must be PopulationState")
        if not isinstance(self.final_state, PopulationState):
            raise TypeError("population trajectory final state must be PopulationState")
        if not isinstance(self.rounds, tuple):
            raise TypeError("population trajectory rounds must be a tuple")
        rounds = tuple(self.rounds)
        if not rounds:
            raise ValueError("population trajectory requires at least one round")
        if any(not isinstance(item, NetworkRoundResult) for item in rounds):
            raise TypeError(
                "population trajectory rounds must contain NetworkRoundResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("population trajectory states must bind exact model identity")
        current = self.initial_state
        for result in rounds:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError("population trajectory rounds must bind exact model identity")
            if result.prior_state != current:
                raise ValueError("population trajectory state chain is discontinuous")
            current = result.next_state
        if current != self.final_state:
            raise ValueError("population trajectory final state must equal chain tail")
        object.__setattr__(self, "rounds", rounds)

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


def _validate_model_state(model: NetworkABMModel, state: PopulationState) -> None:
    if not isinstance(model, NetworkABMModel):
        raise TypeError("network simulation requires NetworkABMModel")
    if not isinstance(state, PopulationState):
        raise TypeError("network simulation requires PopulationState")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("population state does not bind exact model identity")
    if {item.agent_id for item in state.agents} != set(model.network.agent_ids):
        raise ValueError("population state must contain the exact model roster")
    profiles = {item.agent_id: item for item in model.agents}
    for item in state.agents:
        expected = item.belief >= profiles[item.agent_id].broadcast_threshold
        if item.broadcasting is not expected:
            raise ValueError("population agent broadcasting is inconsistent with model threshold")


def simulate_round(
    model: NetworkABMModel,
    prior_state: PopulationState,
) -> NetworkRoundResult:
    """Advance every agent synchronously by one network-constrained round."""

    _validate_model_state(model, prior_state)
    next_round = prior_state.round_index + 1
    prior = {item.agent_id: item for item in prior_state.agents}
    profiles = {item.agent_id: item for item in model.agents}

    transmissions = tuple(
        InformationTransmission(
            next_round,
            edge.source_agent_id,
            edge.target_agent_id,
            edge.relation_type,
            edge.influence,
            prior[edge.source_agent_id].belief,
        )
        for edge in model.network.edges
        if edge.active
        and edge.influence > 0.0
        and prior[edge.source_agent_id].broadcasting
    )
    incoming: dict[str, list[InformationTransmission]] = {
        agent_id: [] for agent_id in model.network.agent_ids
    }
    for item in transmissions:
        incoming[item.target_agent_id].append(item)

    next_agents: list[NetworkAgentState] = []
    for agent_id in model.network.agent_ids:
        old = prior[agent_id]
        received = incoming[agent_id]
        belief = old.belief
        if received:
            total_influence = sum(item.influence for item in received)
            neighbor_signal = (
                sum(item.influence * item.signal for item in received)
                / total_influence
            )
            assimilation = profiles[agent_id].receptivity * min(
                1.0,
                total_influence,
            )
            belief = old.belief + assimilation * (neighbor_signal - old.belief)
            belief = min(1.0, max(0.0, belief))
        next_agents.append(
            NetworkAgentState(
                agent_id,
                belief,
                old.exposure_count + len(received),
                belief >= profiles[agent_id].broadcast_threshold,
            )
        )

    next_state = PopulationState(
        model.model_id,
        model.content_hash,
        next_round,
        prior_state.content_hash,
        tuple(next_agents),
    )
    return NetworkRoundResult(
        model.model_id,
        model.content_hash,
        next_round,
        prior_state,
        transmissions,
        next_state,
    )


def simulate_population(
    model: NetworkABMModel,
    initial_state: PopulationState,
    *,
    rounds: int,
) -> PopulationTrajectory:
    """Run a positive number of deterministic synchronous network rounds."""

    _validate_model_state(model, initial_state)
    if not isinstance(rounds, int) or isinstance(rounds, bool):
        raise TypeError("population simulation rounds must be an integer")
    if rounds <= 0:
        raise ValueError("population simulation rounds must be positive")
    current = initial_state
    results: list[NetworkRoundResult] = []
    for _ in range(rounds):
        result = simulate_round(model, current)
        results.append(result)
        current = result.next_state
    return PopulationTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(results),
        current,
    )


__all__ = (
    "InformationTransmission",
    "NetworkRoundResult",
    "PopulationTrajectory",
    "simulate_round",
    "simulate_population",
)
