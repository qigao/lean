from __future__ import annotations

from dataclasses import dataclass, replace
import math

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    PopulationState,
    SocialNetwork,
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.rewiring_contracts import (
    EdgeTopologyState,
    EndogenousRewiringModel,
    RewiringPopulationState,
)
from narrative_dynamics.abm.simulation import InformationTransmission, simulate_round
from narrative_dynamics.contracts import stable_content_hash


def _edge_key(value: EdgeSelector | EdgeTopologyState | EdgeRewiringUpdate):
    edge = value if isinstance(value, EdgeSelector) else value.edge
    return edge.identity


def _transmission_key(value: InformationTransmission) -> tuple[str, str, str]:
    return (
        value.source_agent_id,
        value.target_agent_id,
        value.relation_type,
    )


def _validate_rewiring_model_state(
    model: EndogenousRewiringModel,
    state: RewiringPopulationState,
) -> None:
    if not isinstance(model, EndogenousRewiringModel):
        raise TypeError("rewiring simulation requires EndogenousRewiringModel")
    if not isinstance(state, RewiringPopulationState):
        raise TypeError("rewiring simulation requires RewiringPopulationState")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("rewiring state does not bind exact model identity")
    if {item.agent_id for item in state.agents} != set(
        model.base_model.network.agent_ids
    ):
        raise ValueError("rewiring state must contain the exact agent roster")
    profiles = {item.agent_id: item for item in model.base_model.agents}
    agents = {item.agent_id: item for item in state.agents}
    for item in state.agents:
        expected = item.belief >= profiles[item.agent_id].broadcast_threshold
        if item.broadcasting is not expected:
            raise ValueError(
                "rewiring agent broadcasting is inconsistent with model threshold"
            )
    catalog_edges = {
        (edge.source_agent_id, edge.target_agent_id, edge.relation_type): edge
        for edge in model.base_model.network.edges
    }
    topology = {item.edge.identity: item for item in state.edge_topology}
    if set(topology) != set(catalog_edges):
        raise ValueError("rewiring state must contain the exact candidate edge set")
    for identity, item in topology.items():
        expected_similarity = 1.0 - abs(
            agents[identity[0]].belief - agents[identity[1]].belief
        )
        if not math.isclose(
            item.last_similarity,
            expected_similarity,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("edge topology similarity is inconsistent with agent beliefs")
        if state.round_index == 0:
            if item.active is not catalog_edges[identity].active:
                raise ValueError("round-zero topology must match base edge activity")
            if item.rewiring_count != 0:
                raise ValueError("round-zero topology cannot contain rewiring history")


def _effective_model(
    model: EndogenousRewiringModel,
    state: RewiringPopulationState,
) -> NetworkABMModel:
    active_by_identity = {
        item.edge.identity: item.active for item in state.edge_topology
    }
    base = model.base_model
    edges = tuple(
        replace(edge, active=active_by_identity[edge.identity])
        for edge in base.network.edges
    )
    return NetworkABMModel(
        f"{model.model_id}:effective",
        model.version,
        base.agents,
        SocialNetwork(base.network.agent_ids, edges),
    )


def rewiring_population_view(
    model: EndogenousRewiringModel,
    state: RewiringPopulationState,
) -> PopulationState:
    """Project rewiring agents and current topology to a V1 population state."""

    _validate_rewiring_model_state(model, state)
    effective = _effective_model(model, state)
    return PopulationState(
        effective.model_id,
        effective.content_hash,
        state.round_index,
        state.parent_state_hash if state.round_index > 0 else None,
        state.agents,
    )


@dataclass(frozen=True)
class EdgeRewiringUpdate:
    """Post-propagation activity decision for one candidate edge."""

    edge: EdgeSelector
    prior_active: bool
    similarity: float
    next_active: bool
    changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeSelector):
            raise TypeError("edge rewiring update requires EdgeSelector")
        for field_name in ("prior_active", "next_active", "changed"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"edge rewiring {field_name} must be boolean")
        object.__setattr__(
            self,
            "similarity",
            _probability(self.similarity, label="edge rewiring similarity"),
        )
        if self.changed is not (self.prior_active != self.next_active):
            raise ValueError("edge rewiring changed must match activity transition")

    def to_dict(self) -> dict[str, object]:
        return {
            "edge": self.edge.to_dict(),
            "prior_active": self.prior_active,
            "similarity": self.similarity,
            "next_active": self.next_active,
            "changed": self.changed,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RewiringRoundResult:
    """One synchronous propagation followed by topology rewiring."""

    model_id: str
    model_hash: str
    round_index: int
    prior_state: RewiringPopulationState
    transmissions: tuple[InformationTransmission, ...]
    edge_updates: tuple[EdgeRewiringUpdate, ...]
    next_state: RewiringPopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="rewiring round model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="rewiring round model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="rewiring round index",
        )
        if round_index == 0:
            raise ValueError("rewiring round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        if not isinstance(self.prior_state, RewiringPopulationState):
            raise TypeError("rewiring round prior state must be RewiringPopulationState")
        if not isinstance(self.next_state, RewiringPopulationState):
            raise TypeError("rewiring round next state must be RewiringPopulationState")
        if not isinstance(self.transmissions, tuple):
            raise TypeError("rewiring round transmissions must be a tuple")
        if any(not isinstance(item, InformationTransmission) for item in self.transmissions):
            raise TypeError(
                "rewiring round transmissions must contain InformationTransmission values"
            )
        transmission_ids = tuple(_transmission_key(item) for item in self.transmissions)
        if len(set(transmission_ids)) != len(transmission_ids):
            raise ValueError("rewiring round transmission identities must be unique")
        if any(item.round_index != round_index for item in self.transmissions):
            raise ValueError("rewiring transmissions must bind exact round index")
        if not isinstance(self.edge_updates, tuple):
            raise TypeError("rewiring round edge updates must be a tuple")
        if not self.edge_updates:
            raise ValueError("rewiring round requires candidate edge updates")
        if any(not isinstance(item, EdgeRewiringUpdate) for item in self.edge_updates):
            raise TypeError(
                "rewiring round updates must contain EdgeRewiringUpdate values"
            )
        update_ids = tuple(item.edge.identity for item in self.edge_updates)
        if len(set(update_ids)) != len(update_ids):
            raise ValueError("rewiring round update edge identities must be unique")
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("rewiring round states must bind exact model identity")
        if round_index != self.prior_state.round_index + 1:
            raise ValueError("rewiring round must immediately follow prior state")
        if self.next_state.round_index != round_index:
            raise ValueError("rewiring next state must bind exact round index")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("rewiring next state must bind exact prior state hash")
        prior_topology = {
            item.edge.identity: item for item in self.prior_state.edge_topology
        }
        next_topology = {
            item.edge.identity: item for item in self.next_state.edge_topology
        }
        updates = {item.edge.identity: item for item in self.edge_updates}
        if set(prior_topology) != set(next_topology) or set(updates) != set(next_topology):
            raise ValueError("rewiring round must update the exact candidate edge set")
        for identity, update in updates.items():
            prior = prior_topology[identity]
            next_item = next_topology[identity]
            if update.prior_active is not prior.active:
                raise ValueError("rewiring update prior activity does not match prior state")
            if update.next_active is not next_item.active:
                raise ValueError("rewiring update next activity does not match next state")
            if update.similarity != next_item.last_similarity:
                raise ValueError("rewiring update similarity does not match next state")
            if next_item.rewiring_count != prior.rewiring_count + int(update.changed):
                raise ValueError("rewiring count does not match edge activity change")
        object.__setattr__(
            self,
            "transmissions",
            tuple(sorted(self.transmissions, key=_transmission_key)),
        )
        object.__setattr__(
            self,
            "edge_updates",
            tuple(sorted(self.edge_updates, key=_edge_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "prior_state_hash": self.prior_state.content_hash,
            "transmissions": [item.to_dict() for item in self.transmissions],
            "edge_updates": [item.to_dict() for item in self.edge_updates],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RewiringTrajectory:
    """Validated positive multi-round rewiring state chain."""

    model_id: str
    model_hash: str
    initial_state: RewiringPopulationState
    rounds: tuple[RewiringRoundResult, ...]
    final_state: RewiringPopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="rewiring trajectory model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="rewiring trajectory model hash"),
        )
        if not isinstance(self.initial_state, RewiringPopulationState):
            raise TypeError(
                "rewiring trajectory initial state must be RewiringPopulationState"
            )
        if not isinstance(self.final_state, RewiringPopulationState):
            raise TypeError(
                "rewiring trajectory final state must be RewiringPopulationState"
            )
        if not isinstance(self.rounds, tuple):
            raise TypeError("rewiring trajectory rounds must be a tuple")
        if not self.rounds:
            raise ValueError("rewiring trajectory requires at least one round")
        if any(not isinstance(item, RewiringRoundResult) for item in self.rounds):
            raise TypeError(
                "rewiring trajectory rounds must contain RewiringRoundResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("rewiring trajectory states must bind exact model identity")
        current = self.initial_state
        for result in self.rounds:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError("rewiring trajectory rounds must bind exact model identity")
            if result.prior_state != current:
                raise ValueError("rewiring trajectory state chain is discontinuous")
            current = result.next_state
        if current != self.final_state:
            raise ValueError("rewiring trajectory final state must equal chain tail")

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


def simulate_rewiring_round(
    model: EndogenousRewiringModel,
    prior_state: RewiringPopulationState,
) -> RewiringRoundResult:
    """Propagate over prior topology, then update topology from next beliefs."""

    _validate_rewiring_model_state(model, prior_state)
    effective_model = _effective_model(model, prior_state)
    prior_view = PopulationState(
        effective_model.model_id,
        effective_model.content_hash,
        prior_state.round_index,
        prior_state.parent_state_hash if prior_state.round_index > 0 else None,
        prior_state.agents,
    )
    base_result = simulate_round(effective_model, prior_view)
    next_agents = base_result.next_state.agents
    by_id = {item.agent_id: item for item in next_agents}
    updates: list[EdgeRewiringUpdate] = []
    next_topology: list[EdgeTopologyState] = []
    for prior in prior_state.edge_topology:
        similarity = 1.0 - abs(
            by_id[prior.edge.source_agent_id].belief
            - by_id[prior.edge.target_agent_id].belief
        )
        if similarity >= model.formation_similarity:
            next_active = True
        elif similarity <= model.dissolution_similarity:
            next_active = False
        else:
            next_active = prior.active
        changed = prior.active != next_active
        updates.append(
            EdgeRewiringUpdate(
                prior.edge,
                prior.active,
                similarity,
                next_active,
                changed,
            )
        )
        next_topology.append(
            EdgeTopologyState(
                prior.edge,
                next_active,
                similarity,
                prior.rewiring_count + int(changed),
            )
        )
    next_state = RewiringPopulationState(
        model.model_id,
        model.content_hash,
        prior_state.round_index + 1,
        prior_state.content_hash,
        next_agents,
        tuple(next_topology),
    )
    return RewiringRoundResult(
        model.model_id,
        model.content_hash,
        next_state.round_index,
        prior_state,
        base_result.transmissions,
        tuple(updates),
        next_state,
    )


def simulate_rewiring_population(
    model: EndogenousRewiringModel,
    initial_state: RewiringPopulationState,
    *,
    rounds: int,
) -> RewiringTrajectory:
    """Run a positive number of propagation–rewiring rounds."""

    _validate_rewiring_model_state(model, initial_state)
    if not isinstance(rounds, int) or isinstance(rounds, bool):
        raise TypeError("rewiring simulation rounds must be an integer")
    if rounds <= 0:
        raise ValueError("rewiring simulation rounds must be positive")
    current = initial_state
    results: list[RewiringRoundResult] = []
    for _ in range(rounds):
        result = simulate_rewiring_round(model, current)
        results.append(result)
        current = result.next_state
    return RewiringTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(results),
        current,
    )


__all__ = (
    "EdgeRewiringUpdate",
    "RewiringRoundResult",
    "RewiringTrajectory",
    "rewiring_population_view",
    "simulate_rewiring_round",
    "simulate_rewiring_population",
)
