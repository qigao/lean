from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.abm.adaptive_contracts import EdgeTrustState
from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
    initialize_population,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleMemberState,
    LifecycleStatus,
)
from narrative_dynamics.abm.rewiring_contracts import EdgeTopologyState
from narrative_dynamics.contracts import stable_content_hash


def _member_key(value: LifecycleMemberState) -> str:
    return value.agent_id


def _trust_key(value: EdgeTrustState) -> tuple[str, str, str]:
    return value.edge.identity


def _topology_key(value: EdgeTopologyState) -> tuple[str, str, str]:
    return value.edge.identity


@dataclass(frozen=True)
class EvolvingNetworkModel:
    """Unified parameters for lifecycle, trust learning, and rewiring."""

    model_id: str
    version: str
    base_model: NetworkABMModel
    initial_active_agent_ids: tuple[str, ...]
    learning_rate: float
    initial_trust: float
    dissolution_similarity: float
    formation_similarity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="evolving network model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="evolving network model version"),
        )
        if not isinstance(self.base_model, NetworkABMModel):
            raise TypeError("evolving network base model must be NetworkABMModel")
        if not self.base_model.network.edges:
            raise ValueError("evolving network model requires a candidate edge")
        if not isinstance(self.initial_active_agent_ids, tuple):
            raise TypeError("evolving initially active agent ids must be a tuple")
        active_ids = tuple(
            _text(item, label="evolving initially active agent id")
            for item in self.initial_active_agent_ids
        )
        if not active_ids:
            raise ValueError("evolving network requires an initially active agent")
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("evolving initially active agent ids must be unique")
        if not set(active_ids).issubset(self.base_model.network.agent_ids):
            raise ValueError("evolving initially active ids contain an unknown agent")
        object.__setattr__(self, "initial_active_agent_ids", tuple(sorted(active_ids)))
        for field_name in (
            "learning_rate",
            "initial_trust",
            "dissolution_similarity",
            "formation_similarity",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"evolving network {field_name}",
                ),
            )
        if self.dissolution_similarity >= self.formation_similarity:
            raise ValueError(
                "evolving dissolution similarity must be strictly below formation similarity"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "base_model": self.base_model.to_dict(),
            "initial_active_agent_ids": list(self.initial_active_agent_ids),
            "learning_rate": self.learning_rate,
            "initial_trust": self.initial_trust,
            "dissolution_similarity": self.dissolution_similarity,
            "formation_similarity": self.formation_similarity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class EvolvingPopulationState:
    """One unified snapshot of membership, trust, and topology."""

    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    members: tuple[LifecycleMemberState, ...]
    edge_trust: tuple[EdgeTrustState, ...]
    edge_topology: tuple[EdgeTopologyState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="evolving population model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="evolving population model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="evolving population round index",
        )
        object.__setattr__(self, "round_index", round_index)
        if round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("evolving population round zero cannot have a parent")
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(
                    self.parent_state_hash,
                    label="evolving population parent state hash",
                ),
            )
        if not isinstance(self.members, tuple):
            raise TypeError("evolving population members must be a tuple")
        members = tuple(self.members)
        if not members:
            raise ValueError("evolving population requires catalog members")
        if any(not isinstance(item, LifecycleMemberState) for item in members):
            raise TypeError(
                "evolving population members must contain LifecycleMemberState values"
            )
        member_ids = tuple(item.agent_id for item in members)
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("evolving population member ids must be unique")
        if not isinstance(self.edge_trust, tuple):
            raise TypeError("evolving population edge trust must be a tuple")
        edge_trust = tuple(self.edge_trust)
        if not edge_trust:
            raise ValueError("evolving population requires edge trust")
        if any(not isinstance(item, EdgeTrustState) for item in edge_trust):
            raise TypeError("evolving population trust must contain EdgeTrustState values")
        trust_ids = tuple(item.edge.identity for item in edge_trust)
        if len(set(trust_ids)) != len(trust_ids):
            raise ValueError("evolving population trust edge identities must be unique")
        if not isinstance(self.edge_topology, tuple):
            raise TypeError("evolving population edge topology must be a tuple")
        edge_topology = tuple(self.edge_topology)
        if not edge_topology:
            raise ValueError("evolving population requires edge topology")
        if any(not isinstance(item, EdgeTopologyState) for item in edge_topology):
            raise TypeError(
                "evolving population topology must contain EdgeTopologyState values"
            )
        topology_ids = tuple(item.edge.identity for item in edge_topology)
        if len(set(topology_ids)) != len(topology_ids):
            raise ValueError("evolving population topology edge identities must be unique")
        if set(trust_ids) != set(topology_ids):
            raise ValueError(
                "evolving population trust and topology must cover the same edge identities"
            )
        object.__setattr__(self, "members", tuple(sorted(members, key=_member_key)))
        object.__setattr__(self, "edge_trust", tuple(sorted(edge_trust, key=_trust_key)))
        object.__setattr__(
            self,
            "edge_topology",
            tuple(sorted(edge_topology, key=_topology_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "members": [item.to_dict() for item in self.members],
            "edge_trust": [item.to_dict() for item in self.edge_trust],
            "edge_topology": [item.to_dict() for item in self.edge_topology],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_evolving_population(
    model: EvolvingNetworkModel,
    *,
    beliefs: Mapping[str, float] | None = None,
) -> EvolvingPopulationState:
    """Initialize every unified state dimension from one catalog snapshot."""

    if not isinstance(model, EvolvingNetworkModel):
        raise TypeError("evolving initialization requires EvolvingNetworkModel")
    base_state = initialize_population(model.base_model, beliefs=beliefs)
    active_ids = set(model.initial_active_agent_ids)
    members = tuple(
        LifecycleMemberState(
            item.agent_id,
            item.belief,
            item.exposure_count,
            item.broadcasting if item.agent_id in active_ids else False,
            (
                LifecycleStatus.ACTIVE
                if item.agent_id in active_ids
                else LifecycleStatus.INACTIVE
            ),
            1 if item.agent_id in active_ids else 0,
            0,
        )
        for item in base_state.agents
    )
    by_id = {item.agent_id: item for item in base_state.agents}
    edge_trust: list[EdgeTrustState] = []
    edge_topology: list[EdgeTopologyState] = []
    for edge in model.base_model.network.edges:
        selector = EdgeSelector(
            edge.source_agent_id,
            edge.target_agent_id,
            edge.relation_type,
        )
        edge_trust.append(EdgeTrustState(selector, model.initial_trust, 0))
        edge_topology.append(
            EdgeTopologyState(
                selector,
                edge.active,
                1.0
                - abs(
                    by_id[edge.source_agent_id].belief
                    - by_id[edge.target_agent_id].belief
                ),
                0,
            )
        )
    return EvolvingPopulationState(
        model.model_id,
        model.content_hash,
        0,
        None,
        members,
        tuple(edge_trust),
        tuple(edge_topology),
    )


__all__ = (
    "EvolvingNetworkModel",
    "EvolvingPopulationState",
    "initialize_evolving_population",
)
