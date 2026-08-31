from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentState,
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
    initialize_population,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.contracts import stable_content_hash


def _topology_key(value: EdgeTopologyState) -> tuple[str, str, str]:
    return value.edge.identity


@dataclass(frozen=True)
class EndogenousRewiringModel:
    """Fixed candidate network plus hysteretic belief-similarity thresholds."""

    model_id: str
    version: str
    base_model: NetworkABMModel
    dissolution_similarity: float
    formation_similarity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="endogenous rewiring model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="endogenous rewiring model version"),
        )
        if not isinstance(self.base_model, NetworkABMModel):
            raise TypeError("endogenous rewiring base model must be NetworkABMModel")
        if not self.base_model.network.edges:
            raise ValueError("endogenous rewiring model requires a candidate edge")
        object.__setattr__(
            self,
            "dissolution_similarity",
            _probability(
                self.dissolution_similarity,
                label="rewiring dissolution similarity",
            ),
        )
        object.__setattr__(
            self,
            "formation_similarity",
            _probability(
                self.formation_similarity,
                label="rewiring formation similarity",
            ),
        )
        if self.dissolution_similarity >= self.formation_similarity:
            raise ValueError(
                "rewiring dissolution similarity must be strictly below formation similarity"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "base_model": self.base_model.to_dict(),
            "dissolution_similarity": self.dissolution_similarity,
            "formation_similarity": self.formation_similarity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class EdgeTopologyState:
    """Persistent activity and rewiring history for one candidate edge."""

    edge: EdgeSelector
    active: bool
    last_similarity: float
    rewiring_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeSelector):
            raise TypeError("edge topology state requires EdgeSelector")
        if not isinstance(self.active, bool):
            raise TypeError("edge topology active must be boolean")
        object.__setattr__(
            self,
            "last_similarity",
            _probability(
                self.last_similarity,
                label="edge topology last similarity",
            ),
        )
        object.__setattr__(
            self,
            "rewiring_count",
            _nonnegative_integer(
                self.rewiring_count,
                label="edge topology rewiring count",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "edge": self.edge.to_dict(),
            "active": self.active,
            "last_similarity": self.last_similarity,
            "rewiring_count": self.rewiring_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RewiringPopulationState:
    """Agent and candidate-edge topology snapshot bound to a rewiring model."""

    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    agents: tuple[NetworkAgentState, ...]
    edge_topology: tuple[EdgeTopologyState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="rewiring population model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="rewiring population model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="rewiring population round index",
        )
        object.__setattr__(self, "round_index", round_index)
        if round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("rewiring population round zero cannot have a parent")
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(
                    self.parent_state_hash,
                    label="rewiring population parent state hash",
                ),
            )
        if not isinstance(self.agents, tuple):
            raise TypeError("rewiring population agents must be a tuple")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("rewiring population requires at least one agent")
        if any(not isinstance(item, NetworkAgentState) for item in agents):
            raise TypeError(
                "rewiring population agents must contain NetworkAgentState values"
            )
        agent_ids = tuple(item.agent_id for item in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("rewiring population agent ids must be unique")
        if not isinstance(self.edge_topology, tuple):
            raise TypeError("rewiring population edge topology must be a tuple")
        edge_topology = tuple(self.edge_topology)
        if not edge_topology:
            raise ValueError("rewiring population requires candidate edge topology")
        if any(not isinstance(item, EdgeTopologyState) for item in edge_topology):
            raise TypeError(
                "rewiring population topology must contain EdgeTopologyState values"
            )
        identities = tuple(item.edge.identity for item in edge_topology)
        if len(set(identities)) != len(identities):
            raise ValueError("rewiring population edge identities must be unique")
        object.__setattr__(
            self,
            "agents",
            tuple(sorted(agents, key=lambda item: item.agent_id)),
        )
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
            "agents": [item.to_dict() for item in self.agents],
            "edge_topology": [item.to_dict() for item in self.edge_topology],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_rewiring_population(
    model: EndogenousRewiringModel,
    *,
    beliefs: Mapping[str, float] | None = None,
) -> RewiringPopulationState:
    """Create round-zero agents and topology for every candidate edge."""

    if not isinstance(model, EndogenousRewiringModel):
        raise TypeError("rewiring initialization requires EndogenousRewiringModel")
    base_state = initialize_population(model.base_model, beliefs=beliefs)
    by_id = {item.agent_id: item for item in base_state.agents}
    topology = tuple(
        EdgeTopologyState(
            EdgeSelector(
                edge.source_agent_id,
                edge.target_agent_id,
                edge.relation_type,
            ),
            edge.active,
            1.0
            - abs(
                by_id[edge.source_agent_id].belief
                - by_id[edge.target_agent_id].belief
            ),
            0,
        )
        for edge in model.base_model.network.edges
    )
    return RewiringPopulationState(
        model.model_id,
        model.content_hash,
        0,
        None,
        base_state.agents,
        topology,
    )


__all__ = (
    "EndogenousRewiringModel",
    "EdgeTopologyState",
    "RewiringPopulationState",
    "initialize_rewiring_population",
)
