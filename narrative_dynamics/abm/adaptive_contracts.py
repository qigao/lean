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


def _trust_key(value: EdgeTrustState) -> tuple[str, str, str]:
    return value.edge.identity


def _feedback_key(value: TruthFeedback) -> tuple[str, str, str]:
    return value.edge.identity


@dataclass(frozen=True)
class AdaptiveTrustModel:
    """One fixed network plus the parameters of edge-local trust learning."""

    model_id: str
    version: str
    base_model: NetworkABMModel
    learning_rate: float
    initial_trust: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="adaptive trust model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="adaptive trust model version"),
        )
        if not isinstance(self.base_model, NetworkABMModel):
            raise TypeError("adaptive trust model base model must be NetworkABMModel")
        object.__setattr__(
            self,
            "learning_rate",
            _probability(self.learning_rate, label="adaptive trust learning rate"),
        )
        object.__setattr__(
            self,
            "initial_trust",
            _probability(self.initial_trust, label="adaptive initial trust"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "base_model": self.base_model.to_dict(),
            "learning_rate": self.learning_rate,
            "initial_trust": self.initial_trust,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class EdgeTrustState:
    """Persistent learned reliability for one exact directed social edge."""

    edge: EdgeSelector
    trust: float
    feedback_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeSelector):
            raise TypeError("edge trust state requires EdgeSelector")
        object.__setattr__(
            self,
            "trust",
            _probability(self.trust, label="edge trust"),
        )
        object.__setattr__(
            self,
            "feedback_count",
            _nonnegative_integer(
                self.feedback_count,
                label="edge trust feedback count",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "edge": self.edge.to_dict(),
            "trust": self.trust,
            "feedback_count": self.feedback_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class TruthFeedback:
    """Externally verified truth for one edge transmission in one round."""

    edge: EdgeSelector
    observed_truth: float

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeSelector):
            raise TypeError("truth feedback requires EdgeSelector")
        object.__setattr__(
            self,
            "observed_truth",
            _probability(self.observed_truth, label="truth feedback observation"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "edge": self.edge.to_dict(),
            "observed_truth": self.observed_truth,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AdaptivePopulationState:
    """Synchronous agent and edge-trust snapshot bound to an adaptive model."""

    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    agents: tuple[NetworkAgentState, ...]
    edge_trust: tuple[EdgeTrustState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="adaptive population model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="adaptive population model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="adaptive population round index",
        )
        object.__setattr__(self, "round_index", round_index)
        if round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("adaptive population round zero cannot have a parent")
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(
                    self.parent_state_hash,
                    label="adaptive population parent state hash",
                ),
            )
        if not isinstance(self.agents, tuple):
            raise TypeError("adaptive population agents must be a tuple")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("adaptive population requires at least one agent")
        if any(not isinstance(item, NetworkAgentState) for item in agents):
            raise TypeError(
                "adaptive population agents must contain NetworkAgentState values"
            )
        agent_ids = tuple(item.agent_id for item in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("adaptive population agent ids must be unique")
        if not isinstance(self.edge_trust, tuple):
            raise TypeError("adaptive population edge trust must be a tuple")
        edge_trust = tuple(self.edge_trust)
        if any(not isinstance(item, EdgeTrustState) for item in edge_trust):
            raise TypeError(
                "adaptive population edge trust must contain EdgeTrustState values"
            )
        identities = tuple(item.edge.identity for item in edge_trust)
        if len(set(identities)) != len(identities):
            raise ValueError("adaptive population edge identities must be unique")
        object.__setattr__(
            self,
            "agents",
            tuple(sorted(agents, key=lambda item: item.agent_id)),
        )
        object.__setattr__(
            self,
            "edge_trust",
            tuple(sorted(edge_trust, key=_trust_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "agents": [item.to_dict() for item in self.agents],
            "edge_trust": [item.to_dict() for item in self.edge_trust],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_adaptive_population(
    model: AdaptiveTrustModel,
    *,
    beliefs: Mapping[str, float] | None = None,
) -> AdaptivePopulationState:
    """Create round-zero agents and one trust state per exact base edge."""

    if not isinstance(model, AdaptiveTrustModel):
        raise TypeError("adaptive initialization requires AdaptiveTrustModel")
    base_state = initialize_population(model.base_model, beliefs=beliefs)
    edge_trust = tuple(
        EdgeTrustState(
            EdgeSelector(
                edge.source_agent_id,
                edge.target_agent_id,
                edge.relation_type,
            ),
            model.initial_trust,
        )
        for edge in model.base_model.network.edges
    )
    return AdaptivePopulationState(
        model.model_id,
        model.content_hash,
        0,
        None,
        base_state.agents,
        edge_trust,
    )


__all__ = (
    "AdaptiveTrustModel",
    "EdgeTrustState",
    "AdaptivePopulationState",
    "TruthFeedback",
    "initialize_adaptive_population",
)
