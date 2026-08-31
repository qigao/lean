from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re

from narrative_dynamics.contracts import stable_content_hash


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _probability(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def _nonnegative_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _agent_key(value: NetworkAgentSpec) -> str:
    return value.agent_id


def _edge_key(value: SocialEdge) -> tuple[str, str, str]:
    return (value.source_agent_id, value.target_agent_id, value.relation_type)


def _state_key(value: NetworkAgentState) -> str:
    return value.agent_id


@dataclass(frozen=True)
class NetworkAgentSpec:
    """Fixed agent identity and thresholds for the V1 network ABM."""

    agent_id: str
    role: str
    receptivity: float
    adoption_threshold: float
    broadcast_threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="agent id"))
        object.__setattr__(self, "role", _text(self.role, label="agent role"))
        object.__setattr__(
            self,
            "receptivity",
            _probability(self.receptivity, label="agent receptivity"),
        )
        object.__setattr__(
            self,
            "adoption_threshold",
            _probability(
                self.adoption_threshold,
                label="agent adoption threshold",
            ),
        )
        object.__setattr__(
            self,
            "broadcast_threshold",
            _probability(
                self.broadcast_threshold,
                label="agent broadcast threshold",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "receptivity": self.receptivity,
            "adoption_threshold": self.adoption_threshold,
            "broadcast_threshold": self.broadcast_threshold,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SocialEdge:
    """One directed, typed information channel between two agents."""

    source_agent_id: str
    target_agent_id: str
    relation_type: str
    influence: float = 1.0
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_agent_id",
            _text(self.source_agent_id, label="edge source agent id"),
        )
        object.__setattr__(
            self,
            "target_agent_id",
            _text(self.target_agent_id, label="edge target agent id"),
        )
        object.__setattr__(
            self,
            "relation_type",
            _text(self.relation_type, label="edge relation type"),
        )
        object.__setattr__(
            self,
            "influence",
            _probability(self.influence, label="edge influence"),
        )
        if not isinstance(self.active, bool):
            raise TypeError("edge active must be boolean")
        if self.source_agent_id == self.target_agent_id:
            raise ValueError("social network self-edge is not allowed")

    @property
    def identity(self) -> tuple[str, str, str]:
        return _edge_key(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "relation_type": self.relation_type,
            "influence": self.influence,
            "active": self.active,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SocialNetwork:
    """Canonical fixed-roster social topology."""

    agent_ids: tuple[str, ...]
    edges: tuple[SocialEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.agent_ids, tuple):
            raise TypeError("social network agent ids must be a tuple")
        if not isinstance(self.edges, tuple):
            raise TypeError("social network edges must be a tuple")
        agent_ids = tuple(_text(item, label="network agent id") for item in self.agent_ids)
        if not agent_ids:
            raise ValueError("social network requires at least one agent")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("social network agent ids must be unique")
        edges = tuple(self.edges)
        if any(not isinstance(item, SocialEdge) for item in edges):
            raise TypeError("social network edges must contain SocialEdge values")
        identities = tuple(item.identity for item in edges)
        if len(set(identities)) != len(identities):
            raise ValueError("social network edge identities must be unique")
        roster = set(agent_ids)
        if any(
            item.source_agent_id not in roster or item.target_agent_id not in roster
            for item in edges
        ):
            raise ValueError("social network edge endpoint must reference a network agent")
        object.__setattr__(self, "agent_ids", tuple(sorted(agent_ids)))
        object.__setattr__(self, "edges", tuple(sorted(edges, key=_edge_key)))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_ids": list(self.agent_ids),
            "edges": [item.to_dict() for item in self.edges],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NetworkABMModel:
    """Exact population profiles and topology for one ABM model."""

    model_id: str
    version: str
    agents: tuple[NetworkAgentSpec, ...]
    network: SocialNetwork

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="ABM model id"))
        object.__setattr__(self, "version", _text(self.version, label="ABM model version"))
        if not isinstance(self.agents, tuple):
            raise TypeError("ABM model agents must be a tuple")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("ABM model requires at least one agent")
        if any(not isinstance(item, NetworkAgentSpec) for item in agents):
            raise TypeError("ABM model agents must contain NetworkAgentSpec values")
        agent_ids = tuple(item.agent_id for item in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("ABM model agent ids must be unique")
        if not isinstance(self.network, SocialNetwork):
            raise TypeError("ABM model network must be SocialNetwork")
        if set(agent_ids) != set(self.network.agent_ids):
            raise ValueError("ABM model and network must contain the same agent ids")
        object.__setattr__(self, "agents", tuple(sorted(agents, key=_agent_key)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "agents": [item.to_dict() for item in self.agents],
            "network": self.network.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NetworkAgentState:
    """Persistent information-dependent state for one agent."""

    agent_id: str
    belief: float
    exposure_count: int
    broadcasting: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="state agent id"))
        object.__setattr__(
            self,
            "belief",
            _probability(self.belief, label="agent belief"),
        )
        object.__setattr__(
            self,
            "exposure_count",
            _nonnegative_integer(self.exposure_count, label="agent exposure count"),
        )
        if not isinstance(self.broadcasting, bool):
            raise TypeError("agent broadcasting must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "belief": self.belief,
            "exposure_count": self.exposure_count,
            "broadcasting": self.broadcasting,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PopulationState:
    """One immutable round snapshot bound to an exact ABM model."""

    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    agents: tuple[NetworkAgentState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="population model id"))
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="population model hash"),
        )
        object.__setattr__(
            self,
            "round_index",
            _nonnegative_integer(self.round_index, label="population round index"),
        )
        if self.round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("population round zero cannot have a parent state hash")
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(self.parent_state_hash, label="population parent state hash"),
            )
        if not isinstance(self.agents, tuple):
            raise TypeError("population agents must be a tuple")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("population state requires at least one agent")
        if any(not isinstance(item, NetworkAgentState) for item in agents):
            raise TypeError("population agents must contain NetworkAgentState values")
        agent_ids = tuple(item.agent_id for item in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("population state agent ids must be unique")
        object.__setattr__(self, "agents", tuple(sorted(agents, key=_state_key)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "agents": [item.to_dict() for item in self.agents],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_population(
    model: NetworkABMModel,
    *,
    beliefs: Mapping[str, float] | None = None,
) -> PopulationState:
    """Create the canonical round-zero state for an exact model."""

    if not isinstance(model, NetworkABMModel):
        raise TypeError("population initialization requires NetworkABMModel")
    if beliefs is None:
        beliefs = {}
    if not isinstance(beliefs, Mapping):
        raise TypeError("initial beliefs must be a mapping")
    unknown = set(beliefs) - set(model.network.agent_ids)
    if unknown:
        raise ValueError("initial beliefs contain an unknown agent")
    profiles = {item.agent_id: item for item in model.agents}
    states: list[NetworkAgentState] = []
    for agent_id in model.network.agent_ids:
        seeded = agent_id in beliefs
        belief = _probability(
            beliefs[agent_id] if seeded else 0.0,
            label=f"initial belief for {agent_id}",
        )
        states.append(
            NetworkAgentState(
                agent_id,
                belief,
                1 if seeded else 0,
                belief >= profiles[agent_id].broadcast_threshold,
            )
        )
    return PopulationState(
        model.model_id,
        model.content_hash,
        0,
        None,
        tuple(states),
    )


__all__ = (
    "NetworkAgentSpec",
    "SocialEdge",
    "SocialNetwork",
    "NetworkABMModel",
    "NetworkAgentState",
    "PopulationState",
    "initialize_population",
)
