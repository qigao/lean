from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
    initialize_population,
)
from narrative_dynamics.contracts import stable_content_hash


class LifecycleStatus(str, Enum):
    """Membership status of one catalogued agent."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    DEAD = "dead"


class LifecycleEventKind(str, Enum):
    """Supported lifecycle transition at a round boundary."""

    ENTER = "enter"
    EXIT = "exit"
    DEATH = "death"


def _member_key(value: LifecycleMemberState) -> str:
    return value.agent_id


@dataclass(frozen=True)
class PopulationLifecycleEvent:
    """One requested membership transition before a propagation round."""

    agent_id: str
    kind: LifecycleEventKind
    belief: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="lifecycle event agent id"),
        )
        if not isinstance(self.kind, LifecycleEventKind):
            raise TypeError("lifecycle event kind must be LifecycleEventKind")
        if self.belief is not None:
            if self.kind is not LifecycleEventKind.ENTER:
                raise ValueError("lifecycle event belief is allowed only ENTER")
            object.__setattr__(
                self,
                "belief",
                _probability(self.belief, label="lifecycle entry belief"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "kind": self.kind.value,
            "belief": self.belief,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PopulationLifecycleModel:
    """Fixed identity catalog plus initial effective population."""

    model_id: str
    version: str
    catalog_model: NetworkABMModel
    initial_active_agent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="population lifecycle model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="population lifecycle model version"),
        )
        if not isinstance(self.catalog_model, NetworkABMModel):
            raise TypeError("population lifecycle catalog model must be NetworkABMModel")
        if not isinstance(self.initial_active_agent_ids, tuple):
            raise TypeError("initial active agent ids must be a tuple")
        agent_ids = tuple(
            _text(item, label="initial active agent id")
            for item in self.initial_active_agent_ids
        )
        if not agent_ids:
            raise ValueError("population lifecycle requires at least one initially active agent")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("initial active agent ids must be unique")
        if not set(agent_ids).issubset(self.catalog_model.network.agent_ids):
            raise ValueError("initial active agent ids contain an unknown catalog agent")
        object.__setattr__(self, "initial_active_agent_ids", tuple(sorted(agent_ids)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "catalog_model": self.catalog_model.to_dict(),
            "initial_active_agent_ids": list(self.initial_active_agent_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class LifecycleMemberState:
    """Persistent information and membership state for one catalogued agent."""

    agent_id: str
    belief: float
    exposure_count: int
    broadcasting: bool
    status: LifecycleStatus
    entry_count: int
    exit_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="lifecycle member agent id"),
        )
        object.__setattr__(
            self,
            "belief",
            _probability(self.belief, label="lifecycle member belief"),
        )
        object.__setattr__(
            self,
            "exposure_count",
            _nonnegative_integer(
                self.exposure_count,
                label="lifecycle member exposure count",
            ),
        )
        if not isinstance(self.broadcasting, bool):
            raise TypeError("lifecycle member broadcasting must be boolean")
        if not isinstance(self.status, LifecycleStatus):
            raise TypeError("lifecycle member status must be LifecycleStatus")
        object.__setattr__(
            self,
            "entry_count",
            _nonnegative_integer(self.entry_count, label="lifecycle member entry count"),
        )
        object.__setattr__(
            self,
            "exit_count",
            _nonnegative_integer(self.exit_count, label="lifecycle member exit count"),
        )
        if self.status is not LifecycleStatus.ACTIVE and self.broadcasting:
            raise ValueError("inactive or dead lifecycle member cannot broadcast")

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "belief": self.belief,
            "exposure_count": self.exposure_count,
            "broadcasting": self.broadcasting,
            "status": self.status.value,
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PopulationLifecycleState:
    """Complete catalog snapshot with a changing effective population."""

    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    members: tuple[LifecycleMemberState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="population lifecycle state model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="population lifecycle state model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="population lifecycle state round index",
        )
        object.__setattr__(self, "round_index", round_index)
        if round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("population lifecycle round zero cannot have a parent")
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(
                    self.parent_state_hash,
                    label="population lifecycle parent state hash",
                ),
            )
        if not isinstance(self.members, tuple):
            raise TypeError("population lifecycle members must be a tuple")
        members = tuple(self.members)
        if not members:
            raise ValueError("population lifecycle state requires catalog members")
        if any(not isinstance(item, LifecycleMemberState) for item in members):
            raise TypeError(
                "population lifecycle members must contain LifecycleMemberState values"
            )
        agent_ids = tuple(item.agent_id for item in members)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("population lifecycle member ids must be unique")
        object.__setattr__(self, "members", tuple(sorted(members, key=_member_key)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "members": [item.to_dict() for item in self.members],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_lifecycle_population(
    model: PopulationLifecycleModel,
    *,
    beliefs: Mapping[str, float] | None = None,
) -> PopulationLifecycleState:
    """Create a round-zero complete-catalog lifecycle snapshot."""

    if not isinstance(model, PopulationLifecycleModel):
        raise TypeError("lifecycle initialization requires PopulationLifecycleModel")
    base_state = initialize_population(model.catalog_model, beliefs=beliefs)
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
    return PopulationLifecycleState(
        model.model_id,
        model.content_hash,
        0,
        None,
        members,
    )


__all__ = (
    "LifecycleStatus",
    "LifecycleEventKind",
    "PopulationLifecycleEvent",
    "PopulationLifecycleModel",
    "LifecycleMemberState",
    "PopulationLifecycleState",
    "initialize_lifecycle_population",
)
