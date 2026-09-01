"""Immutable V19 contracts for situated multiplex-network runtime views."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel
from narrative_dynamics.abm.situated_cognition_contracts import SituatedCognitiveState
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedSocialMemoryModel,
    SituatedSocialMemoryState,
)
from narrative_dynamics.abm.situated_story import SituatedStory


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a content hash")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return 0.0 if result == 0.0 else result


def _unit(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a probability in [0, 1]")
    return result


def _affinity(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between -1 and 1")
    return result


@dataclass(frozen=True)
class SituatedNetworkRuntimeModel:
    model_id: str
    version: str
    percept_memory_model: SituatedPerceptMemoryCognitiveModel
    social_memory_model: SituatedSocialMemoryModel
    tracked_hypothesis_id: str
    adoption_threshold: float
    relationship_trust_threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated network model id"))
        object.__setattr__(self, "version", _text(self.version, label="situated network model version"))
        if not isinstance(self.percept_memory_model, SituatedPerceptMemoryCognitiveModel):
            raise TypeError("situated network model requires SituatedPerceptMemoryCognitiveModel")
        if not isinstance(self.social_memory_model, SituatedSocialMemoryModel):
            raise TypeError("situated network model requires SituatedSocialMemoryModel")
        cognitive = self.percept_memory_model.cognitive_model
        social_cognitive = self.social_memory_model.memory_cognitive_model.cognitive_model
        if social_cognitive != cognitive:
            raise ValueError("situated network models must bind the exact cognitive model")
        if self.percept_memory_model.perception_model.world_model != cognitive.world_model:
            raise ValueError("situated network models must bind the exact perception world")
        roster = tuple(agent.agent_id for agent in cognitive.agents)
        if roster != tuple(agent.agent_id for agent in social_cognitive.agents):
            raise ValueError("situated network models must bind the exact agent roster")
        object.__setattr__(self, "tracked_hypothesis_id", _text(
            self.tracked_hypothesis_id, label="situated network tracked hypothesis id"
        ))
        if any(
            self.tracked_hypothesis_id not in {item.hypothesis_id for item in agent.hypotheses}
            for agent in cognitive.agents
        ):
            raise ValueError("situated network tracked hypothesis must occur in every cognitive agent")
        object.__setattr__(self, "adoption_threshold", _unit(
            self.adoption_threshold, label="situated network adoption threshold"
        ))
        object.__setattr__(self, "relationship_trust_threshold", _unit(
            self.relationship_trust_threshold, label="situated network relationship trust threshold"
        ))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "percept_memory_model_hash": self.percept_memory_model.content_hash,
            "social_memory_model_hash": self.social_memory_model.content_hash,
            "tracked_hypothesis_id": self.tracked_hypothesis_id,
            "adoption_threshold": self.adoption_threshold,
            "relationship_trust_threshold": self.relationship_trust_threshold,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkAgentNode:
    agent_id: str
    role_id: str
    place_id: str
    tracked_belief_probability: float
    active_claim_count: int

    def __post_init__(self) -> None:
        for name in ("agent_id", "role_id", "place_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"network node {name.replace('_', ' ')}"))
        object.__setattr__(self, "tracked_belief_probability", _unit(
            self.tracked_belief_probability, label="network node tracked belief probability"
        ))
        object.__setattr__(self, "active_claim_count", _nonnegative_integer(
            self.active_claim_count, label="network node active claim count"
        ))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "role_id": self.role_id,
            "place_id": self.place_id,
            "tracked_belief_probability": self.tracked_belief_probability,
            "active_claim_count": self.active_claim_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkRelationshipEdge:
    source_agent_id: str
    observer_agent_id: str
    trust: float
    affinity: float
    confirmation_count: int
    contradiction_count: int
    active: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_agent_id", _text(self.source_agent_id, label="network relationship source agent id"))
        object.__setattr__(self, "observer_agent_id", _text(self.observer_agent_id, label="network relationship observer agent id"))
        if self.source_agent_id == self.observer_agent_id:
            raise ValueError("network relationship source and observer must differ")
        object.__setattr__(self, "trust", _unit(self.trust, label="network relationship trust"))
        object.__setattr__(self, "affinity", _affinity(self.affinity, label="network relationship affinity"))
        for name in ("confirmation_count", "contradiction_count"):
            object.__setattr__(self, name, _nonnegative_integer(
                getattr(self, name), label=f"network relationship {name.replace('_', ' ')}"
            ))
        if not isinstance(self.active, bool):
            raise TypeError("network relationship active must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_agent_id": self.source_agent_id,
            "observer_agent_id": self.observer_agent_id,
            "trust": self.trust,
            "affinity": self.affinity,
            "confirmation_count": self.confirmation_count,
            "contradiction_count": self.contradiction_count,
            "active": self.active,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkAccessEdge:
    source_agent_id: str
    observer_agent_id: str
    visual_cost: float | None
    auditory_loss: float | None
    direct_interaction: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_agent_id", _text(self.source_agent_id, label="network access source agent id"))
        object.__setattr__(self, "observer_agent_id", _text(self.observer_agent_id, label="network access observer agent id"))
        if self.source_agent_id == self.observer_agent_id:
            raise ValueError("network access source and observer must differ")
        for name in ("visual_cost", "auditory_loss"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(
                    value, label=f"network access {name.replace('_', ' ')}", minimum=0.0
                ))
        if not isinstance(self.direct_interaction, bool):
            raise TypeError("network access direct interaction must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_agent_id": self.source_agent_id,
            "observer_agent_id": self.observer_agent_id,
            "visual_cost": self.visual_cost,
            "auditory_loss": self.auditory_loss,
            "direct_interaction": self.direct_interaction,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkTransmission:
    event_id: str
    source_agent_id: str
    observer_agent_id: str
    fidelity: SituatedPerceptFidelity
    channels: tuple[ObservationChannel, ...]

    def __post_init__(self) -> None:
        for name in ("event_id", "source_agent_id", "observer_agent_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"network transmission {name.replace('_', ' ')}"))
        if self.source_agent_id == self.observer_agent_id:
            raise ValueError("network transmission source and observer must be distinct")
        if not isinstance(self.fidelity, SituatedPerceptFidelity):
            raise TypeError("network transmission fidelity must be SituatedPerceptFidelity")
        if self.fidelity is SituatedPerceptFidelity.DETECTED:
            raise ValueError(
                "network transmission requires a disclosed source"
            )
        if not isinstance(self.channels, tuple) or any(
            not isinstance(item, ObservationChannel) for item in self.channels
        ):
            raise TypeError("network transmission channels must be a tuple of ObservationChannel values")
        if not self.channels:
            raise ValueError("network transmission requires disclosed channels")
        if any(item not in (ObservationChannel.VISUAL, ObservationChannel.AUDITORY) for item in self.channels):
            raise ValueError("network transmission channels must be visual or auditory")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("network transmission channels must be unique")
        object.__setattr__(self, "channels", tuple(sorted(self.channels, key=lambda item: item.value)))

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "source_agent_id": self.source_agent_id,
            "observer_agent_id": self.observer_agent_id,
            "fidelity": self.fidelity.value,
            "channels": [item.value for item in self.channels],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkSnapshot:
    model_id: str
    model_hash: str
    round_index: int
    story_hash: str
    cognitive_state_hash: str
    social_state_hash: str
    nodes: tuple[SituatedNetworkAgentNode, ...]
    relationship_edges: tuple[SituatedNetworkRelationshipEdge, ...]
    access_edges: tuple[SituatedNetworkAccessEdge, ...]
    transmissions: tuple[SituatedNetworkTransmission, ...]
    latest_tell_event_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="network snapshot model id"))
        for name in ("model_hash", "story_hash", "cognitive_state_hash", "social_state_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), label=f"network snapshot {name.replace('_', ' ')}"))
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="network snapshot round index"))
        for name, item_type, identity, unique_label in (
            ("nodes", SituatedNetworkAgentNode, lambda item: item.agent_id, "node ids"),
            ("relationship_edges", SituatedNetworkRelationshipEdge, lambda item: (item.source_agent_id, item.observer_agent_id), "relationship pairs"),
            ("access_edges", SituatedNetworkAccessEdge, lambda item: (item.source_agent_id, item.observer_agent_id), "access pairs"),
            ("transmissions", SituatedNetworkTransmission, lambda item: (item.event_id, item.source_agent_id, item.observer_agent_id), "transmission identities"),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, item_type) for item in values):
                raise TypeError(f"network snapshot {name.replace('_', ' ')} must be a tuple of {item_type.__name__} values")
            identities = tuple(identity(item) for item in values)
            if len(set(identities)) != len(identities):
                raise ValueError(f"network snapshot {unique_label} must be unique")
            object.__setattr__(self, name, tuple(sorted(values, key=identity)))
        object.__setattr__(
            self,
            "latest_tell_event_count",
            _nonnegative_integer(
                self.latest_tell_event_count,
                label="network snapshot latest tell event count",
            ),
        )
        if self.latest_tell_event_count < len({item.event_id for item in self.transmissions}):
            raise ValueError(
                "network snapshot latest tell event count must cover transmitted events"
            )
        if self.round_index == 0 and self.latest_tell_event_count != 0:
            raise ValueError("network snapshot round zero latest tell event count must be zero")
        node_ids = {item.agent_id for item in self.nodes}
        if any(
            item.source_agent_id not in node_ids or item.observer_agent_id not in node_ids
            for edges in (self.relationship_edges, self.access_edges, self.transmissions)
            for item in edges
        ):
            raise ValueError("network snapshot edges must reference snapshot nodes")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "story_hash": self.story_hash,
            "cognitive_state_hash": self.cognitive_state_hash,
            "social_state_hash": self.social_state_hash,
            "nodes": [item.to_dict() for item in self.nodes],
            "relationship_edges": [item.to_dict() for item in self.relationship_edges],
            "access_edges": [item.to_dict() for item in self.access_edges],
            "transmissions": [item.to_dict() for item in self.transmissions],
            "latest_tell_event_count": self.latest_tell_event_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkEmergenceMetrics:
    snapshot_hash: str
    round_index: int
    population_size: int
    occupied_place_count: int
    adopted_count: int
    adoption_rate: float
    tracked_belief_mean: float
    tracked_belief_variance: float
    active_relationship_edge_count: int
    mean_relationship_trust: float
    direct_interaction_pair_count: int
    latest_tell_event_count: int
    transmission_count: int
    reached_observer_count: int
    exact_transmission_count: int
    detected_transmission_count: int
    identified_transmission_count: int
    active_claim_count: int
    confirmed_claim_count: int
    contradicted_claim_count: int
    superseded_claim_count: int
    forgotten_claim_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_hash", _hash(self.snapshot_hash, label="network metrics snapshot hash"))
        for name in (
            "round_index", "population_size", "occupied_place_count", "adopted_count",
            "active_relationship_edge_count", "direct_interaction_pair_count", "latest_tell_event_count",
            "transmission_count", "reached_observer_count", "exact_transmission_count",
            "detected_transmission_count", "identified_transmission_count", "active_claim_count",
            "confirmed_claim_count", "contradicted_claim_count", "superseded_claim_count",
            "forgotten_claim_count",
        ):
            object.__setattr__(self, name, _nonnegative_integer(getattr(self, name), label=f"network metrics {name.replace('_', ' ')}"))
        if self.population_size == 0:
            raise ValueError("network metrics population size must be positive")
        if self.occupied_place_count > self.population_size or self.adopted_count > self.population_size:
            raise ValueError("network metrics population counts cannot exceed population size")
        object.__setattr__(self, "adoption_rate", _unit(self.adoption_rate, label="network metrics adoption rate"))
        if not math.isclose(self.adoption_rate, self.adopted_count / self.population_size, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("network metrics adoption rate must equal adopted count divided by population")
        object.__setattr__(self, "tracked_belief_mean", _unit(self.tracked_belief_mean, label="network metrics tracked belief mean"))
        object.__setattr__(self, "tracked_belief_variance", _finite(self.tracked_belief_variance, label="network metrics tracked belief variance", minimum=0.0))
        object.__setattr__(self, "mean_relationship_trust", _unit(self.mean_relationship_trust, label="network metrics mean relationship trust"))
        if self.direct_interaction_pair_count > self.population_size * (self.population_size - 1):
            raise ValueError("network metrics direct interaction pairs exceed ordered population pairs")
        if self.reached_observer_count > self.transmission_count:
            raise ValueError("network metrics reached observers cannot exceed transmissions")
        if self.exact_transmission_count + self.detected_transmission_count + self.identified_transmission_count != self.transmission_count:
            raise ValueError("network metrics fidelity counts must equal transmission count")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkRuntimeState:
    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    memory_store_hash: str
    story: SituatedStory
    cognitive_state: SituatedCognitiveState
    social_state: SituatedSocialMemoryState
    snapshot: SituatedNetworkSnapshot
    metrics: SituatedNetworkEmergenceMetrics
    checkpoint: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="network runtime state model id"))
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="network runtime state model hash"))
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="network runtime state round index"))
        if not isinstance(self.checkpoint, bool):
            raise TypeError("network runtime state checkpoint must be boolean")
        if self.parent_state_hash is not None:
            object.__setattr__(self, "parent_state_hash", _hash(self.parent_state_hash, label="network runtime state parent hash"))
        object.__setattr__(
            self,
            "memory_store_hash",
            _hash(
                self.memory_store_hash,
                label="network runtime state memory store hash",
            ),
        )
        if self.checkpoint and self.parent_state_hash is not None:
            raise ValueError("network runtime state checkpoint must not have a parent hash")
        if not self.checkpoint and self.round_index == 0 and self.parent_state_hash is not None:
            raise ValueError("network runtime state initial round must not have a parent hash")
        if not self.checkpoint and self.round_index != 0 and self.parent_state_hash is None:
            raise ValueError("network runtime state non-initial rounds require an exact parent hash")
        if not isinstance(self.story, SituatedStory) or not isinstance(self.cognitive_state, SituatedCognitiveState) or not isinstance(self.social_state, SituatedSocialMemoryState):
            raise TypeError("network runtime state requires exact story, cognitive state, and social state")
        if not isinstance(self.snapshot, SituatedNetworkSnapshot) or not isinstance(self.metrics, SituatedNetworkEmergenceMetrics):
            raise TypeError("network runtime state requires a network snapshot and metrics")
        if (
            self.story.current_state.round_index != self.round_index
            or self.cognitive_state.round_index != self.round_index
            or self.social_state.round_index != self.round_index
            or self.snapshot.round_index != self.round_index
            or self.metrics.round_index != self.round_index
        ):
            raise ValueError("network runtime state values must share one exact round")
        if self.cognitive_state.story_hash != self.story.content_hash:
            raise ValueError("network runtime state cognitive state must bind exact story")
        if self.social_state.cognitive_state_hash != self.cognitive_state.content_hash:
            raise ValueError("network runtime state social state must bind exact cognitive state")
        if (
            self.checkpoint != self.cognitive_state.checkpoint
            or self.checkpoint != self.social_state.checkpoint
        ):
            raise ValueError(
                "network runtime state checkpoint must bind exact cognitive and social checkpoints"
            )
        if self.snapshot.model_id != self.model_id or self.snapshot.model_hash != self.model_hash:
            raise ValueError("network runtime state snapshot must bind the exact model")
        if (
            self.snapshot.story_hash != self.story.content_hash
            or self.snapshot.cognitive_state_hash != self.cognitive_state.content_hash
            or self.snapshot.social_state_hash != self.social_state.content_hash
        ):
            raise ValueError("network runtime state snapshot hashes must bind exact state values")
        if self.metrics.snapshot_hash != self.snapshot.content_hash:
            raise ValueError("network runtime state metrics must bind the exact snapshot")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "memory_store_hash": self.memory_store_hash,
            "story_hash": self.story.content_hash,
            "cognitive_state_hash": self.cognitive_state.content_hash,
            "social_state_hash": self.social_state.content_hash,
            "snapshot": self.snapshot.to_dict(),
            "metrics": self.metrics.to_dict(),
            "checkpoint": self.checkpoint,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validate_underlying_branch(
    prior_state: SituatedNetworkRuntimeState,
    next_state: SituatedNetworkRuntimeState,
) -> None:
    prior_story = prior_state.story
    next_story = next_state.story
    if (
        next_story.model_id != prior_story.model_id
        or next_story.model_hash != prior_story.model_hash
        or next_story.initial_state != prior_story.initial_state
        or next_story.perception_model != prior_story.perception_model
        or len(next_story.rounds) != len(prior_story.rounds) + 1
        or next_story.rounds[:-1] != prior_story.rounds
    ):
        raise ValueError(
            "network round must extend the exact underlying branch"
        )
    if (
        next_state.cognitive_state.parent_state_hash
        != prior_state.cognitive_state.content_hash
    ):
        raise ValueError(
            "network round cognitive state must extend the exact underlying branch"
        )
    if (
        next_state.social_state.parent_state_hash
        != prior_state.social_state.content_hash
    ):
        raise ValueError(
            "network round social state must extend the exact underlying branch"
        )


@dataclass(frozen=True)
class SituatedNetworkRoundResult:
    model_id: str
    model_hash: str
    prior_state: SituatedNetworkRuntimeState
    next_state: SituatedNetworkRuntimeState

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="network round model id"))
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="network round model hash"))
        if not isinstance(self.prior_state, SituatedNetworkRuntimeState) or not isinstance(self.next_state, SituatedNetworkRuntimeState):
            raise TypeError("network round requires prior and next runtime states")
        if any(state.model_id != self.model_id or state.model_hash != self.model_hash for state in (self.prior_state, self.next_state)):
            raise ValueError("network round states must bind the exact model")
        if self.next_state.round_index != self.prior_state.round_index + 1:
            raise ValueError("network round next state must advance exactly one round")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("network round next state must bind the exact parent")
        _validate_underlying_branch(self.prior_state, self.next_state)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "prior_state_hash": self.prior_state.content_hash,
            "next_state_hash": self.next_state.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedNetworkTrajectory:
    model_id: str
    model_hash: str
    initial_state: SituatedNetworkRuntimeState
    rounds: tuple[SituatedNetworkRoundResult, ...]
    final_state: SituatedNetworkRuntimeState

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="network trajectory model id"))
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="network trajectory model hash"))
        if not isinstance(self.initial_state, SituatedNetworkRuntimeState) or not isinstance(self.final_state, SituatedNetworkRuntimeState):
            raise TypeError("network trajectory requires initial and final runtime states")
        if not isinstance(self.rounds, tuple) or any(not isinstance(item, SituatedNetworkRoundResult) for item in self.rounds):
            raise TypeError("network trajectory rounds must be a tuple of network round results")
        if not self.rounds:
            raise ValueError("network trajectory requires at least one round")
        state = self.initial_state
        for item in self.rounds:
            if item.model_id != self.model_id or item.model_hash != self.model_hash:
                raise ValueError("network trajectory rounds must bind the exact model")
            if item.prior_state != state:
                raise ValueError("network trajectory rounds must form one exact chain")
            _validate_underlying_branch(item.prior_state, item.next_state)
            state = item.next_state
        if state != self.final_state:
            raise ValueError("network trajectory final state must equal its exact chain")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "initial_state_hash": self.initial_state.content_hash,
            "rounds": [item.to_dict() for item in self.rounds],
            "final_state_hash": self.final_state.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = (
    "SituatedNetworkRuntimeModel",
    "SituatedNetworkAgentNode",
    "SituatedNetworkRelationshipEdge",
    "SituatedNetworkAccessEdge",
    "SituatedNetworkTransmission",
    "SituatedNetworkSnapshot",
    "SituatedNetworkEmergenceMetrics",
    "SituatedNetworkRuntimeState",
    "SituatedNetworkRoundResult",
    "SituatedNetworkTrajectory",
)
