"""Immutable, transport-neutral contracts for audience-scoped simulation output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import TypeAlias

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_cognition import SituatedCognitiveDecision
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkEmergenceMetrics,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPercept


SIMULATION_OUTPUT_RECORD_SCHEMA = "narrative-dynamics.simulation-output-record/v1"
SIMULATION_OUTPUT_BATCH_SCHEMA = "narrative-dynamics.simulation-output-batch/v1"
SIMULATION_OUTPUT_VIEW_SCHEMA = "narrative-dynamics.simulation-output-view/v1"

_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class SimulationOutputAudience(str, Enum):
    PUBLIC = "public"
    OBJECTIVE = "objective"
    AGENT = "agent"
    ANALYST = "analyst"
    INTERNAL = "internal"


class SimulationOutputKind(str, Enum):
    STATE_DELTA = "state.delta"
    EVENT_OBJECTIVE = "event.objective"
    PERCEPT_PRIVATE = "percept.private"
    AGENT_DECISION = "agent.decision"
    MEMORY_UPDATE = "memory.update"
    SOCIAL_UPDATE = "social.update"
    NETWORK_METRICS = "network.metrics"
    STORY_PROGRESS = "story.progress"
    NARRATIVE_SCENE = "narrative.scene"
    BLENDER_DELTA = "blender.delta"
    COMMAND_RESULT = "command.result"
    DIAGNOSTIC = "diagnostic"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label=label)


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a content hash")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _canonical_text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TypeError(f"{label} must be a tuple of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} identities must be unique")
    return tuple(sorted(value))


class _PayloadHash:
    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(self.content_hash)


@dataclass(frozen=True)
class SimulationStateDeltaPayload(_PayloadHash):
    prior_snapshot_hash: str
    next_snapshot_hash: str
    changed_agent_ids: tuple[str, ...]
    changed_passage_ids: tuple[str, ...]
    changed_object_ids: tuple[str, ...]

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "prior_snapshot_hash", _hash(self.prior_snapshot_hash, label="state delta prior snapshot hash"))
        object.__setattr__(self, "next_snapshot_hash", _hash(self.next_snapshot_hash, label="state delta next snapshot hash"))
        for name in ("changed_agent_ids", "changed_passage_ids", "changed_object_ids"):
            object.__setattr__(self, name, _canonical_text_tuple(getattr(self, name), label=f"state delta {name.replace('_', ' ')}"))

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_snapshot_hash": self.prior_snapshot_hash,
            "next_snapshot_hash": self.next_snapshot_hash,
            "changed_agent_ids": list(self.changed_agent_ids),
            "changed_passage_ids": list(self.changed_passage_ids),
            "changed_object_ids": list(self.changed_object_ids),
        }


@dataclass(frozen=True)
class SimulationObjectiveEventPayload(_PayloadHash):
    event_id: str
    event_hash: str
    action_id: str
    action_kind: str
    actor_agent_id: str
    place_id: str
    target_id: str | None
    success: bool
    cause_event_ids: tuple[str, ...]

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        for name in ("event_id", "action_id", "action_kind", "actor_agent_id", "place_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"objective event {name.replace('_', ' ')}"))
        object.__setattr__(self, "event_hash", _hash(self.event_hash, label="objective event hash"))
        object.__setattr__(self, "target_id", _optional_text(self.target_id, label="objective event target id"))
        object.__setattr__(self, "success", _bool(self.success, label="objective event success"))
        object.__setattr__(self, "cause_event_ids", _canonical_text_tuple(self.cause_event_ids, label="objective event cause event ids"))

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_hash": self.event_hash,
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "actor_agent_id": self.actor_agent_id,
            "place_id": self.place_id,
            "target_id": self.target_id,
            "success": self.success,
            "cause_event_ids": list(self.cause_event_ids),
        }


@dataclass(frozen=True)
class SimulationPrivatePerceptPayload(_PayloadHash):
    percept: SituatedPercept

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        if not isinstance(self.percept, SituatedPercept):
            raise TypeError("private percept payload requires a SituatedPercept")

    def to_dict(self) -> dict[str, object]:
        return {"percept": self.percept.to_dict()}


@dataclass(frozen=True)
class SimulationAgentDecisionPayload(_PayloadHash):
    decision: SituatedCognitiveDecision

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SituatedCognitiveDecision):
            raise TypeError("agent decision payload requires a SituatedCognitiveDecision")

    def to_dict(self) -> dict[str, object]:
        return {"decision": self.decision.to_dict()}


@dataclass(frozen=True)
class SimulationMemoryUpdatePayload(_PayloadHash):
    agent_id: str
    prior_mind_hash: str
    next_mind_hash: str
    recalled_memory_ids: tuple[str, ...]
    admitted_memory_ids: tuple[str, ...]

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="memory update agent id"))
        object.__setattr__(self, "prior_mind_hash", _hash(self.prior_mind_hash, label="memory update prior mind hash"))
        object.__setattr__(self, "next_mind_hash", _hash(self.next_mind_hash, label="memory update next mind hash"))
        for name in ("recalled_memory_ids", "admitted_memory_ids"):
            object.__setattr__(self, name, _canonical_text_tuple(getattr(self, name), label=f"memory update {name.replace('_', ' ')}"))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "prior_mind_hash": self.prior_mind_hash,
            "next_mind_hash": self.next_mind_hash,
            "recalled_memory_ids": list(self.recalled_memory_ids),
            "admitted_memory_ids": list(self.admitted_memory_ids),
        }


@dataclass(frozen=True)
class SimulationSocialUpdatePayload(_PayloadHash):
    observer_agent_id: str
    prior_claim_hashes: tuple[str, ...]
    next_claim_hashes: tuple[str, ...]
    prior_relationship_hashes: tuple[str, ...]
    next_relationship_hashes: tuple[str, ...]
    admitted_evidence_ids: tuple[str, ...]

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "observer_agent_id", _text(self.observer_agent_id, label="social update observer agent id"))
        for name in (
            "prior_claim_hashes", "next_claim_hashes",
            "prior_relationship_hashes", "next_relationship_hashes",
        ):
            values = _canonical_text_tuple(getattr(self, name), label=f"social update {name.replace('_', ' ')}")
            for value in values:
                _hash(value, label=f"social update {name.replace('_', ' ')} value")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "admitted_evidence_ids", _canonical_text_tuple(self.admitted_evidence_ids, label="social update admitted evidence ids"))

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_agent_id": self.observer_agent_id,
            "prior_claim_hashes": list(self.prior_claim_hashes),
            "next_claim_hashes": list(self.next_claim_hashes),
            "prior_relationship_hashes": list(self.prior_relationship_hashes),
            "next_relationship_hashes": list(self.next_relationship_hashes),
            "admitted_evidence_ids": list(self.admitted_evidence_ids),
        }


@dataclass(frozen=True)
class SimulationNetworkMetricsPayload(_PayloadHash):
    metrics: SituatedNetworkEmergenceMetrics

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, SituatedNetworkEmergenceMetrics):
            raise TypeError("network metrics payload requires SituatedNetworkEmergenceMetrics")

    def to_dict(self) -> dict[str, object]:
        return {"metrics": self.metrics.to_dict()}


@dataclass(frozen=True)
class SimulationStoryProgressPayload(_PayloadHash):
    active_scene_id: str | None
    completed_scene_ids: tuple[str, ...]
    status: str

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_scene_id", _optional_text(self.active_scene_id, label="story progress active scene id"))
        object.__setattr__(self, "completed_scene_ids", _canonical_text_tuple(self.completed_scene_ids, label="story progress completed scene ids"))
        object.__setattr__(self, "status", _text(self.status, label="story progress status"))

    def to_dict(self) -> dict[str, object]:
        return {
            "active_scene_id": self.active_scene_id,
            "completed_scene_ids": list(self.completed_scene_ids),
            "status": self.status,
        }


@dataclass(frozen=True)
class SimulationNarrativeScenePayload(_PayloadHash):
    scene_id: str
    projection_hash: str
    realization_hash: str

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _text(self.scene_id, label="narrative scene id"))
        object.__setattr__(self, "projection_hash", _hash(self.projection_hash, label="narrative scene projection hash"))
        object.__setattr__(self, "realization_hash", _hash(self.realization_hash, label="narrative scene realization hash"))

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "projection_hash": self.projection_hash,
            "realization_hash": self.realization_hash,
        }


def _canonical_agent_places(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in value
    ):
        raise TypeError("Blender delta agent places must be a tuple of (agent id, place id) tuples")
    result = tuple(
        (_text(item[0], label="Blender delta agent id"), _text(item[1], label="Blender delta agent place id"))
        for item in value
    )
    if len({item[0] for item in result}) != len(result):
        raise ValueError("Blender delta agent identities must be unique")
    return tuple(sorted(result))


def _canonical_passage_states(value: object) -> tuple[tuple[str, bool], ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in value
    ):
        raise TypeError("Blender delta passage states must be a tuple of (passage id, open) tuples")
    result = tuple(
        (_text(item[0], label="Blender delta passage id"), _bool(item[1], label="Blender delta passage open"))
        for item in value
    )
    if len({item[0] for item in result}) != len(result):
        raise ValueError("Blender delta passage identities must be unique")
    return tuple(sorted(result))


def _canonical_object_placements(
    value: object,
) -> tuple[tuple[str, str | None, str | None], ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple) or len(item) != 3 for item in value
    ):
        raise TypeError("Blender delta object placements must be a tuple of placement tuples")
    result = tuple(
        (
            _text(item[0], label="Blender delta object id"),
            _optional_text(item[1], label="Blender delta object place id"),
            _optional_text(item[2], label="Blender delta holder agent id"),
        )
        for item in value
    )
    if len({item[0] for item in result}) != len(result):
        raise ValueError("Blender delta object identities must be unique")
    return tuple(sorted(result, key=lambda item: item[0]))


@dataclass(frozen=True)
class SimulationBlenderDeltaPayload(_PayloadHash):
    agent_places: tuple[tuple[str, str], ...]
    passage_states: tuple[tuple[str, bool], ...]
    object_placements: tuple[tuple[str, str | None, str | None], ...]

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_places", _canonical_agent_places(self.agent_places))
        object.__setattr__(self, "passage_states", _canonical_passage_states(self.passage_states))
        object.__setattr__(self, "object_placements", _canonical_object_placements(self.object_placements))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_places": [list(item) for item in self.agent_places],
            "passage_states": [list(item) for item in self.passage_states],
            "object_placements": [list(item) for item in self.object_placements],
        }


@dataclass(frozen=True)
class SimulationCommandResultPayload(_PayloadHash):
    command_id: str
    accepted: bool
    reason_code: str

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _text(self.command_id, label="command result command id"))
        object.__setattr__(self, "accepted", _bool(self.accepted, label="command result accepted"))
        object.__setattr__(self, "reason_code", _text(self.reason_code, label="command result reason code"))

    def to_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class SimulationDiagnosticPayload(_PayloadHash):
    code: str
    message: str

    __hash__ = _PayloadHash.__hash__

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, label="diagnostic code"))
        object.__setattr__(self, "message", _text(self.message, label="diagnostic message"))

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message}


SimulationOutputPayload: TypeAlias = (
    SimulationStateDeltaPayload
    | SimulationObjectiveEventPayload
    | SimulationPrivatePerceptPayload
    | SimulationAgentDecisionPayload
    | SimulationMemoryUpdatePayload
    | SimulationSocialUpdatePayload
    | SimulationNetworkMetricsPayload
    | SimulationStoryProgressPayload
    | SimulationNarrativeScenePayload
    | SimulationBlenderDeltaPayload
    | SimulationCommandResultPayload
    | SimulationDiagnosticPayload
)

_PAYLOAD_KIND = {
    SimulationStateDeltaPayload: SimulationOutputKind.STATE_DELTA,
    SimulationObjectiveEventPayload: SimulationOutputKind.EVENT_OBJECTIVE,
    SimulationPrivatePerceptPayload: SimulationOutputKind.PERCEPT_PRIVATE,
    SimulationAgentDecisionPayload: SimulationOutputKind.AGENT_DECISION,
    SimulationMemoryUpdatePayload: SimulationOutputKind.MEMORY_UPDATE,
    SimulationSocialUpdatePayload: SimulationOutputKind.SOCIAL_UPDATE,
    SimulationNetworkMetricsPayload: SimulationOutputKind.NETWORK_METRICS,
    SimulationStoryProgressPayload: SimulationOutputKind.STORY_PROGRESS,
    SimulationNarrativeScenePayload: SimulationOutputKind.NARRATIVE_SCENE,
    SimulationBlenderDeltaPayload: SimulationOutputKind.BLENDER_DELTA,
    SimulationCommandResultPayload: SimulationOutputKind.COMMAND_RESULT,
    SimulationDiagnosticPayload: SimulationOutputKind.DIAGNOSTIC,
}

_PRIVATE_OWNER_ATTRIBUTE = {
    SimulationPrivatePerceptPayload: lambda payload: payload.percept.agent_id,
    SimulationAgentDecisionPayload: lambda payload: payload.decision.agent_id,
    SimulationMemoryUpdatePayload: lambda payload: payload.agent_id,
    SimulationSocialUpdatePayload: lambda payload: payload.observer_agent_id,
}

_KIND_RANK = {
    SimulationOutputKind.COMMAND_RESULT: 0,
    SimulationOutputKind.EVENT_OBJECTIVE: 1,
    SimulationOutputKind.STATE_DELTA: 2,
    SimulationOutputKind.PERCEPT_PRIVATE: 3,
    SimulationOutputKind.AGENT_DECISION: 4,
    SimulationOutputKind.MEMORY_UPDATE: 5,
    SimulationOutputKind.SOCIAL_UPDATE: 6,
    SimulationOutputKind.NETWORK_METRICS: 7,
    SimulationOutputKind.STORY_PROGRESS: 8,
    SimulationOutputKind.NARRATIVE_SCENE: 9,
    SimulationOutputKind.BLENDER_DELTA: 10,
    SimulationOutputKind.DIAGNOSTIC: 11,
}


def _payload_identity(payload: SimulationOutputPayload) -> str:
    if isinstance(payload, SimulationObjectiveEventPayload):
        return payload.event_id
    if isinstance(payload, SimulationPrivatePerceptPayload):
        return payload.percept.percept_id
    if isinstance(payload, SimulationAgentDecisionPayload):
        return payload.decision.selected_action_id
    if isinstance(payload, SimulationMemoryUpdatePayload):
        return payload.agent_id
    if isinstance(payload, SimulationSocialUpdatePayload):
        return payload.observer_agent_id
    if isinstance(payload, SimulationStoryProgressPayload):
        return "" if payload.active_scene_id is None else payload.active_scene_id
    if isinstance(payload, SimulationNarrativeScenePayload):
        return payload.scene_id
    if isinstance(payload, SimulationCommandResultPayload):
        return payload.command_id
    if isinstance(payload, SimulationDiagnosticPayload):
        return payload.code
    return payload.content_hash


@dataclass(frozen=True)
class SimulationOutputRecord:
    stream_id: str
    scenario_hash: str
    sequence: int
    round_index: int
    state_hash: str
    kind: SimulationOutputKind
    audience: SimulationOutputAudience
    owner_agent_id: str | None
    source_artifact_hashes: tuple[str, ...]
    payload: SimulationOutputPayload
    schema: str = SIMULATION_OUTPUT_RECORD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _text(self.stream_id, label="output record stream id"))
        object.__setattr__(self, "scenario_hash", _hash(self.scenario_hash, label="output record scenario hash"))
        object.__setattr__(self, "sequence", _positive_integer(self.sequence, label="output record sequence"))
        object.__setattr__(self, "round_index", _positive_integer(self.round_index, label="output record round index"))
        object.__setattr__(self, "state_hash", _hash(self.state_hash, label="output record state hash"))
        if not isinstance(self.kind, SimulationOutputKind):
            raise TypeError("output record kind must be SimulationOutputKind")
        if not isinstance(self.audience, SimulationOutputAudience):
            raise TypeError("output record audience must be SimulationOutputAudience")
        object.__setattr__(self, "owner_agent_id", _optional_text(self.owner_agent_id, label="output record owner agent id"))
        sources = _canonical_text_tuple(self.source_artifact_hashes, label="output record source artifact hashes")
        for value in sources:
            _hash(value, label="output record source artifact hash")
        object.__setattr__(self, "source_artifact_hashes", sources)
        if type(self.payload) not in _PAYLOAD_KIND:
            raise TypeError("output record payload must be a typed simulation output payload")
        if _PAYLOAD_KIND[type(self.payload)] is not self.kind:
            raise ValueError("output record kind must match payload kind")
        owner_getter = _PRIVATE_OWNER_ATTRIBUTE.get(type(self.payload))
        if owner_getter is not None:
            if self.audience is not SimulationOutputAudience.AGENT:
                raise ValueError("private output payload audience must be agent")
            if self.owner_agent_id != owner_getter(self.payload):
                raise ValueError("private output payload owner must match its Agent")
        elif self.audience is SimulationOutputAudience.AGENT:
            raise ValueError("agent audience payload must name a matching owner")
        elif self.owner_agent_id is not None:
            raise ValueError("non-agent output audience must not name an owner")
        if self.schema != SIMULATION_OUTPUT_RECORD_SCHEMA:
            raise ValueError("output record schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "scenario_hash": self.scenario_hash,
            "sequence": self.sequence,
            "round_index": self.round_index,
            "state_hash": self.state_hash,
            "kind": self.kind.value,
            "audience": self.audience.value,
            "owner_agent_id": self.owner_agent_id,
            "source_artifact_hashes": list(self.source_artifact_hashes),
            "payload": self.payload.to_dict(),
            "payload_hash": self.payload.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def output_record_sort_key(record: SimulationOutputRecord) -> tuple[int, str, str, int]:
    if not isinstance(record, SimulationOutputRecord):
        raise TypeError("output record sort key requires a SimulationOutputRecord")
    return (
        _KIND_RANK[record.kind],
        "" if record.owner_agent_id is None else record.owner_agent_id,
        _payload_identity(record.payload),
        record.sequence,
    )


@dataclass(frozen=True)
class SimulationOutputBatch:
    stream_id: str
    scenario_hash: str
    prior_state_hash: str
    next_state_hash: str
    round_result_hash: str
    first_sequence: int
    last_sequence: int
    records: tuple[SimulationOutputRecord, ...]
    checkpoint: bool = False
    schema: str = SIMULATION_OUTPUT_BATCH_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _text(self.stream_id, label="output batch stream id"))
        for name in ("scenario_hash", "prior_state_hash", "next_state_hash", "round_result_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), label=f"output batch {name.replace('_', ' ')}"))
        object.__setattr__(self, "first_sequence", _positive_integer(self.first_sequence, label="output batch first sequence"))
        object.__setattr__(self, "last_sequence", _positive_integer(self.last_sequence, label="output batch last sequence"))
        if not isinstance(self.records, tuple) or not self.records or any(
            not isinstance(record, SimulationOutputRecord) for record in self.records
        ):
            raise ValueError("output batch requires a non-empty tuple of records")
        if any(
            type(record.payload) not in _PAYLOAD_KIND
            or _PAYLOAD_KIND[type(record.payload)] is not record.kind
            for record in self.records
        ):
            raise ValueError("output batch record kind must match its payload kind")
        expected_sequences = tuple(range(self.first_sequence, self.last_sequence + 1))
        actual_sequences = tuple(record.sequence for record in self.records)
        if actual_sequences != expected_sequences:
            raise ValueError("output batch record sequences must be contiguous")
        if any(record.stream_id != self.stream_id for record in self.records):
            raise ValueError("output batch records must bind the exact stream")
        if any(record.scenario_hash != self.scenario_hash for record in self.records):
            raise ValueError("output batch records must bind the exact scenario")
        if any(record.state_hash != self.next_state_hash for record in self.records):
            raise ValueError("output batch records must bind the exact next state")
        if len({record.round_index for record in self.records}) != 1:
            raise ValueError("output batch records must bind one exact round")
        if tuple(sorted(self.records, key=output_record_sort_key)) != self.records:
            raise ValueError("output batch records must use canonical order")
        object.__setattr__(self, "checkpoint", _bool(self.checkpoint, label="output batch checkpoint"))
        if self.schema != SIMULATION_OUTPUT_BATCH_SCHEMA:
            raise ValueError("output batch schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "scenario_hash": self.scenario_hash,
            "prior_state_hash": self.prior_state_hash,
            "next_state_hash": self.next_state_hash,
            "round_result_hash": self.round_result_hash,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "record_hashes": [record.content_hash for record in self.records],
            "checkpoint": self.checkpoint,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationAudienceCapability:
    audience: SimulationOutputAudience
    owner_agent_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.audience, SimulationOutputAudience):
            raise TypeError("output capability audience must be SimulationOutputAudience")
        object.__setattr__(self, "owner_agent_id", _optional_text(self.owner_agent_id, label="output capability owner agent id"))
        if self.audience is SimulationOutputAudience.AGENT:
            if self.owner_agent_id is None:
                raise ValueError("agent output capability requires an owner")
        elif self.owner_agent_id is not None:
            raise ValueError("non-agent output capability must not name an owner")

    def allows(self, record: SimulationOutputRecord) -> bool:
        if not isinstance(record, SimulationOutputRecord):
            raise TypeError("output capability requires a SimulationOutputRecord")
        if self.audience is SimulationOutputAudience.PUBLIC:
            return record.audience is SimulationOutputAudience.PUBLIC
        if self.audience is SimulationOutputAudience.OBJECTIVE:
            return record.audience in (
                SimulationOutputAudience.PUBLIC,
                SimulationOutputAudience.OBJECTIVE,
            )
        if self.audience is SimulationOutputAudience.AGENT:
            return record.audience is SimulationOutputAudience.PUBLIC or (
                record.audience is SimulationOutputAudience.AGENT
                and record.owner_agent_id == self.owner_agent_id
            )
        if self.audience is SimulationOutputAudience.ANALYST:
            return record.audience in (
                SimulationOutputAudience.PUBLIC,
                SimulationOutputAudience.OBJECTIVE,
                SimulationOutputAudience.ANALYST,
            )
        return True


@dataclass(frozen=True)
class SimulationOutputView:
    stream_id: str
    scenario_hash: str
    prior_state_hash: str
    next_state_hash: str
    round_result_hash: str
    first_sequence: int
    last_sequence: int
    records: tuple[SimulationOutputRecord, ...]
    source_batch_hash: str
    checkpoint: bool = False
    schema: str = SIMULATION_OUTPUT_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _text(self.stream_id, label="output view stream id"))
        for name in (
            "scenario_hash", "prior_state_hash", "next_state_hash",
            "round_result_hash", "source_batch_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), label=f"output view {name.replace('_', ' ')}"))
        object.__setattr__(self, "first_sequence", _positive_integer(self.first_sequence, label="output view first sequence"))
        object.__setattr__(self, "last_sequence", _positive_integer(self.last_sequence, label="output view last sequence"))
        if self.last_sequence < self.first_sequence:
            raise ValueError("output view sequence bounds must be ordered")
        if not isinstance(self.records, tuple) or any(
            not isinstance(record, SimulationOutputRecord) for record in self.records
        ):
            raise TypeError("output view records must be a tuple of output records")
        sequences = tuple(record.sequence for record in self.records)
        if sequences != tuple(sorted(sequences)) or len(set(sequences)) != len(sequences):
            raise ValueError("output view record sequences must retain unique source order")
        if any(sequence < self.first_sequence or sequence > self.last_sequence for sequence in sequences):
            raise ValueError("output view record sequences must remain within source batch bounds")
        if any(record.stream_id != self.stream_id for record in self.records):
            raise ValueError("output view records must bind the exact stream")
        if any(record.scenario_hash != self.scenario_hash for record in self.records):
            raise ValueError("output view records must bind the exact scenario")
        if any(record.state_hash != self.next_state_hash for record in self.records):
            raise ValueError("output view records must bind the exact next state")
        object.__setattr__(self, "checkpoint", _bool(self.checkpoint, label="output view checkpoint"))
        if self.schema != SIMULATION_OUTPUT_VIEW_SCHEMA:
            raise ValueError("output view schema must match the supported schema")

    @classmethod
    def from_batch(
        cls,
        batch: SimulationOutputBatch,
        capability: SimulationAudienceCapability,
    ) -> "SimulationOutputView":
        if not isinstance(batch, SimulationOutputBatch):
            raise TypeError("output view source must be a SimulationOutputBatch")
        if not isinstance(capability, SimulationAudienceCapability):
            raise TypeError("output view filtering requires a SimulationAudienceCapability")
        return cls(
            batch.stream_id,
            batch.scenario_hash,
            batch.prior_state_hash,
            batch.next_state_hash,
            batch.round_result_hash,
            batch.first_sequence,
            batch.last_sequence,
            tuple(record for record in batch.records if capability.allows(record)),
            batch.content_hash,
            batch.checkpoint,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "scenario_hash": self.scenario_hash,
            "prior_state_hash": self.prior_state_hash,
            "next_state_hash": self.next_state_hash,
            "round_result_hash": self.round_result_hash,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "record_hashes": [record.content_hash for record in self.records],
            "source_batch_hash": self.source_batch_hash,
            "checkpoint": self.checkpoint,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = (
    "SIMULATION_OUTPUT_RECORD_SCHEMA",
    "SIMULATION_OUTPUT_BATCH_SCHEMA",
    "SIMULATION_OUTPUT_VIEW_SCHEMA",
    "SimulationOutputAudience",
    "SimulationOutputKind",
    "SimulationStateDeltaPayload",
    "SimulationObjectiveEventPayload",
    "SimulationPrivatePerceptPayload",
    "SimulationAgentDecisionPayload",
    "SimulationMemoryUpdatePayload",
    "SimulationSocialUpdatePayload",
    "SimulationNetworkMetricsPayload",
    "SimulationStoryProgressPayload",
    "SimulationNarrativeScenePayload",
    "SimulationBlenderDeltaPayload",
    "SimulationCommandResultPayload",
    "SimulationDiagnosticPayload",
    "SimulationOutputPayload",
    "SimulationOutputRecord",
    "SimulationOutputBatch",
    "SimulationAudienceCapability",
    "SimulationOutputView",
    "output_record_sort_key",
)
