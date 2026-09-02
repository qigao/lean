"""Immutable contracts for the single-process scenario coordinator boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentMindState,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkEmergenceMetrics,
    SituatedNetworkRuntimeState,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedConsolidatedClaim,
    SituatedSourceRelationship,
)
from narrative_dynamics.contracts import stable_content_hash


SCENARIO_COMMAND_REQUEST_SCHEMA = "narrative-dynamics.scenario-command-request/v1"
SCENARIO_COMMAND_RESULT_SCHEMA = "narrative-dynamics.scenario-command-result/v1"
SCENARIO_RUN_VIEW_SCHEMA = "narrative-dynamics.scenario-run-view/v1"
SCENARIO_PUBLIC_STATE_VIEW_SCHEMA = "narrative-dynamics.scenario-public-state-view/v1"
SCENARIO_AGENT_STATE_VIEW_SCHEMA = "narrative-dynamics.scenario-agent-state-view/v1"
SCENARIO_CHECKPOINT_SCHEMA = "narrative-dynamics.scenario-checkpoint/v1"
SCENARIO_FORK_REQUEST_SCHEMA = "narrative-dynamics.scenario-fork-request/v1"
SCENARIO_FORK_RESULT_SCHEMA = "narrative-dynamics.scenario-fork-result/v1"

_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class ScenarioRunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


class ScenarioCommandKind(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"
    STOP = "stop"
    CHECKPOINT = "checkpoint"


class ScenarioCommandReason(str, Enum):
    ACCEPTED = "accepted"
    UNAUTHORIZED = "unauthorized"
    RUN_MISMATCH = "run_mismatch"
    SCENARIO_MISMATCH = "scenario_mismatch"
    EPOCH_MISMATCH = "epoch_mismatch"
    STALE_STATE = "stale_state"
    INVALID_STATUS = "invalid_status"
    MAXIMUM_ROUNDS = "maximum_rounds"


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
        raise ValueError(f"{label} must be an exact sha256 content hash")
    return value


def _optional_hash(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _hash(value, label=label)


def _nonnegative_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _enum(value: object, enum_type: type[Enum], *, label: str) -> Enum:
    if not isinstance(value, enum_type):
        raise TypeError(f"{label} must be {enum_type.__name__}")
    return value


def _hash_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple of content hashes")
    result = tuple(_hash(item, label=f"{label} value") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} values must be unique")
    return result


def _agent_places(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in value
    ):
        raise TypeError("public state agent places must be a tuple of placement tuples")
    result = tuple(
        (
            _text(item[0], label="public state agent id"),
            _text(item[1], label="public state agent place id"),
        )
        for item in value
    )
    if len({item[0] for item in result}) != len(result):
        raise ValueError("public state agent identities must be unique")
    return tuple(sorted(result))


def _passage_states(value: object) -> tuple[tuple[str, bool], ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in value
    ):
        raise TypeError("public state passage states must be a tuple of state tuples")
    result = tuple(
        (
            _text(item[0], label="public state passage id"),
            _boolean(item[1], label="public state passage open"),
        )
        for item in value
    )
    if len({item[0] for item in result}) != len(result):
        raise ValueError("public state passage identities must be unique")
    return tuple(sorted(result))


def _object_placements(
    value: object,
) -> tuple[tuple[str, str | None, str | None], ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple) or len(item) != 3 for item in value
    ):
        raise TypeError("public state object placements must be a tuple of placement tuples")
    result = tuple(
        (
            _text(item[0], label="public state object id"),
            _optional_text(item[1], label="public state object place id"),
            _optional_text(item[2], label="public state object holder agent id"),
        )
        for item in value
    )
    if len({item[0] for item in result}) != len(result):
        raise ValueError("public state object identities must be unique")
    return tuple(sorted(result, key=lambda item: item[0]))


class _ContentAddressed:
    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ScenarioCommandCapability(_ContentAddressed):
    authority_id: str
    run_id: str
    allowed_kinds: tuple[ScenarioCommandKind, ...]
    can_fork: bool = False
    can_read_all_audit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "authority_id", _text(self.authority_id, label="command capability authority id"))
        object.__setattr__(self, "run_id", _text(self.run_id, label="command capability run id"))
        if not isinstance(self.allowed_kinds, tuple) or any(
            not isinstance(item, ScenarioCommandKind) for item in self.allowed_kinds
        ):
            raise TypeError("command capability allowed kinds must be a tuple of ScenarioCommandKind values")
        if len(set(self.allowed_kinds)) != len(self.allowed_kinds):
            raise ValueError("command capability allowed kinds must be unique")
        if self.allowed_kinds != tuple(sorted(self.allowed_kinds, key=lambda item: item.value)):
            raise ValueError("command capability allowed kinds must be sorted")
        object.__setattr__(self, "can_fork", _boolean(self.can_fork, label="command capability can fork"))
        object.__setattr__(self, "can_read_all_audit", _boolean(self.can_read_all_audit, label="command capability can read all audit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_id": self.authority_id,
            "run_id": self.run_id,
            "allowed_kinds": [item.value for item in self.allowed_kinds],
            "can_fork": self.can_fork,
            "can_read_all_audit": self.can_read_all_audit,
        }


@dataclass(frozen=True)
class ScenarioCommandRequest(_ContentAddressed):
    command_id: str
    idempotency_key: str
    run_id: str
    scenario_hash: str
    coordinator_epoch: int
    expected_state_hash: str
    authority_id: str
    kind: ScenarioCommandKind
    requested_checkpoint_id: str | None = None
    schema: str = SCENARIO_COMMAND_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        for name in ("command_id", "idempotency_key", "run_id", "authority_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"command request {name.replace('_', ' ')}"))
        object.__setattr__(self, "scenario_hash", _hash(self.scenario_hash, label="command request scenario hash"))
        object.__setattr__(self, "coordinator_epoch", _positive_integer(self.coordinator_epoch, label="command request coordinator epoch"))
        object.__setattr__(self, "expected_state_hash", _hash(self.expected_state_hash, label="command request expected state hash"))
        object.__setattr__(self, "kind", _enum(self.kind, ScenarioCommandKind, label="command request kind"))
        object.__setattr__(self, "requested_checkpoint_id", _optional_text(self.requested_checkpoint_id, label="command request checkpoint id"))
        if self.kind is ScenarioCommandKind.CHECKPOINT:
            if self.requested_checkpoint_id is None:
                raise ValueError("checkpoint command requires a requested checkpoint id")
        elif self.requested_checkpoint_id is not None:
            raise ValueError("non-checkpoint command must not name a checkpoint id")
        if self.schema != SCENARIO_COMMAND_REQUEST_SCHEMA:
            raise ValueError("command request schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "run_id": self.run_id,
            "scenario_hash": self.scenario_hash,
            "coordinator_epoch": self.coordinator_epoch,
            "expected_state_hash": self.expected_state_hash,
            "authority_id": self.authority_id,
            "kind": self.kind.value,
            "requested_checkpoint_id": self.requested_checkpoint_id,
        }


@dataclass(frozen=True)
class ScenarioCommandResult(_ContentAddressed):
    command_id: str
    idempotency_key: str
    request_hash: str
    capability_hash: str
    run_id: str
    scenario_hash: str
    coordinator_epoch: int
    kind: ScenarioCommandKind
    accepted: bool
    reason: ScenarioCommandReason
    prior_status: ScenarioRunStatus
    next_status: ScenarioRunStatus
    prior_state_hash: str
    next_state_hash: str
    round_index: int
    output_batch_hash: str | None = None
    checkpoint_hash: str | None = None
    schema: str = SCENARIO_COMMAND_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("command_id", "idempotency_key", "run_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"command result {name.replace('_', ' ')}"))
        for name in ("request_hash", "capability_hash", "scenario_hash", "prior_state_hash", "next_state_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), label=f"command result {name.replace('_', ' ')}"))
        object.__setattr__(self, "coordinator_epoch", _positive_integer(self.coordinator_epoch, label="command result coordinator epoch"))
        object.__setattr__(self, "kind", _enum(self.kind, ScenarioCommandKind, label="command result kind"))
        object.__setattr__(self, "accepted", _boolean(self.accepted, label="command result accepted"))
        object.__setattr__(self, "reason", _enum(self.reason, ScenarioCommandReason, label="command result reason"))
        object.__setattr__(self, "prior_status", _enum(self.prior_status, ScenarioRunStatus, label="command result prior status"))
        object.__setattr__(self, "next_status", _enum(self.next_status, ScenarioRunStatus, label="command result next status"))
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="command result round index"))
        object.__setattr__(self, "output_batch_hash", _optional_hash(self.output_batch_hash, label="command result output batch hash"))
        object.__setattr__(self, "checkpoint_hash", _optional_hash(self.checkpoint_hash, label="command result checkpoint hash"))
        if self.accepted != (self.reason is ScenarioCommandReason.ACCEPTED):
            raise ValueError("command result accepted flag must match its reason")
        if self.schema != SCENARIO_COMMAND_RESULT_SCHEMA:
            raise ValueError("command result schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "request_hash": self.request_hash,
            "capability_hash": self.capability_hash,
            "run_id": self.run_id,
            "scenario_hash": self.scenario_hash,
            "coordinator_epoch": self.coordinator_epoch,
            "kind": self.kind.value,
            "accepted": self.accepted,
            "reason": self.reason.value,
            "prior_status": self.prior_status.value,
            "next_status": self.next_status.value,
            "prior_state_hash": self.prior_state_hash,
            "next_state_hash": self.next_state_hash,
            "round_index": self.round_index,
            "output_batch_hash": self.output_batch_hash,
            "checkpoint_hash": self.checkpoint_hash,
        }


@dataclass(frozen=True)
class ScenarioRunView(_ContentAddressed):
    run_id: str
    stream_id: str
    scenario_hash: str
    coordinator_epoch: int
    status: ScenarioRunStatus
    round_index: int
    state_hash: str
    next_sequence: int
    output_batch_hashes: tuple[str, ...]
    checkpoint_hashes: tuple[str, ...]
    parent_checkpoint_hash: str | None = None
    schema: str = SCENARIO_RUN_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, label="run view run id"))
        object.__setattr__(self, "stream_id", _text(self.stream_id, label="run view stream id"))
        object.__setattr__(self, "scenario_hash", _hash(self.scenario_hash, label="run view scenario hash"))
        object.__setattr__(self, "coordinator_epoch", _positive_integer(self.coordinator_epoch, label="run view coordinator epoch"))
        object.__setattr__(self, "status", _enum(self.status, ScenarioRunStatus, label="run view status"))
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="run view round index"))
        object.__setattr__(self, "state_hash", _hash(self.state_hash, label="run view state hash"))
        object.__setattr__(self, "next_sequence", _positive_integer(self.next_sequence, label="run view next sequence"))
        object.__setattr__(self, "output_batch_hashes", _hash_tuple(self.output_batch_hashes, label="run view output batch hashes"))
        object.__setattr__(self, "checkpoint_hashes", _hash_tuple(self.checkpoint_hashes, label="run view checkpoint hashes"))
        object.__setattr__(self, "parent_checkpoint_hash", _optional_hash(self.parent_checkpoint_hash, label="run view parent checkpoint hash"))
        if self.schema != SCENARIO_RUN_VIEW_SCHEMA:
            raise ValueError("run view schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "stream_id": self.stream_id,
            "scenario_hash": self.scenario_hash,
            "coordinator_epoch": self.coordinator_epoch,
            "status": self.status.value,
            "round_index": self.round_index,
            "state_hash": self.state_hash,
            "next_sequence": self.next_sequence,
            "output_batch_hashes": list(self.output_batch_hashes),
            "checkpoint_hashes": list(self.checkpoint_hashes),
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
        }


@dataclass(frozen=True)
class ScenarioPublicStateView(_ContentAddressed):
    run_id: str
    scenario_hash: str
    round_index: int
    state_hash: str
    agent_places: tuple[tuple[str, str], ...]
    passage_states: tuple[tuple[str, bool], ...]
    object_placements: tuple[tuple[str, str | None, str | None], ...]
    metrics: SituatedNetworkEmergenceMetrics
    schema: str = SCENARIO_PUBLIC_STATE_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, label="public state run id"))
        object.__setattr__(self, "scenario_hash", _hash(self.scenario_hash, label="public state scenario hash"))
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="public state round index"))
        object.__setattr__(self, "state_hash", _hash(self.state_hash, label="public state state hash"))
        object.__setattr__(self, "agent_places", _agent_places(self.agent_places))
        object.__setattr__(self, "passage_states", _passage_states(self.passage_states))
        object.__setattr__(self, "object_placements", _object_placements(self.object_placements))
        if not isinstance(self.metrics, SituatedNetworkEmergenceMetrics):
            raise TypeError("public state metrics must be SituatedNetworkEmergenceMetrics")
        if self.metrics.round_index != self.round_index:
            raise ValueError("public state metrics must bind the exact round")
        if self.schema != SCENARIO_PUBLIC_STATE_VIEW_SCHEMA:
            raise ValueError("public state schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "scenario_hash": self.scenario_hash,
            "round_index": self.round_index,
            "state_hash": self.state_hash,
            "agent_places": [list(item) for item in self.agent_places],
            "passage_states": [list(item) for item in self.passage_states],
            "object_placements": [list(item) for item in self.object_placements],
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class ScenarioAgentStateView(_ContentAddressed):
    run_id: str
    scenario_hash: str
    round_index: int
    state_hash: str
    agent_id: str
    place_id: str
    mind: SituatedAgentMindState
    relationships: tuple[SituatedSourceRelationship, ...]
    claims: tuple[SituatedConsolidatedClaim, ...]
    schema: str = SCENARIO_AGENT_STATE_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, label="agent state run id"))
        object.__setattr__(self, "scenario_hash", _hash(self.scenario_hash, label="agent state scenario hash"))
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="agent state round index"))
        object.__setattr__(self, "state_hash", _hash(self.state_hash, label="agent state state hash"))
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="agent state agent id"))
        object.__setattr__(self, "place_id", _text(self.place_id, label="agent state place id"))
        if not isinstance(self.mind, SituatedAgentMindState):
            raise TypeError("agent state mind must be SituatedAgentMindState")
        if self.mind.agent_id != self.agent_id:
            raise ValueError("agent state mind must belong to the exact agent")
        if self.mind.own_place_id != self.place_id:
            raise ValueError("agent state mind must bind the exact place")
        if not isinstance(self.relationships, tuple) or any(
            not isinstance(item, SituatedSourceRelationship) for item in self.relationships
        ):
            raise TypeError("agent state relationships must be a tuple of SituatedSourceRelationship values")
        if any(item.observer_agent_id != self.agent_id for item in self.relationships):
            raise ValueError("agent state relationships must belong to the exact observer")
        if len({item.source_agent_id for item in self.relationships}) != len(self.relationships):
            raise ValueError("agent state relationship sources must be unique")
        object.__setattr__(self, "relationships", tuple(sorted(self.relationships, key=lambda item: item.source_agent_id)))
        if not isinstance(self.claims, tuple) or any(
            not isinstance(item, SituatedConsolidatedClaim) for item in self.claims
        ):
            raise TypeError("agent state claims must be a tuple of SituatedConsolidatedClaim values")
        if any(item.observer_agent_id != self.agent_id for item in self.claims):
            raise ValueError("agent state claims must belong to the exact observer")
        if len({item.claim_id for item in self.claims}) != len(self.claims):
            raise ValueError("agent state claim identities must be unique")
        object.__setattr__(self, "claims", tuple(sorted(self.claims, key=lambda item: item.claim_id)))
        if self.schema != SCENARIO_AGENT_STATE_VIEW_SCHEMA:
            raise ValueError("agent state schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "scenario_hash": self.scenario_hash,
            "round_index": self.round_index,
            "state_hash": self.state_hash,
            "agent_id": self.agent_id,
            "place_id": self.place_id,
            "mind": self.mind.to_dict(),
            "relationships": [item.to_dict() for item in self.relationships],
            "claims": [item.to_dict() for item in self.claims],
        }


@dataclass(frozen=True)
class ScenarioCheckpoint(_ContentAddressed):
    checkpoint_id: str
    run_id: str
    scenario_hash: str
    coordinator_epoch: int
    state: SituatedNetworkRuntimeState
    next_sequence: int
    parent_checkpoint_hash: str | None = None
    schema: str = SCENARIO_CHECKPOINT_SCHEMA
    state_hash: str = field(init=False)
    memory_store_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", _text(self.checkpoint_id, label="checkpoint id"))
        object.__setattr__(self, "run_id", _text(self.run_id, label="checkpoint run id"))
        object.__setattr__(self, "scenario_hash", _hash(self.scenario_hash, label="checkpoint scenario hash"))
        object.__setattr__(self, "coordinator_epoch", _positive_integer(self.coordinator_epoch, label="checkpoint coordinator epoch"))
        if not isinstance(self.state, SituatedNetworkRuntimeState):
            raise TypeError("checkpoint state must be SituatedNetworkRuntimeState")
        object.__setattr__(self, "state_hash", _hash(self.state.content_hash, label="checkpoint state hash"))
        object.__setattr__(self, "memory_store_hash", _hash(self.state.memory_store_hash, label="checkpoint memory store hash"))
        object.__setattr__(self, "next_sequence", _positive_integer(self.next_sequence, label="checkpoint next sequence"))
        object.__setattr__(self, "parent_checkpoint_hash", _optional_hash(self.parent_checkpoint_hash, label="checkpoint parent hash"))
        if self.schema != SCENARIO_CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema must match the supported schema")

    def _validate_embedded_state(self) -> None:
        if self.state.content_hash != self.state_hash:
            raise ValueError("checkpoint embedded state does not match its declared state hash")
        if self.state.memory_store_hash != self.memory_store_hash:
            raise ValueError("checkpoint embedded state does not match its declared memory store hash")

    def to_dict(self) -> dict[str, object]:
        self._validate_embedded_state()
        return {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "scenario_hash": self.scenario_hash,
            "coordinator_epoch": self.coordinator_epoch,
            "state_hash": self.state_hash,
            "memory_store_hash": self.memory_store_hash,
            "next_sequence": self.next_sequence,
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
        }


@dataclass(frozen=True)
class ScenarioForkRequest(_ContentAddressed):
    fork_id: str
    idempotency_key: str
    source_run_id: str
    scenario_hash: str
    source_epoch: int
    checkpoint_hash: str
    child_run_id: str
    child_stream_id: str
    schema: str = SCENARIO_FORK_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        for name in ("fork_id", "idempotency_key", "source_run_id", "child_run_id", "child_stream_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"fork request {name.replace('_', ' ')}"))
        object.__setattr__(self, "scenario_hash", _hash(self.scenario_hash, label="fork request scenario hash"))
        object.__setattr__(self, "source_epoch", _positive_integer(self.source_epoch, label="fork request source epoch"))
        object.__setattr__(self, "checkpoint_hash", _hash(self.checkpoint_hash, label="fork request checkpoint hash"))
        if self.child_run_id == self.source_run_id:
            raise ValueError("fork request child run must differ from source run")
        if self.schema != SCENARIO_FORK_REQUEST_SCHEMA:
            raise ValueError("fork request schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fork_id": self.fork_id,
            "idempotency_key": self.idempotency_key,
            "source_run_id": self.source_run_id,
            "scenario_hash": self.scenario_hash,
            "source_epoch": self.source_epoch,
            "checkpoint_hash": self.checkpoint_hash,
            "child_run_id": self.child_run_id,
            "child_stream_id": self.child_stream_id,
        }


@dataclass(frozen=True)
class ScenarioForkResult(_ContentAddressed):
    fork_id: str
    idempotency_key: str
    request_hash: str
    capability_hash: str
    source_run_id: str
    child_run_id: str
    child_stream_id: str
    scenario_hash: str
    child_epoch: int
    checkpoint_hash: str
    child_state_hash: str
    schema: str = SCENARIO_FORK_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("fork_id", "idempotency_key", "source_run_id", "child_run_id", "child_stream_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"fork result {name.replace('_', ' ')}"))
        for name in ("request_hash", "capability_hash", "scenario_hash", "checkpoint_hash", "child_state_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), label=f"fork result {name.replace('_', ' ')}"))
        object.__setattr__(self, "child_epoch", _positive_integer(self.child_epoch, label="fork result child epoch"))
        if self.child_run_id == self.source_run_id:
            raise ValueError("fork result child run must differ from source run")
        if self.schema != SCENARIO_FORK_RESULT_SCHEMA:
            raise ValueError("fork result schema must match the supported schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fork_id": self.fork_id,
            "idempotency_key": self.idempotency_key,
            "request_hash": self.request_hash,
            "capability_hash": self.capability_hash,
            "source_run_id": self.source_run_id,
            "child_run_id": self.child_run_id,
            "child_stream_id": self.child_stream_id,
            "scenario_hash": self.scenario_hash,
            "child_epoch": self.child_epoch,
            "checkpoint_hash": self.checkpoint_hash,
            "child_state_hash": self.child_state_hash,
        }


__all__ = (
    "SCENARIO_COMMAND_REQUEST_SCHEMA",
    "SCENARIO_COMMAND_RESULT_SCHEMA",
    "SCENARIO_RUN_VIEW_SCHEMA",
    "SCENARIO_PUBLIC_STATE_VIEW_SCHEMA",
    "SCENARIO_AGENT_STATE_VIEW_SCHEMA",
    "SCENARIO_CHECKPOINT_SCHEMA",
    "SCENARIO_FORK_REQUEST_SCHEMA",
    "SCENARIO_FORK_RESULT_SCHEMA",
    "ScenarioRunStatus",
    "ScenarioCommandKind",
    "ScenarioCommandReason",
    "ScenarioCommandCapability",
    "ScenarioCommandRequest",
    "ScenarioCommandResult",
    "ScenarioRunView",
    "ScenarioPublicStateView",
    "ScenarioAgentStateView",
    "ScenarioCheckpoint",
    "ScenarioForkRequest",
    "ScenarioForkResult",
)
