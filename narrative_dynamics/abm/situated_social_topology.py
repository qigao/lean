"""Explicit V19 social-memory topology interventions at a checkpoint boundary.

The original SituatedSocialMemoryState, not a synthetic MeshSnapshot, owns
relationship trust. This pure administrative intervention creates a NEW,
validated social-memory checkpoint using the existing runtime projection. It
does not authenticate an actor, modify SQLite memory, or replace ordinary
social-evidence learning during subsequent rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_topology_events import (
    TopologyChangeError, TopologyVersionConflict, TopologyUnknownAgent,
    TopologyDuplicateEvent,
)
from narrative_dynamics.abm.situated_network import (
    _validate_runtime_state,
    initialize_situated_network_runtime,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRuntimeModel, SituatedNetworkRuntimeState,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    validate_situated_social_memory_state,
)

_SCHEMA = "v19-social-trust-checkpoint-intervention-v1"
_RECEIPT_SCHEMA = "v19-social-trust-checkpoint-receipt-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


class SocialTopologyBoundaryError(TopologyChangeError):
    """An intervention cannot splice an ordinary running social-memory round."""


class SocialTopologyTrustConflict(TopologyChangeError):
    """The directed trust intervention does not actually cross the active threshold."""


class SocialTopologyIntegrityError(TopologyChangeError):
    """A recorded checkpoint cannot be reproduced from its authoritative start."""


class SocialTrustOperation(str, Enum):
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"


def _text(value: object, field: str, maximum: int = 256) -> str:
    if (
        type(value) is not str or not value or not value.strip()
        or value.strip() != value or len(value) > maximum
    ):
        raise TopologyChangeError(f"{field} must be a bounded canonical string")
    return value


def _hash(value: object, field: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise TopologyChangeError(f"{field} must be canonical SHA-256")
    return value


@dataclass(frozen=True)
class SocialTrustCheckpointIntervention:
    """One directed social-trust intervention with exact source/state CAS.

    The actor field is provenance metadata only, NOT an authorization token.
    The supplied target trust must cross the actual V19 relationship threshold
    in the direction declared by operation; no synthetic active flag exists.
    """

    event_id: str
    expected_runtime_hash: str
    expected_source_hash: str
    expected_social_state_hash: str
    source_agent_id: str
    observer_agent_id: str
    actor_id: str
    reason: str
    operation: SocialTrustOperation
    target_trust: float

    def __post_init__(self) -> None:
        for key in ("event_id", "source_agent_id", "observer_agent_id", "actor_id"):
            _text(getattr(self, key), key, 128)
        _text(self.reason, "reason", 256)
        for key in ("expected_runtime_hash", "expected_source_hash", "expected_social_state_hash"):
            _hash(getattr(self, key), key)
        if self.source_agent_id == self.observer_agent_id:
            raise TopologyChangeError("social trust intervention requires distinct directed agents")
        if not isinstance(self.operation, SocialTrustOperation):
            raise TopologyChangeError("social trust intervention operation must be typed")
        trust = self.target_trust
        if type(trust) not in (int, float) or not math.isfinite(trust) or not 0 <= trust <= 1:
            raise TopologyChangeError("target_trust must be a finite real in [0,1], not bool")
        object.__setattr__(self, "target_trust", 0.0 if float(trust) == 0 else float(trust))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "event_id": self.event_id,
            "expected_runtime_hash": self.expected_runtime_hash,
            "expected_source_hash": self.expected_source_hash,
            "expected_social_state_hash": self.expected_social_state_hash,
            "source_agent_id": self.source_agent_id,
            "observer_agent_id": self.observer_agent_id,
            "actor_id": self.actor_id,
            "reason": self.reason,
            "operation": self.operation.value,
            "target_trust": self.target_trust,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SocialTrustCheckpointReceipt:
    intervention_hash: str
    before_runtime_hash: str
    after_runtime_hash: str
    before_source_hash: str
    after_source_hash: str
    before_social_hash: str
    after_social_hash: str
    threshold: float
    previous_active: bool
    next_active: bool

    def __post_init__(self) -> None:
        for key in (
            "intervention_hash", "before_runtime_hash", "after_runtime_hash",
            "before_source_hash", "after_source_hash",
            "before_social_hash", "after_social_hash",
        ):
            try:
                _hash(getattr(self, key), key)
            except TopologyChangeError as exc:
                raise SocialTopologyIntegrityError(str(exc)) from exc
        if (
            type(self.previous_active) is not bool
            or type(self.next_active) is not bool
            or self.previous_active == self.next_active
        ):
            raise SocialTopologyIntegrityError(
                "receipt needs opposite typed before/after effective states"
            )
        if (
            type(self.threshold) not in (int, float)
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise SocialTopologyIntegrityError("receipt threshold must be finite in [0,1]")
        if (
            self.before_runtime_hash == self.after_runtime_hash
            or self.before_source_hash == self.after_source_hash
            or self.before_social_hash == self.after_social_hash
        ):
            raise SocialTopologyIntegrityError("receipt must bind changed states")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "intervention_hash": self.intervention_hash,
            "before_runtime_hash": self.before_runtime_hash,
            "after_runtime_hash": self.after_runtime_hash,
            "before_source_hash": self.before_source_hash,
            "after_source_hash": self.after_source_hash,
            "before_social_hash": self.before_social_hash,
            "after_social_hash": self.after_social_hash,
            "threshold": self.threshold,
            "previous_active": self.previous_active,
            "next_active": self.next_active,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SocialTrustCheckpointResult:
    previous_runtime_state: SituatedNetworkRuntimeState
    next_runtime_state: SituatedNetworkRuntimeState
    intervention: SocialTrustCheckpointIntervention
    receipt: SocialTrustCheckpointReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.previous_runtime_state, SituatedNetworkRuntimeState)
            or not isinstance(self.next_runtime_state, SituatedNetworkRuntimeState)
            or type(self.intervention) is not SocialTrustCheckpointIntervention
            or type(self.receipt) is not SocialTrustCheckpointReceipt
        ):
            raise SocialTopologyIntegrityError("checkpoint result requires exact typed records")
        old, new, event, receipt = (
            self.previous_runtime_state, self.next_runtime_state,
            self.intervention, self.receipt,
        )
        if (
            event.expected_runtime_hash != old.content_hash
            or event.expected_source_hash != old.snapshot.content_hash
            or event.expected_social_state_hash != old.social_state.content_hash
            or receipt.intervention_hash != event.content_hash
            or receipt.before_runtime_hash != old.content_hash
            or receipt.after_runtime_hash != new.content_hash
            or receipt.before_source_hash != old.snapshot.content_hash
            or receipt.after_source_hash != new.snapshot.content_hash
            or receipt.before_social_hash != old.social_state.content_hash
            or receipt.after_social_hash != new.social_state.content_hash
            or new.story != old.story
            or new.cognitive_state != old.cognitive_state
            or new.round_index != old.round_index
            or new.memory_store_hash != old.memory_store_hash
            or not new.checkpoint
        ):
            raise SocialTopologyIntegrityError("recorded checkpoint state/receipt chain mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_runtime_hash": self.previous_runtime_state.content_hash,
            "next_runtime_hash": self.next_runtime_state.content_hash,
            "intervention": self.intervention.to_dict(),
            "receipt": self.receipt.to_dict() | {"receipt_hash": self.receipt.content_hash},
        }


def admit_social_trust_checkpoint(
    database_path: str | Path,
    model: SituatedNetworkRuntimeModel,
    runtime: SituatedNetworkRuntimeState,
    event: SocialTrustCheckpointIntervention,
) -> SocialTrustCheckpointResult:
    """Rebind a genuine V19 checkpoint to an explicitly adjusted social trust.

    The canonical social memory is updated before V19 source projection.
    Prior source/story/cognition and SQLite memory are never mutated. The
    result can be admitted by existing simulate_situated_network_round().
    """
    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("checkpoint admission requires the V19 runtime model")
    if not isinstance(runtime, SituatedNetworkRuntimeState):
        raise TypeError("checkpoint admission requires a V19 runtime state")
    if not isinstance(event, SocialTrustCheckpointIntervention):
        raise TypeError("checkpoint admission requires a typed intervention")
    if not runtime.checkpoint:
        raise SocialTopologyBoundaryError(
            "relationship trust intervention requires an explicit V19 checkpoint"
        )
    # Reuse the existing V19 identity, model, memory-store and exact
    # reprojection gate before examining the social-memory intervention.
    _validate_runtime_state(database_path, model, runtime)
    if (
        event.expected_runtime_hash != runtime.content_hash
        or event.expected_source_hash != runtime.snapshot.content_hash
        or event.expected_social_state_hash != runtime.social_state.content_hash
    ):
        raise TopologyVersionConflict(
            "stale V19 runtime/source/social checkpoint identity"
        )
    source_ids = {node.agent_id for node in runtime.snapshot.nodes}
    if event.source_agent_id not in source_ids or event.observer_agent_id not in source_ids:
        raise TopologyUnknownAgent("social-trust endpoint must be a real V19 agent")
    old_relation = next((
        rel for rel in runtime.social_state.relationships
        if rel.source_agent_id == event.source_agent_id
        and rel.observer_agent_id == event.observer_agent_id
    ), None)
    if old_relation is None:
        raise TopologyChangeError("directed social relation missing from authoritative state")
    threshold = model.relationship_trust_threshold
    old_active = old_relation.trust >= threshold
    new_active = event.target_trust >= threshold
    if (
        old_active == new_active
        or (event.operation is SocialTrustOperation.ACTIVATE and not new_active)
        or (event.operation is SocialTrustOperation.DEACTIVATE and new_active)
    ):
        raise SocialTopologyTrustConflict(
            "intervention must flip actual threshold-derived directed relation in requested direction"
        )
    changed = replace(old_relation, trust=event.target_trust)
    new_social_state = replace(
        runtime.social_state,
        relationships=tuple(
            changed if item is old_relation else item
            for item in runtime.social_state.relationships
        ),
    )
    validate_situated_social_memory_state(
        model.social_memory_model, runtime.cognitive_state, new_social_state,
    )
    # This is an explicit rebased checkpoint (same logical story/cognitive
    # round); not a hidden mid-round social-memory child or a new graph owner.
    new_runtime = initialize_situated_network_runtime(
        database_path, model, runtime.story, runtime.cognitive_state,
        new_social_state,
    )
    if (
        new_runtime.snapshot.social_state_hash != new_social_state.content_hash
        or new_runtime.snapshot.content_hash == runtime.snapshot.content_hash
        or new_runtime.story != runtime.story
        or new_runtime.cognitive_state != runtime.cognitive_state
        or new_runtime.memory_store_hash != runtime.memory_store_hash
    ):
        raise AssertionError("V19 checkpoint rebinding violated exact source authority")
    projected = next((
        edge for edge in new_runtime.snapshot.relationship_edges
        if edge.source_agent_id == event.source_agent_id
        and edge.observer_agent_id == event.observer_agent_id
    ), None)
    if projected is None or projected.active is not new_active:
        raise AssertionError("V19 projection did not use updated social-memory trust")
    receipt = SocialTrustCheckpointReceipt(
        event.content_hash,
        runtime.content_hash, new_runtime.content_hash,
        runtime.snapshot.content_hash, new_runtime.snapshot.content_hash,
        runtime.social_state.content_hash, new_social_state.content_hash,
        threshold, old_active, new_active,
    )
    return SocialTrustCheckpointResult(runtime, new_runtime, event, receipt)

def verify_social_trust_checkpoint(
    database_path: str | Path,
    model: SituatedNetworkRuntimeModel,
    original: SituatedNetworkRuntimeState,
    recorded: SocialTrustCheckpointResult,
) -> bool:
    """Independently reapply one trust intervention through the same V19 owner.

    Internal hash-consistent records may still claim a semantically false
    target trust value. Re-run the real social-state admission from the
    caller's original checkpoint and compare the full resulting state and
    receipt. This verifies reproducibility, not actor authentication.
    """
    if not isinstance(original, SituatedNetworkRuntimeState):
        raise TypeError("verification requires a V19 original checkpoint")
    if not isinstance(recorded, SocialTrustCheckpointResult):
        raise TypeError("verification requires a social trust checkpoint record")
    if recorded.previous_runtime_state != original:
        raise TopologyVersionConflict("recorded source checkpoint differs from provided origin")
    if (
        recorded.previous_runtime_state.content_hash != original.content_hash
        or recorded.intervention.expected_runtime_hash != original.content_hash
    ):
        raise TopologyVersionConflict("recorded expected checkpoint hash is stale")
    try:
        actual = admit_social_trust_checkpoint(
            database_path, model, original, recorded.intervention,
        )
    except TopologyChangeError as exc:
        raise SocialTopologyIntegrityError(
            "recorded trust event cannot execute from original V19 checkpoint"
        ) from exc
    if (
        recorded.to_dict() != actual.to_dict()
        or recorded.next_runtime_state.to_dict() != actual.next_runtime_state.to_dict()
    ):
        raise SocialTopologyIntegrityError(
            "recorded trust effect or receipt differs from actual source-authoritative admission"
        )
    return True



def verify_social_trust_replay(
    database_path: str | Path,
    model: SituatedNetworkRuntimeModel,
    original: SituatedNetworkRuntimeState,
    recorded: tuple[SocialTrustCheckpointResult, ...],
) -> bool:
    """Independently verify all claimed V19 social trust checkpoint results.

    The complete bounded series must start at the caller's authoritative
    checkpoint and form a unique, ordered chain. Re-execute every event with
    the single existing admission engine, then compare each original/new
    source/social/runtime snapshot and receipt. This is reproducibility
    checking, not external identity authentication or historical attestation.
    """
    if not isinstance(original, SituatedNetworkRuntimeState):
        raise TypeError("social trust replay verification requires a V19 starting checkpoint")
    if type(recorded) is not tuple:
        raise TypeError("recorded social trust replay must be an immutable tuple")
    if len(recorded) > 128:
        raise TopologyChangeError("R1.1 verified social trust replay budget exceeded")
    _validate_runtime_state(database_path, model, original)
    previous = original
    seen: set[str] = set()
    for index, item in enumerate(recorded):
        if type(item) is not SocialTrustCheckpointResult:
            raise TypeError("recorded social trust replay requires exact typed results")
        if item.intervention.event_id in seen:
            raise TopologyDuplicateEvent("duplicate event in recorded social trust checkpoint replay")
        seen.add(item.intervention.event_id)
        if index == 0 and item.previous_runtime_state != original:
            raise TopologyVersionConflict("recorded checkpoint replay starts from a different V19 source")
        if item.previous_runtime_state != previous:
            raise SocialTopologyIntegrityError("recorded social checkpoints are not consecutive")
        previous = item.next_runtime_state
    try:
        actual = replay_social_trust_checkpoints(
            database_path, model, original,
            tuple(item.intervention for item in recorded),
        )
    except TopologyChangeError as exc:
        raise SocialTopologyIntegrityError(
            "recorded social trust changes cannot replay from the authoritative V19 source"
        ) from exc
    if len(actual) != len(recorded):
        raise SocialTopologyIntegrityError("recorded social replay result length differs")
    for claimed, reproduced in zip(recorded, actual):
        if (
            claimed.to_dict() != reproduced.to_dict()
            or claimed.previous_runtime_state.to_dict() != reproduced.previous_runtime_state.to_dict()
            or claimed.next_runtime_state.to_dict() != reproduced.next_runtime_state.to_dict()
        ):
            raise SocialTopologyIntegrityError(
                "recorded checkpoint effects or receipts differ from authoritative reexecution"
            )
    return True

def replay_social_trust_checkpoints(
    database_path: str | Path,
    model: SituatedNetworkRuntimeModel,
    original: SituatedNetworkRuntimeState,
    events: tuple[SocialTrustCheckpointIntervention, ...],
) -> tuple[SocialTrustCheckpointResult, ...]:
    """Deterministically replay a bounded, unique-event source-trust series.

    Every event binds the *current* runtime/source/social hashes; a reversed,
    duplicate or stale event is rejected before any partial result escapes.
    The ledger consists only of immutable V19 checkpoint results, not another
    graph state owner or an independently implemented update engine.
    """
    if type(events) is not tuple:
        raise TypeError("social trust replay requires an immutable tuple of interventions")
    if len(events) > 128:
        raise TopologyChangeError("R1.1 social trust replay budget exceeded")
    if not isinstance(original, SituatedNetworkRuntimeState):
        raise TypeError("social trust replay requires a V19 original checkpoint")
    seen: set[str] = set()
    current = original
    results: list[SocialTrustCheckpointResult] = []
    for event in events:
        if type(event) is not SocialTrustCheckpointIntervention:
            raise TypeError("social trust replay requires exact typed interventions")
        if event.event_id in seen:
            raise TopologyDuplicateEvent("duplicate social topology intervention id")
        seen.add(event.event_id)
        item = admit_social_trust_checkpoint(
            database_path, model, current, event,
        )
        results.append(item)
        current = item.next_runtime_state
    return tuple(results)
