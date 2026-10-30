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
