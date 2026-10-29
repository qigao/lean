"""Immutable V19 relationship-topology revisions and deterministic event replay.

This is a bounded editing/replay surface over authoritative SituatedNetworkSnapshot
values, NOT another graph owner or a mutation of live social/cognitive memory.
The existing source snapshot and V24 selected projections remain authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSnapshot
from narrative_dynamics.abm.mesh_queries import find_mesh_path
from narrative_dynamics.abm.situated_mesh import mesh_edges
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRelationshipEdge,
    SituatedNetworkSnapshot,
)

_REVISION_SCHEMA = "v19-relationship-topology-change-v1"
_RECEIPT_SCHEMA = "v19-relationship-topology-receipt-v1"
_REPLAY_SCHEMA = "v19-topology-replay-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_EVENTS = 128
_MAX_METRIC_NODES = 24


class TopologyChangeError(ValueError):
    """Base type for rejected topology events; no partially changed snapshot."""


class TopologyVersionConflict(TopologyChangeError):
    """The expected immutable V19 source hash does not match."""


class TopologyDuplicateEvent(TopologyChangeError):
    """The same event identity was replayed twice."""


class TopologyUnknownAgent(TopologyChangeError):
    """An endpoint is absent from the authoritative source roster."""


class TopologyEdgeConflict(TopologyChangeError):
    """The requested operation contradicts the current directed source edge."""


class TopologyUnsupportedRelation(TopologyChangeError):
    """Only relationship topology changes are supported in the R1 slice."""


class TopologyReplayIntegrityError(TopologyChangeError):
    """Untrusted or inconsistent event/receipt lineage cannot be resumed."""


class TopologyOperation(str, Enum):
    ADD = "add"
    DEACTIVATE = "deactivate"


def _label(value: object, name: str, *, max_length: int = 128) -> str:
    if (
        type(value) is not str or not value.strip()
        or len(value) > max_length or value != value.strip()
    ):
        raise TopologyChangeError(f"{name} must be a bounded nonempty canonical string")
    return value


def _source_hash(value: object) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise TopologyChangeError("expected_source_hash must be canonical sha256")
    return value


@dataclass(frozen=True)
class TopologyChange:
    """One source-hash-bound directed relationship change.

    ADD requires the complete explicit relationship edge and rejects any
    existing pair (including an inactive historical pair). DEACTIVATE keeps
    the historical record but marks its existing directed edge inactive.
    No action creates an inverse relation, interaction or transmission event.
    """

    event_id: str
    expected_source_hash: str
    relation: MeshRelation
    operation: TopologyOperation
    source_agent_id: str
    target_agent_id: str
    actor_id: str
    reason: str
    edge: SituatedNetworkRelationshipEdge | None = None

    def __post_init__(self) -> None:
        for key in ("event_id", "source_agent_id", "target_agent_id", "actor_id"):
            _label(getattr(self, key), key)
        _label(self.reason, "reason", max_length=256)
        _source_hash(self.expected_source_hash)
        if self.source_agent_id == self.target_agent_id:
            raise TopologyChangeError("topology change endpoints must differ")
        if not isinstance(self.relation, MeshRelation) or self.relation is not MeshRelation.RELATIONSHIP:
            raise TopologyUnsupportedRelation("R1 only supports relationship topology")
        if not isinstance(self.operation, TopologyOperation):
            raise TopologyChangeError("operation must be TopologyOperation")
        if self.operation is TopologyOperation.ADD:
            if not isinstance(self.edge, SituatedNetworkRelationshipEdge):
                raise TopologyChangeError("ADD requires an explicit relationship record")
            if not self.edge.active:
                raise TopologyChangeError("ADD cannot create an inactive relationship")
            if (self.edge.source_agent_id, self.edge.observer_agent_id) != (
                self.source_agent_id, self.target_agent_id
            ):
                raise TopologyChangeError("ADD record must match the declared directed endpoints")
        elif self.edge is not None:
            raise TopologyChangeError("DEACTIVATE reads the existing source edge; no replacement")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _REVISION_SCHEMA,
            "event_id": self.event_id,
            "expected_source_hash": self.expected_source_hash,
            "relation": self.relation.value,
            "operation": self.operation.value,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "actor_id": self.actor_id,
            "reason": self.reason,
            "edge": self.edge.to_dict() if self.edge is not None else None,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class TopologyRevisionReceipt:
    ordinal: int
    event_id: str
    event_hash: str
    prior_receipt_hash: str | None
    source_hash_before: str
    source_hash_after: str
    operation: TopologyOperation
    source_agent_id: str
    target_agent_id: str

    def __post_init__(self) -> None:
        """Reject malformed receipt fields before their hash becomes reusable."""
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise TopologyReplayIntegrityError("receipt ordinal must be a nonnegative integer")
        for field in ("event_id", "source_agent_id", "target_agent_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip() or value != value.strip() or len(value) > 128:
                raise TopologyReplayIntegrityError(f"receipt {field} must be canonical")
        if self.source_agent_id == self.target_agent_id:
            raise TopologyReplayIntegrityError("receipt source and target must differ")
        for field in ("event_hash", "source_hash_before", "source_hash_after"):
            value = getattr(self, field)
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise TopologyReplayIntegrityError(f"receipt {field} must be canonical sha256")
        if self.source_hash_before == self.source_hash_after:
            raise TopologyReplayIntegrityError("accepted event must change source revision")
        if self.ordinal == 0:
            if self.prior_receipt_hash is not None:
                raise TopologyReplayIntegrityError("first receipt must not name a predecessor")
        elif type(self.prior_receipt_hash) is not str or _HASH.fullmatch(self.prior_receipt_hash) is None:
            raise TopologyReplayIntegrityError("subsequent receipt requires prior receipt sha256")
        if not isinstance(self.operation, TopologyOperation):
            raise TopologyReplayIntegrityError("receipt operation must be typed")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "ordinal": self.ordinal,
            "event_id": self.event_id,
            "event_hash": self.event_hash,
            "prior_receipt_hash": self.prior_receipt_hash,
            "source_hash_before": self.source_hash_before,
            "source_hash_after": self.source_hash_after,
            "operation": self.operation.value,
            "relation": MeshRelation.RELATIONSHIP.value,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class TopologyTransitionResult:
    source_snapshot: SituatedNetworkSnapshot
    receipt: TopologyRevisionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot, SituatedNetworkSnapshot):
            raise TopologyReplayIntegrityError("transition requires authoritative source snapshot")
        if (not isinstance(self.receipt, TopologyRevisionReceipt)
                or self.receipt.source_hash_after != self.source_snapshot.content_hash):
            raise TopologyReplayIntegrityError("transition receipt must match new source hash")


@dataclass(frozen=True)
class TopologyReplayResult:
    """Read-only in-memory event/receipt ledger, never a replacement graph owner."""

    initial_source_hash: str
    source_snapshot: SituatedNetworkSnapshot
    changes: tuple[TopologyChange, ...]
    receipts: tuple[TopologyRevisionReceipt, ...]

    def __post_init__(self) -> None:
        _validate_replay_lineage(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _REPLAY_SCHEMA,
            "initial_source_hash": self.initial_source_hash,
            "final_source_hash": self.source_snapshot.content_hash,
            "round_index": self.source_snapshot.round_index,
            "changes": [change.to_dict() for change in self.changes],
            "receipts": [receipt.to_dict() | {"receipt_hash": receipt.content_hash}
                         for receipt in self.receipts],
        }



def _validate_replay_lineage(replay: TopologyReplayResult) -> None:
    """Check the entire bounded, typed event/receipt chain before resumption.

    Every event must bind the preceding authoritative source hash. Each
    receipt must bind its event hash, consecutive ordinal and predecessor
    receipt hash, and the final receipt must match the final V19 snapshot.
    This is *internal consistency*, not authentication of an outside actor.
    """
    if not isinstance(replay.source_snapshot, SituatedNetworkSnapshot):
        raise TopologyReplayIntegrityError("replay requires an authoritative V19 snapshot")
    if (type(replay.initial_source_hash) is not str
            or _HASH.fullmatch(replay.initial_source_hash) is None):
        raise TopologyReplayIntegrityError("replay initial source hash must be canonical")
    if type(replay.changes) is not tuple or type(replay.receipts) is not tuple:
        raise TopologyReplayIntegrityError("replay requires immutable event/receipt tuples")
    if len(replay.changes) != len(replay.receipts) or len(replay.changes) > _MAX_EVENTS:
        raise TopologyReplayIntegrityError("replay event/receipt length or budget mismatch")

    previous_source = replay.initial_source_hash
    previous_receipt: str | None = None
    seen: set[str] = set()
    for index, (event, receipt) in enumerate(zip(replay.changes, replay.receipts)):
        if type(event) is not TopologyChange or type(receipt) is not TopologyRevisionReceipt:
            raise TopologyReplayIntegrityError("replay entries must have exact event/receipt types")
        if event.event_id in seen:
            raise TopologyReplayIntegrityError("replay contains duplicate event identity")
        seen.add(event.event_id)
        if (
            receipt.ordinal != index
            or receipt.event_id != event.event_id
            or receipt.event_hash != event.content_hash
            or receipt.operation is not event.operation
            or receipt.source_agent_id != event.source_agent_id
            or receipt.target_agent_id != event.target_agent_id
            or event.expected_source_hash != previous_source
            or receipt.source_hash_before != previous_source
            or receipt.prior_receipt_hash != previous_receipt
        ):
            raise TopologyReplayIntegrityError("replay event/receipt lineage mismatch")
        previous_source = receipt.source_hash_after
        previous_receipt = receipt.content_hash
    if previous_source != replay.source_snapshot.content_hash:
        raise TopologyReplayIntegrityError("replay final source is not the committed revision")


def _apply(
    snapshot: SituatedNetworkSnapshot,
    change: TopologyChange,
    *, ordinal: int, prior_receipt_hash: str | None,
) -> TopologyTransitionResult:
    if not isinstance(snapshot, SituatedNetworkSnapshot):
        raise TypeError("topology revision requires authoritative SituatedNetworkSnapshot")
    if not isinstance(change, TopologyChange):
        raise TypeError("topology revision requires a typed TopologyChange")
    before = snapshot.content_hash
    if change.expected_source_hash != before:
        raise TopologyVersionConflict("stale expected_source_hash: authoritative source has changed")
    agents = {node.agent_id for node in snapshot.nodes}
    if change.source_agent_id not in agents or change.target_agent_id not in agents:
        raise TopologyUnknownAgent("directed edge endpoint not present in V19 agent roster")
    pair = (change.source_agent_id, change.target_agent_id)
    current = {
        (edge.source_agent_id, edge.observer_agent_id): edge
        for edge in snapshot.relationship_edges
    }
    prior = current.get(pair)
    if change.operation is TopologyOperation.ADD:
        if prior is not None:
            raise TopologyEdgeConflict("directed edge already has a source record")
        assert change.edge is not None
        current[pair] = change.edge
    else:
        if prior is None or not prior.active:
            raise TopologyEdgeConflict("cannot deactivate an absent/inactive source edge")
        current[pair] = replace(prior, active=False)
    # Do not advance V19 cognitive/story round without its actual state
    # transition. Source content_hash, not round_index, is the edit revision.
    # Snapshot canonicalization preserves sorted record order.
    next_source = replace(snapshot, relationship_edges=tuple(current.values()))
    if next_source.content_hash == before:
        raise AssertionError("accepted topology event must change the source revision")
    receipt = TopologyRevisionReceipt(
        ordinal=ordinal,
        event_id=change.event_id,
        event_hash=change.content_hash,
        prior_receipt_hash=prior_receipt_hash,
        source_hash_before=before,
        source_hash_after=next_source.content_hash,
        operation=change.operation,
        source_agent_id=change.source_agent_id,
        target_agent_id=change.target_agent_id,
    )
    return TopologyTransitionResult(next_source, receipt)


def apply_topology_change(
    snapshot: SituatedNetworkSnapshot, change: TopologyChange,
) -> TopologyTransitionResult:
    """Apply one event; caller retains the original immutable snapshot."""
    return _apply(snapshot, change, ordinal=0, prior_receipt_hash=None)


def begin_topology_replay(snapshot: SituatedNetworkSnapshot) -> TopologyReplayResult:
    if not isinstance(snapshot, SituatedNetworkSnapshot):
        raise TypeError("topology replay requires authoritative SituatedNetworkSnapshot")
    return TopologyReplayResult(snapshot.content_hash, snapshot, (), ())


def append_topology_change(
    replay: TopologyReplayResult, change: TopologyChange,
) -> TopologyReplayResult:
    if not isinstance(replay, TopologyReplayResult):
        raise TypeError("append requires a TopologyReplayResult")
    # Defensive recheck catches even illegally mutated frozen objects.
    _validate_replay_lineage(replay)
    if not isinstance(change, TopologyChange):
        raise TypeError("append requires a TopologyChange")
    if len(replay.changes) >= _MAX_EVENTS:
        raise TopologyChangeError("R1 replay event budget exceeded")
    if any(existing.event_id == change.event_id for existing in replay.changes):
        raise TopologyDuplicateEvent("duplicate topology event_id in replay")
    previous = replay.receipts[-1].content_hash if replay.receipts else None
    result = _apply(replay.source_snapshot, change,
                    ordinal=len(replay.changes), prior_receipt_hash=previous)
    return TopologyReplayResult(
        replay.initial_source_hash, result.source_snapshot,
        (*replay.changes, change), (*replay.receipts, result.receipt),
    )


def replay_topology_changes(
    initial_snapshot: SituatedNetworkSnapshot,
    changes: tuple[TopologyChange, ...],
) -> TopologyReplayResult:
    """Replay ordered version-bound events; any failure aborts without commit."""
    if not isinstance(changes, tuple):
        raise TypeError("topology replay changes must be a tuple")
    if len(changes) > _MAX_EVENTS:
        raise TopologyChangeError("R1 replay event budget exceeded")
    replay = begin_topology_replay(initial_snapshot)
    for change in changes:
        replay = append_topology_change(replay, change)
    return replay


def observe_topology(
    mesh: MeshSnapshot, *, relation: MeshRelation,
) -> dict[str, object]:
    """Finite topology metrics using the EXISTING source-backed path query.

    Strong components are formed by mutual directed reachability, rather than
    an alternate hidden mutable graph. On disconnected graphs the directed
    diameter is undefined (None), not an invented six-hop upper bound.
    """
    if not isinstance(mesh, MeshSnapshot):
        raise TypeError("topology observation requires MeshSnapshot")
    if not isinstance(relation, MeshRelation):
        raise TopologyUnsupportedRelation("topology observation requires selected MeshRelation")
    nodes = tuple(node.agent_id for node in mesh.source_snapshot.nodes)
    if len(nodes) > _MAX_METRIC_NODES:
        raise TopologyChangeError("finite observation node budget exceeded")
    pairs: dict[tuple[str, str], int] = {}
    budget = max(0, len(nodes) - 1)
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            path = find_mesh_path(mesh, source, target, relation=relation,
                                  max_hops=budget)
            if path is not None:
                pairs[(source, target)] = path.hop_count
    components: list[list[str]] = []
    covered: set[str] = set()
    for source in nodes:
        if source in covered:
            continue
        mutual = [
            target for target in nodes
            if target == source or (
                (source, target) in pairs and (target, source) in pairs
            )
        ]
        covered.update(mutual)
        components.append(mutual)
    expected_pairs = len(nodes) * max(0, len(nodes) - 1)
    return {
        "source_snapshot_hash": mesh.source_snapshot.content_hash,
        "mesh_hash": mesh.content_hash,
        "relation": relation.value,
        "agent_count": len(nodes),
        "directed_edges": [list(edge) for edge in mesh_edges(mesh, relation)],
        "reachable_ordered_pair_count": len(pairs),
        "unreachable_ordered_pair_count": expected_pairs - len(pairs),
        "distance_sum": sum(pairs.values()),
        "max_finite_distance": max(pairs.values(), default=0),
        "directed_diameter": (
            max(pairs.values(), default=0)
            if nodes and len(pairs) == expected_pairs else None
        ),
        "strong_components": components,
    }
