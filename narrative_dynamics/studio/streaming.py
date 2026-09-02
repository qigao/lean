"""Bounded, capability-filtered delivery for World Studio output streams."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import re
from threading import RLock
from time import monotonic
from typing import Callable

from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationOutputAudience,
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputView,
)
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.studio.capabilities import StudioCapability
from narrative_dynamics.studio.service import (
    JsonObject,
    StudioAuthorizationError,
    StudioCapacityError,
    StudioConflictError,
    StudioHistoryGapError,
    StudioInvalidParamsError,
    StudioNotFoundError,
    StudioStaleStateError,
    WorldStudioService,
)


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class StudioOutputLimits:
    maximum_subscriptions_per_connection: int = 16
    maximum_subscriptions: int = 64
    maximum_subscriptions_per_run: int = 16
    maximum_retained_batches: int = 256
    maximum_retained_records: int = 20_000
    maximum_retained_bytes: int = 16 * 1024 * 1024
    maximum_seen_batch_identities: int = 20_000
    maximum_released_subscriptions: int = 64
    lease_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_subscriptions_per_connection",
            "maximum_subscriptions",
            "maximum_subscriptions_per_run",
            "maximum_retained_batches",
            "maximum_retained_records",
            "maximum_retained_bytes",
            "maximum_seen_batch_identities",
            "maximum_released_subscriptions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"studio output {name.replace('_', ' ')} must be an integer")
            if value <= 0:
                raise ValueError(f"studio output {name.replace('_', ' ')} must be positive")
        if (
            not isinstance(self.lease_seconds, (int, float))
            or isinstance(self.lease_seconds, bool)
            or self.lease_seconds <= 0
        ):
            raise ValueError("studio output lease seconds must be positive")

    def to_dict(self) -> JsonObject:
        return {
            "maximum_subscriptions_per_connection": self.maximum_subscriptions_per_connection,
            "maximum_subscriptions": self.maximum_subscriptions,
            "maximum_subscriptions_per_run": self.maximum_subscriptions_per_run,
            "maximum_retained_batches": self.maximum_retained_batches,
            "maximum_retained_records": self.maximum_retained_records,
            "maximum_retained_bytes": self.maximum_retained_bytes,
            "maximum_seen_batch_identities": self.maximum_seen_batch_identities,
            "maximum_released_subscriptions": self.maximum_released_subscriptions,
            "lease_seconds": float(self.lease_seconds),
        }


class SubscriptionAuthorizationError(StudioAuthorizationError):
    pass


class SubscriptionCapacityError(StudioCapacityError):
    pass


class SubscriptionConflictError(StudioConflictError):
    pass


class SubscriptionStateError(StudioStaleStateError):
    pass


class SubscriptionLeaseExpired(StudioNotFoundError):
    pass


class SubscriptionHistoryGap(StudioHistoryGapError):
    def __init__(
        self,
        current_sequence: int,
        current_batch_hash: str,
        snapshot_token: str,
    ) -> None:
        self.current_sequence = current_sequence
        self.current_batch_hash = current_batch_hash
        self.snapshot_token = snapshot_token
        super().__init__()
        self.data = self.to_dict()

    def to_dict(self) -> JsonObject:
        return {
            "current_sequence": self.current_sequence,
            "current_batch_hash": self.current_batch_hash,
            "snapshot_token": self.snapshot_token,
        }


@dataclass(frozen=True)
class StudioOutputSubscription:
    subscription_id: str
    run_id: str
    stream_id: str
    capability: StudioCapability
    kinds: tuple[SimulationOutputKind, ...]
    owner_agent_id: str | None
    audience_capability: SimulationAudienceCapability


OutputCallback = Callable[[SimulationOutputView], None]
StreamingOutputCallback = Callable[[str, SimulationOutputView], None]


@dataclass
class _SubscriptionState:
    subscription: StudioOutputSubscription
    connection_id: str
    deadline: float
    on_output: OutputCallback | None
    retained: deque[tuple[SimulationOutputView, int]] = field(default_factory=deque)
    retained_records: int = 0
    retained_bytes: int = 0
    acknowledged: tuple[int, str] | None = None
    latest: tuple[int, str] | None = None
    latest_bounds: tuple[int, int, str] | None = None


@dataclass
class _ReleasedSubscriptionState:
    """Capability-filtered recovery only; never retains a connection or callback."""

    subscription: StudioOutputSubscription
    deadline: float
    release_order: int
    retained: deque[tuple[SimulationOutputView, int]] = field(default_factory=deque)
    retained_records: int = 0
    retained_bytes: int = 0
    acknowledged: tuple[int, str] | None = None
    latest: tuple[int, str] | None = None
    latest_bounds: tuple[int, int, str] | None = None


@dataclass
class _StreamIdentityLedger:
    identities: dict[tuple[int, int], str] = field(default_factory=dict)
    latest: tuple[int, int, str] | None = None


def _identity(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{label} must be a bounded path-free identity")
    return value


def studio_output_view_to_dict(view: SimulationOutputView) -> JsonObject:
    """Return the trusted full wire projection of an already-filtered view."""
    records = []
    for record in view.records:
        row = record.to_dict()
        row["content_hash"] = record.content_hash
        records.append(row)
    return {
        "schema": view.schema,
        "stream_id": view.stream_id,
        "scenario_hash": view.scenario_hash,
        "prior_state_hash": view.prior_state_hash,
        "next_state_hash": view.next_state_hash,
        "round_result_hash": view.round_result_hash,
        "first_sequence": view.first_sequence,
        "last_sequence": view.last_sequence,
        "records": records,
        "source_batch_hash": view.source_batch_hash,
        "checkpoint": view.checkpoint,
        "content_hash": view.content_hash,
    }


def _view_bytes(view: SimulationOutputView) -> int:
    return len(
        json.dumps(
            studio_output_view_to_dict(view),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


class StudioOutputRouter:
    """Retain only immutable output views projected for each exact authority."""

    def __init__(
        self,
        *,
        limits: StudioOutputLimits | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._limits = limits or StudioOutputLimits()
        if not isinstance(self._limits, StudioOutputLimits):
            raise TypeError("studio output limits must be StudioOutputLimits")
        if not callable(clock):
            raise TypeError("studio output clock must be callable")
        self._clock = clock
        self._lock = RLock()
        self._subscriptions: dict[str, _SubscriptionState] = {}
        self._released: dict[str, _ReleasedSubscriptionState] = {}
        self._release_order = 0
        # There is no safe stream-end lifecycle in V22, so accepted identities are
        # deliberately retained until router process restart. Replay views remain
        # independently bounded on each subscription.
        self._seen_streams: dict[tuple[str, str], _StreamIdentityLedger] = {}
        self._seen_batch_identity_count = 0

    @property
    def limits(self) -> StudioOutputLimits:
        return self._limits

    @staticmethod
    def _audience(
        capability: StudioCapability, owner_agent_id: str | None
    ) -> SimulationAudienceCapability:
        if owner_agent_id is not None:
            owner = _identity(owner_agent_id, label="subscription owner agent ID")
            if owner not in capability.agent_ids:
                raise SubscriptionAuthorizationError()
            return SimulationAudienceCapability(SimulationOutputAudience.AGENT, owner)
        if "state.network" in capability.permissions:
            return SimulationAudienceCapability(SimulationOutputAudience.ANALYST)
        return SimulationAudienceCapability(SimulationOutputAudience.PUBLIC)

    @staticmethod
    def _authorize_run(capability: StudioCapability, run_id: str) -> None:
        if "output.read" not in capability.permissions or run_id not in capability.run_ids:
            raise SubscriptionAuthorizationError()

    def _purge_expired_locked(self, now: float) -> set[str]:
        expired = {
            subscription_id
            for subscription_id, state in self._subscriptions.items()
            if state.deadline <= now
        }
        for subscription_id in expired:
            del self._subscriptions[subscription_id]
        released_expired = tuple(
            subscription_id
            for subscription_id, state in self._released.items()
            if state.deadline <= now
        )
        for subscription_id in released_expired:
            del self._released[subscription_id]
        return expired

    def _archive_released_locked(self, state: _SubscriptionState, now: float) -> None:
        self._release_order += 1
        self._released[state.subscription.subscription_id] = _ReleasedSubscriptionState(
            state.subscription,
            now + float(self._limits.lease_seconds),
            self._release_order,
            state.retained,
            state.retained_records,
            state.retained_bytes,
            state.acknowledged,
            state.latest,
            state.latest_bounds,
        )
        while len(self._released) > self._limits.maximum_released_subscriptions:
            oldest_id = min(
                self._released,
                key=lambda item: (
                    self._released[item].release_order,
                    item,
                ),
            )
            del self._released[oldest_id]

    def subscribe(
        self,
        subscription_id: str,
        run_id: str,
        stream_id: str,
        capability: StudioCapability,
        kinds: tuple[SimulationOutputKind, ...],
        *,
        connection_id: str | None = None,
        owner_agent_id: str | None = None,
        on_output: OutputCallback | None = None,
    ) -> StudioOutputSubscription:
        subscription_id = _identity(subscription_id, label="subscription ID")
        run_id = _identity(run_id, label="subscription run ID")
        stream_id = _identity(stream_id, label="subscription stream ID")
        if not isinstance(capability, StudioCapability):
            raise TypeError("subscription capability must be StudioCapability")
        self._authorize_run(capability, run_id)
        if not isinstance(kinds, tuple) or any(
            not isinstance(kind, SimulationOutputKind) for kind in kinds
        ):
            raise TypeError("subscription kinds must be a tuple of SimulationOutputKind")
        if len(set(kinds)) != len(kinds):
            raise ValueError("subscription kinds must be unique")
        canonical_kinds = tuple(kind for kind in SimulationOutputKind if kind in kinds)
        connection_id = _identity(
            subscription_id if connection_id is None else connection_id,
            label="subscription connection ID",
        )
        if on_output is not None and not callable(on_output):
            raise TypeError("subscription output callback must be callable")
        audience = self._audience(capability, owner_agent_id)
        subscription = StudioOutputSubscription(
            subscription_id,
            run_id,
            stream_id,
            capability,
            canonical_kinds,
            owner_agent_id,
            audience,
        )
        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            if subscription_id in self._subscriptions:
                raise SubscriptionConflictError()
            released = self._released.get(subscription_id)
            if released is not None:
                if released.subscription.capability != capability:
                    raise SubscriptionAuthorizationError()
                if released.subscription != subscription:
                    raise SubscriptionConflictError()
            states = tuple(self._subscriptions.values())
            if len(states) >= self._limits.maximum_subscriptions:
                raise SubscriptionCapacityError()
            if (
                sum(state.connection_id == connection_id for state in states)
                >= self._limits.maximum_subscriptions_per_connection
            ):
                raise SubscriptionCapacityError()
            if (
                sum(state.subscription.run_id == run_id for state in states)
                >= self._limits.maximum_subscriptions_per_run
            ):
                raise SubscriptionCapacityError()
            if released is None:
                state = _SubscriptionState(
                    subscription,
                    connection_id,
                    now + float(self._limits.lease_seconds),
                    on_output,
                )
            else:
                state = _SubscriptionState(
                    subscription,
                    connection_id,
                    now + float(self._limits.lease_seconds),
                    on_output,
                    released.retained,
                    released.retained_records,
                    released.retained_bytes,
                    released.acknowledged,
                    released.latest,
                    released.latest_bounds,
                )
                del self._released[subscription_id]
            self._subscriptions[subscription_id] = state
        return subscription

    @staticmethod
    def _filtered_view(
        batch: SimulationOutputBatch, subscription: StudioOutputSubscription
    ) -> SimulationOutputView:
        scoped = SimulationOutputView.from_batch(
            batch, subscription.audience_capability
        )
        return SimulationOutputView(
            scoped.stream_id,
            scoped.scenario_hash,
            scoped.prior_state_hash,
            scoped.next_state_hash,
            scoped.round_result_hash,
            scoped.first_sequence,
            scoped.last_sequence,
            tuple(record for record in scoped.records if record.kind in subscription.kinds),
            scoped.source_batch_hash,
            scoped.checkpoint,
        )

    def _seen_duplicate_locked(
        self,
        stream_key: tuple[str, str],
        identity: tuple[int, int, str],
    ) -> bool:
        ledger = self._seen_streams.get(stream_key)
        if ledger is not None:
            bounds = identity[:2]
            seen_hash = ledger.identities.get(bounds)
            if seen_hash is not None:
                if seen_hash == identity[2]:
                    return True
                raise SubscriptionStateError()
            if ledger.latest is not None and identity[0] != ledger.latest[1] + 1:
                raise SubscriptionStateError()
        if (
            self._seen_batch_identity_count
            >= self._limits.maximum_seen_batch_identities
        ):
            raise SubscriptionCapacityError()
        return False

    def _record_seen_locked(
        self,
        stream_key: tuple[str, str],
        identity: tuple[int, int, str],
    ) -> None:
        ledger = self._seen_streams.setdefault(stream_key, _StreamIdentityLedger())
        ledger.identities[identity[:2]] = identity[2]
        ledger.latest = identity
        self._seen_batch_identity_count += 1

    def publish(self, run_id: str, batch: SimulationOutputBatch) -> tuple[str, ...]:
        run_id = _identity(run_id, label="published run ID")
        if not isinstance(batch, SimulationOutputBatch):
            raise TypeError("published output must be a SimulationOutputBatch")
        now = self._clock()
        identity = (batch.first_sequence, batch.last_sequence, batch.content_hash)
        stream_key = (run_id, batch.stream_id)
        with self._lock:
            self._purge_expired_locked(now)
            snapshot = tuple(
                (
                    state.subscription,
                    state.latest_bounds,
                    frozenset(
                        (
                            view.first_sequence,
                            view.last_sequence,
                            view.source_batch_hash,
                        )
                        for view, _ in state.retained
                    ),
                    False,
                )
                for state in self._subscriptions.values()
                if state.subscription.run_id == run_id
                and state.subscription.stream_id == batch.stream_id
            ) + tuple(
                (
                    state.subscription,
                    state.latest_bounds,
                    frozenset(
                        (
                            view.first_sequence,
                            view.last_sequence,
                            view.source_batch_hash,
                        )
                        for view, _ in state.retained
                    ),
                    True,
                )
                for state in self._released.values()
                if state.subscription.run_id == run_id
                and state.subscription.stream_id == batch.stream_id
            )
            if not snapshot:
                return ()
            if self._seen_duplicate_locked(stream_key, identity):
                return ()

        pending = []
        for subscription, latest_bounds, retained_identities, released in snapshot:
            if identity in retained_identities or latest_bounds == identity:
                continue
            if latest_bounds is not None and batch.first_sequence != latest_bounds[1] + 1:
                raise SubscriptionStateError()
            pending.append((subscription, latest_bounds, released))

        projected = tuple(
            (
                subscription,
                prior_bounds,
                released,
                self._filtered_view(batch, subscription),
            )
            for subscription, prior_bounds, released in pending
        )
        callbacks: list[tuple[OutputCallback, SimulationOutputView]] = []
        delivered: list[str] = []
        with self._lock:
            if self._seen_duplicate_locked(stream_key, identity):
                return ()
            actions: list[
                tuple[_SubscriptionState | _ReleasedSubscriptionState, SimulationOutputView, bool]
            ] = []
            for subscription, prior_bounds, released, view in projected:
                state = (
                    self._released.get(subscription.subscription_id)
                    if released
                    else self._subscriptions.get(subscription.subscription_id)
                )
                if (
                    state is None
                    or state.subscription != subscription
                    or state.deadline <= self._clock()
                ):
                    continue
                if state.latest_bounds == identity:
                    continue
                if any(
                    (
                        retained.first_sequence,
                        retained.last_sequence,
                        retained.source_batch_hash,
                    )
                    == identity
                    for retained, _ in state.retained
                ):
                    continue
                if (
                    state.latest_bounds != prior_bounds
                    or state.latest_bounds is not None
                    and batch.first_sequence != state.latest_bounds[1] + 1
                ):
                    raise SubscriptionStateError()
                actions.append((state, view, released))
            if not actions:
                return ()
            self._record_seen_locked(stream_key, identity)
            for state, view, released in actions:
                size = _view_bytes(view)
                state.retained.append((view, size))
                state.retained_records += len(view.records)
                state.retained_bytes += size
                state.latest = (view.last_sequence, view.source_batch_hash)
                state.latest_bounds = identity
                while state.retained and (
                    len(state.retained) > self._limits.maximum_retained_batches
                    or state.retained_records > self._limits.maximum_retained_records
                    or state.retained_bytes > self._limits.maximum_retained_bytes
                ):
                    removed, removed_size = state.retained.popleft()
                    state.retained_records -= len(removed.records)
                    state.retained_bytes -= removed_size
                if not released:
                    state.deadline = self._clock() + float(self._limits.lease_seconds)
                    active = state
                    if isinstance(active, _SubscriptionState) and active.on_output is not None:
                        callbacks.append((active.on_output, view))
                    delivered.append(state.subscription.subscription_id)
        for callback, view in callbacks:
            try:
                callback(view)
            except Exception:
                continue
        return tuple(delivered)

    def _state(
        self,
        subscription_id: str,
        capability: StudioCapability | None,
        stream_id: str | None = None,
    ) -> _SubscriptionState:
        subscription_id = _identity(subscription_id, label="subscription ID")
        now = self._clock()
        with self._lock:
            state = self._subscriptions.get(subscription_id)
            if state is None:
                raise SubscriptionLeaseExpired()
            if state.deadline <= now:
                del self._subscriptions[subscription_id]
                raise SubscriptionLeaseExpired()
            if capability is not None and state.subscription.capability != capability:
                raise SubscriptionAuthorizationError()
            if stream_id is not None and state.subscription.stream_id != stream_id:
                raise SubscriptionAuthorizationError()
            return state

    @staticmethod
    def _cursor_exists(state: _SubscriptionState, cursor: tuple[int, str]) -> bool:
        return cursor in (state.acknowledged, state.latest) or any(
            (view.last_sequence, view.source_batch_hash) == cursor
            for view, _ in state.retained
        )

    @staticmethod
    def _cursor(last_sequence: object, last_batch_hash: object) -> tuple[int, str]:
        if (
            not isinstance(last_sequence, int)
            or isinstance(last_sequence, bool)
            or last_sequence <= 0
            or not isinstance(last_batch_hash, str)
            or _CONTENT_HASH.fullmatch(last_batch_hash) is None
        ):
            raise SubscriptionStateError()
        return last_sequence, last_batch_hash

    def acknowledge(
        self,
        subscription_id: str,
        last_sequence: int,
        last_batch_hash: str,
        *,
        capability: StudioCapability | None = None,
        stream_id: str | None = None,
    ) -> None:
        cursor = self._cursor(last_sequence, last_batch_hash)
        with self._lock:
            state = self._state(subscription_id, capability, stream_id)
            if not self._cursor_exists(state, cursor):
                raise SubscriptionStateError()
            if state.acknowledged is not None and last_sequence < state.acknowledged[0]:
                raise SubscriptionStateError()
            state.acknowledged = cursor
            state.deadline = self._clock() + float(self._limits.lease_seconds)

    def resume(
        self,
        subscription_id: str,
        last_sequence: int,
        last_batch_hash: str,
        *,
        capability: StudioCapability | None = None,
        stream_id: str | None = None,
    ) -> tuple[SimulationOutputView, ...]:
        cursor = self._cursor(last_sequence, last_batch_hash)
        with self._lock:
            state = self._state(subscription_id, capability, stream_id)
            if state.acknowledged is not None and last_sequence < state.acknowledged[0]:
                raise SubscriptionStateError()
            if not self._cursor_exists(state, cursor):
                raise SubscriptionStateError()
            retained = tuple(view for view, _ in state.retained)
            latest = state.latest
            state.deadline = self._clock() + float(self._limits.lease_seconds)
            if latest is None or cursor == latest:
                return ()
            later = tuple(view for view in retained if view.last_sequence > last_sequence)
            contiguous = bool(later)
            next_sequence = last_sequence + 1
            for view in later:
                if view.first_sequence != next_sequence:
                    contiguous = False
                    break
                next_sequence = view.last_sequence + 1
            if not contiguous or later[-1].last_sequence != latest[0]:
                token = stable_content_hash(
                    {
                        "schema": "narrative-dynamics.studio-snapshot-token/v1",
                        "run_id": state.subscription.run_id,
                        "stream_id": state.subscription.stream_id,
                        "current_sequence": latest[0],
                        "current_batch_hash": latest[1],
                        "capability_hash": state.subscription.capability.content_hash,
                    }
                )
                raise SubscriptionHistoryGap(latest[0], latest[1], token)
            seen: set[str] = set()
            ordered = []
            for view in later:
                if view.source_batch_hash not in seen:
                    ordered.append(view)
                    seen.add(view.source_batch_hash)
            return tuple(ordered)

    def unsubscribe(
        self,
        subscription_id: str,
        *,
        capability: StudioCapability | None = None,
        connection_id: str | None = None,
    ) -> bool:
        with self._lock:
            state = self._state(subscription_id, capability)
            if connection_id is not None and state.connection_id != connection_id:
                raise SubscriptionAuthorizationError()
            del self._subscriptions[state.subscription.subscription_id]
            return True

    def unsubscribe_connection(self, connection_id: str) -> tuple[str, ...]:
        connection_id = _identity(connection_id, label="subscription connection ID")
        with self._lock:
            self._purge_expired_locked(self._clock())
            removed = tuple(
                sorted(
                    subscription_id
                    for subscription_id, state in self._subscriptions.items()
                    if state.connection_id == connection_id
                )
            )
            now = self._clock()
            for subscription_id in removed:
                state = self._subscriptions.pop(subscription_id)
                self._archive_released_locked(state, now)
            return removed


class StudioStreamingService(WorldStudioService):
    """Transport-neutral strict application service for stream controls."""

    def __init__(
        self,
        router: StudioOutputRouter,
        connection_id: str,
        *,
        on_output: StreamingOutputCallback | None = None,
    ) -> None:
        if not isinstance(router, StudioOutputRouter):
            raise TypeError("streaming service requires StudioOutputRouter")
        self._router = router
        self._connection_id = _identity(connection_id, label="stream connection ID")
        self._on_output = on_output
        self._methods = {
            "stream.subscribe": self._subscribe,
            "stream.acknowledge": self._acknowledge,
            "stream.resume": self._resume,
            "stream.unsubscribe": self._unsubscribe,
        }

    def is_state_changing(self, method: str) -> bool:
        return method in self._methods

    def _subscribe(self, params: JsonObject, capability: StudioCapability):
        allowed = {"subscription_id", "run_id", "stream_id", "kinds", "owner_agent_id"}
        required = {"subscription_id", "run_id", "stream_id", "kinds"}
        if set(params) - allowed or not required.issubset(params):
            raise StudioInvalidParamsError()
        try:
            raw_kinds = params["kinds"]
            if not isinstance(raw_kinds, list):
                raise ValueError
            kinds = tuple(SimulationOutputKind(item) for item in raw_kinds)
            subscription_id = _identity(
                params["subscription_id"], label="subscription ID"
            )
            callback = None
            if self._on_output is not None:
                callback = lambda view: self._on_output(subscription_id, view)
            subscription = self._router.subscribe(
                subscription_id,
                params["run_id"],
                params["stream_id"],
                capability,
                kinds,
                connection_id=self._connection_id,
                owner_agent_id=params.get("owner_agent_id"),
                on_output=callback,
            )
        except (TypeError, ValueError) as error:
            if isinstance(
                error,
                (StudioAuthorizationError, StudioCapacityError, StudioConflictError),
            ):
                raise
            raise StudioInvalidParamsError() from None
        return {
            "subscription_id": subscription.subscription_id,
            "run_id": subscription.run_id,
            "stream_id": subscription.stream_id,
            "kinds": [kind.value for kind in subscription.kinds],
        }

    @staticmethod
    def _cursor_params(params: JsonObject) -> tuple[str, str, int, str]:
        if set(params) != {"subscription_id", "stream_id", "last_sequence", "last_batch_hash"}:
            raise StudioInvalidParamsError()
        try:
            sequence = params["last_sequence"]
            batch_hash = params["last_batch_hash"]
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence <= 0
                or not isinstance(batch_hash, str)
                or _CONTENT_HASH.fullmatch(batch_hash) is None
            ):
                raise ValueError
            return (
                _identity(params["subscription_id"], label="subscription ID"),
                _identity(params["stream_id"], label="subscription stream ID"),
                sequence,
                batch_hash,
            )
        except (TypeError, ValueError):
            raise StudioInvalidParamsError() from None

    def _acknowledge(self, params: JsonObject, capability: StudioCapability):
        subscription_id, stream_id, sequence, batch_hash = self._cursor_params(params)
        self._router.acknowledge(
            subscription_id,
            sequence,
            batch_hash,
            capability=capability,
            stream_id=stream_id,
        )
        return {"subscription_id": subscription_id, "acknowledged": True}

    def _resume(self, params: JsonObject, capability: StudioCapability):
        subscription_id, stream_id, sequence, batch_hash = self._cursor_params(params)
        views = self._router.resume(
            subscription_id,
            sequence,
            batch_hash,
            capability=capability,
            stream_id=stream_id,
        )
        if self._on_output is not None:
            for view in views:
                self._on_output(subscription_id, view)
        current_sequence = sequence if not views else views[-1].last_sequence
        current_batch_hash = batch_hash if not views else views[-1].source_batch_hash
        return {
            "subscription_id": subscription_id,
            "resumed_from_sequence": sequence,
            "replayed_count": len(views),
            "current_sequence": current_sequence,
            "current_batch_hash": current_batch_hash,
        }

    def _unsubscribe(self, params: JsonObject, capability: StudioCapability):
        if set(params) != {"subscription_id"}:
            raise StudioInvalidParamsError()
        try:
            subscription_id = _identity(params["subscription_id"], label="subscription ID")
        except (TypeError, ValueError):
            raise StudioInvalidParamsError() from None
        self._router.unsubscribe(
            subscription_id,
            capability=capability,
            connection_id=self._connection_id,
        )
        return {"subscription_id": subscription_id, "unsubscribed": True}


__all__ = (
    "StudioOutputLimits",
    "StudioOutputRouter",
    "StudioOutputSubscription",
    "StudioStreamingService",
    "SubscriptionAuthorizationError",
    "SubscriptionCapacityError",
    "SubscriptionConflictError",
    "SubscriptionHistoryGap",
    "SubscriptionLeaseExpired",
    "SubscriptionStateError",
    "studio_output_view_to_dict",
)
