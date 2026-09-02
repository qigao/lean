"""Synchronous, capability-scoped delivery of immutable simulation output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from threading import RLock, get_ident

from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputView,
)
from narrative_dynamics.contracts import stable_content_hash


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_DELIVERY_FAILURE_CODES = frozenset(
    {"callback_error", "non_none_return", "reentrant_publish"}
)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a content hash")
    return value


def _canonical_text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple of strings")
    result = tuple(_text(item, label=f"{label} value") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} values must be unique")
    return tuple(sorted(result))


class _ContentAddressed:
    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())  # type: ignore[attr-defined]


@dataclass(frozen=True)
class SimulationOutputSubscription(_ContentAddressed):
    subscription_id: str
    kinds: tuple[SimulationOutputKind, ...]
    capability: SimulationAudienceCapability

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subscription_id",
            _text(self.subscription_id, label="output subscription id"),
        )
        if not isinstance(self.kinds, tuple) or any(
            not isinstance(kind, SimulationOutputKind) for kind in self.kinds
        ):
            raise TypeError(
                "output subscription kinds must be a tuple of SimulationOutputKind values"
            )
        if not self.kinds:
            raise ValueError("output subscription kinds must be non-empty")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("output subscription kinds must be unique")
        object.__setattr__(
            self,
            "kinds",
            tuple(sorted(self.kinds, key=lambda kind: kind.value)),
        )
        if not isinstance(self.capability, SimulationAudienceCapability):
            raise TypeError(
                "output subscription capability must be SimulationAudienceCapability"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "subscription_id": self.subscription_id,
            "kinds": [kind.value for kind in self.kinds],
            "capability": {
                "audience": self.capability.audience.value,
                "owner_agent_id": self.capability.owner_agent_id,
            },
        }


@dataclass(frozen=True)
class SimulationDeliveryFailure(_ContentAddressed):
    subscription_id: str
    code: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subscription_id",
            _text(self.subscription_id, label="delivery failure subscription id"),
        )
        object.__setattr__(
            self,
            "code",
            _text(self.code, label="delivery failure code"),
        )
        if self.code not in _DELIVERY_FAILURE_CODES:
            raise ValueError("delivery failure code is not supported")

    def to_dict(self) -> dict[str, object]:
        return {"subscription_id": self.subscription_id, "code": self.code}


@dataclass(frozen=True)
class SimulationDeliveryReport(_ContentAddressed):
    batch_hash: str
    delivered_subscription_ids: tuple[str, ...]
    failures: tuple[SimulationDeliveryFailure, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "batch_hash",
            _content_hash(self.batch_hash, label="delivery report batch hash"),
        )
        delivered = _canonical_text_tuple(
            self.delivered_subscription_ids,
            label="delivery report delivered subscription ids",
        )
        object.__setattr__(self, "delivered_subscription_ids", delivered)
        if not isinstance(self.failures, tuple) or any(
            not isinstance(failure, SimulationDeliveryFailure)
            for failure in self.failures
        ):
            raise TypeError(
                "delivery report failures must be a tuple of SimulationDeliveryFailure values"
            )
        failure_ids = tuple(failure.subscription_id for failure in self.failures)
        if len(set(failure_ids)) != len(failure_ids):
            raise ValueError("delivery report failure subscription ids must be unique")
        if set(delivered).intersection(failure_ids):
            raise ValueError(
                "delivery report subscriptions cannot be both delivered and failed"
            )
        object.__setattr__(
            self,
            "failures",
            tuple(
                sorted(
                    self.failures,
                    key=lambda failure: (failure.subscription_id, failure.code),
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_hash": self.batch_hash,
            "delivered_subscription_ids": list(self.delivered_subscription_ids),
            "failures": [failure.to_dict() for failure in self.failures],
        }


SimulationOutputCallback = Callable[[SimulationOutputView], object]


class SimulationOutputBus:
    """Deliver output views synchronously from a stable subscription snapshot."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._subscriptions: dict[
            str, tuple[SimulationOutputSubscription, SimulationOutputCallback]
        ] = {}
        self._publishing_thread_id: int | None = None

    def subscribe(
        self,
        subscription_id: str,
        kinds: tuple[SimulationOutputKind, ...],
        capability: SimulationAudienceCapability,
        callback: SimulationOutputCallback,
    ) -> SimulationOutputSubscription:
        subscription = SimulationOutputSubscription(
            subscription_id,
            kinds,
            capability,
        )
        if not callable(callback):
            raise TypeError("output subscription callback must be callable")
        with self._lock:
            if subscription.subscription_id in self._subscriptions:
                raise ValueError("output subscription id already exists")
            self._subscriptions[subscription.subscription_id] = (
                subscription,
                callback,
            )
            return subscription

    def unsubscribe(self, subscription_id: str) -> None:
        normalized_id = _text(subscription_id, label="output subscription id")
        with self._lock:
            self._subscriptions.pop(normalized_id, None)

    def publish(self, batch: SimulationOutputBatch) -> SimulationDeliveryReport:
        with self._lock:
            thread_id = get_ident()
            if self._publishing_thread_id == thread_id:
                raise RuntimeError("reentrant_publish")
            if not isinstance(batch, SimulationOutputBatch):
                raise TypeError("output publication requires a SimulationOutputBatch")

            snapshot = tuple(
                sorted(
                    self._subscriptions.values(),
                    key=lambda item: item[0].subscription_id,
                )
            )
            delivered: list[str] = []
            failures: list[SimulationDeliveryFailure] = []
            self._publishing_thread_id = thread_id
            try:
                for subscription, callback in snapshot:
                    scoped = SimulationOutputView.from_batch(
                        batch,
                        subscription.capability,
                    )
                    view = SimulationOutputView(
                        scoped.stream_id,
                        scoped.scenario_hash,
                        scoped.prior_state_hash,
                        scoped.next_state_hash,
                        scoped.round_result_hash,
                        scoped.first_sequence,
                        scoped.last_sequence,
                        tuple(
                            record
                            for record in scoped.records
                            if record.kind in subscription.kinds
                        ),
                        scoped.source_batch_hash,
                        scoped.checkpoint,
                    )
                    try:
                        result = callback(view)
                    except Exception:
                        failures.append(
                            SimulationDeliveryFailure(
                                subscription.subscription_id,
                                "callback_error",
                            )
                        )
                        continue
                    if result is not None:
                        failures.append(
                            SimulationDeliveryFailure(
                                subscription.subscription_id,
                                "non_none_return",
                            )
                        )
                        continue
                    delivered.append(subscription.subscription_id)
            finally:
                self._publishing_thread_id = None

            return SimulationDeliveryReport(
                batch.content_hash,
                tuple(delivered),
                tuple(failures),
            )


__all__ = (
    "SimulationOutputSubscription",
    "SimulationDeliveryFailure",
    "SimulationDeliveryReport",
    "SimulationOutputBus",
)
