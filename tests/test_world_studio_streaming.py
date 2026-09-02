from __future__ import annotations

import pytest

from narrative_dynamics.abm import (
    ObservationChannel,
    SimulationAudienceCapability,
    SimulationCommandResultPayload,
    SimulationOutputAudience,
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputRecord,
    SimulationPrivatePerceptPayload,
    SituatedPercept,
    SituatedPerceptFidelity,
)
from narrative_dynamics.studio import StudioCapability
from narrative_dynamics.studio.streaming import (
    StudioOutputLimits,
    StudioOutputRouter,
    SubscriptionAuthorizationError,
    SubscriptionCapacityError,
    SubscriptionConflictError,
    SubscriptionHistoryGap,
    SubscriptionLeaseExpired,
    SubscriptionStateError,
)


HASHES = tuple("sha256:" + str(index) * 64 for index in range(1, 9))


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def capability(
    authority_id: str = "operator",
    *,
    run_ids: tuple[str, ...] = ("run-1",),
    agent_ids: tuple[str, ...] = (),
    permissions: tuple[str, ...] = ("output.read",),
) -> StudioCapability:
    return StudioCapability(
        authority_id,
        run_ids=run_ids,
        agent_ids=agent_ids,
        permissions=permissions,
    )


def public_batch(sequence: int, *, stream_id: str = "stream-1") -> SimulationOutputBatch:
    next_hash = HASHES[sequence % 3 + 2]
    record = SimulationOutputRecord(
        stream_id,
        HASHES[0],
        sequence,
        sequence,
        next_hash,
        SimulationOutputKind.COMMAND_RESULT,
        SimulationOutputAudience.PUBLIC,
        None,
        (),
        SimulationCommandResultPayload(f"command-{sequence}", True, "accepted"),
    )
    return SimulationOutputBatch(
        stream_id,
        HASHES[0],
        HASHES[1],
        next_hash,
        HASHES[5],
        sequence,
        sequence,
        (record,),
    )


def private_batch(sequence: int = 1) -> SimulationOutputBatch:
    percept = SituatedPercept(
        "percept-1",
        sequence,
        "alice",
        "event-1",
        HASHES[6],
        (ObservationChannel.AUDITORY,),
        SituatedPerceptFidelity.DETECTED,
    )
    record = SimulationOutputRecord(
        "stream-1",
        HASHES[0],
        sequence,
        sequence,
        HASHES[2],
        SimulationOutputKind.PERCEPT_PRIVATE,
        SimulationOutputAudience.AGENT,
        "alice",
        (),
        SimulationPrivatePerceptPayload(percept),
    )
    return SimulationOutputBatch(
        "stream-1",
        HASHES[0],
        HASHES[1],
        HASHES[2],
        HASHES[5],
        sequence,
        sequence,
        (record,),
    )


def test_acknowledged_cursor_resumes_with_only_later_batches() -> None:
    router = StudioOutputRouter()
    allowed = capability()
    router.subscribe(
        "sub-1", "run-1", "stream-1", allowed, tuple(SimulationOutputKind)
    )
    first = public_batch(1)
    second = public_batch(2)
    router.publish("run-1", first)
    router.acknowledge(
        "sub-1", first.last_sequence, first.content_hash, capability=allowed
    )
    router.publish("run-1", second)

    replayed = router.resume(
        "sub-1",
        first.last_sequence,
        first.content_hash,
        capability=allowed,
        stream_id="stream-1",
    )

    assert tuple(item.source_batch_hash for item in replayed) == (second.content_hash,)


def test_capability_is_exact_and_acknowledgements_are_delivered_monotonic_cursors() -> None:
    router = StudioOutputRouter()
    allowed = capability()
    router.subscribe("sub-1", "run-1", "stream-1", allowed, tuple(SimulationOutputKind))
    first = public_batch(1)
    second = public_batch(2)
    router.publish("run-1", first)
    router.publish("run-1", second)

    with pytest.raises(SubscriptionAuthorizationError):
        router.resume(
            "sub-1", 1, first.content_hash, capability=capability("intruder")
        )
    with pytest.raises(SubscriptionStateError):
        router.acknowledge("sub-1", 2, HASHES[7], capability=allowed)
    router.acknowledge("sub-1", 2, second.content_hash, capability=allowed)
    with pytest.raises(SubscriptionStateError):
        router.acknowledge("sub-1", 1, first.content_hash, capability=allowed)
    with pytest.raises(SubscriptionStateError):
        router.resume("sub-1", "2", second.content_hash, capability=allowed)  # type: ignore[arg-type]


def test_duplicate_ids_and_connection_run_and_global_capacity_fail_closed() -> None:
    allowed = capability(run_ids=("run-1", "run-2"))
    router = StudioOutputRouter(
        limits=StudioOutputLimits(
            maximum_subscriptions_per_connection=1,
            maximum_subscriptions=2,
            maximum_subscriptions_per_run=1,
        )
    )
    router.subscribe(
        "sub-1", "run-1", "stream-1", allowed, (), connection_id="connection-1"
    )
    with pytest.raises(SubscriptionConflictError):
        router.subscribe(
            "sub-1", "run-2", "stream-2", allowed, (), connection_id="connection-2"
        )
    with pytest.raises(SubscriptionCapacityError):
        router.subscribe(
            "sub-2", "run-2", "stream-2", allowed, (), connection_id="connection-1"
        )
    with pytest.raises(SubscriptionCapacityError):
        router.subscribe(
            "sub-2", "run-1", "stream-2", allowed, (), connection_id="connection-2"
        )
    router.subscribe(
        "sub-2", "run-2", "stream-2", allowed, (), connection_id="connection-2"
    )
    with pytest.raises(SubscriptionCapacityError):
        router.subscribe(
            "sub-3", "run-3", "stream-3", capability(run_ids=("run-3",)), ()
        )


def test_expired_lease_rejects_operations_and_can_be_released_by_connection() -> None:
    clock = Clock()
    router = StudioOutputRouter(
        limits=StudioOutputLimits(lease_seconds=5.0), clock=clock
    )
    allowed = capability()
    router.subscribe(
        "sub-1", "run-1", "stream-1", allowed, (), connection_id="connection-1"
    )
    clock.value += 6.0

    with pytest.raises(SubscriptionLeaseExpired):
        router.resume("sub-1", 1, HASHES[0], capability=allowed)

    assert router.unsubscribe_connection("connection-1") == ()


def test_retained_gap_is_scoped_and_public_buffer_never_contains_private_records() -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter(
        limits=StudioOutputLimits(
            maximum_retained_batches=1,
            maximum_retained_records=10,
            maximum_retained_bytes=1_000_000,
        )
    )
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    first = public_batch(1)
    second = public_batch(3)
    router.publish("run-1", first)
    router.acknowledge("sub-1", 1, first.content_hash, capability=allowed)
    router.publish("run-1", private_batch(2))
    router.publish("run-1", second)

    assert delivered[1].records == ()
    assert all(
        record.audience is SimulationOutputAudience.PUBLIC
        for view in delivered
        for record in view.records
    )
    with pytest.raises(SubscriptionHistoryGap) as raised:
        router.resume("sub-1", 1, first.content_hash, capability=allowed)
    gap = raised.value
    assert gap.current_sequence == 3
    assert gap.current_batch_hash == second.content_hash
    assert gap.snapshot_token.startswith("sha256:")
    assert set(gap.to_dict()) == {
        "current_sequence",
        "current_batch_hash",
        "snapshot_token",
    }
    assert "percept" not in str(gap.to_dict()).lower()


def test_owner_projection_requires_exact_agent_membership() -> None:
    router = StudioOutputRouter()
    allowed = capability(agent_ids=("alice",))
    with pytest.raises(SubscriptionAuthorizationError):
        router.subscribe(
            "sub-bob", "run-1", "stream-1", allowed, (), owner_agent_id="bob"
        )
    captured = []
    subscription = router.subscribe(
        "sub-alice",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        owner_agent_id="alice",
        on_output=captured.append,
    )
    router.publish("run-1", private_batch())

    assert subscription.audience_capability == SimulationAudienceCapability(
        SimulationOutputAudience.AGENT, "alice"
    )
    assert captured[0].records[0].owner_agent_id == "alice"


def test_delivered_oversize_batch_can_still_be_acknowledged() -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter(
        limits=StudioOutputLimits(maximum_retained_bytes=1)
    )
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    batch = public_batch(1)
    router.publish("run-1", batch)

    assert delivered[0].source_batch_hash == batch.content_hash
    router.acknowledge("sub-1", 1, batch.content_hash, capability=allowed)
    assert router.resume("sub-1", 1, batch.content_hash, capability=allowed) == ()
