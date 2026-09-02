from __future__ import annotations

from dataclasses import replace
from threading import Event, Thread

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


def public_batch_range(first: int, last: int) -> SimulationOutputBatch:
    records = tuple(
        SimulationOutputRecord(
            "stream-1",
            HASHES[0],
            sequence,
            1,
            HASHES[2],
            SimulationOutputKind.COMMAND_RESULT,
            SimulationOutputAudience.PUBLIC,
            None,
            (),
            SimulationCommandResultPayload(
                f"command-{sequence:04d}", True, "accepted"
            ),
        )
        for sequence in range(first, last + 1)
    )
    return SimulationOutputBatch(
        "stream-1",
        HASHES[0],
        HASHES[1],
        HASHES[2],
        HASHES[5],
        first,
        last,
        records,
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
            "sub-bob", "run-1", "stream-1", allowed, (),
            audience=SimulationOutputAudience.AGENT,
            owner_agent_id="bob",
        )
    captured = []
    subscription = router.subscribe(
        "sub-alice",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        audience=SimulationOutputAudience.AGENT,
        owner_agent_id="alice",
        on_output=captured.append,
    )
    router.publish("run-1", private_batch())

    assert subscription.audience_capability == SimulationAudienceCapability(
        SimulationOutputAudience.AGENT, "alice"
    )
    assert captured[0].records[0].owner_agent_id == "alice"


def test_explicit_closed_audience_never_escalates_from_capability() -> None:
    router = StudioOutputRouter()
    privileged = capability(
        agent_ids=("alice",), permissions=("output.read", "state.network")
    )
    public = router.subscribe(
        "sub-public", "run-1", "stream-1", privileged, (),
        audience=SimulationOutputAudience.PUBLIC,
    )
    analyst = router.subscribe(
        "sub-analyst", "run-1", "stream-2", privileged, (),
        audience=SimulationOutputAudience.ANALYST,
    )

    assert public.audience is SimulationOutputAudience.PUBLIC
    assert public.audience_capability == SimulationAudienceCapability(
        SimulationOutputAudience.PUBLIC
    )
    assert analyst.audience is SimulationOutputAudience.ANALYST
    with pytest.raises(SubscriptionAuthorizationError):
        router.subscribe(
            "sub-denied-analyst", "run-1", "stream-3", capability(), (),
            audience=SimulationOutputAudience.ANALYST,
        )
    with pytest.raises((TypeError, ValueError)):
        router.subscribe(
            "sub-invalid", "run-1", "stream-4", privileged, (),
            audience=SimulationOutputAudience.OBJECTIVE,
        )
    with pytest.raises(SubscriptionAuthorizationError):
        router.subscribe(
            "sub-owner-on-public", "run-1", "stream-5", privileged, (),
            audience=SimulationOutputAudience.PUBLIC,
            owner_agent_id="alice",
        )


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


def test_exact_duplicate_publication_is_idempotent_and_hash_exact() -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter()
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    batch = public_batch_range(1, 1)

    assert router.publish("run-1", batch) == ("sub-1",)
    assert router.publish("run-1", batch) == ()
    assert tuple(view.source_batch_hash for view in delivered) == (batch.content_hash,)
    router.acknowledge("sub-1", 1, batch.content_hash, capability=allowed)
    with pytest.raises(SubscriptionStateError):
        router.acknowledge("sub-1", 1, HASHES[7], capability=allowed)


def test_delayed_exact_duplicate_is_deduplicated_not_treated_as_out_of_order() -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter()
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    first = public_batch_range(1, 1)
    second = public_batch_range(2, 2)
    router.publish("run-1", first)
    router.publish("run-1", second)

    assert router.publish("run-1", first) == ()
    assert tuple(view.last_sequence for view in delivered) == (1, 2)


def test_seen_identity_survives_replay_eviction() -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter(
        limits=StudioOutputLimits(maximum_retained_batches=1)
    )
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    first = public_batch_range(1, 1)
    second = public_batch_range(2, 2)
    router.publish("run-1", first)
    router.publish("run-1", second)

    assert router.publish("run-1", first) == ()
    assert tuple(view.source_batch_hash for view in delivered) == (
        first.content_hash,
        second.content_hash,
    )


def test_evicted_bounds_with_different_hash_still_reject() -> None:
    delivered = []
    router = StudioOutputRouter(
        limits=StudioOutputLimits(maximum_retained_batches=1)
    )
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        capability(),
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    first = public_batch_range(1, 1)
    second = public_batch_range(2, 2)
    altered_record = replace(
        first.records[0],
        payload=SimulationCommandResultPayload("command-altered", True, "accepted"),
    )
    altered = replace(first, records=(altered_record,))
    router.publish("run-1", first)
    router.publish("run-1", second)

    with pytest.raises(SubscriptionStateError):
        router.publish("run-1", altered)

    assert tuple(view.source_batch_hash for view in delivered) == (
        first.content_hash,
        second.content_hash,
    )


def test_seen_identity_capacity_rejects_atomically() -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter(
        limits=StudioOutputLimits(maximum_seen_batch_identities=2)
    )
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    first = public_batch_range(1, 1)
    second = public_batch_range(2, 2)
    rejected = public_batch_range(3, 3)
    router.publish("run-1", first)
    router.publish("run-1", second)

    with pytest.raises(SubscriptionCapacityError):
        router.publish("run-1", rejected)

    assert tuple(view.source_batch_hash for view in delivered) == (
        first.content_hash,
        second.content_hash,
    )
    router.acknowledge("sub-1", 2, second.content_hash, capability=allowed)
    assert router.resume("sub-1", 2, second.content_hash, capability=allowed) == ()


def test_seen_identity_limit_is_validated_and_serialized() -> None:
    limits = StudioOutputLimits(maximum_seen_batch_identities=7)

    assert limits.to_dict()["maximum_seen_batch_identities"] == 7
    with pytest.raises(ValueError):
        StudioOutputLimits(maximum_seen_batch_identities=0)


def test_same_sequence_different_hash_rejects_without_delivery() -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter()
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    first = public_batch_range(1, 1)
    altered_record = replace(
        first.records[0],
        payload=SimulationCommandResultPayload("command-altered", True, "accepted"),
    )
    altered = replace(first, records=(altered_record,))
    router.publish("run-1", first)

    with pytest.raises(SubscriptionStateError):
        router.publish("run-1", altered)

    assert tuple(view.source_batch_hash for view in delivered) == (first.content_hash,)


@pytest.mark.parametrize(
    ("first", "invalid"),
    (
        (public_batch_range(1, 2), public_batch_range(2, 3)),
        (public_batch_range(1, 1), public_batch_range(3, 3)),
        (public_batch_range(2, 2), public_batch_range(1, 1)),
    ),
)
def test_noncontiguous_publication_rejects_without_retention_or_callback(
    first: SimulationOutputBatch, invalid: SimulationOutputBatch
) -> None:
    delivered = []
    allowed = capability()
    router = StudioOutputRouter()
    router.subscribe(
        "sub-1",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        on_output=delivered.append,
    )
    router.publish("run-1", first)

    with pytest.raises(SubscriptionStateError):
        router.publish("run-1", invalid)

    assert tuple(view.source_batch_hash for view in delivered) == (first.content_hash,)
    router.acknowledge(
        "sub-1", first.last_sequence, first.content_hash, capability=allowed
    )
    assert router.resume(
        "sub-1", first.last_sequence, first.content_hash, capability=allowed
    ) == ()


def test_disconnect_rebinds_exact_released_subscription_and_replays_later_output() -> None:
    router = StudioOutputRouter()
    allowed = capability()
    first = public_batch_range(1, 1)
    second = public_batch_range(2, 2)
    router.subscribe(
        "sub-reconnect",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        connection_id="connection-1",
    )
    router.publish("run-1", first)
    router.acknowledge(
        "sub-reconnect",
        1,
        first.content_hash,
        capability=allowed,
        stream_id="stream-1",
    )
    assert router.unsubscribe_connection("connection-1") == ("sub-reconnect",)

    router.publish("run-1", second)
    replayed_live = []
    router.subscribe(
        "sub-reconnect",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        connection_id="connection-2",
        on_output=replayed_live.append,
    )
    replayed = router.resume(
        "sub-reconnect",
        1,
        first.content_hash,
        capability=allowed,
        stream_id="stream-1",
    )

    assert replayed_live == []
    assert tuple(item.source_batch_hash for item in replayed) == (second.content_hash,)


def test_publish_commits_once_when_released_subscription_rebinds_during_projection() -> None:
    class ProjectionBarrierRouter(StudioOutputRouter):
        def __init__(self) -> None:
            super().__init__()
            self.projecting = Event()
            self.continue_projection = Event()
            self.block_projection = False

        def _filtered_view(self, batch, subscription):
            if self.block_projection:
                self.projecting.set()
                assert self.continue_projection.wait(timeout=2.0)
            return super()._filtered_view(batch, subscription)

    router = ProjectionBarrierRouter()
    allowed = capability()
    first = public_batch_range(1, 1)
    second = public_batch_range(2, 2)
    third = public_batch_range(3, 3)
    router.subscribe(
        "sub-race", "run-1", "stream-1", allowed,
        tuple(SimulationOutputKind), connection_id="connection-1",
    )
    router.publish("run-1", first)
    router.acknowledge("sub-race", 1, first.content_hash, capability=allowed)
    assert router.unsubscribe_connection("connection-1") == ("sub-race",)
    router.block_projection = True

    delivered = []
    publish_result = []
    publish_error = []

    def publish_second() -> None:
        try:
            publish_result.append(router.publish("run-1", second))
        except BaseException as error:  # retain the worker failure for the main assertion
            publish_error.append(error)

    worker = Thread(target=publish_second)
    worker.start()
    assert router.projecting.wait(timeout=2.0)
    router.subscribe(
        "sub-race", "run-1", "stream-1", allowed,
        tuple(SimulationOutputKind), connection_id="connection-2",
        on_output=delivered.append,
    )
    router.continue_projection.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert publish_error == []
    assert publish_result == [("sub-race",)]
    assert [view.source_batch_hash for view in delivered] == [second.content_hash]
    assert router.publish("run-1", third) == ("sub-race",)
    assert [view.source_batch_hash for view in delivered] == [
        second.content_hash, third.content_hash,
    ]


def test_released_subscription_requires_exact_full_private_binding() -> None:
    router = StudioOutputRouter()
    allowed = capability(agent_ids=("alice",))
    router.subscribe(
        "sub-private",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        connection_id="connection-private",
        audience=SimulationOutputAudience.AGENT,
        owner_agent_id="alice",
    )
    router.publish("run-1", private_batch())
    router.unsubscribe_connection("connection-private")

    with pytest.raises(SubscriptionConflictError):
        router.subscribe(
            "sub-private",
            "run-1",
            "stream-1",
            allowed,
            tuple(SimulationOutputKind),
            connection_id="connection-public",
        )

    restored = router.subscribe(
        "sub-private",
        "run-1",
        "stream-1",
        allowed,
        tuple(SimulationOutputKind),
        connection_id="connection-restored",
        audience=SimulationOutputAudience.AGENT,
        owner_agent_id="alice",
    )
    assert restored.owner_agent_id == "alice"
    assert router.resume(
        "sub-private",
        1,
        private_batch().content_hash,
        capability=allowed,
        stream_id="stream-1",
    ) == ()


def test_released_recovery_expires_and_is_bounded_deterministically() -> None:
    clock = Clock()
    allowed = capability(run_ids=("run-1", "run-2", "run-3"))
    router = StudioOutputRouter(
        limits=StudioOutputLimits(
            maximum_released_subscriptions=1,
            lease_seconds=5.0,
        ),
        clock=clock,
    )
    first = public_batch_range(1, 1)
    router.subscribe(
        "sub-oldest", "run-1", "stream-1", allowed,
        tuple(SimulationOutputKind), connection_id="connection-oldest",
    )
    router.publish("run-1", first)
    router.acknowledge("sub-oldest", 1, first.content_hash, capability=allowed)
    router.unsubscribe_connection("connection-oldest")

    router.subscribe(
        "sub-newest", "run-2", "stream-2", allowed,
        tuple(SimulationOutputKind), connection_id="connection-newest",
    )
    second_stream = public_batch(1, stream_id="stream-2")
    router.publish("run-2", second_stream)
    router.acknowledge("sub-newest", 1, second_stream.content_hash, capability=allowed)
    router.unsubscribe_connection("connection-newest")

    router.subscribe(
        "sub-oldest", "run-1", "stream-1", allowed,
        tuple(SimulationOutputKind), connection_id="connection-fresh",
    )
    with pytest.raises(SubscriptionStateError):
        router.resume("sub-oldest", 1, first.content_hash, capability=allowed)
    router.unsubscribe("sub-oldest", capability=allowed)

    clock.value += 6.0
    router.subscribe(
        "sub-newest", "run-2", "stream-2", allowed,
        tuple(SimulationOutputKind), connection_id="connection-expired",
    )
    with pytest.raises(SubscriptionStateError):
        router.resume("sub-newest", 1, second_stream.content_hash, capability=allowed)


def test_released_limit_is_validated_and_serialized() -> None:
    limits = StudioOutputLimits(maximum_released_subscriptions=7)

    assert limits.to_dict()["maximum_released_subscriptions"] == 7
    with pytest.raises(ValueError):
        StudioOutputLimits(maximum_released_subscriptions=0)
