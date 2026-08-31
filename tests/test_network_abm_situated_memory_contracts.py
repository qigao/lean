from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_memory_contracts import (
    MemoryChannelPolicy,
    SituatedMemoryPolicy,
    SituatedMemoryQuery,
    SituatedMemoryRecord,
    standard_situated_memory_policy,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


def memory_record(*, details=(), cause_event_ids=(), active=True):
    return SituatedMemoryRecord(
        memory_id="r0001:e0001:o:alice",
        agent_id="alice",
        observation_id="r0001:e0001:o:alice",
        story_model_id="small-office",
        story_model_hash=HASH_A,
        event_id="r0001:e0001",
        event_hash=HASH_B,
        round_index=1,
        sequence=1,
        action_id="inspect-memo",
        kind=SituatedActionKind.INSPECT,
        actor_agent_id="alice",
        place_id="records",
        target_id="memo",
        success=True,
        outcome="inspected",
        details=details,
        cause_event_ids=cause_event_ids,
        channel=ObservationChannel.INSPECTION,
        confidence=1.0,
        salience=1.0,
        policy_hash=HASH_C,
        summary="round 1 alice inspect at records target memo outcome inspected",
        active=active,
    )


class SituatedMemoryPolicyTests(unittest.TestCase):
    def test_standard_policy_declares_every_channel_weight(self):
        policy = standard_situated_memory_policy()

        self.assertEqual(policy.policy_id, "situated-memory-standard")
        self.assertEqual(policy.version, "1")
        self.assertEqual(
            tuple((item.channel.value, item.confidence, item.salience) for item in policy.channels),
            (
                ("auditory", 0.7, 0.75),
                ("inspection", 1.0, 1.0),
                ("self", 1.0, 0.5),
                ("visual", 0.9, 0.65),
            ),
        )
        self.assertEqual(policy.for_channel(ObservationChannel.AUDITORY).confidence, 0.7)

    def test_policy_rejects_missing_or_duplicate_channel(self):
        complete = standard_situated_memory_policy().channels
        with self.assertRaisesRegex(ValueError, "exactly one policy for every observation channel"):
            SituatedMemoryPolicy("incomplete", "1", complete[:-1])
        with self.assertRaisesRegex(ValueError, "exactly one policy for every observation channel"):
            SituatedMemoryPolicy("duplicate", "1", complete + (complete[0],))

    def test_channel_policy_rejects_out_of_range_weight(self):
        with self.assertRaisesRegex(ValueError, "confidence must be between zero and one"):
            MemoryChannelPolicy(ObservationChannel.SELF, 1.01, 0.5)
        with self.assertRaisesRegex(ValueError, "salience must be between zero and one"):
            MemoryChannelPolicy(ObservationChannel.SELF, 1.0, -0.01)


class SituatedMemoryRecordTests(unittest.TestCase):
    def test_record_canonicalizes_details_and_causes(self):
        record = memory_record(
            details=(EvidenceFact("zeta", "last"), EvidenceFact("alpha", "first")),
            cause_event_ids=("event-z", "event-a"),
        )

        self.assertEqual(tuple(item.name for item in record.details), ("alpha", "zeta"))
        self.assertEqual(record.cause_event_ids, ("event-a", "event-z"))
        self.assertEqual(
            record.to_dict()["details"],
            [{"name": "alpha", "value": "first"}, {"name": "zeta", "value": "last"}],
        )

    def test_record_hash_is_stable_but_activation_is_auditable_state(self):
        first = memory_record(
            details=(EvidenceFact("zeta", "last"), EvidenceFact("alpha", "first")),
            cause_event_ids=("event-z", "event-a"),
        )
        reordered = memory_record(
            details=(EvidenceFact("alpha", "first"), EvidenceFact("zeta", "last")),
            cause_event_ids=("event-a", "event-z"),
        )

        self.assertEqual(first.content_hash, reordered.content_hash)
        self.assertNotEqual(first.content_hash, memory_record(active=False).content_hash)

    def test_record_rejects_mismatched_observer_identity(self):
        with self.assertRaisesRegex(ValueError, "memory id must equal observation id"):
            replace(memory_record(), memory_id="different")


class SituatedMemoryQueryTests(unittest.TestCase):
    def test_query_canonicalizes_enum_filters(self):
        query = SituatedMemoryQuery(
            "alice",
            text="重组消息",
            event_kinds=(SituatedActionKind.TELL, SituatedActionKind.INSPECT),
            channels=(ObservationChannel.VISUAL, ObservationChannel.AUDITORY),
            min_round=2,
            max_round=8,
            min_confidence=0.7,
            limit=50,
        )

        self.assertEqual(tuple(item.value for item in query.event_kinds), ("inspect", "tell"))
        self.assertEqual(tuple(item.value for item in query.channels), ("auditory", "visual"))
        self.assertEqual(query.text, "重组消息")

    def test_query_rejects_ambiguous_or_unbounded_input(self):
        with self.assertRaisesRegex(ValueError, "query text must be non-empty"):
            SituatedMemoryQuery("alice", text="  ")
        with self.assertRaisesRegex(ValueError, "minimum round cannot exceed maximum round"):
            SituatedMemoryQuery("alice", min_round=3, max_round=2)
        with self.assertRaisesRegex(ValueError, "limit must be between 1 and 1000"):
            SituatedMemoryQuery("alice", limit=0)
        with self.assertRaisesRegex(ValueError, "minimum confidence must be between zero and one"):
            SituatedMemoryQuery("alice", min_confidence=1.1)


if __name__ == "__main__":
    unittest.main()
