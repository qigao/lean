from dataclasses import replace
from contextlib import closing
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_memory import (
    SituatedMemoryConflictError,
    ingest_situated_memory,
    initialize_situated_memory,
    list_situated_memories,
    rebuild_situated_memory_index,
    search_situated_memories,
    set_situated_memory_active,
)
from narrative_dynamics.abm.situated_memory_contracts import (
    SituatedMemoryQuery,
    SituatedMemoryRecord,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


def record(
    agent_id,
    observation_id,
    event_id,
    *,
    round_index=1,
    sequence=1,
    action_id="inspect-memo",
    kind=SituatedActionKind.INSPECT,
    actor_agent_id="alice",
    place_id="records",
    target_id="memo",
    outcome="inspected",
    details=(EvidenceFact("message", "办公室重组消息已批准 / restructuring approved"),),
    causes=(),
    channel=ObservationChannel.INSPECTION,
    confidence=1.0,
    salience=1.0,
    summary="round 1 alice inspect at records target memo; 办公室重组消息已批准; restructuring approved",
):
    return SituatedMemoryRecord(
        memory_id=observation_id,
        agent_id=agent_id,
        observation_id=observation_id,
        story_model_id="small-office",
        story_model_hash=HASH_A,
        event_id=event_id,
        event_hash=HASH_B,
        round_index=round_index,
        sequence=sequence,
        action_id=action_id,
        kind=kind,
        actor_agent_id=actor_agent_id,
        place_id=place_id,
        target_id=target_id,
        success=True,
        outcome=outcome,
        details=details,
        cause_event_ids=causes,
        channel=channel,
        confidence=confidence,
        salience=salience,
        policy_hash=HASH_C,
        summary=summary,
    )


class SituatedMemoryDatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.database_path = f"{self.temporary.name}/memory.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()


class SituatedMemoryStorageTests(SituatedMemoryDatabaseTestCase):
    def test_initialization_creates_authoritative_schema_and_fts5(self):
        initialize_situated_memory(self.database_path)

        with closing(sqlite3.connect(self.database_path)) as connection:
            version = connection.execute(
                "SELECT value FROM memory_metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
            objects = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
                )
            }
        self.assertEqual(version, "1")
        self.assertTrue({"memory_records", "memory_fts", "memory_records_ai"}.issubset(objects))

    def test_ingestion_persists_complete_records_and_is_idempotent(self):
        memories = (
            record("alice", "obs-a", "event-a"),
            record(
                "alice",
                "obs-b",
                "event-b",
                round_index=2,
                sequence=2,
                kind=SituatedActionKind.TELL,
                action_id="tell-bob",
                target_id=None,
                outcome="told",
                causes=("event-a",),
                channel=ObservationChannel.SELF,
                confidence=1.0,
                salience=0.5,
                summary="round 2 alice tell at open; restructuring approved",
            ),
        )

        first = ingest_situated_memory(self.database_path, "alice", memories)
        second = ingest_situated_memory(self.database_path, "alice", memories)
        loaded = list_situated_memories(self.database_path, "alice")

        self.assertEqual((first.inserted_count, first.existing_count), (2, 0))
        self.assertEqual((second.inserted_count, second.existing_count), (0, 2))
        self.assertEqual(tuple(item.memory_id for item in loaded), ("obs-a", "obs-b"))
        self.assertEqual(loaded, memories)

    def test_ingestion_rejects_cross_agent_rows_and_identity_conflicts(self):
        original = record("alice", "obs-a", "event-a")
        ingest_situated_memory(self.database_path, "alice", (original,))

        with self.assertRaisesRegex(ValueError, "all memories must belong to the scoped agent"):
            ingest_situated_memory(
                self.database_path,
                "alice",
                (record("bob", "obs-b", "event-b"),),
            )
        with self.assertRaisesRegex(SituatedMemoryConflictError, "different content"):
            ingest_situated_memory(
                self.database_path,
                "alice",
                (replace(original, summary="changed historical content"),),
            )


class SituatedMemoryRetrievalTests(SituatedMemoryDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.alice_inspection = record("alice", "alice-inspect", "event-inspect")
        self.alice_telling = record(
            "alice",
            "alice-tell",
            "event-tell",
            round_index=4,
            sequence=1,
            action_id="tell-team",
            kind=SituatedActionKind.TELL,
            place_id="open",
            target_id=None,
            outcome="told",
            details=(EvidenceFact("message", "办公室重组消息已批准 / restructuring approved"),),
            causes=("event-inspect",),
            channel=ObservationChannel.SELF,
            salience=0.5,
            summary="round 4 alice tell at open; 办公室重组消息已批准; restructuring approved",
        )
        self.bob_telling = replace(
            self.alice_telling,
            memory_id="bob-heard",
            agent_id="bob",
            observation_id="bob-heard",
            channel=ObservationChannel.AUDITORY,
            confidence=0.7,
            salience=0.75,
        )
        ingest_situated_memory(self.database_path, "alice", (self.alice_inspection, self.alice_telling))
        ingest_situated_memory(self.database_path, "bob", (self.bob_telling,))

    def test_text_search_is_literal_substring_and_agent_scoped(self):
        alice = search_situated_memories(
            self.database_path, SituatedMemoryQuery("alice", text="重组消息")
        )
        bob = search_situated_memories(
            self.database_path, SituatedMemoryQuery("bob", text="structur")
        )
        short = search_situated_memories(
            self.database_path, SituatedMemoryQuery("alice", text="批")
        )
        hostile_literal = search_situated_memories(
            self.database_path, SituatedMemoryQuery("alice", text='" OR *')
        )

        self.assertEqual({item.memory.agent_id for item in alice}, {"alice"})
        self.assertEqual({item.memory.memory_id for item in alice}, {"alice-inspect", "alice-tell"})
        self.assertEqual(tuple(item.memory.memory_id for item in bob), ("bob-heard",))
        self.assertEqual({item.memory.memory_id for item in short}, {"alice-inspect", "alice-tell"})
        self.assertEqual(hostile_literal, ())
        self.assertTrue(all(item.lexical_rank is not None for item in alice + bob))
        self.assertTrue(all(item.lexical_rank is None for item in short))

    def test_structured_filters_are_combined_deterministically(self):
        hits = search_situated_memories(
            self.database_path,
            SituatedMemoryQuery(
                "alice",
                actor_agent_id="alice",
                place_id="open",
                event_kinds=(SituatedActionKind.TELL,),
                channels=(ObservationChannel.SELF,),
                min_round=3,
                max_round=5,
                min_confidence=0.9,
            ),
        )

        self.assertEqual(tuple(item.memory.memory_id for item in hits), ("alice-tell",))
        self.assertIsNone(hits[0].lexical_rank)

    def test_activation_is_private_soft_deletion(self):
        deactivated = set_situated_memory_active(
            self.database_path, "alice", "alice-inspect", False
        )

        self.assertFalse(deactivated.active)
        self.assertEqual(
            {item.memory_id for item in list_situated_memories(self.database_path, "alice")},
            {"alice-tell"},
        )
        self.assertEqual(
            {item.memory_id for item in list_situated_memories(self.database_path, "alice", include_inactive=True)},
            {"alice-inspect", "alice-tell"},
        )
        self.assertEqual(
            search_situated_memories(
                self.database_path, SituatedMemoryQuery("alice", text="重组消息")
            )[0].memory.memory_id,
            "alice-tell",
        )
        with self.assertRaisesRegex(ValueError, "memory must belong to the scoped agent"):
            set_situated_memory_active(self.database_path, "bob", "alice-inspect", True)

        reactivated = set_situated_memory_active(
            self.database_path, "alice", "alice-inspect", True
        )
        self.assertTrue(reactivated.active)

    def test_fts_rebuild_preserves_scoped_results(self):
        before = tuple(
            item.memory.memory_id
            for item in search_situated_memories(
                self.database_path, SituatedMemoryQuery("alice", text="重组消息")
            )
        )
        report = rebuild_situated_memory_index(self.database_path)
        after = tuple(
            item.memory.memory_id
            for item in search_situated_memories(
                self.database_path, SituatedMemoryQuery("alice", text="重组消息")
            )
        )

        self.assertEqual(report.indexed_count, 3)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
