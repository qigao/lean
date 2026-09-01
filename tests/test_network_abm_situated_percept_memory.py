from dataclasses import fields, replace
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
)
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_memory import initialize_situated_memory
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from narrative_dynamics.abm.situated_percept_memory import (
    SituatedPerceptMemoryConflictError,
    SituatedPerceptMemoryStorageError,
    ingest_situated_percept_story,
    initialize_situated_percept_memory,
    list_situated_percept_memories,
    rebuild_situated_percept_memory_index,
    search_situated_percept_memories,
    set_situated_percept_memory_active,
    situated_percept_memory_schema_snapshot,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    SituatedPerceptMemoryFidelityPolicy,
    SituatedPerceptMemoryPolicy,
    SituatedPerceptMemoryQuery,
    SituatedPerceptMemoryRecord,
    standard_situated_percept_memory_policy,
)
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from tests.test_network_abm_situated_percept_cognition import (
    SECRET,
    initial_state,
    perception_model,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64


def exact_record(**changes) -> SituatedPerceptMemoryRecord:
    values = {
        "memory_id": "event-1:p:bob",
        "agent_id": "bob",
        "percept_id": "event-1:p:bob",
        "perception_model_id": "office-perception",
        "perception_model_hash": HASH_A,
        "story_model_id": "small-office",
        "story_model_hash": HASH_B,
        "projection_hash": HASH_C,
        "source_event_id": "event-1",
        "source_event_hash": HASH_D,
        "round_index": 1,
        "channels": (ObservationChannel.AUDITORY,),
        "fidelity": SituatedPerceptFidelity.EXACT,
        "actor_agent_id": "alice",
        "kind": SituatedActionKind.TELL,
        "place_id": "records",
        "outcome": "told",
        "details": (EvidenceFact("message", "The restructuring is approved."),),
        "confidence": 1.0,
        "salience": 0.8,
        "policy_hash": HASH_A,
        "summary": "round 1; alice; tell; at records; outcome told",
    }
    values.update(changes)
    return SituatedPerceptMemoryRecord(**values)


class SituatedPerceptMemoryContractTests(unittest.TestCase):
    def test_public_schema_snapshot_is_stable_and_complete(self) -> None:
        snapshot = situated_percept_memory_schema_snapshot()

        self.assertIsInstance(snapshot, tuple)
        self.assertEqual(
            {item[1] for item in snapshot},
            {
                "percept_memory_fts",
                "percept_memory_fts_config",
                "percept_memory_fts_data",
                "percept_memory_fts_docsize",
                "percept_memory_fts_idx",
                "percept_memory_metadata",
                "percept_memory_records",
                "percept_memory_records_ad",
                "percept_memory_records_ai",
                "percept_memory_records_au",
                "percept_memory_records_agent_round",
            },
        )
        self.assertEqual(snapshot, situated_percept_memory_schema_snapshot())

    def test_policy_requires_exactly_one_entry_per_fidelity(self) -> None:
        policy = standard_situated_percept_memory_policy()

        self.assertEqual(
            tuple(item.fidelity for item in policy.fidelities),
            (
                SituatedPerceptFidelity.DETECTED,
                SituatedPerceptFidelity.EXACT,
                SituatedPerceptFidelity.IDENTIFIED,
            ),
        )
        self.assertEqual(policy.for_fidelity(SituatedPerceptFidelity.DETECTED).confidence, 0.25)
        with self.assertRaisesRegex(ValueError, "every percept fidelity"):
            SituatedPerceptMemoryPolicy(
                "incomplete",
                "1",
                (SituatedPerceptMemoryFidelityPolicy(
                    SituatedPerceptFidelity.EXACT,
                    1.0,
                    1.0,
                ),),
            )

    def test_record_shape_has_no_undisclosed_objective_event_fields(self) -> None:
        names = {item.name for item in fields(SituatedPerceptMemoryRecord)}

        self.assertFalse({
            "event",
            "action_id",
            "target_id",
            "success",
            "cause_event_ids",
            "database_path",
            "rowid",
        } & names)

    def test_record_enforces_percept_fidelity_disclosure(self) -> None:
        detected = exact_record(
            memory_id="event-1:p:carol",
            agent_id="carol",
            percept_id="event-1:p:carol",
            fidelity=SituatedPerceptFidelity.DETECTED,
            actor_agent_id=None,
            kind=None,
            place_id=None,
            outcome=None,
            details=(),
        )
        identified = exact_record(
            fidelity=SituatedPerceptFidelity.IDENTIFIED,
            outcome=None,
            details=(),
        )

        self.assertIsNone(detected.actor_agent_id)
        self.assertIsNone(identified.outcome)
        with self.assertRaisesRegex(ValueError, "detected"):
            replace(detected, actor_agent_id="alice")
        with self.assertRaisesRegex(ValueError, "identified"):
            replace(identified, details=(EvidenceFact("message", "secret"),))

    def test_record_identity_and_hash_inputs_are_canonical(self) -> None:
        record = exact_record(
            channels=(ObservationChannel.VISUAL, ObservationChannel.AUDITORY),
            details=(
                EvidenceFact("z", "last"),
                EvidenceFact("a", "first"),
            ),
        )

        self.assertEqual(
            record.channels,
            (ObservationChannel.AUDITORY, ObservationChannel.VISUAL),
        )
        self.assertEqual(tuple(item.name for item in record.details), ("a", "z"))
        self.assertEqual(record, replace(record, channels=tuple(reversed(record.channels))))
        self.assertEqual(record.content_hash, replace(record, channels=tuple(reversed(record.channels))).content_hash)
        with self.assertRaisesRegex(ValueError, "memory id must equal percept id"):
            replace(record, memory_id="different")


def private_story(perception, *, door_open: bool):
    story = initialize_situated_story(
        perception.world_model,
        initial_state(perception, door_open=door_open),
    )
    story = advance_situated_story(
        perception.world_model,
        story,
        (SituatedActionIntent(
            "alice-inspect",
            "alice",
            SituatedActionKind.INSPECT,
            "memo",
        ),),
    )
    inspected = next(
        item
        for item in story.rounds[-1].events
        if item.actor_agent_id == "alice" and item.kind is SituatedActionKind.INSPECT
    )
    return advance_situated_story(
        perception.world_model,
        story,
        (SituatedActionIntent(
            "alice-tell",
            "alice",
            SituatedActionKind.TELL,
            message=SECRET,
            source_event_ids=(inspected.event_id,),
        ),),
    )


class SituatedPerceptMemoryStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.database = f"{self.temporary.name}/percepts.sqlite3"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_incompatible_schema_is_rejected_before_database_modification(self) -> None:
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "CREATE TABLE percept_memory_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO percept_memory_metadata(key, value) VALUES ('schema_version', '999')"
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(SituatedPerceptMemoryStorageError, "schema version 999"):
            initialize_situated_percept_memory(self.database)

        connection = sqlite3.connect(self.database)
        try:
            tables = tuple(
                item[0]
                for item in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                )
            )
            version = connection.execute(
                "SELECT value FROM percept_memory_metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(tables, ("percept_memory_metadata",))
        self.assertEqual(version, "999")

    def test_story_ingestion_persists_only_disclosed_private_fields(self) -> None:
        detected_model = perception_model()
        detected_story = private_story(detected_model, door_open=False)
        identified_model = perception_model(visual=True)
        identified_story = private_story(identified_model, door_open=False)

        initialize_situated_percept_memory(self.database)
        alice_report = ingest_situated_percept_story(
            self.database,
            detected_model,
            detected_story,
            "alice",
        )
        bob_report = ingest_situated_percept_story(
            self.database,
            detected_model,
            detected_story,
            "bob",
        )
        carol_report = ingest_situated_percept_story(
            self.database,
            identified_model,
            identified_story,
            "carol",
        )

        self.assertGreater(alice_report.inserted_count, 0)
        self.assertGreater(bob_report.inserted_count, 0)
        self.assertGreater(carol_report.inserted_count, 0)
        alice = list_situated_percept_memories(self.database, "alice")
        bob = list_situated_percept_memories(self.database, "bob")
        carol = list_situated_percept_memories(self.database, "carol")
        inspection_id = next(
            item.source_event_id
            for item in alice
            if item.kind is SituatedActionKind.INSPECT
        )
        self.assertNotIn(inspection_id, {item.source_event_id for item in bob})
        self.assertNotIn(inspection_id, {item.source_event_id for item in carol})

        bob_tell = next(
            item for item in bob
            if item.round_index == 2 and item.fidelity is SituatedPerceptFidelity.DETECTED
        )
        self.assertIsNone(bob_tell.actor_agent_id)
        self.assertIsNone(bob_tell.kind)
        self.assertIsNone(bob_tell.place_id)
        self.assertIsNone(bob_tell.outcome)
        self.assertEqual(bob_tell.details, ())
        self.assertNotIn(SECRET, bob_tell.summary)

        carol_tell = next(
            item for item in carol
            if item.round_index == 2 and item.fidelity is SituatedPerceptFidelity.IDENTIFIED
        )
        self.assertEqual(carol_tell.actor_agent_id, "alice")
        self.assertIs(carol_tell.kind, SituatedActionKind.TELL)
        self.assertEqual(carol_tell.place_id, "records")
        self.assertIsNone(carol_tell.outcome)
        self.assertEqual(carol_tell.details, ())

        connection = sqlite3.connect(self.database)
        try:
            row = connection.execute(
                "SELECT * FROM percept_memory_records WHERE agent_id = ? AND percept_id = ?",
                ("bob", bob_tell.percept_id),
            ).fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(row)
        self.assertNotIn(SECRET, repr(tuple(row)))

    def test_ingestion_is_idempotent_and_conflicts_on_same_private_identity(self) -> None:
        model = perception_model()
        story = private_story(model, door_open=False)

        first = ingest_situated_percept_story(self.database, model, story, "bob")
        second = ingest_situated_percept_story(self.database, model, story, "bob")

        self.assertGreater(first.inserted_count, 0)
        self.assertEqual(second.inserted_count, 0)
        self.assertEqual(second.existing_count, first.inserted_count)
        conflicting = perception_model(visual=True)
        with self.assertRaises(SituatedPerceptMemoryConflictError):
            ingest_situated_percept_story(
                self.database,
                conflicting,
                private_story(conflicting, door_open=False),
                "bob",
            )

    def test_tampered_authoritative_row_is_rejected_by_every_read_path(self) -> None:
        model = perception_model()
        story = private_story(model, door_open=False)
        ingest_situated_percept_story(self.database, model, story, "bob")
        detected = next(
            item
            for item in list_situated_percept_memories(self.database, "bob")
            if item.fidelity is SituatedPerceptFidelity.DETECTED
        )
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE percept_memory_records SET summary = ?, confidence = 1.0 "
                "WHERE agent_id = ? AND percept_id = ?",
                (SECRET, "bob", detected.percept_id),
            )
            connection.commit()
        finally:
            connection.close()

        operations = (
            lambda: list_situated_percept_memories(self.database, "bob"),
            lambda: search_situated_percept_memories(
                self.database,
                SituatedPerceptMemoryQuery("bob", text=SECRET),
            ),
            lambda: ingest_situated_percept_story(
                self.database, model, story, "bob"
            ),
            lambda: set_situated_percept_memory_active(
                self.database, "bob", detected.memory_id, False
            ),
            lambda: rebuild_situated_percept_memory_index(self.database),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    SituatedPerceptMemoryConflictError,
                    "integrity",
                ):
                    operation()

    def test_percept_tables_coexist_without_mutating_legacy_memory_rows(self) -> None:
        initialize_situated_memory(self.database)
        model = perception_model()
        story = private_story(model, door_open=False)

        ingest_situated_percept_story(self.database, model, story, "bob")

        connection = sqlite3.connect(self.database)
        try:
            legacy_count = connection.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
            tables = {
                item[0]
                for item in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
        finally:
            connection.close()
        self.assertEqual(legacy_count, 0)
        self.assertIn("percept_memory_records", tables)
        self.assertIn("memory_records", tables)


class SituatedPerceptMemoryQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.database = f"{self.temporary.name}/percepts.sqlite3"
        self.model = perception_model()
        self.story = private_story(self.model, door_open=False)
        for agent_id in ("alice", "bob"):
            ingest_situated_percept_story(
                self.database,
                self.model,
                self.story,
                agent_id,
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_search_is_private_and_detected_memory_cannot_match_secret_text(self) -> None:
        bob_secret = search_situated_percept_memories(
            self.database,
            SituatedPerceptMemoryQuery("bob", text="restructuring approved"),
        )
        alice_secret = search_situated_percept_memories(
            self.database,
            SituatedPerceptMemoryQuery("alice", text="restructuring approved"),
        )
        bob_detected = search_situated_percept_memories(
            self.database,
            SituatedPerceptMemoryQuery(
                "bob",
                text="detected",
                fidelities=(SituatedPerceptFidelity.DETECTED,),
                channels=(ObservationChannel.AUDITORY,),
            ),
        )

        self.assertEqual(bob_secret, ())
        self.assertTrue(alice_secret)
        self.assertEqual(len(bob_detected), 1)
        self.assertIs(bob_detected[0].memory.fidelity, SituatedPerceptFidelity.DETECTED)
        self.assertIsNotNone(bob_detected[0].lexical_rank)

    def test_filters_activation_restart_and_rebuild_preserve_authoritative_rows(self) -> None:
        before = list_situated_percept_memories(self.database, "alice")
        tell = next(item for item in before if item.kind is SituatedActionKind.TELL)

        deactivated = set_situated_percept_memory_active(
            self.database,
            "alice",
            tell.memory_id,
            False,
        )
        visible = list_situated_percept_memories(self.database, "alice")
        all_rows = list_situated_percept_memories(
            self.database,
            "alice",
            include_inactive=True,
        )
        rebuilt = rebuild_situated_percept_memory_index(self.database)
        restarted = list_situated_percept_memories(
            self.database,
            "alice",
            include_inactive=True,
        )

        self.assertFalse(deactivated.active)
        self.assertNotIn(tell.memory_id, {item.memory_id for item in visible})
        self.assertEqual(len(all_rows), len(before))
        self.assertEqual(rebuilt.record_count, len(before) + len(
            list_situated_percept_memories(
                self.database,
                "bob",
                include_inactive=True,
            )
        ))
        self.assertEqual(all_rows, restarted)
        exact_only = search_situated_percept_memories(
            self.database,
            SituatedPerceptMemoryQuery(
                "alice",
                fidelities=(SituatedPerceptFidelity.EXACT,),
                event_kinds=(SituatedActionKind.INSPECT,),
                max_round=1,
                story_model_hash=self.story.model_hash,
                excluded_memory_ids=(tell.memory_id,),
            ),
        )
        self.assertEqual(len(exact_only), 1)
        self.assertIs(exact_only[0].memory.kind, SituatedActionKind.INSPECT)


if __name__ == "__main__":
    unittest.main()
