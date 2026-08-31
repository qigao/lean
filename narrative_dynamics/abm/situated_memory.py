"""SQLite persistence and scoped retrieval for V12 situated memories."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_memory_contracts import (
    SituatedMemoryIndexReport,
    SituatedMemoryPolicy,
    SituatedMemoryQuery,
    SituatedMemoryRecord,
    SituatedMemorySearchHit,
    SituatedMemoryWriteReport,
    standard_situated_memory_policy,
)
from narrative_dynamics.abm.situated_story import (
    SituatedPerspectiveEvent,
    SituatedStory,
    perspective_timeline,
)


_SCHEMA_VERSION = "1"


class SituatedMemoryStorageError(RuntimeError):
    """Raised when the local SQLite build cannot provide V12 storage."""


class SituatedMemoryConflictError(ValueError):
    """Raised when an existing memory identity is reused with different history."""


_SCHEMA = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS memory_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_records (
    rowid INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    story_model_id TEXT NOT NULL,
    story_model_hash TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    action_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    actor_agent_id TEXT NOT NULL,
    place_id TEXT NOT NULL,
    target_id TEXT,
    success INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    details_json TEXT NOT NULL,
    details_text TEXT NOT NULL,
    causes_json TEXT NOT NULL,
    channel TEXT NOT NULL,
    confidence REAL NOT NULL,
    salience REAL NOT NULL,
    policy_hash TEXT NOT NULL,
    summary TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE (agent_id, observation_id)
);

CREATE INDEX IF NOT EXISTS memory_records_agent_round
ON memory_records(agent_id, round_index, sequence, memory_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    summary,
    actor_agent_id,
    place_id,
    outcome,
    details_text,
    content='memory_records',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS memory_records_ai AFTER INSERT ON memory_records BEGIN
    INSERT INTO memory_fts(rowid, summary, actor_agent_id, place_id, outcome, details_text)
    VALUES (new.rowid, new.summary, new.actor_agent_id, new.place_id, new.outcome, new.details_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_records_ad AFTER DELETE ON memory_records BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, summary, actor_agent_id, place_id, outcome, details_text)
    VALUES ('delete', old.rowid, old.summary, old.actor_agent_id, old.place_id, old.outcome, old.details_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_records_au AFTER UPDATE ON memory_records BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, summary, actor_agent_id, place_id, outcome, details_text)
    VALUES ('delete', old.rowid, old.summary, old.actor_agent_id, old.place_id, old.outcome, old.details_text);
    INSERT INTO memory_fts(rowid, summary, actor_agent_id, place_id, outcome, details_text)
    VALUES (new.rowid, new.summary, new.actor_agent_id, new.place_id, new.outcome, new.details_text);
END;

INSERT OR IGNORE INTO memory_metadata(key, value) VALUES ('schema_version', '1');

COMMIT;
"""


_INSERT = """
INSERT INTO memory_records (
    memory_id, source_hash, agent_id, observation_id, story_model_id,
    story_model_hash, event_id, event_hash, round_index, sequence, action_id,
    kind, actor_agent_id, place_id, target_id, success, outcome, details_json,
    details_text, causes_json, channel, confidence, salience, policy_hash,
    summary, active
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""


def _database_path(value: str | Path) -> str:
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("situated memory database path must be non-empty")
    return value


def _connect(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path(database_path))
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _transaction(database_path: str | Path):
    connection = _connect(database_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_hash(memory: SituatedMemoryRecord) -> str:
    payload = memory.to_dict()
    payload.pop("active")
    return stable_content_hash(payload)


def _details_text(memory: SituatedMemoryRecord) -> str:
    return " ".join(f"{item.name} {item.value}" for item in memory.details)


def _insert_values(memory: SituatedMemoryRecord) -> tuple[object, ...]:
    return (
        memory.memory_id,
        _source_hash(memory),
        memory.agent_id,
        memory.observation_id,
        memory.story_model_id,
        memory.story_model_hash,
        memory.event_id,
        memory.event_hash,
        memory.round_index,
        memory.sequence,
        memory.action_id,
        memory.kind.value,
        memory.actor_agent_id,
        memory.place_id,
        memory.target_id,
        int(memory.success),
        memory.outcome,
        _canonical_json([item.to_dict() for item in memory.details]),
        _details_text(memory),
        _canonical_json(list(memory.cause_event_ids)),
        memory.channel.value,
        memory.confidence,
        memory.salience,
        memory.policy_hash,
        memory.summary,
        int(memory.active),
    )


def _row_to_memory(row: sqlite3.Row) -> SituatedMemoryRecord:
    details = tuple(EvidenceFact(item["name"], item["value"]) for item in json.loads(row["details_json"]))
    return SituatedMemoryRecord(
        memory_id=row["memory_id"],
        agent_id=row["agent_id"],
        observation_id=row["observation_id"],
        story_model_id=row["story_model_id"],
        story_model_hash=row["story_model_hash"],
        event_id=row["event_id"],
        event_hash=row["event_hash"],
        round_index=row["round_index"],
        sequence=row["sequence"],
        action_id=row["action_id"],
        kind=SituatedActionKind(row["kind"]),
        actor_agent_id=row["actor_agent_id"],
        place_id=row["place_id"],
        target_id=row["target_id"],
        success=bool(row["success"]),
        outcome=row["outcome"],
        details=details,
        cause_event_ids=tuple(json.loads(row["causes_json"])),
        channel=ObservationChannel(row["channel"]),
        confidence=row["confidence"],
        salience=row["salience"],
        policy_hash=row["policy_hash"],
        summary=row["summary"],
        active=bool(row["active"]),
    )


def initialize_situated_memory(database_path: str | Path) -> SituatedMemoryIndexReport:
    """Create or validate the V12 schema at ``database_path``."""

    try:
        with _transaction(database_path) as connection:
            metadata_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_metadata'"
            ).fetchone()
            if metadata_exists is not None:
                existing = connection.execute(
                    "SELECT value FROM memory_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if existing is not None and existing["value"] != _SCHEMA_VERSION:
                    raise SituatedMemoryStorageError(
                        f"unsupported situated memory schema version {existing['value']}"
                    )
            connection.executescript(_SCHEMA)
            count = connection.execute("SELECT COUNT(*) AS count FROM memory_records").fetchone()["count"]
        return SituatedMemoryIndexReport(count)
    except sqlite3.Error as error:
        raise SituatedMemoryStorageError(
            "SQLite with FTS5 trigram support is required for situated memory"
        ) from error


def ingest_situated_memory(
    database_path: str | Path,
    agent_id: str,
    memories: tuple[SituatedMemoryRecord, ...],
) -> SituatedMemoryWriteReport:
    """Atomically persist immutable source records for one explicitly scoped agent."""

    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("situated memory scoped agent id must be non-empty")
    if not isinstance(memories, tuple) or any(not isinstance(item, SituatedMemoryRecord) for item in memories):
        raise TypeError("situated memories must be a tuple of SituatedMemoryRecord values")
    if any(item.agent_id != agent_id for item in memories):
        raise ValueError("all memories must belong to the scoped agent")
    ids = tuple(item.memory_id for item in memories)
    if len(set(ids)) != len(ids):
        raise ValueError("situated memory batch ids must be unique")

    initialize_situated_memory(database_path)
    inserted = 0
    existing = 0
    try:
        with _transaction(database_path) as connection:
            for memory in sorted(memories, key=lambda item: (item.round_index, item.sequence, item.memory_id)):
                stored = connection.execute(
                    "SELECT source_hash FROM memory_records WHERE agent_id = ? AND observation_id = ?",
                    (agent_id, memory.observation_id),
                ).fetchone()
                if stored is not None:
                    if stored["source_hash"] != _source_hash(memory):
                        raise SituatedMemoryConflictError(
                            f"memory {memory.memory_id} already exists for agent {agent_id} with different content"
                        )
                    existing += 1
                    continue
                connection.execute(_INSERT, _insert_values(memory))
                inserted += 1
    except sqlite3.Error as error:
        raise SituatedMemoryStorageError("failed to persist situated memories") from error
    return SituatedMemoryWriteReport(agent_id, inserted, existing, ids)


def _perspective_summary(item: SituatedPerspectiveEvent) -> str:
    event = item.event
    parts = [
        f"round {event.round_index}",
        event.actor_agent_id,
        event.kind.value,
        f"at {event.place_id}",
    ]
    if event.target_id is not None:
        parts.append(f"target {event.target_id}")
    parts.extend(("success" if event.success else "failed", f"outcome {event.outcome}"))
    parts.extend(f"{detail.name} {detail.value}" for detail in event.details)
    return "; ".join(parts)


def _perspective_memory(
    story: SituatedStory,
    item: SituatedPerspectiveEvent,
    policy: SituatedMemoryPolicy,
) -> SituatedMemoryRecord:
    event = item.event
    observation = item.observation
    attribution = policy.for_channel(observation.channel)
    return SituatedMemoryRecord(
        memory_id=observation.observation_id,
        agent_id=observation.agent_id,
        observation_id=observation.observation_id,
        story_model_id=story.model_id,
        story_model_hash=story.model_hash,
        event_id=event.event_id,
        event_hash=event.content_hash,
        round_index=event.round_index,
        sequence=event.sequence,
        action_id=event.action_id,
        kind=event.kind,
        actor_agent_id=event.actor_agent_id,
        place_id=event.place_id,
        target_id=event.target_id,
        success=event.success,
        outcome=event.outcome,
        details=event.details,
        cause_event_ids=event.cause_event_ids,
        channel=observation.channel,
        confidence=attribution.confidence,
        salience=attribution.salience,
        policy_hash=policy.content_hash,
        summary=_perspective_summary(item),
    )


def ingest_situated_story(
    database_path: str | Path,
    story: SituatedStory,
    agent_id: str,
    policy: SituatedMemoryPolicy | None = None,
) -> SituatedMemoryWriteReport:
    """Persist only the observations available in one agent's story perspective."""

    if not isinstance(story, SituatedStory):
        raise TypeError("situated memory story ingestion requires a SituatedStory")
    selected_policy = standard_situated_memory_policy() if policy is None else policy
    if not isinstance(selected_policy, SituatedMemoryPolicy):
        raise TypeError("situated memory story ingestion policy must be SituatedMemoryPolicy")
    perspective = perspective_timeline(story, agent_id)
    memories = tuple(_perspective_memory(story, item, selected_policy) for item in perspective)
    return ingest_situated_memory(database_path, agent_id, memories)


def list_situated_memories(
    database_path: str | Path,
    agent_id: str,
    *,
    include_inactive: bool = False,
) -> tuple[SituatedMemoryRecord, ...]:
    """Load one agent's memories in stable chronological order."""

    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("situated memory scoped agent id must be non-empty")
    if not isinstance(include_inactive, bool):
        raise TypeError("situated memory include inactive must be boolean")
    initialize_situated_memory(database_path)
    active_clause = "" if include_inactive else " AND active = 1"
    with _transaction(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM memory_records WHERE agent_id = ?"
            + active_clause
            + " ORDER BY round_index, sequence, memory_id",
            (agent_id,),
        ).fetchall()
    return tuple(_row_to_memory(row) for row in rows)


def _private_literal_rank(summary: str, text: str) -> float:
    """Return a lower-is-better score independent of every other agent's corpus."""

    folded_summary = summary.casefold()
    occurrence_count = folded_summary.count(text.casefold())
    return -float(occurrence_count) / max(len(folded_summary), 1)


def search_situated_memories(
    database_path: str | Path,
    query: SituatedMemoryQuery,
) -> tuple[SituatedMemorySearchHit, ...]:
    """Search one agent's records using literal text and structured filters."""

    if not isinstance(query, SituatedMemoryQuery):
        raise TypeError("situated memory search requires a SituatedMemoryQuery")
    initialize_situated_memory(database_path)

    text_uses_fts = query.text is not None and len(query.text) >= 3
    if text_uses_fts:
        select = (
            "SELECT m.*, NULL AS lexical_rank "
            "FROM memory_fts JOIN memory_records AS m ON m.rowid = memory_fts.rowid"
        )
        clauses = ["memory_fts MATCH ?", "m.agent_id = ?"]
        literal_phrase = '"' + query.text.replace('"', '""') + '"'
        parameters: list[object] = [literal_phrase, query.agent_id]
    else:
        select = "SELECT m.*, NULL AS lexical_rank FROM memory_records AS m"
        clauses = ["m.agent_id = ?"]
        parameters = [query.agent_id]
        if query.text is not None:
            escaped = query.text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append(
                "(m.summary LIKE ? ESCAPE '\\' OR m.actor_agent_id LIKE ? ESCAPE '\\' "
                "OR m.place_id LIKE ? ESCAPE '\\' OR m.outcome LIKE ? ESCAPE '\\' "
                "OR m.details_text LIKE ? ESCAPE '\\')"
            )
            parameters.extend([f"%{escaped}%"] * 5)

    if not query.include_inactive:
        clauses.append("m.active = 1")
    if query.actor_agent_id is not None:
        clauses.append("m.actor_agent_id = ?")
        parameters.append(query.actor_agent_id)
    if query.place_id is not None:
        clauses.append("m.place_id = ?")
        parameters.append(query.place_id)
    if query.event_kinds:
        placeholders = ", ".join("?" for _ in query.event_kinds)
        clauses.append(f"m.kind IN ({placeholders})")
        parameters.extend(item.value for item in query.event_kinds)
    if query.channels:
        placeholders = ", ".join("?" for _ in query.channels)
        clauses.append(f"m.channel IN ({placeholders})")
        parameters.extend(item.value for item in query.channels)
    if query.min_round is not None:
        clauses.append("m.round_index >= ?")
        parameters.append(query.min_round)
    if query.max_round is not None:
        clauses.append("m.round_index <= ?")
        parameters.append(query.max_round)
    clauses.append("m.confidence >= ?")
    parameters.append(query.min_confidence)

    statement = select + " WHERE " + " AND ".join(clauses)
    try:
        with _transaction(database_path) as connection:
            rows = connection.execute(statement, tuple(parameters)).fetchall()
    except sqlite3.Error as error:
        raise SituatedMemoryStorageError("failed to search situated memories") from error
    hits = [
        SituatedMemorySearchHit(
            _row_to_memory(row),
            _private_literal_rank(row["summary"], query.text) if text_uses_fts else None,
        )
        for row in rows
    ]
    hits.sort(key=lambda item: (
        item.lexical_rank if item.lexical_rank is not None else 0.0,
        -item.memory.salience,
        -item.memory.round_index,
        item.memory.memory_id,
    ))
    return tuple(hits[:query.limit])


def set_situated_memory_active(
    database_path: str | Path,
    agent_id: str,
    memory_id: str,
    active: bool,
) -> SituatedMemoryRecord:
    """Soft-deactivate or reactivate exactly one memory owned by ``agent_id``."""

    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("situated memory scoped agent id must be non-empty")
    if not isinstance(memory_id, str) or not memory_id.strip():
        raise ValueError("situated memory id must be non-empty")
    if not isinstance(active, bool):
        raise TypeError("situated memory active must be boolean")
    initialize_situated_memory(database_path)
    try:
        with _transaction(database_path) as connection:
            changed = connection.execute(
                "UPDATE memory_records SET active = ? WHERE memory_id = ? AND agent_id = ?",
                (int(active), memory_id, agent_id),
            )
            if changed.rowcount != 1:
                raise ValueError("memory must belong to the scoped agent")
            row = connection.execute(
                "SELECT * FROM memory_records WHERE memory_id = ? AND agent_id = ?",
                (memory_id, agent_id),
            ).fetchone()
    except sqlite3.Error as error:
        raise SituatedMemoryStorageError("failed to update situated memory activation") from error
    return _row_to_memory(row)


def rebuild_situated_memory_index(database_path: str | Path) -> SituatedMemoryIndexReport:
    """Rebuild the derived FTS5 index from authoritative memory rows."""

    initialize_situated_memory(database_path)
    try:
        with _transaction(database_path) as connection:
            connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            count = connection.execute("SELECT COUNT(*) AS count FROM memory_records").fetchone()["count"]
    except sqlite3.Error as error:
        raise SituatedMemoryStorageError("failed to rebuild situated memory index") from error
    return SituatedMemoryIndexReport(count)


__all__ = (
    "SituatedMemoryStorageError",
    "SituatedMemoryConflictError",
    "initialize_situated_memory",
    "ingest_situated_memory",
    "ingest_situated_story",
    "list_situated_memories",
    "search_situated_memories",
    "set_situated_memory_active",
    "rebuild_situated_memory_index",
)
