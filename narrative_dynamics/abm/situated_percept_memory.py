"""SQLite persistence for sanitized V15 percept memories."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptionModel,
    SituatedPercept,
    SituatedPerceptFidelity,
)
from narrative_dynamics.abm.situated_percept_cognition import (
    project_situated_story_percepts,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    SituatedPerceptMemoryHit,
    SituatedPerceptMemoryIndexReport,
    SituatedPerceptMemoryPolicy,
    SituatedPerceptMemoryQuery,
    SituatedPerceptMemoryRecord,
    SituatedPerceptMemoryWriteReport,
    standard_situated_percept_memory_policy,
)
from narrative_dynamics.abm.situated_story import SituatedStory


class SituatedPerceptMemoryStorageError(RuntimeError):
    """Raised when the sanitized percept store cannot complete an operation."""


class SituatedPerceptMemoryConflictError(ValueError):
    """Raised when one private percept identity is reused with different content."""


_SCHEMA_VERSION = "1"

_SCHEMA = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS percept_memory_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS percept_memory_records (
    rowid INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    percept_id TEXT NOT NULL,
    perception_model_id TEXT NOT NULL,
    perception_model_hash TEXT NOT NULL,
    story_model_id TEXT NOT NULL,
    story_model_hash TEXT NOT NULL,
    projection_hash TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    source_event_hash TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    channels_json TEXT NOT NULL,
    fidelity TEXT NOT NULL,
    actor_agent_id TEXT,
    kind TEXT,
    place_id TEXT,
    outcome TEXT,
    details_json TEXT NOT NULL,
    details_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    salience REAL NOT NULL,
    policy_hash TEXT NOT NULL,
    summary TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE (agent_id, percept_id)
);

CREATE INDEX IF NOT EXISTS percept_memory_records_agent_round
ON percept_memory_records(agent_id, round_index, memory_id);

CREATE VIRTUAL TABLE IF NOT EXISTS percept_memory_fts USING fts5(
    summary,
    actor_agent_id,
    place_id,
    outcome,
    details_text,
    content='percept_memory_records',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS percept_memory_records_ai
AFTER INSERT ON percept_memory_records BEGIN
    INSERT INTO percept_memory_fts(
        rowid, summary, actor_agent_id, place_id, outcome, details_text
    ) VALUES (
        new.rowid, new.summary, new.actor_agent_id, new.place_id,
        new.outcome, new.details_text
    );
END;

CREATE TRIGGER IF NOT EXISTS percept_memory_records_ad
AFTER DELETE ON percept_memory_records BEGIN
    INSERT INTO percept_memory_fts(
        percept_memory_fts, rowid, summary, actor_agent_id,
        place_id, outcome, details_text
    ) VALUES (
        'delete', old.rowid, old.summary, old.actor_agent_id,
        old.place_id, old.outcome, old.details_text
    );
END;

CREATE TRIGGER IF NOT EXISTS percept_memory_records_au
AFTER UPDATE ON percept_memory_records BEGIN
    INSERT INTO percept_memory_fts(
        percept_memory_fts, rowid, summary, actor_agent_id,
        place_id, outcome, details_text
    ) VALUES (
        'delete', old.rowid, old.summary, old.actor_agent_id,
        old.place_id, old.outcome, old.details_text
    );
    INSERT INTO percept_memory_fts(
        rowid, summary, actor_agent_id, place_id, outcome, details_text
    ) VALUES (
        new.rowid, new.summary, new.actor_agent_id, new.place_id,
        new.outcome, new.details_text
    );
END;

INSERT OR IGNORE INTO percept_memory_metadata(key, value)
VALUES ('schema_version', '1');

COMMIT;
"""

_INSERT = """
INSERT INTO percept_memory_records (
    memory_id, source_hash, agent_id, percept_id, perception_model_id,
    perception_model_hash, story_model_id, story_model_hash, projection_hash,
    source_event_id, source_event_hash, round_index, channels_json, fidelity,
    actor_agent_id, kind, place_id, outcome, details_json, details_text,
    confidence, salience, policy_hash, summary, active
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""


def _database_path(value: str | Path) -> str:
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("percept memory database path must be non-empty")
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
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_hash(memory: SituatedPerceptMemoryRecord) -> str:
    payload = memory.to_dict()
    payload.pop("active")
    return stable_content_hash(payload)


def _details_text(memory: SituatedPerceptMemoryRecord) -> str:
    return " ".join(f"{item.name} {item.value}" for item in memory.details)


def _insert_values(memory: SituatedPerceptMemoryRecord) -> tuple[object, ...]:
    return (
        memory.memory_id,
        _source_hash(memory),
        memory.agent_id,
        memory.percept_id,
        memory.perception_model_id,
        memory.perception_model_hash,
        memory.story_model_id,
        memory.story_model_hash,
        memory.projection_hash,
        memory.source_event_id,
        memory.source_event_hash,
        memory.round_index,
        _canonical_json([item.value for item in memory.channels]),
        memory.fidelity.value,
        memory.actor_agent_id,
        None if memory.kind is None else memory.kind.value,
        memory.place_id,
        memory.outcome,
        _canonical_json([item.to_dict() for item in memory.details]),
        _details_text(memory),
        memory.confidence,
        memory.salience,
        memory.policy_hash,
        memory.summary,
        int(memory.active),
    )


def _row_to_memory(row: sqlite3.Row) -> SituatedPerceptMemoryRecord:
    return SituatedPerceptMemoryRecord(
        memory_id=row["memory_id"],
        agent_id=row["agent_id"],
        percept_id=row["percept_id"],
        perception_model_id=row["perception_model_id"],
        perception_model_hash=row["perception_model_hash"],
        story_model_id=row["story_model_id"],
        story_model_hash=row["story_model_hash"],
        projection_hash=row["projection_hash"],
        source_event_id=row["source_event_id"],
        source_event_hash=row["source_event_hash"],
        round_index=row["round_index"],
        channels=tuple(
            ObservationChannel(item) for item in json.loads(row["channels_json"])
        ),
        fidelity=SituatedPerceptFidelity(row["fidelity"]),
        actor_agent_id=row["actor_agent_id"],
        kind=None if row["kind"] is None else SituatedActionKind(row["kind"]),
        place_id=row["place_id"],
        outcome=row["outcome"],
        details=tuple(
            EvidenceFact(item["name"], item["value"])
            for item in json.loads(row["details_json"])
        ),
        confidence=row["confidence"],
        salience=row["salience"],
        policy_hash=row["policy_hash"],
        summary=row["summary"],
        active=bool(row["active"]),
    )


def initialize_situated_percept_memory(
    database_path: str | Path,
) -> SituatedPerceptMemoryIndexReport:
    try:
        with _transaction(database_path) as connection:
            metadata_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'percept_memory_metadata'"
            ).fetchone()
            if metadata_exists is not None:
                version = connection.execute(
                    "SELECT value FROM percept_memory_metadata "
                    "WHERE key = 'schema_version'"
                ).fetchone()
                if version is None or version["value"] != _SCHEMA_VERSION:
                    actual = "missing" if version is None else version["value"]
                    raise SituatedPerceptMemoryStorageError(
                        f"unsupported situated percept memory schema version {actual}"
                    )
            connection.executescript(_SCHEMA)
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM percept_memory_records"
            ).fetchone()["count"]
    except sqlite3.Error as error:
        raise SituatedPerceptMemoryStorageError(
            "failed to initialize situated percept memory"
        ) from error
    return SituatedPerceptMemoryIndexReport(count)


def _summary(percept: SituatedPercept) -> str:
    channels = ",".join(item.value for item in percept.channels)
    if percept.fidelity is SituatedPerceptFidelity.DETECTED:
        return f"round {percept.round_index}; detected event via {channels}"
    parts = [
        f"round {percept.round_index}",
        percept.actor_agent_id,
        percept.kind.value,
        f"at {percept.place_id}",
        percept.fidelity.value,
        f"via {channels}",
    ]
    if percept.fidelity is SituatedPerceptFidelity.EXACT:
        parts.append(f"outcome {percept.outcome}")
        parts.extend(f"{item.name} {item.value}" for item in percept.details)
    return "; ".join(parts)


def _memory_from_percept(
    perception_model: SituatedPerceptionModel,
    story: SituatedStory,
    projection_hash: str,
    percept: SituatedPercept,
    policy: SituatedPerceptMemoryPolicy,
) -> SituatedPerceptMemoryRecord:
    attribution = policy.for_fidelity(percept.fidelity)
    return SituatedPerceptMemoryRecord(
        memory_id=percept.percept_id,
        agent_id=percept.agent_id,
        percept_id=percept.percept_id,
        perception_model_id=perception_model.model_id,
        perception_model_hash=perception_model.content_hash,
        story_model_id=story.model_id,
        story_model_hash=story.model_hash,
        projection_hash=projection_hash,
        source_event_id=percept.source_event_id,
        source_event_hash=percept.source_event_hash,
        round_index=percept.round_index,
        channels=percept.channels,
        fidelity=percept.fidelity,
        actor_agent_id=percept.actor_agent_id,
        kind=percept.kind,
        place_id=percept.place_id,
        outcome=percept.outcome,
        details=percept.details,
        confidence=attribution.confidence,
        salience=attribution.salience,
        policy_hash=policy.content_hash,
        summary=_summary(percept),
    )


def ingest_situated_percept_story(
    database_path: str | Path,
    perception_model: SituatedPerceptionModel,
    story: SituatedStory,
    agent_id: str,
    policy: SituatedPerceptMemoryPolicy | None = None,
) -> SituatedPerceptMemoryWriteReport:
    """Derive and persist only one agent's sanitized story percepts."""

    selected_policy = (
        standard_situated_percept_memory_policy() if policy is None else policy
    )
    if not isinstance(selected_policy, SituatedPerceptMemoryPolicy):
        raise TypeError("percept story ingestion requires a percept memory policy")
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("percept story ingestion agent id must be non-empty")
    if agent_id not in {item.agent_id for item in perception_model.world_model.agents}:
        raise ValueError("percept story ingestion agent must belong to the world")
    projections = project_situated_story_percepts(perception_model, story)
    memories = tuple(
        _memory_from_percept(
            perception_model,
            story,
            projection.content_hash,
            percept,
            selected_policy,
        )
        for projection in projections
        for percept in projection.percepts
        if percept.agent_id == agent_id
    )
    initialize_situated_percept_memory(database_path)
    inserted = 0
    existing = 0
    try:
        with _transaction(database_path) as connection:
            for memory in memories:
                row = connection.execute(
                    "SELECT source_hash FROM percept_memory_records "
                    "WHERE agent_id = ? AND percept_id = ?",
                    (memory.agent_id, memory.percept_id),
                ).fetchone()
                source_hash = _source_hash(memory)
                if row is not None:
                    if row["source_hash"] != source_hash:
                        raise SituatedPerceptMemoryConflictError(
                            "percept memory identity has conflicting private content"
                        )
                    existing += 1
                    continue
                connection.execute(_INSERT, _insert_values(memory))
                inserted += 1
    except sqlite3.Error as error:
        raise SituatedPerceptMemoryStorageError(
            "failed to persist situated percept memories"
        ) from error
    return SituatedPerceptMemoryWriteReport(
        agent_id,
        inserted,
        existing,
        tuple(item.memory_id for item in memories),
    )


def list_situated_percept_memories(
    database_path: str | Path,
    agent_id: str,
    *,
    include_inactive: bool = False,
) -> tuple[SituatedPerceptMemoryRecord, ...]:
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("percept memory list agent id must be non-empty")
    if not isinstance(include_inactive, bool):
        raise TypeError("percept memory include inactive must be boolean")
    initialize_situated_percept_memory(database_path)
    try:
        with _transaction(database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM percept_memory_records WHERE agent_id = ? "
                + ("" if include_inactive else "AND active = 1 ")
                + "ORDER BY round_index, memory_id",
                (agent_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise SituatedPerceptMemoryStorageError(
            "failed to list situated percept memories"
        ) from error
    return tuple(_row_to_memory(row) for row in rows)


def _private_literal_rank(summary: str, text: str) -> float:
    """Rank within one private record without leaking corpus-wide statistics."""

    folded_summary = summary.casefold()
    return -float(folded_summary.count(text.casefold())) / max(len(folded_summary), 1)


def search_situated_percept_memories(
    database_path: str | Path,
    query: SituatedPerceptMemoryQuery,
) -> tuple[SituatedPerceptMemoryHit, ...]:
    """Search only sanitized rows owned by the query's agent."""

    if not isinstance(query, SituatedPerceptMemoryQuery):
        raise TypeError("percept memory search requires a SituatedPerceptMemoryQuery")
    initialize_situated_percept_memory(database_path)
    text_uses_fts = query.text is not None and len(query.text) >= 3
    if text_uses_fts:
        select = (
            "SELECT m.* FROM percept_memory_fts "
            "JOIN percept_memory_records AS m ON m.rowid = percept_memory_fts.rowid"
        )
        literal_phrase = '"' + query.text.replace('"', '""') + '"'
        clauses = ["percept_memory_fts MATCH ?", "m.agent_id = ?"]
        parameters: list[object] = [literal_phrase, query.agent_id]
    else:
        select = "SELECT m.* FROM percept_memory_records AS m"
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
    if query.fidelities:
        placeholders = ", ".join("?" for _ in query.fidelities)
        clauses.append(f"m.fidelity IN ({placeholders})")
        parameters.extend(item.value for item in query.fidelities)
    if query.event_kinds:
        placeholders = ", ".join("?" for _ in query.event_kinds)
        clauses.append(f"m.kind IN ({placeholders})")
        parameters.extend(item.value for item in query.event_kinds)
    if query.min_round is not None:
        clauses.append("m.round_index >= ?")
        parameters.append(query.min_round)
    if query.max_round is not None:
        clauses.append("m.round_index <= ?")
        parameters.append(query.max_round)
    if query.story_model_hash is not None:
        clauses.append("m.story_model_hash = ?")
        parameters.append(query.story_model_hash)
    clauses.append("m.confidence >= ?")
    parameters.append(query.min_confidence)
    statement = select + " WHERE " + " AND ".join(clauses)
    try:
        with _transaction(database_path) as connection:
            rows = connection.execute(statement, tuple(parameters)).fetchall()
    except sqlite3.Error as error:
        raise SituatedPerceptMemoryStorageError(
            "failed to search situated percept memories"
        ) from error
    excluded = set(query.excluded_memory_ids)
    included = set(query.included_memory_ids)
    required_channels = set(query.channels)
    hits = []
    for row in rows:
        memory = _row_to_memory(row)
        if memory.memory_id in excluded or (included and memory.memory_id not in included):
            continue
        if required_channels and not required_channels.intersection(memory.channels):
            continue
        hits.append(
            SituatedPerceptMemoryHit(
                memory,
                _private_literal_rank(memory.summary, query.text) if text_uses_fts else None,
            )
        )
    hits.sort(
        key=lambda item: (
            item.lexical_rank if item.lexical_rank is not None else 0.0,
            -item.memory.salience,
            -item.memory.round_index,
            item.memory.memory_id,
        )
    )
    return tuple(hits[: query.limit])


def set_situated_percept_memory_active(
    database_path: str | Path,
    agent_id: str,
    memory_id: str,
    active: bool,
) -> SituatedPerceptMemoryRecord:
    """Soft-deactivate or reactivate one memory owned by ``agent_id``."""

    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("percept memory scoped agent id must be non-empty")
    if not isinstance(memory_id, str) or not memory_id.strip():
        raise ValueError("percept memory id must be non-empty")
    if not isinstance(active, bool):
        raise TypeError("percept memory active must be boolean")
    initialize_situated_percept_memory(database_path)
    try:
        with _transaction(database_path) as connection:
            changed = connection.execute(
                "UPDATE percept_memory_records SET active = ? "
                "WHERE memory_id = ? AND agent_id = ?",
                (int(active), memory_id, agent_id),
            )
            if changed.rowcount != 1:
                raise ValueError("percept memory must belong to the scoped agent")
            row = connection.execute(
                "SELECT * FROM percept_memory_records "
                "WHERE memory_id = ? AND agent_id = ?",
                (memory_id, agent_id),
            ).fetchone()
    except sqlite3.Error as error:
        raise SituatedPerceptMemoryStorageError(
            "failed to update situated percept memory activation"
        ) from error
    return _row_to_memory(row)


def rebuild_situated_percept_memory_index(
    database_path: str | Path,
) -> SituatedPerceptMemoryIndexReport:
    """Rebuild derived FTS rows from authoritative sanitized memories."""

    initialize_situated_percept_memory(database_path)
    try:
        with _transaction(database_path) as connection:
            connection.execute(
                "INSERT INTO percept_memory_fts(percept_memory_fts) VALUES ('rebuild')"
            )
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM percept_memory_records"
            ).fetchone()["count"]
    except sqlite3.Error as error:
        raise SituatedPerceptMemoryStorageError(
            "failed to rebuild situated percept memory index"
        ) from error
    return SituatedPerceptMemoryIndexReport(count)


__all__ = (
    "SituatedPerceptMemoryStorageError",
    "SituatedPerceptMemoryConflictError",
    "initialize_situated_percept_memory",
    "ingest_situated_percept_story",
    "list_situated_percept_memories",
    "search_situated_percept_memories",
    "set_situated_percept_memory_active",
    "rebuild_situated_percept_memory_index",
)
