"""SQLite persistence for revisioned World Studio scenario projects."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import sqlite3
from typing import Protocol, runtime_checkable

from narrative_dynamics.studio.contracts import (
    DraftOperationKind,
    ScenarioDiagnostic,
    ScenarioDiagnosticReport,
    ScenarioDraftDocument,
    ScenarioDraftOperation,
    ScenarioDraftOperationResult,
    ScenarioDraftSnapshot,
    ScenarioWorkspaceLimits,
)


class ScenarioProjectError(ValueError):
    """Base for stable, path-free project-authority failures."""


class ScenarioProjectNotFoundError(ScenarioProjectError):
    pass


class ScenarioProjectConflictError(ScenarioProjectError):
    pass


class ScenarioProjectCapacityError(ScenarioProjectError):
    pass


class ScenarioProjectStorageError(ScenarioProjectError):
    pass


class ScenarioProjectValidationError(ScenarioProjectError):
    def __init__(
        self,
        message: str,
        diagnostic_report: ScenarioDiagnosticReport | None = None,
    ) -> None:
        self.diagnostic_report = diagnostic_report
        super().__init__(message)


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _document_from_dict(value: Mapping[str, object]) -> ScenarioDraftDocument:
    return ScenarioDraftDocument(
        value["role"],  # type: ignore[arg-type]
        value.get("logical_id"),  # type: ignore[arg-type]
        value["value"],  # type: ignore[arg-type]
        value["content_hash"],  # type: ignore[arg-type]
    )


def _diagnostic_from_dict(value: Mapping[str, object]) -> ScenarioDiagnostic:
    return ScenarioDiagnostic(
        value["severity"],  # type: ignore[arg-type]
        value["code"],  # type: ignore[arg-type]
        value["document_role"],  # type: ignore[arg-type]
        value.get("logical_id"),  # type: ignore[arg-type]
        value["pointer"],  # type: ignore[arg-type]
        tuple(value.get("related_ids", ())),  # type: ignore[arg-type]
        value.get("message_key"),  # type: ignore[arg-type]
    )


def _report_from_dict(value: Mapping[str, object]) -> ScenarioDiagnosticReport:
    raw = value.get("diagnostics", ())
    return ScenarioDiagnosticReport(
        value["project_id"],  # type: ignore[arg-type]
        value["revision"],  # type: ignore[arg-type]
        tuple(_diagnostic_from_dict(item) for item in raw),  # type: ignore[arg-type]
    )


def _snapshot_from_dict(value: Mapping[str, object]) -> ScenarioDraftSnapshot:
    raw = value.get("documents", ())
    return ScenarioDraftSnapshot(
        value["project_id"],  # type: ignore[arg-type]
        value["revision"],  # type: ignore[arg-type]
        tuple(_document_from_dict(item) for item in raw),  # type: ignore[arg-type]
        value["layout"],  # type: ignore[arg-type]
        value["diagnostic_report_hash"],  # type: ignore[arg-type]
        value.get("compiled_scenario_hash"),  # type: ignore[arg-type]
        scenario_id=value.get("scenario_id"),  # type: ignore[arg-type]
        version=value.get("version"),  # type: ignore[arg-type]
    )


def _result_from_dict(value: Mapping[str, object]) -> ScenarioDraftOperationResult:
    return ScenarioDraftOperationResult(
        value["operation_hash"],  # type: ignore[arg-type]
        value["prior_revision"],  # type: ignore[arg-type]
        _snapshot_from_dict(value["next_snapshot"]),  # type: ignore[arg-type]
        _report_from_dict(value["diagnostic_report"]),  # type: ignore[arg-type]
    )


@runtime_checkable
class ScenarioProjectStore(Protocol):
    def create(self, project_id: str, snapshot: ScenarioDraftSnapshot) -> None: ...

    def load(self, project_id: str) -> ScenarioDraftSnapshot: ...

    def load_authority(
        self, project_id: str
    ) -> tuple[ScenarioDraftSnapshot, ScenarioDiagnosticReport]: ...

    def apply(
        self,
        operation: ScenarioDraftOperation,
        next_snapshot: ScenarioDraftSnapshot,
        report: ScenarioDiagnosticReport,
    ) -> ScenarioDraftOperationResult: ...

    def operation_result(
        self, project_id: str, idempotency_key: str
    ) -> ScenarioDraftOperationResult | None: ...

    def undo(
        self, project_id: str, expected_revision: int, expected_hash: str
    ) -> ScenarioDraftSnapshot: ...

    def redo(
        self, project_id: str, expected_revision: int, expected_hash: str
    ) -> ScenarioDraftSnapshot: ...


class SQLiteScenarioProjectStore:
    def __init__(self, database_path: str | Path, limits: ScenarioWorkspaceLimits) -> None:
        if not isinstance(limits, ScenarioWorkspaceLimits):
            raise TypeError("scenario workspace limits must be ScenarioWorkspaceLimits")
        if isinstance(database_path, Path):
            path = database_path
        elif isinstance(database_path, str) and database_path.strip():
            path = Path(database_path)
        else:
            raise ValueError("scenario project database path must be non-empty")
        folded = str(path).casefold().replace(" ", "")
        if str(path) == ":memory:" or (
            folded.startswith("file:") and "mode=memory" in folded
        ):
            raise ValueError("scenario project database must be file-backed")
        self._path = path.resolve()
        self._limits = limits
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                self._initialize(connection)
        except (OSError, sqlite3.Error):
            raise ScenarioProjectStorageError(
                "scenario project storage is unavailable"
            ) from None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS studio_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO studio_metadata(key, value)
            VALUES ('schema_version', '1');
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                report_json TEXT NOT NULL,
                journal_cursor INTEGER,
                redo_entry INTEGER,
                accepted_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS documents (
                project_id TEXT NOT NULL,
                role TEXT NOT NULL,
                logical_id TEXT NOT NULL,
                value_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY(project_id, role, logical_id),
                FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS diagnostics (
                project_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                diagnostic_json TEXT NOT NULL,
                PRIMARY KEY(project_id, ordinal),
                FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS operations (
                project_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                operation_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                PRIMARY KEY(project_id, operation_id),
                UNIQUE(project_id, idempotency_key),
                FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS journal (
                project_id TEXT NOT NULL,
                entry_index INTEGER NOT NULL,
                parent_index INTEGER,
                before_snapshot_json TEXT NOT NULL,
                before_report_json TEXT NOT NULL,
                after_snapshot_json TEXT NOT NULL,
                after_report_json TEXT NOT NULL,
                PRIMARY KEY(project_id, entry_index),
                FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            );
            """
        )

    @staticmethod
    def _project_row(connection: sqlite3.Connection, project_id: str):
        return connection.execute(
            "SELECT snapshot_json, report_json, journal_cursor, redo_entry, accepted_count "
            "FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()

    @staticmethod
    def _decode_snapshot(text: str) -> ScenarioDraftSnapshot:
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError
        return _snapshot_from_dict(value)

    @staticmethod
    def _decode_report(text: str) -> ScenarioDiagnosticReport:
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError
        return _report_from_dict(value)

    @staticmethod
    def _write_dependents(
        connection: sqlite3.Connection,
        snapshot: ScenarioDraftSnapshot,
        report: ScenarioDiagnosticReport,
    ) -> None:
        connection.execute("DELETE FROM documents WHERE project_id = ?", (snapshot.project_id,))
        connection.executemany(
            "INSERT INTO documents(project_id, role, logical_id, value_json, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (
                    snapshot.project_id,
                    document.role,
                    document.logical_id or "",
                    _json_text(document.to_dict()["value"]),
                    document.content_hash,
                )
                for document in snapshot.documents
            ),
        )
        connection.execute("DELETE FROM diagnostics WHERE project_id = ?", (snapshot.project_id,))
        connection.executemany(
            "INSERT INTO diagnostics(project_id, ordinal, diagnostic_json) VALUES (?, ?, ?)",
            (
                (snapshot.project_id, index, _json_text(item.to_dict()))
                for index, item in enumerate(report.diagnostics)
            ),
        )

    def create(self, project_id: str, snapshot: ScenarioDraftSnapshot) -> None:
        if project_id != snapshot.project_id:
            raise ValueError("scenario project identities must match")
        report = ScenarioDiagnosticReport(
            project_id,
            snapshot.revision,
            (
                ScenarioDiagnostic(
                    "error", "manifest_missing", "package", None, ""
                ),
            ),
        )
        if snapshot.diagnostic_report_hash != report.content_hash:
            raise ValueError("initial scenario diagnostic hash must match")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO projects(project_id, snapshot_json, report_json) VALUES (?, ?, ?)",
                    (project_id, _json_text(snapshot.to_dict()), _json_text(report.to_dict())),
                )
                self._write_dependents(connection, snapshot, report)
        except sqlite3.IntegrityError:
            raise ScenarioProjectConflictError("scenario project already exists") from None
        except (OSError, sqlite3.Error, ValueError, TypeError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    def load(self, project_id: str) -> ScenarioDraftSnapshot:
        try:
            with self._connect() as connection:
                row = self._project_row(connection, project_id)
            if row is None:
                raise ScenarioProjectNotFoundError("scenario project is unknown")
            return self._decode_snapshot(row[0])
        except ScenarioProjectNotFoundError:
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    def load_authority(
        self, project_id: str
    ) -> tuple[ScenarioDraftSnapshot, ScenarioDiagnosticReport]:
        """Decode one snapshot/report pair from the same SQLite row read."""

        try:
            with self._connect() as connection:
                row = self._project_row(connection, project_id)
            if row is None:
                raise ScenarioProjectNotFoundError("scenario project is unknown")
            snapshot = self._decode_snapshot(row[0])
            report = self._decode_report(row[1])
            self._validate_snapshot_report(snapshot, report)
            return snapshot, report
        except ScenarioProjectNotFoundError:
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    def diagnostic_report(self, project_id: str) -> ScenarioDiagnosticReport:
        try:
            with self._connect() as connection:
                row = self._project_row(connection, project_id)
            if row is None:
                raise ScenarioProjectNotFoundError("scenario project is unknown")
            return self._decode_report(row[1])
        except (ScenarioProjectNotFoundError, ScenarioProjectStorageError):
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    def operation_result(
        self, project_id: str, idempotency_key: str
    ) -> ScenarioDraftOperationResult | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT result_json FROM operations "
                    "WHERE project_id = ? AND idempotency_key = ?",
                    (project_id, idempotency_key),
                ).fetchone()
            if row is None:
                return None
            raw = json.loads(row[0])
            if not isinstance(raw, Mapping):
                raise ValueError
            return _result_from_dict(raw)
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    def operation_by_id(
        self, project_id: str, operation_id: str
    ) -> tuple[str, ScenarioDraftOperationResult] | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT operation_hash, result_json FROM operations "
                    "WHERE project_id = ? AND operation_id = ?",
                    (project_id, operation_id),
                ).fetchone()
            if row is None:
                return None
            raw = json.loads(row[1])
            if not isinstance(raw, Mapping):
                raise ValueError
            return row[0], _result_from_dict(raw)
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    def import_snapshot(
        self,
        snapshot: ScenarioDraftSnapshot,
        report: ScenarioDiagnosticReport,
        expected_revision: int,
        expected_hash: str,
    ) -> ScenarioDraftSnapshot:
        self._validate_snapshot_report(snapshot, report)
        if snapshot.revision != expected_revision + 1:
            raise ValueError("scenario import revision must increase by one")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = self._project_row(connection, snapshot.project_id)
                if row is None:
                    raise ScenarioProjectNotFoundError("scenario project is unknown")
                current = self._decode_snapshot(row[0])
                if current.revision != expected_revision or current.content_hash != expected_hash:
                    raise ScenarioProjectConflictError("scenario project revision is stale")
                if snapshot.revision != current.revision + 1:
                    raise ValueError("scenario import revision must increase by one")
                connection.execute(
                    "UPDATE projects SET snapshot_json = ?, report_json = ?, "
                    "journal_cursor = NULL, redo_entry = NULL WHERE project_id = ?",
                    (
                        _json_text(snapshot.to_dict()),
                        _json_text(report.to_dict()),
                        snapshot.project_id,
                    ),
                )
                self._write_dependents(connection, snapshot, report)
            return snapshot
        except (ScenarioProjectNotFoundError, ScenarioProjectConflictError):
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    def apply(
        self,
        operation: ScenarioDraftOperation,
        next_snapshot: ScenarioDraftSnapshot,
        report: ScenarioDiagnosticReport,
    ) -> ScenarioDraftOperationResult:
        if not isinstance(operation, ScenarioDraftOperation):
            raise TypeError("scenario store apply requires ScenarioDraftOperation")
        self._validate_snapshot_report(next_snapshot, report)
        if operation.project_id != next_snapshot.project_id:
            raise ValueError("scenario transition project identities must match")
        if next_snapshot.revision != operation.expected_revision + 1:
            raise ValueError("scenario transition revision must increase by one")
        result = ScenarioDraftOperationResult(
            operation.content_hash,
            operation.expected_revision,
            next_snapshot,
            report,
        )
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                collision = connection.execute(
                    "SELECT operation_id, idempotency_key, operation_hash, result_json "
                    "FROM operations WHERE project_id = ? "
                    "AND (operation_id = ? OR idempotency_key = ?)",
                    (
                        operation.project_id,
                        operation.operation_id,
                        operation.idempotency_key,
                    ),
                ).fetchone()
                if collision is not None:
                    if (
                        collision[0] == operation.operation_id
                        and collision[1] == operation.idempotency_key
                        and collision[2] == operation.content_hash
                    ):
                        raw = json.loads(collision[3])
                        return _result_from_dict(raw)
                    if collision[1] == operation.idempotency_key:
                        raise ScenarioProjectConflictError(
                            "scenario project idempotency key conflicts"
                        )
                    raise ScenarioProjectConflictError(
                        "scenario project operation ID conflicts"
                    )
                row = self._project_row(connection, operation.project_id)
                if row is None:
                    raise ScenarioProjectNotFoundError("scenario project is unknown")
                current = self._decode_snapshot(row[0])
                if (
                    current.revision != operation.expected_revision
                    or current.content_hash != operation.expected_snapshot_hash
                ):
                    raise ScenarioProjectConflictError("scenario project revision is stale")
                if (
                    next_snapshot.project_id != current.project_id
                    or next_snapshot.revision != current.revision + 1
                ):
                    raise ValueError("scenario transition state is inconsistent")
                if row[4] >= self._limits.accepted_operation_journal:
                    raise ScenarioProjectCapacityError(
                        "scenario project operation journal is full"
                    )
                entry_index = row[4] + 1
                connection.execute(
                    "INSERT INTO journal(project_id, entry_index, parent_index, "
                    "before_snapshot_json, before_report_json, after_snapshot_json, after_report_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        operation.project_id,
                        entry_index,
                        row[2],
                        row[0],
                        row[1],
                        _json_text(next_snapshot.to_dict()),
                        _json_text(report.to_dict()),
                    ),
                )
                connection.execute(
                    "INSERT INTO operations(project_id, operation_id, idempotency_key, "
                    "operation_hash, result_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        operation.project_id,
                        operation.operation_id,
                        operation.idempotency_key,
                        operation.content_hash,
                        _json_text(result.to_dict()),
                    ),
                )
                connection.execute(
                    "UPDATE projects SET snapshot_json = ?, report_json = ?, "
                    "journal_cursor = ?, redo_entry = NULL, accepted_count = ? "
                    "WHERE project_id = ?",
                    (
                        _json_text(next_snapshot.to_dict()),
                        _json_text(report.to_dict()),
                        entry_index,
                        entry_index,
                        operation.project_id,
                    ),
                )
                self._write_dependents(connection, next_snapshot, report)
            return result
        except (
            ScenarioProjectNotFoundError,
            ScenarioProjectConflictError,
            ScenarioProjectCapacityError,
        ):
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None

    @staticmethod
    def _validate_snapshot_report(
        snapshot: ScenarioDraftSnapshot,
        report: ScenarioDiagnosticReport,
    ) -> None:
        if not isinstance(snapshot, ScenarioDraftSnapshot):
            raise TypeError("scenario transition snapshot must be ScenarioDraftSnapshot")
        if not isinstance(report, ScenarioDiagnosticReport):
            raise TypeError("scenario transition report must be ScenarioDiagnosticReport")
        if snapshot.project_id != report.project_id:
            raise ValueError("scenario transition project identities must match")
        if snapshot.revision != report.revision:
            raise ValueError("scenario transition revisions must match")
        if snapshot.diagnostic_report_hash != report.content_hash:
            raise ValueError("scenario transition diagnostic hash must match")

    @staticmethod
    def _revised_state(
        project_id: str,
        revision: int,
        snapshot_text: str,
        report_text: str,
    ) -> tuple[ScenarioDraftSnapshot, ScenarioDiagnosticReport]:
        old_snapshot = SQLiteScenarioProjectStore._decode_snapshot(snapshot_text)
        old_report = SQLiteScenarioProjectStore._decode_report(report_text)
        report = ScenarioDiagnosticReport(project_id, revision, old_report.diagnostics)
        snapshot = ScenarioDraftSnapshot(
            project_id,
            revision,
            old_snapshot.documents,
            old_snapshot.layout,
            report.content_hash,
            old_snapshot.compiled_scenario_hash,
            scenario_id=old_snapshot.scenario_id,
            version=old_snapshot.version,
        )
        return snapshot, report

    def undo(
        self, project_id: str, expected_revision: int, expected_hash: str
    ) -> ScenarioDraftSnapshot:
        return self._navigate(project_id, expected_revision, expected_hash, redo=False)

    def redo(
        self, project_id: str, expected_revision: int, expected_hash: str
    ) -> ScenarioDraftSnapshot:
        return self._navigate(project_id, expected_revision, expected_hash, redo=True)

    def _navigate(
        self,
        project_id: str,
        expected_revision: int,
        expected_hash: str,
        *,
        redo: bool,
    ) -> ScenarioDraftSnapshot:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = self._project_row(connection, project_id)
                if row is None:
                    raise ScenarioProjectNotFoundError("scenario project is unknown")
                current = self._decode_snapshot(row[0])
                if current.revision != expected_revision or current.content_hash != expected_hash:
                    raise ScenarioProjectConflictError("scenario project revision is stale")
                entry_index = row[3] if redo else row[2]
                if entry_index is None:
                    action = "redo" if redo else "undo"
                    raise ScenarioProjectConflictError(
                        f"scenario project has no operation to {action}"
                    )
                journal = connection.execute(
                    "SELECT parent_index, before_snapshot_json, before_report_json, "
                    "after_snapshot_json, after_report_json FROM journal "
                    "WHERE project_id = ? AND entry_index = ?",
                    (project_id, entry_index),
                ).fetchone()
                if journal is None:
                    raise ScenarioProjectStorageError("scenario project storage failed")
                if redo:
                    snapshot_text, report_text = journal[3], journal[4]
                    next_redo = connection.execute(
                        "SELECT MAX(entry_index) FROM journal "
                        "WHERE project_id = ? AND parent_index = ?",
                        (project_id, entry_index),
                    ).fetchone()
                    cursor = entry_index
                    redo_entry = None if next_redo is None else next_redo[0]
                else:
                    snapshot_text, report_text = journal[1], journal[2]
                    cursor, redo_entry = journal[0], entry_index
                snapshot, report = self._revised_state(
                    project_id, current.revision + 1, snapshot_text, report_text
                )
                connection.execute(
                    "UPDATE projects SET snapshot_json = ?, report_json = ?, "
                    "journal_cursor = ?, redo_entry = ? WHERE project_id = ?",
                    (
                        _json_text(snapshot.to_dict()),
                        _json_text(report.to_dict()),
                        cursor,
                        redo_entry,
                        project_id,
                    ),
                )
                self._write_dependents(connection, snapshot, report)
            return snapshot
        except (
            ScenarioProjectNotFoundError,
            ScenarioProjectConflictError,
            ScenarioProjectStorageError,
        ):
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            raise ScenarioProjectStorageError("scenario project storage failed") from None


__all__ = (
    "SQLiteScenarioProjectStore",
    "ScenarioProjectCapacityError",
    "ScenarioProjectConflictError",
    "ScenarioProjectError",
    "ScenarioProjectNotFoundError",
    "ScenarioProjectStorageError",
    "ScenarioProjectStore",
    "ScenarioProjectValidationError",
)
