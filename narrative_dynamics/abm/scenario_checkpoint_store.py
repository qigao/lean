"""Path-safe atomic SQLite checkpoints for local scenario coordinators."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sqlite3
from tempfile import mkstemp

from narrative_dynamics.abm.scenario_coordinator_contracts import ScenarioCheckpoint
from narrative_dynamics.abm.situated_percept_memory import (
    hash_situated_percept_memory_store,
)


_CONTENT_HASH = re.compile(r"^sha256:([0-9a-f]{64})$")


def _checkpoint_hash(value: object) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ValueError("scenario checkpoint hash must be a content hash")
    matched = _CONTENT_HASH.fullmatch(value)
    if matched is None:
        raise ValueError("scenario checkpoint hash must be a content hash")
    return value, matched.group(1)


def _database_path(value: object, *, label: str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        raise ValueError(f"{label} must be a non-empty path")
    return path


def _flush_file(path: Path) -> None:
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _backup_sqlite(source: Path, destination: Path, *, message: str) -> None:
    source_connection = destination_connection = None
    try:
        source_connection = sqlite3.connect(str(source))
        destination_connection = sqlite3.connect(str(destination))
        source_connection.backup(destination_connection)
    except (OSError, sqlite3.Error) as error:
        raise RuntimeError(message) from None
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()


def _logical_hash(path: Path, *, message: str) -> str:
    try:
        return hash_situated_percept_memory_store(path)
    except Exception as error:
        raise ValueError(message) from None


class LocalScenarioCheckpointStore:
    """Own immutable typed metadata and atomic content-addressed SQLite snapshots."""

    def __init__(self, root: str | Path) -> None:
        exact_root = _database_path(root, label="scenario checkpoint root")
        try:
            exact_root.mkdir(parents=True, exist_ok=True)
            self._root = exact_root.resolve(strict=True)
            if not self._root.is_dir():
                raise OSError("not a directory")
        except OSError as error:
            raise RuntimeError("scenario checkpoint root is unavailable") from None
        self._checkpoints_by_hash: dict[str, ScenarioCheckpoint] = {}
        self._hash_by_checkpoint_id: dict[str, str] = {}

    def _artifact_path(self, checkpoint_hash: str) -> Path:
        _, digest = _checkpoint_hash(checkpoint_hash)
        return self._root / f"checkpoint-{digest}.sqlite3"

    @staticmethod
    def _stage_path(directory: Path, name: str) -> Path:
        try:
            descriptor, value = mkstemp(
                prefix=f".{name}.stage-",
                dir=str(directory),
            )
            os.close(descriptor)
            return Path(value)
        except OSError as error:
            raise RuntimeError("scenario checkpoint stage creation failed") from None

    @staticmethod
    def _remove_stage(stage: Path | None) -> None:
        if stage is None:
            return
        try:
            stage.unlink(missing_ok=True)
        except OSError:
            pass

    def create(
        self,
        checkpoint: ScenarioCheckpoint,
        source_database_path: str | Path,
    ) -> ScenarioCheckpoint:
        if not isinstance(checkpoint, ScenarioCheckpoint):
            raise TypeError("scenario checkpoint store requires ScenarioCheckpoint")
        checkpoint_hash = checkpoint.content_hash
        prior_hash = self._hash_by_checkpoint_id.get(checkpoint.checkpoint_id)
        if prior_hash is not None and prior_hash != checkpoint_hash:
            raise ValueError("scenario checkpoint id was reused for different content")

        source = _database_path(
            source_database_path,
            label="scenario checkpoint source database",
        )
        if not source.is_file():
            raise RuntimeError("scenario checkpoint source database is unavailable")
        source_hash = _logical_hash(
            source,
            message="scenario checkpoint source database failed integrity validation",
        )
        if source_hash != checkpoint.memory_store_hash:
            raise ValueError("scenario checkpoint source memory hash does not match")

        existing = self._checkpoints_by_hash.get(checkpoint_hash)
        if existing is not None:
            self._validate_artifact(checkpoint_hash, existing)
            return existing

        artifact = self._artifact_path(checkpoint_hash)
        stage: Path | None = None
        try:
            stage = self._stage_path(self._root, artifact.name)
            _backup_sqlite(
                source,
                stage,
                message="scenario checkpoint snapshot failed",
            )
            if _logical_hash(
                stage,
                message="scenario checkpoint snapshot failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint snapshot failed integrity validation")
            _flush_file(stage)
            os.replace(stage, artifact)
            stage = None
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError("scenario checkpoint publication failed") from None
        finally:
            self._remove_stage(stage)

        try:
            if _logical_hash(
                artifact,
                message="scenario checkpoint snapshot failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint snapshot failed integrity validation")
        except Exception:
            try:
                artifact.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        self._checkpoints_by_hash[checkpoint_hash] = checkpoint
        self._hash_by_checkpoint_id[checkpoint.checkpoint_id] = checkpoint_hash
        return checkpoint

    def load(self, checkpoint_hash: str) -> ScenarioCheckpoint:
        exact_hash, _ = _checkpoint_hash(checkpoint_hash)
        try:
            checkpoint = self._checkpoints_by_hash[exact_hash]
        except KeyError:
            raise KeyError("unknown scenario checkpoint") from None
        self._validate_artifact(exact_hash, checkpoint)
        return checkpoint

    def _validate_artifact(
        self,
        checkpoint_hash: str,
        checkpoint: ScenarioCheckpoint,
    ) -> None:
        artifact = self._artifact_path(checkpoint_hash)
        if not artifact.is_file():
            raise ValueError("scenario checkpoint snapshot failed integrity validation")
        actual = _logical_hash(
            artifact,
            message="scenario checkpoint snapshot failed integrity validation",
        )
        if actual != checkpoint.memory_store_hash:
            raise ValueError("scenario checkpoint snapshot failed integrity validation")

    def restore(
        self,
        checkpoint_hash: str,
        target_database_path: str | Path,
    ) -> None:
        checkpoint = self.load(checkpoint_hash)
        target = _database_path(
            target_database_path,
            label="scenario checkpoint restore target",
        )
        if target.exists() or target.is_symlink():
            raise ValueError("scenario checkpoint restore target already exists")
        if not target.parent.is_dir():
            raise RuntimeError("scenario checkpoint restore target directory is unavailable")

        artifact = self._artifact_path(checkpoint.content_hash)
        stage: Path | None = None
        published = False
        try:
            stage = self._stage_path(target.parent, target.name)
            _backup_sqlite(
                artifact,
                stage,
                message="scenario checkpoint restore failed",
            )
            if _logical_hash(
                stage,
                message="scenario checkpoint restore failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint restore failed integrity validation")
            _flush_file(stage)
            os.replace(stage, target)
            stage = None
            published = True
            if _logical_hash(
                target,
                message="scenario checkpoint restore failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint restore failed integrity validation")
        except ValueError:
            if published:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        except Exception as error:
            if published:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
            raise RuntimeError("scenario checkpoint restore failed") from None
        finally:
            self._remove_stage(stage)

    def discard(self, checkpoint_hash: str) -> None:
        exact_hash, _ = _checkpoint_hash(checkpoint_hash)
        checkpoint = self._checkpoints_by_hash.get(exact_hash)
        if checkpoint is None:
            return
        artifact = self._artifact_path(exact_hash)
        try:
            artifact.unlink(missing_ok=True)
        except OSError as error:
            raise RuntimeError("scenario checkpoint discard failed") from None
        self._checkpoints_by_hash.pop(exact_hash, None)
        if self._hash_by_checkpoint_id.get(checkpoint.checkpoint_id) == exact_hash:
            self._hash_by_checkpoint_id.pop(checkpoint.checkpoint_id, None)


__all__ = ("LocalScenarioCheckpointStore",)
