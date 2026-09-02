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


class _ScenarioCheckpointRestoreTargetExistsError(ValueError):
    """Identify a restore collision without exposing the target path."""


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


def _path_exists(path: Path) -> bool:
    try:
        return path.exists() or path.is_symlink()
    except Exception:
        return True


def _cleanup_owned_paths(paths: tuple[Path, ...], *, message: str) -> None:
    failed = False
    for path in paths:
        try:
            path.unlink(missing_ok=True)
            if _path_exists(path):
                failed = True
        except Exception:
            failed = True
    if failed:
        raise RuntimeError(message) from None


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
        self._created_artifact_hashes: set[str] = set()

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

    def _validate_artifact_path(self, artifact: Path) -> None:
        try:
            if artifact.is_symlink():
                raise ValueError("scenario checkpoint artifact must not be a symlink")
            resolved = artifact.resolve(strict=True)
        except ValueError:
            raise
        except (OSError, RuntimeError):
            raise ValueError(
                "scenario checkpoint snapshot failed integrity validation"
            ) from None
        if not resolved.is_relative_to(self._root):
            raise ValueError("scenario checkpoint artifact escaped the store root")
        if not artifact.is_file():
            raise ValueError(
                "scenario checkpoint snapshot failed integrity validation"
            )

    def _register(
        self,
        checkpoint_hash: str,
        checkpoint: ScenarioCheckpoint,
    ) -> ScenarioCheckpoint:
        self._checkpoints_by_hash[checkpoint_hash] = checkpoint
        self._hash_by_checkpoint_id[checkpoint.checkpoint_id] = checkpoint_hash
        return checkpoint

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
        if _path_exists(artifact):
            self._validate_artifact(checkpoint_hash, checkpoint)
            return self._register(checkpoint_hash, checkpoint)

        stage = self._stage_path(self._root, artifact.name)
        published = False
        try:
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
            try:
                os.link(stage, artifact)
                published = True
            except FileExistsError:
                pass
            except OSError:
                raise RuntimeError("scenario checkpoint publication failed") from None
            self._validate_artifact(checkpoint_hash, checkpoint)
        except Exception:
            cleanup = (artifact, stage) if published else (stage,)
            _cleanup_owned_paths(
                cleanup,
                message="scenario checkpoint publication cleanup failed",
            )
            raise

        try:
            _cleanup_owned_paths(
                (stage,),
                message="scenario checkpoint publication cleanup failed",
            )
        except RuntimeError:
            cleanup = (artifact, stage) if published else (stage,)
            _cleanup_owned_paths(
                cleanup,
                message="scenario checkpoint publication cleanup failed",
            )
            raise RuntimeError(
                "scenario checkpoint publication cleanup failed"
            ) from None
        if published:
            self._created_artifact_hashes.add(checkpoint_hash)
        return self._register(checkpoint_hash, checkpoint)

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
        self._validate_artifact_path(artifact)
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
        if _path_exists(target):
            raise _ScenarioCheckpointRestoreTargetExistsError(
                "scenario checkpoint restore target already exists"
            )
        if not target.parent.is_dir():
            raise RuntimeError("scenario checkpoint restore target directory is unavailable")

        artifact = self._artifact_path(checkpoint.content_hash)
        stage = self._stage_path(target.parent, target.name)
        published = False
        try:
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
            try:
                os.link(stage, target)
                published = True
            except FileExistsError:
                raise _ScenarioCheckpointRestoreTargetExistsError(
                    "scenario checkpoint restore target already exists"
                ) from None
            except OSError:
                raise RuntimeError("scenario checkpoint restore publication failed") from None
            if target.is_symlink() or not target.is_file():
                raise ValueError(
                    "scenario checkpoint restore failed integrity validation"
                )
            if _logical_hash(
                target,
                message="scenario checkpoint restore failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint restore failed integrity validation")
        except _ScenarioCheckpointRestoreTargetExistsError:
            _cleanup_owned_paths(
                (stage,),
                message="scenario checkpoint restore cleanup failed",
            )
            raise
        except Exception:
            cleanup = (target, stage) if published else (stage,)
            _cleanup_owned_paths(
                cleanup,
                message="scenario checkpoint restore cleanup failed",
            )
            raise

        try:
            _cleanup_owned_paths(
                (stage,),
                message="scenario checkpoint restore cleanup failed",
            )
        except RuntimeError:
            _cleanup_owned_paths(
                (target, stage),
                message="scenario checkpoint restore cleanup failed",
            )
            raise RuntimeError("scenario checkpoint restore cleanup failed") from None

    def discard(self, checkpoint_hash: str) -> None:
        exact_hash, _ = _checkpoint_hash(checkpoint_hash)
        checkpoint = self._checkpoints_by_hash.get(exact_hash)
        if checkpoint is None:
            return
        artifact = self._artifact_path(exact_hash)
        if (
            exact_hash in self._created_artifact_hashes
            and _path_exists(artifact)
        ):
            self._validate_artifact(exact_hash, checkpoint)
            _cleanup_owned_paths(
                (artifact,),
                message="scenario checkpoint discard failed",
            )
        self._created_artifact_hashes.discard(exact_hash)
        self._checkpoints_by_hash.pop(exact_hash, None)
        if self._hash_by_checkpoint_id.get(checkpoint.checkpoint_id) == exact_hash:
            self._hash_by_checkpoint_id.pop(checkpoint.checkpoint_id, None)


__all__ = ("LocalScenarioCheckpointStore",)
