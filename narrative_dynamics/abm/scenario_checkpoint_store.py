"""Path-safe atomic SQLite checkpoints for local scenario coordinators."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sqlite3
import stat
from tempfile import mkstemp

from narrative_dynamics.abm.scenario_coordinator_contracts import ScenarioCheckpoint
from narrative_dynamics.abm.situated_percept_memory import (
    hash_situated_percept_memory_store,
)


_CONTENT_HASH = re.compile(r"^sha256:([0-9a-f]{64})$")


class _ScenarioCheckpointRestoreTargetExistsError(ValueError):
    """Identify a restore collision without exposing the target path."""


@dataclass(frozen=True)
class _PhysicalFileOwnershipToken:
    device: int
    inode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_PhysicalFileOwnershipToken":
        return cls(value.st_dev, value.st_ino)


class _ScenarioCheckpointOwnedRestoreError(RuntimeError):
    """Carry target ownership when restore cleanup could not finish."""

    def __init__(
        self,
        message: str,
        ownership_token: _PhysicalFileOwnershipToken,
    ) -> None:
        super().__init__(message)
        self.ownership_token = ownership_token


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


def _no_follow_stat(
    path: Path,
    *,
    missing_ok: bool,
    message: str,
) -> os.stat_result | None:
    try:
        return os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise RuntimeError(message) from None
    except (OSError, ValueError):
        raise RuntimeError(message) from None


def _regular_file_token(
    path: Path,
    *,
    missing_ok: bool = False,
    message: str,
) -> _PhysicalFileOwnershipToken | None:
    metadata = _no_follow_stat(path, missing_ok=missing_ok, message=message)
    if metadata is None:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(message)
    return _PhysicalFileOwnershipToken.from_stat(metadata)


def _require_file_token(
    path: Path,
    expected: _PhysicalFileOwnershipToken,
    *,
    message: str,
) -> None:
    actual = _regular_file_token(path, message=message)
    if actual != expected:
        raise RuntimeError(message)


def _flush_file(path: Path, expected: _PhysicalFileOwnershipToken) -> None:
    _require_file_token(
        path,
        expected,
        message="scenario checkpoint stage ownership changed",
    )
    with path.open("rb+") as stream:
        if _PhysicalFileOwnershipToken.from_stat(os.fstat(stream.fileno())) != expected:
            raise RuntimeError("scenario checkpoint stage ownership changed")
        os.fsync(stream.fileno())
    _require_file_token(
        path,
        expected,
        message="scenario checkpoint stage ownership changed",
    )


def _backup_sqlite(
    source: Path,
    destination: Path,
    *,
    message: str,
    source_token: _PhysicalFileOwnershipToken | None = None,
    destination_token: _PhysicalFileOwnershipToken | None = None,
) -> None:
    if source_token is not None:
        _require_file_token(source, source_token, message=message)
    if destination_token is not None:
        _require_file_token(destination, destination_token, message=message)
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
    if source_token is not None:
        _require_file_token(source, source_token, message=message)
    if destination_token is not None:
        _require_file_token(destination, destination_token, message=message)


def _logical_hash(path: Path, *, message: str) -> str:
    try:
        return hash_situated_percept_memory_store(path)
    except Exception as error:
        raise ValueError(message) from None


def _owned_logical_hash(
    path: Path,
    expected: _PhysicalFileOwnershipToken,
    *,
    message: str,
) -> str:
    _require_file_token(path, expected, message=message)
    result = _logical_hash(path, message=message)
    _require_file_token(path, expected, message=message)
    return result


def _path_exists(path: Path) -> bool:
    return _no_follow_stat(
        path,
        missing_ok=True,
        message="scenario checkpoint path inspection failed",
    ) is not None


def _cleanup_owned_paths(
    paths: tuple[tuple[Path, _PhysicalFileOwnershipToken], ...],
    *,
    message: str,
) -> None:
    failed = False
    for path, expected in paths:
        try:
            actual = _regular_file_token(
                path,
                missing_ok=True,
                message=message,
            )
            if actual is None:
                continue
            if actual != expected:
                failed = True
                continue
            path.unlink()
            remaining = _regular_file_token(
                path,
                missing_ok=True,
                message=message,
            )
            if remaining is not None:
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
        self._artifact_tokens_by_hash: dict[
            str, _PhysicalFileOwnershipToken
        ] = {}
        self._owned_artifact_tokens_by_hash: dict[
            str, _PhysicalFileOwnershipToken
        ] = {}

    def _artifact_path(self, checkpoint_hash: str) -> Path:
        _, digest = _checkpoint_hash(checkpoint_hash)
        return self._root / f"checkpoint-{digest}.sqlite3"

    @staticmethod
    def _stage_path(
        directory: Path,
        name: str,
    ) -> tuple[Path, _PhysicalFileOwnershipToken]:
        try:
            descriptor, value = mkstemp(
                prefix=f".{name}.stage-",
                dir=str(directory),
            )
            token = _PhysicalFileOwnershipToken.from_stat(os.fstat(descriptor))
            os.close(descriptor)
            path = Path(value)
            try:
                _require_file_token(
                    path,
                    token,
                    message="scenario checkpoint stage ownership changed",
                )
            except Exception:
                _cleanup_owned_paths(
                    ((path, token),),
                    message="scenario checkpoint stage cleanup ownership changed",
                )
                raise
            return path, token
        except OSError as error:
            raise RuntimeError("scenario checkpoint stage creation failed") from None

    def _validate_artifact_path(
        self,
        artifact: Path,
        *,
        expected_token: _PhysicalFileOwnershipToken | None = None,
    ) -> _PhysicalFileOwnershipToken:
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
        token = _regular_file_token(
            artifact,
            message="scenario checkpoint snapshot failed integrity validation",
        )
        if token is None:
            raise ValueError("scenario checkpoint snapshot failed integrity validation")
        if expected_token is not None and token != expected_token:
            raise RuntimeError("scenario checkpoint artifact ownership changed")
        return token

    def _register(
        self,
        checkpoint_hash: str,
        checkpoint: ScenarioCheckpoint,
        artifact_token: _PhysicalFileOwnershipToken,
        *,
        owned: bool,
    ) -> ScenarioCheckpoint:
        self._checkpoints_by_hash[checkpoint_hash] = checkpoint
        self._hash_by_checkpoint_id[checkpoint.checkpoint_id] = checkpoint_hash
        self._artifact_tokens_by_hash[checkpoint_hash] = artifact_token
        if owned:
            self._owned_artifact_tokens_by_hash[checkpoint_hash] = artifact_token
        return checkpoint

    def _drop_registration(
        self,
        checkpoint_hash: str,
        checkpoint: ScenarioCheckpoint,
    ) -> None:
        self._owned_artifact_tokens_by_hash.pop(checkpoint_hash, None)
        self._artifact_tokens_by_hash.pop(checkpoint_hash, None)
        self._checkpoints_by_hash.pop(checkpoint_hash, None)
        if self._hash_by_checkpoint_id.get(checkpoint.checkpoint_id) == checkpoint_hash:
            self._hash_by_checkpoint_id.pop(checkpoint.checkpoint_id, None)

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
            self._validate_artifact(
                checkpoint_hash,
                existing,
                expected_token=self._artifact_tokens_by_hash[checkpoint_hash],
            )
            return existing

        artifact = self._artifact_path(checkpoint_hash)
        if _path_exists(artifact):
            artifact_token = self._validate_artifact(checkpoint_hash, checkpoint)
            return self._register(
                checkpoint_hash,
                checkpoint,
                artifact_token,
                owned=False,
            )

        stage, stage_token = self._stage_path(self._root, artifact.name)
        published = False
        artifact_token: _PhysicalFileOwnershipToken | None = None
        try:
            _backup_sqlite(
                source,
                stage,
                message="scenario checkpoint snapshot failed",
                destination_token=stage_token,
            )
            if _owned_logical_hash(
                stage,
                stage_token,
                message="scenario checkpoint snapshot failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint snapshot failed integrity validation")
            _flush_file(stage, stage_token)
            try:
                os.link(stage, artifact)
                published = True
            except FileExistsError:
                pass
            except OSError:
                raise RuntimeError("scenario checkpoint publication failed") from None
            artifact_token = self._validate_artifact(
                checkpoint_hash,
                checkpoint,
                expected_token=stage_token if published else None,
            )
        except Exception:
            cleanup = (
                ((artifact, stage_token), (stage, stage_token))
                if published
                else ((stage, stage_token),)
            )
            _cleanup_owned_paths(
                cleanup,
                message="scenario checkpoint publication cleanup failed",
            )
            raise

        try:
            _cleanup_owned_paths(
                ((stage, stage_token),),
                message="scenario checkpoint publication cleanup failed",
            )
        except RuntimeError:
            cleanup = (
                ((artifact, stage_token), (stage, stage_token))
                if published
                else ((stage, stage_token),)
            )
            _cleanup_owned_paths(
                cleanup,
                message="scenario checkpoint publication cleanup failed",
            )
            raise RuntimeError(
                "scenario checkpoint publication cleanup failed"
            ) from None
        if artifact_token is None:
            raise RuntimeError("scenario checkpoint publication ownership was not captured")
        return self._register(
            checkpoint_hash,
            checkpoint,
            artifact_token,
            owned=published,
        )

    def load(self, checkpoint_hash: str) -> ScenarioCheckpoint:
        exact_hash, _ = _checkpoint_hash(checkpoint_hash)
        try:
            checkpoint = self._checkpoints_by_hash[exact_hash]
        except KeyError:
            raise KeyError("unknown scenario checkpoint") from None
        self._validate_artifact(
            exact_hash,
            checkpoint,
            expected_token=self._artifact_tokens_by_hash[exact_hash],
        )
        return checkpoint

    def _validate_artifact(
        self,
        checkpoint_hash: str,
        checkpoint: ScenarioCheckpoint,
        *,
        expected_token: _PhysicalFileOwnershipToken | None = None,
    ) -> _PhysicalFileOwnershipToken:
        artifact = self._artifact_path(checkpoint_hash)
        artifact_token = self._validate_artifact_path(
            artifact,
            expected_token=expected_token,
        )
        actual = _owned_logical_hash(
            artifact,
            artifact_token,
            message="scenario checkpoint snapshot failed integrity validation",
        )
        if actual != checkpoint.memory_store_hash:
            raise ValueError("scenario checkpoint snapshot failed integrity validation")
        return artifact_token

    def restore(
        self,
        checkpoint_hash: str,
        target_database_path: str | Path,
    ) -> None:
        try:
            self._restore_owned(checkpoint_hash, target_database_path)
        except _ScenarioCheckpointOwnedRestoreError as error:
            raise RuntimeError(str(error)) from None

    def _restore_owned(
        self,
        checkpoint_hash: str,
        target_database_path: str | Path,
    ) -> _PhysicalFileOwnershipToken:
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
        artifact_token = self._artifact_tokens_by_hash[checkpoint.content_hash]
        stage, stage_token = self._stage_path(target.parent, target.name)
        published = False
        try:
            _backup_sqlite(
                artifact,
                stage,
                message="scenario checkpoint restore failed",
                source_token=artifact_token,
                destination_token=stage_token,
            )
            if _owned_logical_hash(
                stage,
                stage_token,
                message="scenario checkpoint restore failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint restore failed integrity validation")
            _flush_file(stage, stage_token)
            try:
                os.link(stage, target)
                published = True
            except FileExistsError:
                raise _ScenarioCheckpointRestoreTargetExistsError(
                    "scenario checkpoint restore target already exists"
                ) from None
            except OSError:
                raise RuntimeError("scenario checkpoint restore publication failed") from None
            _require_file_token(
                target,
                stage_token,
                message="scenario checkpoint restore ownership changed",
            )
            if _owned_logical_hash(
                target,
                stage_token,
                message="scenario checkpoint restore failed integrity validation",
            ) != checkpoint.memory_store_hash:
                raise ValueError("scenario checkpoint restore failed integrity validation")
        except _ScenarioCheckpointRestoreTargetExistsError:
            _cleanup_owned_paths(
                ((stage, stage_token),),
                message="scenario checkpoint restore cleanup failed",
            )
            raise
        except Exception:
            cleanup = (
                ((target, stage_token), (stage, stage_token))
                if published
                else ((stage, stage_token),)
            )
            try:
                _cleanup_owned_paths(
                    cleanup,
                    message="scenario checkpoint restore cleanup failed",
                )
            except RuntimeError:
                if published:
                    raise _ScenarioCheckpointOwnedRestoreError(
                        "scenario checkpoint restore cleanup failed",
                        stage_token,
                    ) from None
                raise
            raise

        try:
            _cleanup_owned_paths(
                ((stage, stage_token),),
                message="scenario checkpoint restore cleanup failed",
            )
        except RuntimeError:
            try:
                _cleanup_owned_paths(
                    ((target, stage_token), (stage, stage_token)),
                    message="scenario checkpoint restore cleanup failed",
                )
            except RuntimeError:
                raise _ScenarioCheckpointOwnedRestoreError(
                    "scenario checkpoint restore cleanup failed",
                    stage_token,
                ) from None
            raise RuntimeError("scenario checkpoint restore cleanup failed") from None
        return stage_token

    @staticmethod
    def _cleanup_restored_target(
        target_database_path: str | Path,
        ownership_token: _PhysicalFileOwnershipToken,
    ) -> None:
        if not isinstance(ownership_token, _PhysicalFileOwnershipToken):
            raise TypeError("scenario checkpoint restore cleanup requires ownership token")
        target = _database_path(
            target_database_path,
            label="scenario checkpoint restore cleanup target",
        )
        _cleanup_owned_paths(
            ((target, ownership_token),),
            message="scenario checkpoint restore target ownership changed",
        )

    @staticmethod
    def _verify_restored_target(
        target_database_path: str | Path,
        ownership_token: _PhysicalFileOwnershipToken,
    ) -> None:
        if not isinstance(ownership_token, _PhysicalFileOwnershipToken):
            raise TypeError("scenario checkpoint restore verification requires ownership token")
        target = _database_path(
            target_database_path,
            label="scenario checkpoint restore verification target",
        )
        _require_file_token(
            target,
            ownership_token,
            message="scenario checkpoint restore target ownership changed",
        )

    def discard(self, checkpoint_hash: str) -> None:
        exact_hash, _ = _checkpoint_hash(checkpoint_hash)
        checkpoint = self._checkpoints_by_hash.get(exact_hash)
        if checkpoint is None:
            return
        artifact = self._artifact_path(exact_hash)
        owned_token = self._owned_artifact_tokens_by_hash.get(exact_hash)
        if owned_token is None:
            self._drop_registration(exact_hash, checkpoint)
            return
        try:
            current = _regular_file_token(
                artifact,
                missing_ok=True,
                message="scenario checkpoint artifact ownership inspection failed",
            )
        except RuntimeError:
            self._drop_registration(exact_hash, checkpoint)
            raise RuntimeError("scenario checkpoint artifact ownership changed") from None
        if current is None:
            self._drop_registration(exact_hash, checkpoint)
            return
        if current != owned_token:
            self._drop_registration(exact_hash, checkpoint)
            raise RuntimeError("scenario checkpoint artifact ownership changed")
        try:
            _cleanup_owned_paths(
                ((artifact, owned_token),),
                message="scenario checkpoint discard ownership changed",
            )
        except RuntimeError:
            self._drop_registration(exact_hash, checkpoint)
            raise
        self._drop_registration(exact_hash, checkpoint)


__all__ = ("LocalScenarioCheckpointStore",)
