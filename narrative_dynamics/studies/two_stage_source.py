from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from narrative_dynamics.contracts import stable_content_hash


_TASK_VARIANTS = frozenset({"magic_carpet", "spaceship"})
_PURPOSES = frozenset({"scientific_evidence", "transform_metadata"})
_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FEHER_HARE_UPSTREAM_REPOSITORY = "carolfs/muddled_models"
_FEHER_HARE_UPSTREAM_REVISION = "4567763780a2c596fd6510af720ec468a8214a8f"
_FEHER_HARE_LICENSE_REFERENCE = "LICENSE.txt"
_FEHER_HARE_MANIFEST_NAME = "feher-hare-two-stage-v1"
_FEHER_HARE_MANIFEST_VERSION = "1"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _sha1(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-character lowercase Git SHA-1")
    return value


def _relative_path(value: object) -> str:
    path = _text(value, label="two-stage source path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError("two-stage source path must be a normalized relative POSIX path")
    if pure.as_posix() != path:
        raise ValueError("two-stage source path must be canonical POSIX text")
    return path


def _participant(value: object, *, purpose: str) -> str | None:
    if value is None:
        if purpose == "scientific_evidence":
            raise ValueError("scientific evidence requires a participant identity")
        return None
    return _text(value, label="two-stage source participant identity")


def _expected_evidence_path(task_variant: str, participant: str) -> str:
    if task_variant == "magic_carpet":
        return f"results/magic_carpet/choices/{participant}_game.csv"
    return f"results/spaceship/choices/{participant}.csv"


def _expected_metadata_path(task_variant: str, participant: str) -> str:
    if task_variant == "magic_carpet":
        return f"results/magic_carpet/choices/{participant}_config.txt"
    return f"results/spaceship/choices/{participant}_info.txt"


@dataclass(frozen=True)
class TwoStageSourceFile:
    path: str
    git_blob_sha: str
    task_variant: str
    source_participant_id: str | None
    purpose: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(
            self,
            "git_blob_sha",
            _sha1(self.git_blob_sha, label="two-stage source Git blob SHA"),
        )
        task_variant = _text(self.task_variant, label="two-stage task variant")
        if task_variant not in _TASK_VARIANTS:
            raise ValueError(f"unsupported two-stage task variant: {task_variant}")
        object.__setattr__(self, "task_variant", task_variant)
        purpose = _text(self.purpose, label="two-stage source purpose")
        if purpose not in _PURPOSES:
            raise ValueError(f"unsupported two-stage source purpose: {purpose}")
        object.__setattr__(self, "purpose", purpose)
        participant = _participant(self.source_participant_id, purpose=purpose)
        object.__setattr__(self, "source_participant_id", participant)
        if participant is not None:
            expected = (
                _expected_evidence_path(task_variant, participant)
                if purpose == "scientific_evidence"
                else _expected_metadata_path(task_variant, participant)
            )
            if self.path != expected:
                raise ValueError(
                    "two-stage source path does not match task, participant, and purpose"
                )

    def identity_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "git_blob_sha": self.git_blob_sha,
            "task_variant": self.task_variant,
            "source_participant_id": self.source_participant_id,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class TwoStageSourceManifest:
    name: str
    version: str
    repository: str
    revision: str
    license_reference: str
    files: tuple[TwoStageSourceFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="source manifest name"))
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="source manifest version"),
        )
        object.__setattr__(
            self,
            "repository",
            _text(self.repository, label="source manifest repository"),
        )
        object.__setattr__(
            self,
            "revision",
            _sha1(self.revision, label="source manifest revision"),
        )
        object.__setattr__(
            self,
            "license_reference",
            _text(self.license_reference, label="source manifest license reference"),
        )
        values = tuple(self.files)
        if not all(isinstance(item, TwoStageSourceFile) for item in values):
            raise TypeError("source manifest files must contain TwoStageSourceFile values")
        values = tuple(sorted(values, key=lambda item: item.path))
        paths = tuple(item.path for item in values)
        if len(paths) != len(set(paths)):
            raise ValueError("source manifest paths must be unique")
        object.__setattr__(self, "files", values)

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "repository": self.repository,
            "revision": self.revision,
            "license_reference": self.license_reference,
            "files": tuple(item.identity_payload() for item in self.files),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class VerifiedTwoStageSnapshot:
    root: str
    manifest_hash: str
    repository_revision: str
    source_snapshot_hash: str
    verified_file_identities: tuple[tuple[str, str], ...]


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("two-stage source root must be a readable local Git checkout") from error
    return _sha1(result.stdout.strip(), label="local source repository revision")


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _git_tree_files(root: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "ls-tree",
                "-r",
                "HEAD",
                "--",
                "results/magic_carpet/choices",
                "results/spaceship/choices",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("two-stage source Git tree is unreadable") from error

    entries: dict[str, str] = {}
    for line in result.stdout.splitlines():
        try:
            metadata, relative_path = line.split("\t", 1)
            _mode, kind, blob_sha = metadata.split()
        except ValueError as error:
            raise ValueError("two-stage source Git tree entry is malformed") from error
        if kind != "blob":
            continue
        canonical_path = _relative_path(relative_path)
        canonical_sha = _sha1(blob_sha, label="two-stage source tracked Git blob SHA")
        if canonical_path in entries:
            raise ValueError("two-stage source Git tree contains a duplicate path")
        entries[canonical_path] = canonical_sha
    return entries


def _is_magic_evidence_path(path: str) -> bool:
    return (
        path.startswith("results/magic_carpet/choices/")
        and path.endswith("_game.csv")
    )


def _is_spaceship_evidence_path(path: str) -> bool:
    return (
        path.startswith("results/spaceship/choices/")
        and path.endswith(".csv")
        and not path.endswith("_practice.csv")
        and not path.endswith("_game.csv")
    )


def _tracked_evidence_paths(tree_files: dict[str, str]) -> frozenset[str]:
    return frozenset(
        path
        for path in tree_files
        if _is_magic_evidence_path(path) or _is_spaceship_evidence_path(path)
    )


def _evidence_identity(path: str) -> tuple[str, str]:
    name = PurePosixPath(path).name
    if _is_magic_evidence_path(path):
        participant = name.removesuffix("_game.csv")
        return "magic_carpet", _text(
            participant,
            label="Magic Carpet source participant identity",
        )
    if _is_spaceship_evidence_path(path):
        participant = name.removesuffix(".csv")
        return "spaceship", _text(
            participant,
            label="Spaceship source participant identity",
        )
    raise ValueError("two-stage source path is not an approved main-task evidence file")


def _discovered_evidence_paths(root: Path) -> frozenset[str]:
    discovered: set[str] = set()
    magic = root / "results" / "magic_carpet" / "choices"
    if magic.is_dir():
        discovered.update(
            path.relative_to(root).as_posix()
            for path in magic.glob("*_game.csv")
            if path.is_file()
        )
    spaceship = root / "results" / "spaceship" / "choices"
    if spaceship.is_dir():
        discovered.update(
            path.relative_to(root).as_posix()
            for path in spaceship.glob("*.csv")
            if path.is_file()
            and not path.name.endswith("_practice.csv")
            and not path.name.endswith("_game.csv")
        )
    return frozenset(discovered)


def _require_tracked_file(
    root: Path,
    tree_files: dict[str, str],
    relative_path: str,
) -> str:
    try:
        tracked_sha = tree_files[relative_path]
    except KeyError as error:
        raise ValueError(
            f"required two-stage source file is not tracked at HEAD: {relative_path}"
        ) from error
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"required two-stage source file is missing: {relative_path}")
    actual_sha = _git_blob_sha(path)
    if actual_sha != tracked_sha:
        raise ValueError(
            f"two-stage source file differs from pinned HEAD: {relative_path}"
        )
    return tracked_sha


def freeze_feher_hare_v1_source_manifest(root: Path) -> TwoStageSourceManifest:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Feher/Hare source root must be a directory")
    repository_revision = _git_head(root)
    if repository_revision != _FEHER_HARE_UPSTREAM_REVISION:
        raise ValueError(
            "Feher/Hare source checkout must be at the pinned upstream revision"
        )

    tree_files = _git_tree_files(root)
    tracked_evidence = _tracked_evidence_paths(tree_files)
    discovered_evidence = _discovered_evidence_paths(root)
    if not tracked_evidence:
        raise ValueError("Feher/Hare source checkout contains no approved main-task files")
    if discovered_evidence != tracked_evidence:
        missing = tuple(sorted(tracked_evidence - discovered_evidence))
        extra = tuple(sorted(discovered_evidence - tracked_evidence))
        raise ValueError(
            "Feher/Hare local main-task files do not match pinned HEAD: "
            f"missing={missing}, untracked_or_extra={extra}"
        )

    files: list[TwoStageSourceFile] = []
    participant_keys: set[tuple[str, str]] = set()
    for evidence_path in sorted(tracked_evidence):
        task_variant, participant = _evidence_identity(evidence_path)
        participant_key = (task_variant, participant)
        if participant_key in participant_keys:
            raise ValueError("Feher/Hare source participant appears more than once per task")
        participant_keys.add(participant_key)

        evidence_sha = _require_tracked_file(root, tree_files, evidence_path)
        metadata_path = _expected_metadata_path(task_variant, participant)
        metadata_sha = _require_tracked_file(root, tree_files, metadata_path)
        files.extend(
            (
                TwoStageSourceFile(
                    path=evidence_path,
                    git_blob_sha=evidence_sha,
                    task_variant=task_variant,
                    source_participant_id=participant,
                    purpose="scientific_evidence",
                ),
                TwoStageSourceFile(
                    path=metadata_path,
                    git_blob_sha=metadata_sha,
                    task_variant=task_variant,
                    source_participant_id=participant,
                    purpose="transform_metadata",
                ),
            )
        )

    manifest = TwoStageSourceManifest(
        name=_FEHER_HARE_MANIFEST_NAME,
        version=_FEHER_HARE_MANIFEST_VERSION,
        repository=_FEHER_HARE_UPSTREAM_REPOSITORY,
        revision=repository_revision,
        license_reference=_FEHER_HARE_LICENSE_REFERENCE,
        files=tuple(files),
    )
    verify_two_stage_snapshot(root, manifest)
    return manifest


def verify_two_stage_snapshot(
    root: Path,
    manifest: TwoStageSourceManifest,
) -> VerifiedTwoStageSnapshot:
    if not isinstance(root, Path):
        root = Path(root)
    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError("two-stage snapshot verification requires TwoStageSourceManifest")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("two-stage source root must be a directory")

    repository_revision = _git_head(root)
    if repository_revision != manifest.revision:
        raise ValueError("two-stage source repository revision does not match manifest")

    identities: list[tuple[str, str]] = []
    for item in manifest.files:
        path = root / item.path
        if not path.is_file():
            raise ValueError(f"declared two-stage source file is missing: {item.path}")
        actual_sha = _git_blob_sha(path)
        if actual_sha != item.git_blob_sha:
            raise ValueError(f"declared two-stage source blob changed: {item.path}")
        identities.append((item.path, actual_sha))

    declared_evidence = frozenset(
        item.path for item in manifest.files if item.purpose == "scientific_evidence"
    )
    discovered_evidence = _discovered_evidence_paths(root)
    if declared_evidence != discovered_evidence:
        missing = tuple(sorted(discovered_evidence - declared_evidence))
        extra = tuple(sorted(declared_evidence - discovered_evidence))
        raise ValueError(
            "two-stage scientific evidence manifest does not match local main-task files: "
            f"unmanifested={missing}, invalid_or_missing={extra}"
        )

    verified_file_identities = tuple(sorted(identities))
    source_snapshot_hash = stable_content_hash(
        {
            "manifest_hash": manifest.content_hash,
            "repository_revision": repository_revision,
            "verified_file_identities": verified_file_identities,
        }
    )
    return VerifiedTwoStageSnapshot(
        root=str(root),
        manifest_hash=manifest.content_hash,
        repository_revision=repository_revision,
        source_snapshot_hash=source_snapshot_hash,
        verified_file_identities=verified_file_identities,
    )


def _manifest_payload(manifest: TwoStageSourceManifest) -> dict[str, object]:
    payload = manifest.identity_payload()
    payload["files"] = [item.identity_payload() for item in manifest.files]
    payload["content_hash"] = manifest.content_hash
    return payload


def write_two_stage_source_manifest(
    path: Path,
    manifest: TwoStageSourceManifest,
) -> None:
    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError("manifest writer requires TwoStageSourceManifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_manifest_payload(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_two_stage_source_manifest(path: Path) -> TwoStageSourceManifest:
    path = Path(path)
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("two-stage source manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("two-stage source manifest root must be an object")
    expected_keys = {
        "name",
        "version",
        "repository",
        "revision",
        "license_reference",
        "files",
        "content_hash",
    }
    if set(payload) != expected_keys:
        raise ValueError("two-stage source manifest keys are not canonical")
    raw_files = payload["files"]
    if not isinstance(raw_files, list):
        raise ValueError("two-stage source manifest files must be a list")
    files: list[TwoStageSourceFile] = []
    file_keys = {
        "path",
        "git_blob_sha",
        "task_variant",
        "source_participant_id",
        "purpose",
    }
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != file_keys:
            raise ValueError("two-stage source file entry keys are not canonical")
        files.append(
            TwoStageSourceFile(
                path=raw["path"],
                git_blob_sha=raw["git_blob_sha"],
                task_variant=raw["task_variant"],
                source_participant_id=raw["source_participant_id"],
                purpose=raw["purpose"],
            )
        )
    manifest = TwoStageSourceManifest(
        name=payload["name"],
        version=payload["version"],
        repository=payload["repository"],
        revision=payload["revision"],
        license_reference=payload["license_reference"],
        files=tuple(files),
    )
    if payload["content_hash"] != manifest.content_hash:
        raise ValueError("two-stage source manifest declared content hash does not match")
    return manifest


__all__ = [
    "TwoStageSourceFile",
    "TwoStageSourceManifest",
    "VerifiedTwoStageSnapshot",
    "freeze_feher_hare_v1_source_manifest",
    "load_two_stage_source_manifest",
    "verify_two_stage_snapshot",
    "write_two_stage_source_manifest",
]
