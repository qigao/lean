from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

from narrative_dynamics.contracts import TraceEvent, stable_content_hash


IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION = 1
REPOSITORY_IDENTITY_SCHEMA_VERSION = 1
RESULT_ARTIFACT_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SCP_REMOTE_PATTERN = re.compile(r"^(?:[^@]+@)?([^:]+):(.+)$")
_MAX_IMPLEMENTATION_ARTIFACT_BYTES = 16 * 1024 * 1024


class ImplementationAttestationUnavailable(RuntimeError):
    """The trusted parent cannot map an execution target to readable source bytes."""


def _validated_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot have surrounding whitespace")
    return value


def _validated_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _validated_commit(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a git commit or None")
    normalized = value.lower()
    if _GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a full 40-character git commit")
    return normalized


@dataclass(frozen=True)
class ImplementationArtifact:
    """Digest of one logical Python module without exposing host-local paths."""

    locator: str
    sha256: str
    byte_count: int
    kind: str = "python_source"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "locator",
            _validated_text(self.locator, label="implementation artifact locator"),
        )
        object.__setattr__(
            self,
            "sha256",
            _validated_hash(self.sha256, label="implementation artifact hash"),
        )
        if (
            not isinstance(self.byte_count, int)
            or isinstance(self.byte_count, bool)
            or self.byte_count < 0
        ):
            raise ValueError("implementation artifact byte count must be non-negative")
        object.__setattr__(
            self,
            "kind",
            _validated_text(self.kind, label="implementation artifact kind"),
        )

    def manifest_identity(self) -> dict[str, object]:
        return {
            "locator": self.locator,
            "kind": self.kind,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class ImplementationAttestation:
    """Measured identity of directly located Python module files.

    V1 deliberately does not claim a transitive dependency closure, package
    signature verification, native-library attestation, or proof that a worker
    loaded the same bytes after this trusted-parent snapshot.
    """

    target: str
    artifacts: tuple[ImplementationArtifact, ...]
    schema_version: int = IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target",
            _validated_text(self.target, label="implementation target"),
        )
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError(
                "implementation attestation schema version must be positive"
            )
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("implementation attestation requires at least one artifact")
        if any(not isinstance(item, ImplementationArtifact) for item in artifacts):
            raise TypeError(
                "implementation attestation artifacts must be ImplementationArtifact values"
            )
        artifacts = tuple(sorted(artifacts, key=lambda item: item.locator))
        if len({item.locator for item in artifacts}) != len(artifacts):
            raise ValueError("implementation artifact locators must be unique")
        object.__setattr__(self, "artifacts", artifacts)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "scope": "direct_python_modules",
            "artifacts": tuple(item.manifest_identity() for item in self.artifacts),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._payload())

    def manifest_identity(self) -> dict[str, object]:
        return {**self._payload(), "content_hash": self.content_hash}


@dataclass(frozen=True)
class RepositoryIdentity:
    """Repository checkout identity measured by the trusted parent runtime."""

    provider: str
    repository: str | None = None
    checkout_commit: str | None = None
    source_commit: str | None = None
    base_commit: str | None = None
    ref: str | None = None
    dirty: bool | None = None
    schema_version: int = REPOSITORY_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            _validated_text(self.provider, label="repository identity provider"),
        )
        if self.repository is not None:
            object.__setattr__(
                self,
                "repository",
                _validated_text(self.repository, label="repository identity name"),
            )
        if self.ref is not None:
            object.__setattr__(
                self,
                "ref",
                _validated_text(self.ref, label="repository identity ref"),
            )
        object.__setattr__(
            self,
            "checkout_commit",
            _validated_commit(self.checkout_commit, label="repository checkout commit"),
        )
        object.__setattr__(
            self,
            "source_commit",
            _validated_commit(self.source_commit, label="repository source commit"),
        )
        object.__setattr__(
            self,
            "base_commit",
            _validated_commit(self.base_commit, label="repository base commit"),
        )
        if self.dirty is not None and not isinstance(self.dirty, bool):
            raise TypeError("repository dirty flag must be boolean or None")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("repository identity schema version must be positive")

    @property
    def available(self) -> bool:
        return self.checkout_commit is not None or self.source_commit is not None

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "status": "measured" if self.available else "unavailable",
            "repository": self.repository,
            "checkout_commit": self.checkout_commit,
            "source_commit": self.source_commit,
            "base_commit": self.base_commit,
            "ref": self.ref,
            "dirty": self.dirty,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._payload())

    def manifest_identity(self) -> dict[str, object]:
        return {**self._payload(), "content_hash": self.content_hash}


@dataclass(frozen=True)
class ResultArtifact:
    """Content identity for every canonical event and the complete outcome."""

    events_hash: str
    outcome_hash: str
    model_run_hash: str
    schema_version: int = RESULT_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("result artifact schema version must be positive")
        for attribute, label in (
            ("events_hash", "result events hash"),
            ("outcome_hash", "result outcome hash"),
            ("model_run_hash", "result model-run hash"),
        ):
            object.__setattr__(
                self,
                attribute,
                _validated_hash(getattr(self, attribute), label=label),
            )

    @staticmethod
    def _event_payload(events: Sequence[TraceEvent]) -> tuple[dict[str, object], ...]:
        values = tuple(events)
        if any(not isinstance(event, TraceEvent) for event in values):
            raise TypeError("result artifact events must contain TraceEvent values")
        return tuple(
            {"tick": event.tick, "kind": event.kind, "data": event.data}
            for event in values
        )

    @classmethod
    def from_result(
        cls,
        events: Sequence[TraceEvent],
        outcome: Mapping[str, object],
    ) -> "ResultArtifact":
        if not isinstance(outcome, Mapping):
            raise TypeError("result artifact outcome must be a mapping")
        event_payload = cls._event_payload(events)
        return cls(
            events_hash=stable_content_hash(event_payload),
            outcome_hash=stable_content_hash(outcome),
            model_run_hash=stable_content_hash(
                {"events": event_payload, "outcome": outcome}
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_content_hash(
            {
                "schema_version": self.schema_version,
                "events_hash": self.events_hash,
                "outcome_hash": self.outcome_hash,
                "model_run_hash": self.model_run_hash,
            }
        )

    def manifest_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "events_hash": self.events_hash,
            "outcome_hash": self.outcome_hash,
            "model_run_hash": self.model_run_hash,
            "content_hash": self.content_hash,
        }

    def matches(
        self,
        events: Sequence[TraceEvent],
        outcome: Mapping[str, object],
    ) -> bool:
        return self == self.from_result(events, outcome)


def _module_source_path(module_name: str) -> Path:
    """Resolve an ordinary filesystem module without importing external code."""

    candidates: list[Path] = []
    loaded = sys.modules.get(module_name)
    loaded_path = getattr(loaded, "__file__", None)
    if isinstance(loaded_path, str) and loaded_path:
        candidates.append(Path(loaded_path))

    parts = module_name.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ImplementationAttestationUnavailable(
            f"implementation module {module_name!r} is not a valid module name"
        )
    for entry in sys.path:
        base = Path(entry or os.curdir)
        candidates.append(base.joinpath(*parts[:-1], f"{parts[-1]}.py"))
        candidates.append(base.joinpath(*parts, "__init__.py"))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate.suffix in {".pyc", ".pyo"}:
            try:
                candidate = Path(importlib.util.source_from_cache(str(candidate)))
            except (ValueError, NotImplementedError):
                pass
        marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            return resolved

    raise ImplementationAttestationUnavailable(
        f"implementation module {module_name!r} has no readable source file"
    )


def _measure_module(module_name: str) -> ImplementationArtifact:
    module_name = _validated_text(module_name, label="implementation module name")
    path = _module_source_path(module_name)
    try:
        byte_count = path.stat().st_size
        if byte_count > _MAX_IMPLEMENTATION_ARTIFACT_BYTES:
            raise ImplementationAttestationUnavailable(
                f"implementation module {module_name!r} exceeds the measurement limit"
            )
        payload = path.read_bytes()
    except ImplementationAttestationUnavailable:
        raise
    except OSError as error:
        raise ImplementationAttestationUnavailable(
            f"cannot read implementation module {module_name!r}"
        ) from error
    if len(payload) > _MAX_IMPLEMENTATION_ARTIFACT_BYTES:
        raise ImplementationAttestationUnavailable(
            f"implementation module {module_name!r} exceeds the measurement limit"
        )
    return ImplementationArtifact(
        locator=f"python-module:{module_name}",
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        byte_count=len(payload),
    )


def _unwrap_registered_source(source: object) -> object:
    nested = getattr(source, "source", None)
    contract = getattr(source, "contract", None)
    return nested if nested is not None and contract is not None else source


def measure_implementation(source: object) -> ImplementationAttestation:
    """Measure directly located module bytes for a model, factory, or process source."""

    source = _unwrap_registered_source(source)
    factory_path = getattr(source, "factory", None)
    worker_module = getattr(source, "worker_module", None)
    if isinstance(factory_path, str) and isinstance(worker_module, str):
        factory_module, separator, factory_attribute = factory_path.partition(":")
        if not separator or not factory_module or not factory_attribute:
            raise ImplementationAttestationUnavailable(
                "subprocess implementation factory is not a module:attribute path"
            )
        artifacts = {
            item.locator: item
            for item in (
                _measure_module(factory_module),
                _measure_module(worker_module),
            )
        }
        return ImplementationAttestation(
            target=f"python-subprocess-factory:{factory_path}",
            artifacts=tuple(artifacts.values()),
        )

    create = getattr(source, "create", None)
    if callable(create):
        module_name = getattr(create, "__module__", None)
        qualname = getattr(create, "__qualname__", None)
        if not isinstance(module_name, str) or not module_name:
            raise ImplementationAttestationUnavailable(
                "model factory callable has no module identity"
            )
        if not isinstance(qualname, str) or not qualname:
            qualname = create.__class__.__qualname__
        return ImplementationAttestation(
            target=f"python-callable:{module_name}.{qualname}",
            artifacts=(_measure_module(module_name),),
        )

    implementation_type = source if isinstance(source, type) else source.__class__
    module_name = getattr(implementation_type, "__module__", None)
    qualname = getattr(implementation_type, "__qualname__", None)
    if not isinstance(module_name, str) or not module_name:
        raise ImplementationAttestationUnavailable(
            "implementation class has no module identity"
        )
    if not isinstance(qualname, str) or not qualname:
        raise ImplementationAttestationUnavailable(
            "implementation class has no qualified name"
        )
    return ImplementationAttestation(
        target=f"python-class:{module_name}.{qualname}",
        artifacts=(_measure_module(module_name),),
    )


def implementation_attestation_identity(source: object) -> dict[str, object]:
    """Return an explicit measured or unavailable manifest identity."""

    try:
        measured = measure_implementation(source)
    except ImplementationAttestationUnavailable as error:
        return {
            "schema_version": IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": str(error),
        }
    return {**measured.manifest_identity(), "status": "measured"}


def _pull_request_commits(path: str | None) -> tuple[str | None, str | None]:
    if not path:
        return None, None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, None
    pull_request = payload.get("pull_request") if isinstance(payload, Mapping) else None
    if not isinstance(pull_request, Mapping):
        return None, None

    def read(side: str) -> str | None:
        value = pull_request.get(side)
        raw = value.get("sha") if isinstance(value, Mapping) else None
        try:
            return _validated_commit(raw, label=f"pull request {side} commit")
        except (TypeError, ValueError):
            return None

    return read("head"), read("base")


def _git_output(cwd: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(cwd), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _sanitize_repository_remote(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    scp_match = _SCP_REMOTE_PATTERN.fullmatch(candidate)
    if scp_match is not None and "://" not in candidate:
        host, path = scp_match.groups()
        return f"{host}/{path.removesuffix('.git')}"
    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit(
            (parsed.scheme, host, parsed.path.removesuffix(".git"), "", "")
        )
    return Path(candidate).name.removesuffix(".git") or None


def detect_repository_identity(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> RepositoryIdentity:
    """Measure GitHub Actions PR identity, then fall back to local git."""

    environment = os.environ if environ is None else environ
    if environment.get("GITHUB_ACTIONS", "").lower() == "true":
        try:
            checkout = _validated_commit(
                environment.get("GITHUB_SHA"),
                label="GitHub Actions checkout commit",
            )
        except (TypeError, ValueError):
            checkout = None
        source, base = _pull_request_commits(environment.get("GITHUB_EVENT_PATH"))
        return RepositoryIdentity(
            provider="github_actions",
            repository=environment.get("GITHUB_REPOSITORY") or None,
            checkout_commit=checkout,
            source_commit=source or checkout,
            base_commit=base,
            ref=environment.get("GITHUB_REF") or None,
            dirty=None,
        )

    root = Path.cwd() if cwd is None else Path(cwd)
    try:
        checkout = _validated_commit(
            _git_output(root, "rev-parse", "HEAD"),
            label="local repository checkout commit",
        )
    except (TypeError, ValueError):
        checkout = None
    if checkout is None:
        return RepositoryIdentity(provider="unavailable")

    status = _git_output(root, "status", "--porcelain", "--untracked-files=normal")
    return RepositoryIdentity(
        provider="git",
        repository=_sanitize_repository_remote(
            _git_output(root, "remote", "get-url", "origin")
        ),
        checkout_commit=checkout,
        source_commit=checkout,
        ref=_git_output(root, "rev-parse", "--abbrev-ref", "HEAD"),
        dirty=bool(status),
    )


__all__ = [
    "IMPLEMENTATION_ATTESTATION_SCHEMA_VERSION",
    "REPOSITORY_IDENTITY_SCHEMA_VERSION",
    "RESULT_ARTIFACT_SCHEMA_VERSION",
    "ImplementationArtifact",
    "ImplementationAttestation",
    "ImplementationAttestationUnavailable",
    "RepositoryIdentity",
    "ResultArtifact",
    "detect_repository_identity",
    "implementation_attestation_identity",
    "measure_implementation",
]
