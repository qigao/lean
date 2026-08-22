from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit

from narrative_dynamics.attestation._common import (
    validated_commit,
    validated_text,
)
from narrative_dynamics.contracts import stable_content_hash


REPOSITORY_IDENTITY_SCHEMA_VERSION = 1
_SCP_REMOTE_PATTERN = re.compile(r"^(?:[^@]+@)?([^:]+):(.+)$")


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
            validated_text(self.provider, label="repository identity provider"),
        )
        if self.repository is not None:
            object.__setattr__(
                self,
                "repository",
                validated_text(self.repository, label="repository identity name"),
            )
        if self.ref is not None:
            object.__setattr__(
                self,
                "ref",
                validated_text(self.ref, label="repository identity ref"),
            )
        object.__setattr__(
            self,
            "checkout_commit",
            validated_commit(self.checkout_commit, label="repository checkout commit"),
        )
        object.__setattr__(
            self,
            "source_commit",
            validated_commit(self.source_commit, label="repository source commit"),
        )
        object.__setattr__(
            self,
            "base_commit",
            validated_commit(self.base_commit, label="repository base commit"),
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
            return validated_commit(raw, label=f"pull request {side} commit")
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
    """Strip credentials and host-local path details from a git remote."""

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
            checkout = validated_commit(
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
        checkout = validated_commit(
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
    "REPOSITORY_IDENTITY_SCHEMA_VERSION",
    "RepositoryIdentity",
    "detect_repository_identity",
]
