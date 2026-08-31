"""Authoritative append-only Git attempt ledger for transfer V1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Protocol

from narrative_dynamics.contracts import stable_content_hash


LEDGER_BRANCH = "ledger/narrative-cross-dataset-transfer-v1"
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_ATTEMPT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class TransferAttemptEventType(str, Enum):
    FINAL_STARTED = "FINAL_STARTED"
    FINAL_VAULT_OPENED = "FINAL_VAULT_OPENED"
    RUN_PROGRESS = "RUN_PROGRESS"
    PREDICTION_SEALED = "PREDICTION_SEALED"
    SCORE_SEALED = "SCORE_SEALED"
    FINAL_COMPLETED = "FINAL_COMPLETED"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    EXACT_REPLAY_ALLOWED = "EXACT_REPLAY_ALLOWED"


_TERMINAL_TYPES = frozenset(
    {
        TransferAttemptEventType.FINAL_COMPLETED,
        TransferAttemptEventType.REVISION_REQUIRED,
        TransferAttemptEventType.EXACT_REPLAY_ALLOWED,
    }
)
_DETAIL_SCHEMAS = {
    TransferAttemptEventType.FINAL_STARTED: ("authorization_receipt_hash",),
    TransferAttemptEventType.FINAL_VAULT_OPENED: ("final_commitment_hash",),
    TransferAttemptEventType.RUN_PROGRESS: ("completed_model_runs",),
    TransferAttemptEventType.PREDICTION_SEALED: ("prediction_artifact_hash",),
    TransferAttemptEventType.SCORE_SEALED: ("score_artifact_hash",),
    TransferAttemptEventType.FINAL_COMPLETED: ("report_hash",),
    TransferAttemptEventType.REVISION_REQUIRED: ("failure_class",),
    TransferAttemptEventType.EXACT_REPLAY_ALLOWED: ("failure_class",),
}


class ConcurrentAttemptError(RuntimeError):
    """Raised when the authoritative ledger head changes before append."""


def _strict_fields(
    payload: object,
    *,
    expected: tuple[str, ...],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = frozenset(payload)
    required = frozenset(expected)
    if actual != required:
        missing = tuple(sorted(required - actual))
        unknown = tuple(sorted(actual - required))
        raise ValueError(
            f"{label} fields mismatch: missing={missing!r}, unknown={unknown!r}"
        )
    return payload


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return value


def _revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase 40-hex revision")
    return value


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp_utc must be an immutable UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("timestamp_utc must be canonical ISO-8601 UTC") from exc
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise ValueError("timestamp_utc must be canonical ISO-8601 UTC")
    return value


def _event_type(value: object) -> TransferAttemptEventType:
    if isinstance(value, TransferAttemptEventType):
        return value
    try:
        return TransferAttemptEventType(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown transfer attempt event type") from exc


def _canonical_details(
    event_type: TransferAttemptEventType,
    details: object,
) -> tuple[tuple[str, object], ...]:
    values = _strict_fields(
        details,
        expected=_DETAIL_SCHEMAS[event_type],
        label=f"{event_type.value} details",
    )
    canonical: list[tuple[str, object]] = []
    for name in _DETAIL_SCHEMAS[event_type]:
        value = values[name]
        if name.endswith("_hash"):
            value = _hash(value, label=name)
        elif name == "completed_model_runs":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("completed_model_runs must be a positive integer")
        elif name == "failure_class":
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("failure_class must be non-empty trimmed text")
            if (
                event_type is TransferAttemptEventType.EXACT_REPLAY_ALLOWED
                and value != "INFRASTRUCTURE"
            ):
                raise ValueError("EXACT_REPLAY_ALLOWED must be infrastructure-only")
        canonical.append((name, value))
    return tuple(canonical)


@dataclass(frozen=True)
class TransferAttemptEvent:
    event_type: TransferAttemptEventType
    parent_ledger_head: str
    scientific_revision: str
    attempt_id: str
    timestamp_utc: str
    repository_receipt_hash: str
    run_receipt_hash: str
    job_receipt_hash: str
    details: tuple[tuple[str, object], ...]
    event_payload_hash: str
    attestation_hash: str

    def __post_init__(self) -> None:
        event_type = _event_type(self.event_type)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(
            self,
            "parent_ledger_head",
            _revision(self.parent_ledger_head, label="parent_ledger_head"),
        )
        object.__setattr__(
            self,
            "scientific_revision",
            _revision(self.scientific_revision, label="scientific_revision"),
        )
        if not isinstance(self.attempt_id, str) or _ATTEMPT_PATTERN.fullmatch(self.attempt_id) is None:
            raise ValueError("attempt_id must be a canonical opaque identifier")
        object.__setattr__(self, "timestamp_utc", _timestamp(self.timestamp_utc))
        for field_name in (
            "repository_receipt_hash",
            "run_receipt_hash",
            "job_receipt_hash",
            "event_payload_hash",
            "attestation_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        try:
            raw_details = dict(self.details)
        except (TypeError, ValueError) as exc:
            raise ValueError("event details must contain name/value pairs") from exc
        canonical = _canonical_details(event_type, raw_details)
        if len(canonical) != len(tuple(self.details)):
            raise ValueError("event details contain duplicate fields")
        object.__setattr__(self, "details", canonical)
        if self.event_payload_hash != self.recompute_event_payload_hash():
            raise ValueError("event payload hash mismatch")
        if self.attestation_hash != self.recompute_attestation_hash():
            raise ValueError("event attestation hash mismatch")

    def _unsigned_payload(self) -> dict[str, object]:
        return {
            "event_type": self.event_type.value,
            "parent_ledger_head": self.parent_ledger_head,
            "scientific_revision": self.scientific_revision,
            "attempt_id": self.attempt_id,
            "timestamp_utc": self.timestamp_utc,
            "repository_receipt_hash": self.repository_receipt_hash,
            "run_receipt_hash": self.run_receipt_hash,
            "job_receipt_hash": self.job_receipt_hash,
            "details": dict(self.details),
        }

    def recompute_event_payload_hash(self) -> str:
        return stable_content_hash(self._unsigned_payload())

    def recompute_attestation_hash(self) -> str:
        return stable_content_hash(
            {
                "domain": "cross-dataset-transfer-attempt-attestation-v1",
                "event_payload_hash": self.event_payload_hash,
                "repository_receipt_hash": self.repository_receipt_hash,
                "run_receipt_hash": self.run_receipt_hash,
                "job_receipt_hash": self.job_receipt_hash,
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unsigned_payload(),
            "event_payload_hash": self.event_payload_hash,
            "attestation_hash": self.attestation_hash,
        }

    @classmethod
    def create(
        cls,
        *,
        event_type: TransferAttemptEventType,
        parent_ledger_head: str,
        scientific_revision: str,
        attempt_id: str,
        timestamp_utc: str,
        repository_receipt_hash: str,
        run_receipt_hash: str,
        job_receipt_hash: str,
        details: Mapping[str, object],
    ) -> TransferAttemptEvent:
        canonical_type = _event_type(event_type)
        canonical_details = _canonical_details(canonical_type, details)
        unsigned = {
            "event_type": canonical_type.value,
            "parent_ledger_head": _revision(
                parent_ledger_head,
                label="parent_ledger_head",
            ),
            "scientific_revision": _revision(
                scientific_revision,
                label="scientific_revision",
            ),
            "attempt_id": attempt_id,
            "timestamp_utc": _timestamp(timestamp_utc),
            "repository_receipt_hash": _hash(
                repository_receipt_hash,
                label="repository_receipt_hash",
            ),
            "run_receipt_hash": _hash(run_receipt_hash, label="run_receipt_hash"),
            "job_receipt_hash": _hash(job_receipt_hash, label="job_receipt_hash"),
            "details": dict(canonical_details),
        }
        payload_hash = stable_content_hash(unsigned)
        attestation = stable_content_hash(
            {
                "domain": "cross-dataset-transfer-attempt-attestation-v1",
                "event_payload_hash": payload_hash,
                "repository_receipt_hash": unsigned["repository_receipt_hash"],
                "run_receipt_hash": unsigned["run_receipt_hash"],
                "job_receipt_hash": unsigned["job_receipt_hash"],
            }
        )
        return cls(
            event_type=canonical_type,
            parent_ledger_head=unsigned["parent_ledger_head"],
            scientific_revision=unsigned["scientific_revision"],
            attempt_id=attempt_id,
            timestamp_utc=unsigned["timestamp_utc"],
            repository_receipt_hash=unsigned["repository_receipt_hash"],
            run_receipt_hash=unsigned["run_receipt_hash"],
            job_receipt_hash=unsigned["job_receipt_hash"],
            details=canonical_details,
            event_payload_hash=payload_hash,
            attestation_hash=attestation,
        )

    @classmethod
    def from_payload(cls, payload: object) -> TransferAttemptEvent:
        expected = (
            "event_type",
            "parent_ledger_head",
            "scientific_revision",
            "attempt_id",
            "timestamp_utc",
            "repository_receipt_hash",
            "run_receipt_hash",
            "job_receipt_hash",
            "details",
            "event_payload_hash",
            "attestation_hash",
        )
        values = _strict_fields(payload, expected=expected, label="attempt event")
        if not isinstance(values["details"], Mapping):
            raise TypeError("event details must be a mapping")
        return cls(
            event_type=values["event_type"],
            parent_ledger_head=values["parent_ledger_head"],
            scientific_revision=values["scientific_revision"],
            attempt_id=values["attempt_id"],
            timestamp_utc=values["timestamp_utc"],
            repository_receipt_hash=values["repository_receipt_hash"],
            run_receipt_hash=values["run_receipt_hash"],
            job_receipt_hash=values["job_receipt_hash"],
            details=tuple(values["details"].items()),
            event_payload_hash=values["event_payload_hash"],
            attestation_hash=values["attestation_hash"],
        )


@dataclass(frozen=True)
class TransferAttemptHistory:
    genesis_hash: str
    head_hash: str
    events: tuple[TransferAttemptEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "genesis_hash", _revision(self.genesis_hash, label="genesis_hash"))
        object.__setattr__(self, "head_hash", _revision(self.head_hash, label="head_hash"))
        events = tuple(self.events)
        if not all(isinstance(event, TransferAttemptEvent) for event in events):
            raise TypeError("history events must be TransferAttemptEvent values")
        started_by_attempt: set[str] = set()
        terminal_by_attempt: set[str] = set()
        for event in events:
            if event.event_payload_hash != event.recompute_event_payload_hash():
                raise ValueError("corrupt ledger event payload hash")
            if event.attestation_hash != event.recompute_attestation_hash():
                raise ValueError("corrupt ledger event attestation")
            if event.event_type is TransferAttemptEventType.FINAL_STARTED:
                if event.attempt_id in started_by_attempt:
                    raise ValueError("duplicate FINAL_STARTED for attempt")
                started_by_attempt.add(event.attempt_id)
            elif event.attempt_id not in started_by_attempt:
                raise ValueError("attempt event precedes FINAL_STARTED")
            if event.event_type in _TERMINAL_TYPES:
                if event.attempt_id in terminal_by_attempt:
                    raise ValueError("attempt has more than one terminal event")
                terminal_by_attempt.add(event.attempt_id)
            elif event.attempt_id in terminal_by_attempt:
                raise ValueError("non-terminal event follows terminal event")
        object.__setattr__(self, "events", events)

    @property
    def final_started_count(self) -> int:
        return sum(event.event_type is TransferAttemptEventType.FINAL_STARTED for event in self.events)

    @property
    def authorization_consumptions(self) -> int:
        return self.final_started_count

    @property
    def final_projection_openings(self) -> int:
        return sum(
            event.event_type is TransferAttemptEventType.FINAL_VAULT_OPENED
            for event in self.events
        )

    @property
    def completed_model_runs(self) -> int:
        counts = tuple(
            int(dict(event.details)["completed_model_runs"])
            for event in self.events
            if event.event_type is TransferAttemptEventType.RUN_PROGRESS
        )
        return max(counts, default=0)

    @property
    def prediction_artifact_hashes(self) -> tuple[str, ...]:
        return tuple(
            str(dict(event.details)["prediction_artifact_hash"])
            for event in self.events
            if event.event_type is TransferAttemptEventType.PREDICTION_SEALED
        )

    @property
    def score_artifact_hashes(self) -> tuple[str, ...]:
        return tuple(
            str(dict(event.details)["score_artifact_hash"])
            for event in self.events
            if event.event_type is TransferAttemptEventType.SCORE_SEALED
        )

    @property
    def terminal_count(self) -> int:
        return sum(event.event_type in _TERMINAL_TYPES for event in self.events)

    @property
    def last_event(self) -> TransferAttemptEvent | None:
        return self.events[-1] if self.events else None


class TransferAttemptStore(Protocol):
    def head(self) -> str:
        ...

    def history(self) -> TransferAttemptHistory:
        ...

    def compare_and_append(
        self,
        expected_head_hash: str,
        event: TransferAttemptEvent,
    ) -> str:
        ...


class GitTransferAttemptStore:
    """A local client for the authoritative non-force Git ledger branch."""

    branch_name = LEDGER_BRANCH

    def __init__(self, *, remote: Path, workspace: Path) -> None:
        self._remote = Path(remote).resolve()
        self._workspace = Path(workspace).resolve()
        if not self._remote.is_dir():
            raise ValueError("ledger remote must be an existing Git directory")
        if self._workspace.exists():
            raise ValueError("ledger workspace must not already exist")
        self._run(
            ("git", "clone", "--no-checkout", str(self._remote), str(self._workspace)),
            cwd=self._workspace.parent,
        )
        if self._remote_head_or_none() is None:
            self._initialize_genesis()
        self._fetch()
        self._run(
            (
                "git",
                "checkout",
                "-B",
                "transfer-ledger-client",
                f"refs/remotes/origin/{self.branch_name}",
            )
        )

    def _git_environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "GIT_AUTHOR_NAME": "Cross Dataset Transfer Ledger",
            "GIT_AUTHOR_EMAIL": "transfer-ledger@example.invalid",
            "GIT_COMMITTER_NAME": "Cross Dataset Transfer Ledger",
            "GIT_COMMITTER_EMAIL": "transfer-ledger@example.invalid",
            "GIT_AUTHOR_DATE": "2026-08-30T12:00:00Z",
            "GIT_COMMITTER_DATE": "2026-08-30T12:00:00Z",
        }

    def _run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self._workspace if cwd is None else cwd,
            check=check,
            capture_output=True,
            text=True,
            env=self._git_environment(),
        )

    def _remote_head_or_none(self) -> str | None:
        result = self._run(
            (
                "git",
                "ls-remote",
                "--heads",
                "origin",
                f"refs/heads/{self.branch_name}",
            ),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"unable to read authoritative ledger head: {result.stderr.strip()}")
        line = result.stdout.strip()
        if not line:
            return None
        head = line.split()[0]
        return _revision(head, label="authoritative ledger head")

    def _initialize_genesis(self) -> None:
        self._run(("git", "checkout", "--orphan", "transfer-ledger-genesis"))
        genesis_path = self._workspace / "ledger" / "genesis.json"
        genesis_path.parent.mkdir(parents=True, exist_ok=True)
        genesis_path.write_text(
            json.dumps(
                {
                    "branch": self.branch_name,
                    "schema_version": "cross-dataset-transfer-ledger-v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        self._run(("git", "add", "ledger/genesis.json"))
        self._run(("git", "commit", "-m", "initialize transfer attempt ledger"))
        result = self._run(
            (
                "git",
                "push",
                "origin",
                f"HEAD:refs/heads/{self.branch_name}",
            ),
            check=False,
        )
        if result.returncode != 0 and self._remote_head_or_none() is None:
            raise RuntimeError(f"unable to initialize transfer ledger: {result.stderr.strip()}")

    def _fetch(self) -> None:
        result = self._run(
            (
                "git",
                "fetch",
                "--no-tags",
                "origin",
                f"refs/heads/{self.branch_name}:refs/remotes/origin/{self.branch_name}",
            ),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"unable to fetch authoritative ledger: {result.stderr.strip()}")

    def head(self) -> str:
        head = self._remote_head_or_none()
        if head is None:
            raise ValueError("authoritative ledger branch is empty")
        return head

    @property
    def genesis_hash(self) -> str:
        self._fetch()
        result = self._run(
            (
                "git",
                "rev-list",
                "--max-parents=0",
                f"refs/remotes/origin/{self.branch_name}",
            )
        )
        roots = tuple(line for line in result.stdout.splitlines() if line)
        if len(roots) != 1:
            raise ValueError("corrupt ledger genesis history")
        return _revision(roots[0], label="genesis_hash")

    def history(self) -> TransferAttemptHistory:
        self._fetch()
        head = _revision(
            self._run(
                ("git", "rev-parse", f"refs/remotes/origin/{self.branch_name}")
            ).stdout.strip(),
            label="fetched ledger head",
        )
        result = self._run(
            (
                "git",
                "rev-list",
                "--reverse",
                f"refs/remotes/origin/{self.branch_name}",
            )
        )
        commits = tuple(line for line in result.stdout.splitlines() if line)
        if not commits:
            raise ValueError("corrupt ledger branch has no genesis")
        genesis = commits[0]
        events: list[TransferAttemptEvent] = []
        previous = genesis
        for commit in commits[1:]:
            parent_result = self._run(("git", "rev-parse", f"{commit}^"), check=False)
            if parent_result.returncode != 0 or parent_result.stdout.strip() != previous:
                raise ValueError("corrupt ledger parent chain")
            paths_result = self._run(
                ("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
            )
            paths = tuple(path for path in paths_result.stdout.splitlines() if path)
            if len(paths) != 1 or not re.fullmatch(
                r"events/sha256_[0-9a-f]{64}\.json",
                paths[0],
            ):
                raise ValueError("corrupt ledger commit does not contain one canonical event")
            payload_result = self._run(("git", "show", f"{commit}:{paths[0]}"), check=False)
            if payload_result.returncode != 0:
                raise ValueError("corrupt ledger event object is missing")
            try:
                raw_payload = json.loads(payload_result.stdout)
                event = TransferAttemptEvent.from_payload(raw_payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("corrupt ledger event payload") from exc
            if event.parent_ledger_head != previous:
                raise ValueError("corrupt ledger event parent binding")
            expected_path = f"events/{event.event_payload_hash.replace(':', '_')}.json"
            if paths[0] != expected_path:
                raise ValueError("corrupt ledger event filename binding")
            events.append(event)
            previous = commit
        if previous != head:
            raise ValueError("corrupt ledger head binding")
        return TransferAttemptHistory(
            genesis_hash=genesis,
            head_hash=head,
            events=tuple(events),
        )

    def compare_and_append(
        self,
        expected_head_hash: str,
        event: TransferAttemptEvent,
    ) -> str:
        expected = _revision(expected_head_hash, label="expected_head_hash")
        if not isinstance(event, TransferAttemptEvent):
            raise TypeError("event must be TransferAttemptEvent")
        authoritative = self.head()
        if authoritative != expected:
            raise ConcurrentAttemptError("authoritative transfer ledger head changed")
        if event.parent_ledger_head != expected:
            raise ValueError("event parent does not match expected ledger head")
        current_history = self.history()
        if current_history.head_hash != expected:
            raise ConcurrentAttemptError("authoritative transfer ledger changed during validation")
        TransferAttemptHistory(
            genesis_hash=current_history.genesis_hash,
            head_hash=expected,
            events=current_history.events + (event,),
        )

        self._fetch()
        self._run(("git", "checkout", "-B", "transfer-ledger-client", expected))
        event_path = self._workspace / "events" / f"{event.event_payload_hash.replace(':', '_')}.json"
        if event_path.exists():
            raise ValueError("ledger event already exists")
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(
            json.dumps(event.to_payload(), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        relative_path = event_path.relative_to(self._workspace).as_posix()
        self._run(("git", "add", "--", relative_path))
        self._run(("git", "commit", "-m", f"transfer event: {event.event_type.value}"))
        new_head = _revision(
            self._run(("git", "rev-parse", "HEAD")).stdout.strip(),
            label="new ledger head",
        )
        push = self._run(
            (
                "git",
                "push",
                "origin",
                f"{new_head}:refs/heads/{self.branch_name}",
            ),
            check=False,
        )
        if push.returncode != 0:
            self._fetch()
            raise ConcurrentAttemptError("non-fast-forward transfer ledger append rejected")
        verified = self.head()
        if verified != new_head:
            raise ConcurrentAttemptError("authoritative ledger head did not retain appended event")
        return new_head
