"""Exact post-preflight owner authorization for transfer V1 locked FINAL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Mapping

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_ledger import GitTransferAttemptStore
from narrative_dynamics.cross_dataset_release import DualTransferPreflight


AUTHORIZATION_HEADER = (
    "I authorize Cross-dataset Transfer V1 locked FINAL_TEST execution."
)
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_LOGIN_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")


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


def _login(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _LOGIN_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical GitHub login")
    return value


def _timestamp(value: object, *, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from exc
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    return value, parsed


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _expected_text(
    scientific_sha: str,
    lock_commit: str,
    preflight_hash: str,
    brier_release_hash: str,
    log_release_hash: str,
) -> str:
    return "\n".join(
        (
            AUTHORIZATION_HEADER,
            f"scientific_sha={scientific_sha}",
            f"lock_commit={lock_commit}",
            f"preflight_hash={preflight_hash}",
            f"brier_release_hash={brier_release_hash}",
            f"log_release_hash={log_release_hash}",
        )
    )


@dataclass(frozen=True)
class TransferAuthorizationReceipt:
    issue_number: int
    comment_id: int
    comment_url: str
    author_login: str
    comment_created_at: str
    exact_text: str
    scientific_revision: str
    lock_commit: str
    preflight_hash: str
    brier_release_hash: str
    log_release_hash: str
    pre_authorization_ledger_head: str

    def __post_init__(self) -> None:
        if self.issue_number != 43:
            raise ValueError("transfer authorization is restricted to issue 43")
        object.__setattr__(
            self,
            "comment_id",
            _positive_integer(self.comment_id, label="comment_id"),
        )
        expected_url = (
            "https://github.com/qigao/lean/issues/43#issuecomment-"
            f"{self.comment_id}"
        )
        if self.comment_url != expected_url:
            raise ValueError("authorization comment URL does not bind issue/comment ID")
        object.__setattr__(
            self,
            "author_login",
            _login(self.author_login, label="author_login"),
        )
        canonical_time, _ = _timestamp(
            self.comment_created_at,
            label="comment_created_at",
        )
        object.__setattr__(self, "comment_created_at", canonical_time)
        object.__setattr__(
            self,
            "scientific_revision",
            _revision(self.scientific_revision, label="scientific_revision"),
        )
        object.__setattr__(
            self,
            "lock_commit",
            _revision(self.lock_commit, label="lock_commit"),
        )
        for field_name in (
            "preflight_hash",
            "brier_release_hash",
            "log_release_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        object.__setattr__(
            self,
            "pre_authorization_ledger_head",
            _revision(
                self.pre_authorization_ledger_head,
                label="pre_authorization_ledger_head",
            ),
        )
        expected_text = _expected_text(
            self.scientific_revision,
            self.lock_commit,
            self.preflight_hash,
            self.brier_release_hash,
            self.log_release_hash,
        )
        if self.exact_text != expected_text:
            raise ValueError("authorization receipt text is not the exact six-line text")

    def to_payload(self) -> dict[str, object]:
        return {
            "issue_number": self.issue_number,
            "comment_id": self.comment_id,
            "comment_url": self.comment_url,
            "author_login": self.author_login,
            "comment_created_at": self.comment_created_at,
            "exact_text": self.exact_text,
            "scientific_revision": self.scientific_revision,
            "lock_commit": self.lock_commit,
            "preflight_hash": self.preflight_hash,
            "brier_release_hash": self.brier_release_hash,
            "log_release_hash": self.log_release_hash,
            "pre_authorization_ledger_head": self.pre_authorization_ledger_head,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> TransferAuthorizationReceipt:
        expected = (
            "issue_number",
            "comment_id",
            "comment_url",
            "author_login",
            "comment_created_at",
            "exact_text",
            "scientific_revision",
            "lock_commit",
            "preflight_hash",
            "brier_release_hash",
            "log_release_hash",
            "pre_authorization_ledger_head",
        )
        values = _strict_fields(
            payload,
            expected=expected,
            label="transfer authorization receipt",
        )
        return cls(
            issue_number=values["issue_number"],
            comment_id=values["comment_id"],
            comment_url=values["comment_url"],
            author_login=values["author_login"],
            comment_created_at=values["comment_created_at"],
            exact_text=values["exact_text"],
            scientific_revision=values["scientific_revision"],
            lock_commit=values["lock_commit"],
            preflight_hash=values["preflight_hash"],
            brier_release_hash=values["brier_release_hash"],
            log_release_hash=values["log_release_hash"],
            pre_authorization_ledger_head=values[
                "pre_authorization_ledger_head"
            ],
        )


def parse_transfer_authorization(
    *,
    issue_number: int,
    comment: Mapping[str, object],
    authorized_owner: str,
    preflight: DualTransferPreflight,
    lock_commit: str,
    store: GitTransferAttemptStore,
) -> TransferAuthorizationReceipt:
    if issue_number != 43:
        raise ValueError("transfer authorization must come from issue 43")
    owner = _login(authorized_owner, label="authorized_owner")
    if not isinstance(preflight, DualTransferPreflight):
        raise TypeError("preflight must be DualTransferPreflight")
    canonical_lock = _revision(lock_commit, label="lock_commit")
    if not isinstance(store, GitTransferAttemptStore):
        raise TypeError("authorization requires GitTransferAttemptStore")
    current_head = store.head()
    history = store.history()
    if (
        current_head != preflight.ledger_head_hash
        or history.head_hash != current_head
        or history.events
    ):
        raise ValueError("authorization preflight has a stale ledger head or was consumed")

    expected_comment_fields = (
        "comment_id",
        "html_url",
        "author_login",
        "created_at",
        "updated_at",
        "body",
        "deleted",
    )
    values = _strict_fields(
        comment,
        expected=expected_comment_fields,
        label="GitHub authorization comment",
    )
    comment_id = _positive_integer(values["comment_id"], label="comment_id")
    expected_url = (
        "https://github.com/qigao/lean/issues/43#issuecomment-"
        f"{comment_id}"
    )
    if values["html_url"] != expected_url:
        raise ValueError("authorization comment URL identity changed")
    author = _login(values["author_login"], label="comment author login")
    if author != owner:
        raise ValueError("authorization comment author is not the authorized owner")
    if values["deleted"] is not False:
        raise ValueError("authorization comment is deleted")
    created_text, created = _timestamp(values["created_at"], label="created_at")
    updated_text, _updated = _timestamp(values["updated_at"], label="updated_at")
    if created_text != updated_text:
        raise ValueError("authorization comment was edited after creation")
    _preflight_text, preflight_completed = _timestamp(
        preflight.completed_at_utc,
        label="preflight completed_at_utc",
    )
    if created <= preflight_completed:
        raise ValueError("authorization comment must be created strictly after preflight")

    expected_text = _expected_text(
        preflight.scientific_revision,
        canonical_lock,
        preflight.content_hash,
        preflight.brier_release_hash,
        preflight.log_release_hash,
    )
    body = values["body"]
    if not isinstance(body, str) or body != expected_text:
        raise ValueError("authorization comment must equal the exact six-line text")
    return TransferAuthorizationReceipt(
        issue_number=issue_number,
        comment_id=comment_id,
        comment_url=expected_url,
        author_login=author,
        comment_created_at=created_text,
        exact_text=body,
        scientific_revision=preflight.scientific_revision,
        lock_commit=canonical_lock,
        preflight_hash=preflight.content_hash,
        brier_release_hash=preflight.brier_release_hash,
        log_release_hash=preflight.log_release_hash,
        pre_authorization_ledger_head=preflight.ledger_head_hash,
    )
