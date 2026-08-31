"""Role-isolated projections and a sealed FINAL vault for transfer V1."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import secrets
from typing import Mapping, Protocol

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_privacy import (
    ParticipantRole,
    PublicSplitManifest,
    RoleAssignmentIndex,
)
from narrative_dynamics.cross_dataset_source import (
    CanonicalTransferTrial,
    validate_canonical_trials,
)


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_CAPABILITY_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _projection_hash(rows: tuple[CanonicalTransferTrial, ...], role: str) -> str:
    return stable_content_hash(
        {
            "role": role,
            "row_commitments": [row.row_commitment for row in rows],
        }
    )


@dataclass(frozen=True, repr=False)
class TrainProjection:
    """A TRAIN-only capability intended for refit construction."""

    _rows: tuple[CanonicalTransferTrial, ...] = field(repr=False)
    projection_hash: str

    def __post_init__(self) -> None:
        rows = validate_canonical_trials(self._rows)
        expected = _projection_hash(rows, ParticipantRole.TRAIN.value)
        if _hash(self.projection_hash, label="projection_hash") != expected:
            raise ValueError("TRAIN projection hash mismatch")
        object.__setattr__(self, "_rows", rows)

    @property
    def trial_count(self) -> int:
        return len(self._rows)

    def identity_payload(self) -> dict[str, object]:
        return {
            "role": ParticipantRole.TRAIN.value,
            "trial_count": self.trial_count,
            "projection_hash": self.projection_hash,
        }

    def consume_for_refit(self) -> tuple[CanonicalTransferTrial, ...]:
        return self._rows

    def __repr__(self) -> str:
        return (
            "TrainProjection(rows=<sealed>, "
            f"projection_hash={self.projection_hash!r})"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("TrainProjection is not serializable")


@dataclass(frozen=True, repr=False)
class SelectionProjection:
    """A SELECTION_VALIDATION-only capability for refit construction."""

    _rows: tuple[CanonicalTransferTrial, ...] = field(repr=False)
    projection_hash: str

    def __post_init__(self) -> None:
        rows = validate_canonical_trials(self._rows)
        expected = _projection_hash(
            rows,
            ParticipantRole.SELECTION_VALIDATION.value,
        )
        if _hash(self.projection_hash, label="projection_hash") != expected:
            raise ValueError("SELECTION_VALIDATION projection hash mismatch")
        object.__setattr__(self, "_rows", rows)

    @property
    def trial_count(self) -> int:
        return len(self._rows)

    def identity_payload(self) -> dict[str, object]:
        return {
            "role": ParticipantRole.SELECTION_VALIDATION.value,
            "trial_count": self.trial_count,
            "projection_hash": self.projection_hash,
        }

    def consume_for_refit(self) -> tuple[CanonicalTransferTrial, ...]:
        return self._rows

    def __repr__(self) -> str:
        return (
            "SelectionProjection(rows=<sealed>, "
            f"projection_hash={self.projection_hash!r})"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("SelectionProjection is not serializable")


@dataclass(frozen=True)
class FinalProjectionCommitment:
    source_identity_hash: str
    split_manifest_hash: str
    trial_count: int
    projection_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_identity_hash",
            "split_manifest_hash",
            "projection_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        object.__setattr__(
            self,
            "trial_count",
            _positive_integer(self.trial_count, label="trial_count"),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_identity_hash": self.source_identity_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "trial_count": self.trial_count,
            "projection_hash": self.projection_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> FinalProjectionCommitment:
        expected = (
            "source_identity_hash",
            "split_manifest_hash",
            "trial_count",
            "projection_hash",
        )
        values = _strict_fields(
            payload,
            expected=expected,
            label="FINAL projection commitment",
        )
        return cls(
            source_identity_hash=values["source_identity_hash"],
            split_manifest_hash=values["split_manifest_hash"],
            trial_count=values["trial_count"],
            projection_hash=values["projection_hash"],
        )


class FinalVaultHandle:
    """Opaque object capability; only its provisioning backend may consume it."""

    __slots__ = ("_capability_id", "commitment_hash")

    def __init__(self, capability_id: str, commitment_hash: str) -> None:
        if (
            not isinstance(capability_id, str)
            or _CAPABILITY_PATTERN.fullmatch(capability_id) is None
        ):
            raise ValueError("capability_id must be 64 lowercase hex characters")
        self._capability_id = capability_id
        self.commitment_hash = _hash(
            commitment_hash,
            label="commitment_hash",
        )

    @classmethod
    def _for_backend(
        cls,
        *,
        capability_id: str,
        commitment_hash: str,
    ) -> FinalVaultHandle:
        return cls(capability_id, commitment_hash)

    def __repr__(self) -> str:
        return f"FinalVaultHandle(commitment_hash={self.commitment_hash!r})"

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("FinalVaultHandle is not serializable")


@dataclass(frozen=True)
class FinalUnlockGrant:
    scientific_revision: str
    ledger_head_hash: str
    preflight_hash: str
    authorization_receipt_hash: str
    brier_release_hash: str
    log_release_hash: str
    lock_commit: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scientific_revision",
            _revision(self.scientific_revision, label="scientific_revision"),
        )
        for field_name in (
            "ledger_head_hash",
            "preflight_hash",
            "authorization_receipt_hash",
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
            "lock_commit",
            _revision(self.lock_commit, label="lock_commit"),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "scientific_revision": self.scientific_revision,
            "ledger_head_hash": self.ledger_head_hash,
            "preflight_hash": self.preflight_hash,
            "authorization_receipt_hash": self.authorization_receipt_hash,
            "brier_release_hash": self.brier_release_hash,
            "log_release_hash": self.log_release_hash,
            "lock_commit": self.lock_commit,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> FinalUnlockGrant:
        expected = (
            "scientific_revision",
            "ledger_head_hash",
            "preflight_hash",
            "authorization_receipt_hash",
            "brier_release_hash",
            "log_release_hash",
            "lock_commit",
        )
        values = _strict_fields(payload, expected=expected, label="FINAL unlock grant")
        return cls(
            scientific_revision=values["scientific_revision"],
            ledger_head_hash=values["ledger_head_hash"],
            preflight_hash=values["preflight_hash"],
            authorization_receipt_hash=values["authorization_receipt_hash"],
            brier_release_hash=values["brier_release_hash"],
            log_release_hash=values["log_release_hash"],
            lock_commit=values["lock_commit"],
        )


class FinalWorkerProjection:
    """Worker-only one-shot projection of sealed FINAL trials."""

    __slots__ = ("_consumed", "_rows", "commitment_hash")

    def __init__(
        self,
        *,
        rows: tuple[CanonicalTransferTrial, ...],
        commitment_hash: str,
    ) -> None:
        self._rows = validate_canonical_trials(rows)
        self.commitment_hash = _hash(
            commitment_hash,
            label="commitment_hash",
        )
        self._consumed = False

    def consume_once(self) -> tuple[CanonicalTransferTrial, ...]:
        if self._consumed:
            raise RuntimeError("FINAL worker projection already consumed")
        self._consumed = True
        rows = self._rows
        self._rows = ()
        return rows

    def __repr__(self) -> str:
        state = "consumed" if self._consumed else "sealed"
        return (
            "FinalWorkerProjection(rows=<sealed>, "
            f"commitment_hash={self.commitment_hash!r}, state={state!r})"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("FinalWorkerProjection is not serializable")


@dataclass(frozen=True)
class PreparedCrossDatasetTransferV1:
    source_identity_hash: str
    split_manifest: PublicSplitManifest
    train: TrainProjection
    selection_validation: SelectionProjection
    final_commitment: FinalProjectionCommitment
    final_vault_handle: FinalVaultHandle

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_identity_hash",
            _hash(self.source_identity_hash, label="source_identity_hash"),
        )
        if not isinstance(self.split_manifest, PublicSplitManifest):
            raise TypeError("split_manifest must be PublicSplitManifest")
        if not isinstance(self.train, TrainProjection):
            raise TypeError("train must be TrainProjection")
        if not isinstance(self.selection_validation, SelectionProjection):
            raise TypeError("selection_validation must be SelectionProjection")
        if not isinstance(self.final_commitment, FinalProjectionCommitment):
            raise TypeError("final_commitment must be FinalProjectionCommitment")
        if not isinstance(self.final_vault_handle, FinalVaultHandle):
            raise TypeError("final_vault_handle must be FinalVaultHandle")
        if self.final_commitment.source_identity_hash != self.source_identity_hash:
            raise ValueError("FINAL commitment source identity mismatch")
        if self.final_commitment.split_manifest_hash != self.split_manifest.content_hash:
            raise ValueError("FINAL commitment split manifest mismatch")
        if self.final_vault_handle.commitment_hash != self.final_commitment.content_hash:
            raise ValueError("FINAL vault handle commitment mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_identity_hash": self.source_identity_hash,
            "split_manifest": self.split_manifest.to_payload(),
            "train": self.train.identity_payload(),
            "selection_validation": self.selection_validation.identity_payload(),
            "final_commitment": self.final_commitment.to_payload(),
            "final_vault_handle": {
                "commitment_hash": self.final_vault_handle.commitment_hash,
            },
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


class FinalVaultBackend(Protocol):
    @property
    def final_projection_opened(self) -> bool:
        ...

    def bind_unlock_policy(
        self,
        handle: FinalVaultHandle,
        grant: FinalUnlockGrant,
    ) -> None:
        ...

    def unlock(
        self,
        handle: FinalVaultHandle,
        grant: FinalUnlockGrant,
    ) -> FinalWorkerProjection:
        ...


class _ServiceOwnedFinalVaultBackend:
    __slots__ = (
        "_consumed",
        "_expected_grant",
        "_handle",
        "_opened",
        "_rows",
    )

    def __init__(
        self,
        *,
        handle: FinalVaultHandle,
        rows: tuple[CanonicalTransferTrial, ...],
    ) -> None:
        self._handle = handle
        self._rows = rows
        self._expected_grant: FinalUnlockGrant | None = None
        self._opened = False
        self._consumed = False

    @property
    def final_projection_opened(self) -> bool:
        return self._opened

    def _require_exact_handle(self, handle: FinalVaultHandle) -> None:
        if not isinstance(handle, FinalVaultHandle) or handle is not self._handle:
            raise ValueError("unknown FINAL vault handle")

    def bind_unlock_policy(
        self,
        handle: FinalVaultHandle,
        grant: FinalUnlockGrant,
    ) -> None:
        self._require_exact_handle(handle)
        if self._expected_grant is not None:
            raise RuntimeError("FINAL vault policy already bound")
        if not isinstance(grant, FinalUnlockGrant):
            raise TypeError("grant must be FinalUnlockGrant")
        self._expected_grant = grant

    def unlock(
        self,
        handle: FinalVaultHandle,
        grant: FinalUnlockGrant,
    ) -> FinalWorkerProjection:
        self._require_exact_handle(handle)
        if self._consumed:
            raise RuntimeError("FINAL vault handle already consumed")
        if self._expected_grant is None:
            raise RuntimeError("FINAL vault unlock policy is not bound")
        if not isinstance(grant, FinalUnlockGrant):
            raise TypeError("grant must be FinalUnlockGrant")
        if grant != self._expected_grant:
            raise ValueError("FINAL unlock grant mismatch")
        self._opened = True
        self._consumed = True
        rows = self._rows
        self._rows = ()
        return FinalWorkerProjection(
            rows=rows,
            commitment_hash=handle.commitment_hash,
        )


def provision_transfer_capabilities(
    *,
    source_identity_hash: str,
    trials: tuple[CanonicalTransferTrial, ...],
    role_index: RoleAssignmentIndex,
    split_manifest: PublicSplitManifest,
) -> tuple[PreparedCrossDatasetTransferV1, FinalVaultBackend]:
    source_hash = _hash(source_identity_hash, label="source_identity_hash")
    rows = validate_canonical_trials(trials)
    if not isinstance(role_index, RoleAssignmentIndex):
        raise TypeError("role_index must be RoleAssignmentIndex")
    if not isinstance(split_manifest, PublicSplitManifest):
        raise TypeError("split_manifest must be PublicSplitManifest")

    assignments = role_index._private_assignments()
    assigned_identities = {
        (row.participant_id, row.source_stratum) for row in assignments
    }
    trial_identities = {
        (row.participant_key, row.source_stratum) for row in rows
    }
    if trial_identities != assigned_identities:
        raise ValueError("canonical trials must cover every assigned participant")

    by_role: dict[ParticipantRole, list[CanonicalTransferTrial]] = {
        role: [] for role in ParticipantRole
    }
    for row in rows:
        role = role_index.role_for(row.participant_key, row.source_stratum)
        by_role[role].append(row)

    expected_counts = {
        (stratum, role): count
        for stratum, train, selection, final in split_manifest.role_counts
        for role, count in (
            (ParticipantRole.TRAIN, train),
            (ParticipantRole.SELECTION_VALIDATION, selection),
            (ParticipantRole.FINAL_TEST, final),
        )
    }
    actual_counts: dict[tuple[str, ParticipantRole], int] = {}
    for assignment in assignments:
        key = (assignment.source_stratum, assignment.role)
        actual_counts[key] = actual_counts.get(key, 0) + 1
    if actual_counts != expected_counts:
        raise ValueError("role index and public split manifest counts disagree")

    train_rows = tuple(by_role[ParticipantRole.TRAIN])
    selection_rows = tuple(by_role[ParticipantRole.SELECTION_VALIDATION])
    final_rows = tuple(by_role[ParticipantRole.FINAL_TEST])
    train = TrainProjection(
        _rows=train_rows,
        projection_hash=_projection_hash(train_rows, ParticipantRole.TRAIN.value),
    )
    selection = SelectionProjection(
        _rows=selection_rows,
        projection_hash=_projection_hash(
            selection_rows,
            ParticipantRole.SELECTION_VALIDATION.value,
        ),
    )
    final_commitment = FinalProjectionCommitment(
        source_identity_hash=source_hash,
        split_manifest_hash=split_manifest.content_hash,
        trial_count=len(final_rows),
        projection_hash=_projection_hash(
            final_rows,
            ParticipantRole.FINAL_TEST.value,
        ),
    )
    handle = FinalVaultHandle._for_backend(
        capability_id=secrets.token_hex(32),
        commitment_hash=final_commitment.content_hash,
    )
    backend = _ServiceOwnedFinalVaultBackend(handle=handle, rows=final_rows)
    prepared = PreparedCrossDatasetTransferV1(
        source_identity_hash=source_hash,
        split_manifest=split_manifest,
        train=train,
        selection_validation=selection,
        final_commitment=final_commitment,
        final_vault_handle=handle,
    )
    return prepared, backend
