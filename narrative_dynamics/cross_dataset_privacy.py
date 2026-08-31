"""Private participant assignment and public split evidence for transfer V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import math
import re
import secrets
from typing import Mapping

from narrative_dynamics.contracts import stable_content_hash


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SECRET_BYTES = 32


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


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty without surrounding whitespace")
    return value


def _hash(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _HASH_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return text


def _count(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


class ParticipantRole(str, Enum):
    TRAIN = "TRAIN"
    SELECTION_VALIDATION = "SELECTION_VALIDATION"
    FINAL_TEST = "FINAL_TEST"


@dataclass(frozen=True, repr=False)
class PrivateParticipant:
    participant_id: str = field(repr=False)
    source_stratum: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_id",
            _text(self.participant_id, label="participant_id"),
        )
        object.__setattr__(
            self,
            "source_stratum",
            _text(self.source_stratum, label="source_stratum"),
        )

    def __repr__(self) -> str:
        return f"PrivateParticipant(participant_id=<redacted>, source_stratum={self.source_stratum!r})"


class RestrictedStudySecret:
    """In-memory HMAC capability with explicit zeroizing close semantics."""

    __slots__ = ("_closed", "_key", "_key_commitment")

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes):
            raise TypeError("study secret must be bytes")
        if len(key) != _SECRET_BYTES:
            raise ValueError("study secret must contain exactly 32 bytes")
        self._key = bytearray(key)
        self._closed = False
        self._key_commitment = "sha256:" + hashlib.sha256(
            b"cross-dataset-study-key-v1\x00" + key
        ).hexdigest()

    @classmethod
    def from_bytes(cls, key: bytes) -> RestrictedStudySecret:
        return cls(key)

    @classmethod
    def generate(cls) -> RestrictedStudySecret:
        return cls(secrets.token_bytes(_SECRET_BYTES))

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def key_commitment(self) -> str:
        return self._key_commitment

    def digest(self, message: bytes) -> bytes:
        if self._closed:
            raise RuntimeError("study secret is closed")
        if not isinstance(message, bytes):
            raise TypeError("HMAC message must be bytes")
        return hmac.new(bytes(self._key), message, hashlib.sha256).digest()

    def commit(self, *parts: str) -> bytes:
        if self._closed:
            raise RuntimeError("study secret is closed")
        if not parts:
            raise ValueError("participant commitment requires message parts")
        canonical = tuple(
            _text(part, label=f"commitment part {index}")
            for index, part in enumerate(parts)
        )
        message = b"cross-dataset-participant-v1\x00" + "\x1f".join(
            canonical
        ).encode("utf-8")
        return self.digest(message)

    def close(self) -> None:
        if self._closed:
            return
        for index in range(len(self._key)):
            self._key[index] = 0
        self._closed = True

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"RestrictedStudySecret(<redacted>, state={state})"

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("RestrictedStudySecret is not serializable")


@dataclass(frozen=True, repr=False)
class _PrivateRoleAssignment:
    participant_id: str = field(repr=False)
    source_stratum: str
    commitment: bytes = field(repr=False)
    role: ParticipantRole

    def __repr__(self) -> str:
        return (
            "_PrivateRoleAssignment(participant_id=<redacted>, "
            f"source_stratum={self.source_stratum!r}, role={self.role.value!r})"
        )


class RoleAssignmentIndex:
    """Private role lookup; this type intentionally has no durable serializer."""

    __slots__ = ("_assignments", "assignment_hash", "role_counts")

    def __init__(
        self,
        assignments: tuple[_PrivateRoleAssignment, ...],
        role_counts: tuple[tuple[str, int, int, int], ...],
    ) -> None:
        self._assignments = assignments
        self.role_counts = role_counts
        self.assignment_hash = stable_content_hash(
            {
                "assignments": [
                    {
                        "participant_id": row.participant_id,
                        "source_stratum": row.source_stratum,
                        "commitment": row.commitment.hex(),
                        "role": row.role.value,
                    }
                    for row in assignments
                ],
                "role_counts": [list(row) for row in role_counts],
            }
        )

    @property
    def has_overlap(self) -> bool:
        identities = tuple(row.participant_id for row in self._assignments)
        return len(identities) != len(set(identities))

    def role_total(self, role: ParticipantRole) -> int:
        if not isinstance(role, ParticipantRole):
            raise TypeError("role must be ParticipantRole")
        return sum(1 for row in self._assignments if row.role is role)

    def role_for(self, participant_id: str, source_stratum: str) -> ParticipantRole:
        participant = _text(participant_id, label="participant_id")
        stratum = _text(source_stratum, label="source_stratum")
        matches = tuple(
            row.role
            for row in self._assignments
            if row.participant_id == participant and row.source_stratum == stratum
        )
        if len(matches) != 1:
            raise KeyError("participant role is not uniquely assigned")
        return matches[0]

    def _private_assignments(self) -> tuple[_PrivateRoleAssignment, ...]:
        return self._assignments

    def __repr__(self) -> str:
        return (
            "RoleAssignmentIndex(assignments=<redacted>, "
            f"role_counts={self.role_counts!r})"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("RoleAssignmentIndex is not serializable")


@dataclass(frozen=True)
class PublicSplitManifest:
    namespace: str
    source_snapshot_hash: str
    role_counts: tuple[tuple[str, int, int, int], ...]
    merkle_root: str
    hmac_algorithm: str
    key_commitment: str
    transform_attestation_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _text(self.namespace, label="namespace"),
        )
        for field_name in (
            "source_snapshot_hash",
            "merkle_root",
            "key_commitment",
            "transform_attestation_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        if self.hmac_algorithm != "HMAC-SHA256":
            raise ValueError("hmac_algorithm must be HMAC-SHA256")
        if not isinstance(self.role_counts, (tuple, list)):
            raise TypeError("role_counts must be a sequence")
        rows: list[tuple[str, int, int, int]] = []
        for index, row in enumerate(self.role_counts):
            if not isinstance(row, (tuple, list)) or len(row) != 4:
                raise ValueError(f"role_counts[{index}] must contain four fields")
            canonical = (
                _text(row[0], label=f"role_counts[{index}].stratum"),
                _count(row[1], label=f"role_counts[{index}].train", minimum=1),
                _count(row[2], label=f"role_counts[{index}].selection", minimum=1),
                _count(row[3], label=f"role_counts[{index}].final", minimum=1),
            )
            rows.append(canonical)
        if not rows or tuple(rows) != tuple(sorted(rows)):
            raise ValueError("role_counts must be non-empty and ordered by stratum")
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("role_counts strata must be unique")
        object.__setattr__(self, "role_counts", tuple(rows))

    def identity_payload(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "source_snapshot_hash": self.source_snapshot_hash,
            "role_counts": [list(row) for row in self.role_counts],
            "merkle_root": self.merkle_root,
            "hmac_algorithm": self.hmac_algorithm,
            "key_commitment": self.key_commitment,
            "transform_attestation_hash": self.transform_attestation_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> PublicSplitManifest:
        expected = (
            "namespace",
            "source_snapshot_hash",
            "role_counts",
            "merkle_root",
            "hmac_algorithm",
            "key_commitment",
            "transform_attestation_hash",
        )
        values = _strict_fields(payload, expected=expected, label="public split manifest")
        return cls(
            namespace=values["namespace"],
            source_snapshot_hash=values["source_snapshot_hash"],
            role_counts=tuple(tuple(row) for row in values["role_counts"]),
            merkle_root=values["merkle_root"],
            hmac_algorithm=values["hmac_algorithm"],
            key_commitment=values["key_commitment"],
            transform_attestation_hash=values["transform_attestation_hash"],
        )


def _role_counts(size: int) -> tuple[int, int, int]:
    train = math.floor(0.60 * size)
    selection = math.floor(0.20 * size)
    final = size - train - selection
    if train == 0 or selection == 0 or final == 0:
        raise ValueError("every source stratum must populate all three roles")
    return train, selection, final


def _merkle_root(assignments: tuple[_PrivateRoleAssignment, ...]) -> str:
    nodes = [
        hashlib.sha256(
            b"cross-dataset-split-leaf-v1\x00"
            + row.commitment
            + b"\x00"
            + row.source_stratum.encode("utf-8")
            + b"\x00"
            + row.role.value.encode("utf-8")
        ).digest()
        for row in assignments
    ]
    nodes.sort()
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"cross-dataset-split-node-v1\x00" + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return "sha256:" + nodes[0].hex()


def assign_participant_roles(
    *,
    namespace: str,
    source_snapshot_hash: str,
    participants: tuple[PrivateParticipant, ...],
    secret: RestrictedStudySecret,
    transform_attestation_hash: str,
) -> tuple[RoleAssignmentIndex, PublicSplitManifest]:
    canonical_namespace = _text(namespace, label="namespace")
    snapshot_hash = _hash(source_snapshot_hash, label="source_snapshot_hash")
    transform_hash = _hash(
        transform_attestation_hash,
        label="transform_attestation_hash",
    )
    if not isinstance(secret, RestrictedStudySecret):
        raise TypeError("secret must be RestrictedStudySecret")
    if secret.closed:
        raise RuntimeError("study secret is closed")
    if not isinstance(participants, (tuple, list)):
        raise TypeError("participants must be a sequence")
    canonical = tuple(participants)
    if not canonical or not all(
        isinstance(row, PrivateParticipant) for row in canonical
    ):
        raise ValueError("participants must contain PrivateParticipant values")
    participant_ids = tuple(row.participant_id for row in canonical)
    if len(participant_ids) != len(set(participant_ids)):
        raise ValueError("participant identifiers must be unique across strata")

    by_stratum: dict[str, list[tuple[bytes, PrivateParticipant]]] = {}
    for participant in canonical:
        commitment = secret.commit(
            canonical_namespace,
            snapshot_hash,
            participant.source_stratum,
            participant.participant_id,
        )
        by_stratum.setdefault(participant.source_stratum, []).append(
            (commitment, participant)
        )

    assignments: list[_PrivateRoleAssignment] = []
    role_count_rows: list[tuple[str, int, int, int]] = []
    for stratum in sorted(by_stratum):
        rows = sorted(by_stratum[stratum], key=lambda row: row[0])
        commitments = tuple(commitment for commitment, _ in rows)
        if len(commitments) != len(set(commitments)):
            raise ValueError("participant commitments must be unique")
        train_count, selection_count, final_count = _role_counts(len(rows))
        role_count_rows.append(
            (stratum, train_count, selection_count, final_count)
        )
        for index, (commitment, participant) in enumerate(rows):
            if index < train_count:
                role = ParticipantRole.TRAIN
            elif index < train_count + selection_count:
                role = ParticipantRole.SELECTION_VALIDATION
            else:
                role = ParticipantRole.FINAL_TEST
            assignments.append(
                _PrivateRoleAssignment(
                    participant_id=participant.participant_id,
                    source_stratum=stratum,
                    commitment=commitment,
                    role=role,
                )
            )

    ordered_assignments = tuple(
        sorted(
            assignments,
            key=lambda row: (row.source_stratum, row.commitment),
        )
    )
    role_counts = tuple(role_count_rows)
    private_index = RoleAssignmentIndex(ordered_assignments, role_counts)
    if private_index.has_overlap:
        raise ValueError("participant roles overlap")
    public_manifest = PublicSplitManifest(
        namespace=canonical_namespace,
        source_snapshot_hash=snapshot_hash,
        role_counts=role_counts,
        merkle_root=_merkle_root(ordered_assignments),
        hmac_algorithm="HMAC-SHA256",
        key_commitment=secret.key_commitment,
        transform_attestation_hash=transform_hash,
    )
    return private_index, public_manifest


__all__ = [
    "ParticipantRole",
    "PrivateParticipant",
    "PublicSplitManifest",
    "RestrictedStudySecret",
    "RoleAssignmentIndex",
    "assign_participant_roles",
]
