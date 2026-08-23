from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.manifest import component_identity

from .dataset import _freeze_mapping
from .preregistration import PreregisteredEvaluationProtocol


PROTOCOL_RELEASE_SCHEMA_VERSION = 1
WITNESS_RECEIPT_SCHEMA_VERSION = 1
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ProtocolReleaseVerificationError(ValueError):
    """A protocol release or its external witness evidence failed validation."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _schema_version(value: object, *, expected: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ValueError(f"unsupported {label} schema version")
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _candidate_hashes(values: object) -> tuple[str, ...]:
    try:
        hashes = tuple(values)
    except TypeError as error:
        raise TypeError("release candidate hashes must be iterable") from error
    if not hashes:
        raise ValueError("release candidate hashes must be non-empty")
    canonical = tuple(_hash(value, label="release candidate hash") for value in hashes)
    if len(set(canonical)) != len(canonical):
        raise ValueError("release candidate hashes must be unique")
    return canonical


@dataclass(frozen=True)
class ProtocolRelease:
    name: str
    version: str
    protocol_hash: str
    dataset_hash: str
    target_spec_hash: str
    candidate_hashes: tuple[str, ...]
    source_revision: Mapping[str, object]
    declared_content_hash: str
    schema_version: int = PROTOCOL_RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="release name"))
        object.__setattr__(self, "version", _text(self.version, label="release version"))
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(
                self.schema_version,
                expected=PROTOCOL_RELEASE_SCHEMA_VERSION,
                label="protocol release",
            ),
        )
        for attribute, label in (
            ("protocol_hash", "release protocol hash"),
            ("dataset_hash", "release dataset hash"),
            ("target_spec_hash", "release target-spec hash"),
        ):
            object.__setattr__(self, attribute, _hash(getattr(self, attribute), label=label))
        object.__setattr__(self, "candidate_hashes", _candidate_hashes(self.candidate_hashes))
        object.__setattr__(
            self,
            "source_revision",
            _freeze_mapping(self.source_revision, label="release source revision"),
        )
        declared = _hash(self.declared_content_hash, label="declared release content hash")
        expected = stable_content_hash(self.identity_payload())
        if declared != expected:
            raise ValueError("declared release content hash does not match payload")
        object.__setattr__(self, "declared_content_hash", declared)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        protocol: PreregisteredEvaluationProtocol,
        source_revision: Mapping[str, object],
    ) -> "ProtocolRelease":
        if not isinstance(protocol, PreregisteredEvaluationProtocol):
            raise TypeError("protocol release requires PreregisteredEvaluationProtocol")
        canonical_name = _text(name, label="release name")
        canonical_version = _text(version, label="release version")
        candidates = tuple(candidate.content_hash for candidate in protocol.candidates)
        frozen_revision = _freeze_mapping(source_revision, label="release source revision")
        payload = {
            "schema_version": PROTOCOL_RELEASE_SCHEMA_VERSION,
            "name": canonical_name,
            "version": canonical_version,
            "protocol_hash": protocol.content_hash,
            "dataset_hash": protocol.dataset_hash,
            "target_spec_hash": protocol.target_spec_hash,
            "candidate_hashes": candidates,
            "source_revision": frozen_revision,
        }
        return cls(
            name=canonical_name,
            version=canonical_version,
            protocol_hash=protocol.content_hash,
            dataset_hash=protocol.dataset_hash,
            target_spec_hash=protocol.target_spec_hash,
            candidate_hashes=candidates,
            source_revision=frozen_revision,
            declared_content_hash=stable_content_hash(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ProtocolRelease":
        if not isinstance(payload, Mapping):
            raise TypeError("protocol release payload must be a mapping")
        required = {
            "schema_version",
            "name",
            "version",
            "protocol_hash",
            "dataset_hash",
            "target_spec_hash",
            "candidate_hashes",
            "source_revision",
            "content_hash",
        }
        if set(payload) != required:
            raise ValueError("protocol release payload fields must match the schema exactly")
        return cls(
            schema_version=payload["schema_version"],
            name=payload["name"],
            version=payload["version"],
            protocol_hash=payload["protocol_hash"],
            dataset_hash=payload["dataset_hash"],
            target_spec_hash=payload["target_spec_hash"],
            candidate_hashes=tuple(payload["candidate_hashes"]),
            source_revision=payload["source_revision"],
            declared_content_hash=payload["content_hash"],
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "protocol_hash": self.protocol_hash,
            "dataset_hash": self.dataset_hash,
            "target_spec_hash": self.target_spec_hash,
            "candidate_hashes": self.candidate_hashes,
            "source_revision": self.source_revision,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            **_thaw(self.identity_payload()),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class WitnessReceipt:
    provider: str
    authority: str
    subject_hash: str
    reference: str
    claimed_at: str
    proof: Mapping[str, object]
    declared_content_hash: str
    schema_version: int = WITNESS_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _text(self.provider, label="witness provider"))
        object.__setattr__(self, "authority", _text(self.authority, label="witness authority"))
        object.__setattr__(self, "reference", _text(self.reference, label="witness reference"))
        object.__setattr__(self, "claimed_at", _text(self.claimed_at, label="witness claimed_at"))
        object.__setattr__(self, "subject_hash", _hash(self.subject_hash, label="witness subject hash"))
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(
                self.schema_version,
                expected=WITNESS_RECEIPT_SCHEMA_VERSION,
                label="witness receipt",
            ),
        )
        object.__setattr__(
            self,
            "proof",
            _freeze_mapping(self.proof, label="witness proof"),
        )
        declared = _hash(self.declared_content_hash, label="declared witness receipt hash")
        expected = stable_content_hash(self.identity_payload())
        if declared != expected:
            raise ValueError("declared witness receipt hash does not match payload")
        object.__setattr__(self, "declared_content_hash", declared)

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        authority: str,
        subject_hash: str,
        reference: str,
        claimed_at: str,
        proof: Mapping[str, object],
    ) -> "WitnessReceipt":
        frozen_proof = _freeze_mapping(proof, label="witness proof")
        payload = {
            "schema_version": WITNESS_RECEIPT_SCHEMA_VERSION,
            "provider": _text(provider, label="witness provider"),
            "authority": _text(authority, label="witness authority"),
            "subject_hash": _hash(subject_hash, label="witness subject hash"),
            "reference": _text(reference, label="witness reference"),
            "claimed_at": _text(claimed_at, label="witness claimed_at"),
            "proof": frozen_proof,
        }
        return cls(
            provider=payload["provider"],
            authority=payload["authority"],
            subject_hash=payload["subject_hash"],
            reference=payload["reference"],
            claimed_at=payload["claimed_at"],
            proof=frozen_proof,
            declared_content_hash=stable_content_hash(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WitnessReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("witness receipt payload must be a mapping")
        required = {
            "schema_version",
            "provider",
            "authority",
            "subject_hash",
            "reference",
            "claimed_at",
            "proof",
            "content_hash",
        }
        if set(payload) != required:
            raise ValueError("witness receipt payload fields must match the schema exactly")
        return cls(
            schema_version=payload["schema_version"],
            provider=payload["provider"],
            authority=payload["authority"],
            subject_hash=payload["subject_hash"],
            reference=payload["reference"],
            claimed_at=payload["claimed_at"],
            proof=payload["proof"],
            declared_content_hash=payload["content_hash"],
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "authority": self.authority,
            "subject_hash": self.subject_hash,
            "reference": self.reference,
            "claimed_at": self.claimed_at,
            "proof": self.proof,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            **_thaw(self.identity_payload()),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class VerifiedProtocolRelease:
    release_hash: str
    protocol_hash: str
    verifier_identity: Mapping[str, object]
    verified_receipt_hashes: tuple[str, ...]
    status: str = "verified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_hash", _hash(self.release_hash, label="verified release hash"))
        object.__setattr__(self, "protocol_hash", _hash(self.protocol_hash, label="verified protocol hash"))
        object.__setattr__(
            self,
            "verifier_identity",
            _freeze_mapping(self.verifier_identity, label="verified release verifier identity"),
        )
        hashes = tuple(
            _hash(value, label="verified receipt hash")
            for value in self.verified_receipt_hashes
        )
        if not hashes:
            raise ValueError("verified protocol release requires at least one receipt")
        if len(set(hashes)) != len(hashes):
            raise ValueError("verified receipt hashes must be unique")
        object.__setattr__(self, "verified_receipt_hashes", tuple(sorted(hashes)))
        if self.status != "verified":
            raise ValueError("verified protocol release status must be 'verified'")

    def require_matches(self, protocol: PreregisteredEvaluationProtocol) -> None:
        if not isinstance(protocol, PreregisteredEvaluationProtocol):
            raise TypeError("verified release comparison requires PreregisteredEvaluationProtocol")
        if self.protocol_hash != protocol.content_hash:
            raise ProtocolReleaseVerificationError("verified release protocol identity changed")

    def identity_payload(self) -> dict[str, object]:
        return {
            "release_hash": self.release_hash,
            "protocol_hash": self.protocol_hash,
            "verifier_identity": self.verifier_identity,
            "verified_receipt_hashes": self.verified_receipt_hashes,
            "status": self.status,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _require_release_matches_protocol(
    release: ProtocolRelease,
    protocol: PreregisteredEvaluationProtocol,
) -> None:
    if release.protocol_hash != protocol.content_hash:
        raise ProtocolReleaseVerificationError("release protocol identity changed")
    if release.dataset_hash != protocol.dataset_hash:
        raise ProtocolReleaseVerificationError("release dataset identity changed")
    if release.target_spec_hash != protocol.target_spec_hash:
        raise ProtocolReleaseVerificationError("release target-spec identity changed")
    expected_candidates = tuple(candidate.content_hash for candidate in protocol.candidates)
    if release.candidate_hashes != expected_candidates:
        raise ProtocolReleaseVerificationError("release candidate identities changed")


def verify_protocol_release(
    release: ProtocolRelease,
    *,
    protocol: PreregisteredEvaluationProtocol,
    receipts: tuple[WitnessReceipt, ...],
    verifier: object,
) -> VerifiedProtocolRelease:
    if not isinstance(release, ProtocolRelease):
        raise TypeError("protocol release verification requires ProtocolRelease")
    if not isinstance(protocol, PreregisteredEvaluationProtocol):
        raise TypeError("protocol release verification requires PreregisteredEvaluationProtocol")
    _require_release_matches_protocol(release, protocol)

    receipt_values = tuple(receipts)
    if not receipt_values:
        raise ProtocolReleaseVerificationError("protocol release requires witness receipts")
    if any(not isinstance(receipt, WitnessReceipt) for receipt in receipt_values):
        raise TypeError("protocol release receipts must be WitnessReceipt values")
    hashes = tuple(receipt.content_hash for receipt in receipt_values)
    if len(set(hashes)) != len(hashes):
        raise ProtocolReleaseVerificationError("protocol release receipts must be unique")
    for receipt in receipt_values:
        if receipt.subject_hash != release.content_hash:
            raise ProtocolReleaseVerificationError("witness receipt targets a different release")

    verify = getattr(verifier, "verify", None)
    if not callable(verify):
        raise TypeError("protocol release verifier must provide verify(release, receipt)")
    verifier_identity = component_identity(verifier)

    accepted: list[str] = []
    for receipt in receipt_values:
        try:
            verified = verify(release, receipt)
        except Exception as error:
            raise ProtocolReleaseVerificationError("external witness verifier failed") from error
        if not isinstance(verified, bool):
            raise ProtocolReleaseVerificationError("external witness verifier must return bool")
        if verified:
            accepted.append(receipt.content_hash)
    if not accepted:
        raise ProtocolReleaseVerificationError("no witness receipt was accepted by the verifier")

    return VerifiedProtocolRelease(
        release_hash=release.content_hash,
        protocol_hash=protocol.content_hash,
        verifier_identity=verifier_identity,
        verified_receipt_hashes=tuple(accepted),
    )


__all__ = [
    "PROTOCOL_RELEASE_SCHEMA_VERSION",
    "WITNESS_RECEIPT_SCHEMA_VERSION",
    "ProtocolRelease",
    "ProtocolReleaseVerificationError",
    "VerifiedProtocolRelease",
    "WitnessReceipt",
    "verify_protocol_release",
]
