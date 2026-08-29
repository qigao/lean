from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.observations.dataset import _freeze_mapping
from narrative_dynamics.observations.external import EXTERNAL_CLAIM_SCOPE
from narrative_dynamics.observations.release import ProtocolRelease, WitnessReceipt


_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_OSF_PROVIDER = "osf-registration"
_OSF_AUTHORITY = "public-registration"
_BUNDLE_VERSION = "two-stage-osf-bundle-v1"
_VERIFIER_VERSION = "two-stage-osf-registration-verifier-v1"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _hashes(values: object, *, label: str) -> tuple[str, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be iterable") from error
    if not raw:
        raise ValueError(f"{label} must be non-empty")
    canonical = tuple(sorted(_hash(value, label=label) for value in raw))
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{label} must be unique")
    return canonical


@dataclass(frozen=True)
class TwoStageOSFBundle:
    source_manifest_hash: str
    source_snapshot_hash: str
    transform_hash: str
    participant_assignment_hash: str
    dataset_hash: str
    final_target_hash: str
    frozen_candidate_hashes: tuple[str, ...]
    brier_protocol_hash: str
    log_protocol_hash: str
    external_preregistration_hash: str
    brier_release_hash: str
    log_release_hash: str
    repository_revision: str
    claim_scope: str
    scientific_contract: Mapping[str, object]

    def __post_init__(self) -> None:
        for field_name, label in (
            ("source_manifest_hash", "OSF bundle source manifest hash"),
            ("source_snapshot_hash", "OSF bundle source snapshot hash"),
            ("transform_hash", "OSF bundle transform hash"),
            ("participant_assignment_hash", "OSF bundle participant assignment hash"),
            ("dataset_hash", "OSF bundle dataset hash"),
            ("final_target_hash", "OSF bundle final target hash"),
            ("brier_protocol_hash", "OSF bundle Brier protocol hash"),
            ("log_protocol_hash", "OSF bundle Log protocol hash"),
            ("external_preregistration_hash", "OSF bundle external preregistration hash"),
            ("brier_release_hash", "OSF bundle Brier release hash"),
            ("log_release_hash", "OSF bundle Log release hash"),
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=label),
            )
        object.__setattr__(
            self,
            "frozen_candidate_hashes",
            _hashes(
                self.frozen_candidate_hashes,
                label="OSF bundle frozen candidate hash",
            ),
        )
        object.__setattr__(
            self,
            "repository_revision",
            _text(self.repository_revision, label="OSF bundle repository revision"),
        )
        if self.claim_scope != EXTERNAL_CLAIM_SCOPE:
            raise ValueError("OSF bundle claim scope is fixed")
        contract = _freeze_mapping(
            self.scientific_contract,
            label="OSF bundle scientific contract",
        )
        if not contract:
            raise ValueError("OSF bundle scientific contract must be non-empty")
        object.__setattr__(self, "scientific_contract", contract)
        if self.brier_protocol_hash == self.log_protocol_hash:
            raise ValueError("OSF bundle sibling protocol hashes must be distinct")
        if self.brier_release_hash == self.log_release_hash:
            raise ValueError("OSF bundle sibling release hashes must be distinct")

    def identity_payload(self) -> dict[str, object]:
        return {
            "version": _BUNDLE_VERSION,
            "source_manifest_hash": self.source_manifest_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "transform_hash": self.transform_hash,
            "participant_assignment_hash": self.participant_assignment_hash,
            "dataset_hash": self.dataset_hash,
            "final_target_hash": self.final_target_hash,
            "frozen_candidate_hashes": self.frozen_candidate_hashes,
            "brier_protocol_hash": self.brier_protocol_hash,
            "log_protocol_hash": self.log_protocol_hash,
            "external_preregistration_hash": self.external_preregistration_hash,
            "brier_release_hash": self.brier_release_hash,
            "log_release_hash": self.log_release_hash,
            "repository_revision": self.repository_revision,
            "claim_scope": self.claim_scope,
            "scientific_contract": self.scientific_contract,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class OSFRegistrationProof:
    registration_reference: str
    registered_at: str
    bundle_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registration_reference",
            _text(self.registration_reference, label="OSF registration reference"),
        )
        object.__setattr__(
            self,
            "registered_at",
            _text(self.registered_at, label="OSF registration time"),
        )
        object.__setattr__(
            self,
            "bundle_hash",
            _hash(self.bundle_hash, label="OSF registration bundle hash"),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "registration_reference": self.registration_reference,
            "registered_at": self.registered_at,
            "bundle_hash": self.bundle_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class OSFRegistrationVerifier:
    proof: OSFRegistrationProof

    def __post_init__(self) -> None:
        if not isinstance(self.proof, OSFRegistrationProof):
            raise TypeError("OSF verifier requires OSFRegistrationProof")

    def manifest_identity(self) -> dict[str, object]:
        return {
            "name": "osf-registration-verifier",
            "version": _VERIFIER_VERSION,
            "registration_reference": self.proof.registration_reference,
            "registered_at": self.proof.registered_at,
            "bundle_hash": self.proof.bundle_hash,
            "proof_hash": self.proof.content_hash,
        }

    def verify(self, release: ProtocolRelease, receipt: WitnessReceipt) -> bool:
        if not isinstance(release, ProtocolRelease):
            raise TypeError("OSF verifier release must be ProtocolRelease")
        if not isinstance(receipt, WitnessReceipt):
            raise TypeError("OSF verifier receipt must be WitnessReceipt")
        if receipt.provider != _OSF_PROVIDER:
            return False
        if receipt.authority != _OSF_AUTHORITY:
            return False
        if receipt.subject_hash != release.content_hash:
            return False
        if receipt.reference != self.proof.registration_reference:
            return False
        if receipt.claimed_at != self.proof.registered_at:
            return False
        if set(receipt.proof) != {"bundle_hash"}:
            return False
        return receipt.proof["bundle_hash"] == self.proof.bundle_hash


__all__ = [
    "OSFRegistrationProof",
    "OSFRegistrationVerifier",
    "TwoStageOSFBundle",
]
