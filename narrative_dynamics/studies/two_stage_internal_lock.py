from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.observations.dataset import _freeze_mapping
from narrative_dynamics.observations.external import EXTERNAL_CLAIM_SCOPE
from narrative_dynamics.observations.preregistration import PreregisteredEvaluationProtocol
from narrative_dynamics.observations.release import ProtocolRelease, WitnessReceipt


INTERNAL_LOCK_PROVIDER = "internal-repository-lock"
INTERNAL_LOCK_AUTHORITY = "qigao/lean"
INTERNAL_LOCK_GOVERNANCE_MODE = "internal_locked_final"
_INTERNAL_LOCK_BUNDLE_VERSION = "two-stage-internal-lock-v1"
_INTERNAL_LOCK_VERIFIER_VERSION = "two-stage-internal-lock-verifier-v1"
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def _commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-character lowercase Git commit SHA")
    return value


@dataclass(frozen=True)
class TwoStageInternalLockBundle:
    source_manifest_hash: str
    source_snapshot_hash: str
    transform_hash: str
    participant_assignment_hash: str
    dataset_hash: str
    final_target_hash: str
    frozen_candidate_hashes: tuple[str, ...]
    brier_protocol_hash: str
    log_protocol_hash: str
    evaluation_preregistration_hash: str
    brier_release_hash: str
    log_release_hash: str
    scientific_repository_revision: str
    train_selection_freeze_digest: str
    claim_scope: str
    scientific_contract: Mapping[str, object]
    governance_mode: str = INTERNAL_LOCK_GOVERNANCE_MODE
    external_registration: bool = False
    final_model_execution: bool = False

    def __post_init__(self) -> None:
        for field_name, label in (
            ("source_manifest_hash", "internal lock source manifest hash"),
            ("source_snapshot_hash", "internal lock source snapshot hash"),
            ("transform_hash", "internal lock transform hash"),
            ("participant_assignment_hash", "internal lock participant assignment hash"),
            ("dataset_hash", "internal lock dataset hash"),
            ("final_target_hash", "internal lock final target hash"),
            ("brier_protocol_hash", "internal lock Brier protocol hash"),
            ("log_protocol_hash", "internal lock Log protocol hash"),
            ("evaluation_preregistration_hash", "internal lock evaluation preregistration hash"),
            ("brier_release_hash", "internal lock Brier release hash"),
            ("log_release_hash", "internal lock Log release hash"),
            ("train_selection_freeze_digest", "internal lock TRAIN/SELECTION freeze digest"),
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
                label="internal lock frozen candidate hash",
            ),
        )
        object.__setattr__(
            self,
            "scientific_repository_revision",
            _text(
                self.scientific_repository_revision,
                label="internal lock scientific repository revision",
            ),
        )
        if self.claim_scope != EXTERNAL_CLAIM_SCOPE:
            raise ValueError("internal lock claim scope is fixed")
        if self.governance_mode != INTERNAL_LOCK_GOVERNANCE_MODE:
            raise ValueError("internal lock governance mode is fixed")
        if self.external_registration is not False:
            raise ValueError("internal lock cannot claim external registration")
        if self.final_model_execution is not False:
            raise ValueError("internal lock bundle must precede FINAL model execution")
        contract = _freeze_mapping(
            self.scientific_contract,
            label="internal lock scientific contract",
        )
        if not contract:
            raise ValueError("internal lock scientific contract must be non-empty")
        object.__setattr__(self, "scientific_contract", contract)
        if self.brier_protocol_hash == self.log_protocol_hash:
            raise ValueError("internal lock sibling protocol hashes must be distinct")
        if self.brier_release_hash == self.log_release_hash:
            raise ValueError("internal lock sibling release hashes must be distinct")

    def identity_payload(self) -> dict[str, object]:
        return {
            "version": _INTERNAL_LOCK_BUNDLE_VERSION,
            "source_manifest_hash": self.source_manifest_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "transform_hash": self.transform_hash,
            "participant_assignment_hash": self.participant_assignment_hash,
            "dataset_hash": self.dataset_hash,
            "final_target_hash": self.final_target_hash,
            "frozen_candidate_hashes": self.frozen_candidate_hashes,
            "brier_protocol_hash": self.brier_protocol_hash,
            "log_protocol_hash": self.log_protocol_hash,
            "evaluation_preregistration_hash": self.evaluation_preregistration_hash,
            "brier_release_hash": self.brier_release_hash,
            "log_release_hash": self.log_release_hash,
            "scientific_repository_revision": self.scientific_repository_revision,
            "train_selection_freeze_digest": self.train_selection_freeze_digest,
            "claim_scope": self.claim_scope,
            "scientific_contract": self.scientific_contract,
            "governance_mode": self.governance_mode,
            "external_registration": self.external_registration,
            "final_model_execution": self.final_model_execution,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def create_internal_locked_protocol_release(
    *,
    name: str,
    version: str,
    protocol: PreregisteredEvaluationProtocol,
    repository_revision: str,
    evidence_hash: str,
    preregistration_hash: str,
    score_role: str,
    train_selection_freeze_digest: str,
) -> ProtocolRelease:
    if not isinstance(protocol, PreregisteredEvaluationProtocol):
        raise TypeError("internal locked release requires PreregisteredEvaluationProtocol")
    if score_role not in {"brier", "log"}:
        raise ValueError("internal locked release score role must be brier or log")
    revision = _text(
        repository_revision,
        label="internal locked release repository revision",
    )
    evidence = _hash(evidence_hash, label="internal locked release evidence hash")
    preregistration = _hash(
        preregistration_hash,
        label="internal locked release preregistration hash",
    )
    freeze_digest = _hash(
        train_selection_freeze_digest,
        label="internal locked release TRAIN/SELECTION freeze digest",
    )
    return ProtocolRelease.create(
        name=name,
        version=version,
        protocol=protocol,
        source_revision={
            "repository_revision": revision,
            "external_evidence_declaration_hash": evidence,
            "external_validation_preregistration_hash": preregistration,
            "score_role": score_role,
            "governance_mode": INTERNAL_LOCK_GOVERNANCE_MODE,
            "external_registration": False,
            "train_selection_freeze_digest": freeze_digest,
        },
    )


@dataclass(frozen=True)
class InternalRepositoryLockProof:
    repository: str
    lock_commit: str
    locked_at: str
    lock_payload_hash: str
    scientific_repository_revision: str
    release_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        repository = _text(self.repository, label="internal lock proof repository")
        if repository != INTERNAL_LOCK_AUTHORITY:
            raise ValueError("internal lock proof repository is fixed")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(
            self,
            "lock_commit",
            _commit(self.lock_commit, label="internal lock proof commit"),
        )
        object.__setattr__(
            self,
            "locked_at",
            _text(self.locked_at, label="internal lock proof timestamp"),
        )
        object.__setattr__(
            self,
            "lock_payload_hash",
            _hash(self.lock_payload_hash, label="internal lock proof payload hash"),
        )
        object.__setattr__(
            self,
            "scientific_repository_revision",
            _text(
                self.scientific_repository_revision,
                label="internal lock proof scientific repository revision",
            ),
        )
        releases = _hashes(
            self.release_hashes,
            label="internal lock proof release hash",
        )
        if len(releases) != 2:
            raise ValueError("internal lock proof requires exactly two sibling releases")
        object.__setattr__(self, "release_hashes", releases)

    @property
    def reference(self) -> str:
        return f"https://github.com/{self.repository}/commit/{self.lock_commit}"

    def identity_payload(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "lock_commit": self.lock_commit,
            "locked_at": self.locked_at,
            "lock_payload_hash": self.lock_payload_hash,
            "scientific_repository_revision": self.scientific_repository_revision,
            "release_hashes": self.release_hashes,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class InternalRepositoryLockVerifier:
    proof: InternalRepositoryLockProof

    def __post_init__(self) -> None:
        if not isinstance(self.proof, InternalRepositoryLockProof):
            raise TypeError("internal repository lock verifier requires InternalRepositoryLockProof")

    def manifest_identity(self) -> dict[str, object]:
        return {
            "name": "internal-repository-lock-verifier",
            "version": _INTERNAL_LOCK_VERIFIER_VERSION,
            "repository": self.proof.repository,
            "reference": self.proof.reference,
            "locked_at": self.proof.locked_at,
            "lock_payload_hash": self.proof.lock_payload_hash,
            "scientific_repository_revision": self.proof.scientific_repository_revision,
            "release_hashes": self.proof.release_hashes,
            "proof_hash": self.proof.content_hash,
        }

    def verify(self, release: ProtocolRelease, receipt: WitnessReceipt) -> bool:
        if not isinstance(release, ProtocolRelease):
            raise TypeError("internal lock verifier release must be ProtocolRelease")
        if not isinstance(receipt, WitnessReceipt):
            raise TypeError("internal lock verifier receipt must be WitnessReceipt")
        if receipt.provider != INTERNAL_LOCK_PROVIDER:
            return False
        if receipt.authority != INTERNAL_LOCK_AUTHORITY:
            return False
        if receipt.subject_hash != release.content_hash:
            return False
        if receipt.subject_hash not in self.proof.release_hashes:
            return False
        if receipt.reference != self.proof.reference:
            return False
        if receipt.claimed_at != self.proof.locked_at:
            return False
        required_proof = {
            "lock_payload_hash",
            "scientific_repository_revision",
            "governance_mode",
            "external_registration",
        }
        if set(receipt.proof) != required_proof:
            return False
        if receipt.proof["lock_payload_hash"] != self.proof.lock_payload_hash:
            return False
        if (
            receipt.proof["scientific_repository_revision"]
            != self.proof.scientific_repository_revision
        ):
            return False
        if receipt.proof["governance_mode"] != INTERNAL_LOCK_GOVERNANCE_MODE:
            return False
        if receipt.proof["external_registration"] is not False:
            return False
        revision = release.source_revision
        if (
            revision.get("repository_revision")
            != self.proof.scientific_repository_revision
        ):
            return False
        if revision.get("governance_mode") != INTERNAL_LOCK_GOVERNANCE_MODE:
            return False
        if revision.get("external_registration") is not False:
            return False
        return True


def create_internal_repository_lock_receipt(
    release: ProtocolRelease,
    proof: InternalRepositoryLockProof,
) -> WitnessReceipt:
    if not isinstance(release, ProtocolRelease):
        raise TypeError("internal lock receipt requires ProtocolRelease")
    if not isinstance(proof, InternalRepositoryLockProof):
        raise TypeError("internal lock receipt requires InternalRepositoryLockProof")
    if release.content_hash not in proof.release_hashes:
        raise ValueError("internal lock receipt release is not bound by the lock proof")
    if release.source_revision.get("repository_revision") != proof.scientific_repository_revision:
        raise ValueError("internal lock receipt release repository revision changed")
    if release.source_revision.get("governance_mode") != INTERNAL_LOCK_GOVERNANCE_MODE:
        raise ValueError("internal lock receipt release governance mode changed")
    if release.source_revision.get("external_registration") is not False:
        raise ValueError("internal lock receipt release claims external registration")
    return WitnessReceipt.create(
        provider=INTERNAL_LOCK_PROVIDER,
        authority=INTERNAL_LOCK_AUTHORITY,
        subject_hash=release.content_hash,
        reference=proof.reference,
        claimed_at=proof.locked_at,
        proof={
            "lock_payload_hash": proof.lock_payload_hash,
            "scientific_repository_revision": proof.scientific_repository_revision,
            "governance_mode": INTERNAL_LOCK_GOVERNANCE_MODE,
            "external_registration": False,
        },
    )


__all__ = [
    "INTERNAL_LOCK_AUTHORITY",
    "INTERNAL_LOCK_GOVERNANCE_MODE",
    "INTERNAL_LOCK_PROVIDER",
    "InternalRepositoryLockProof",
    "InternalRepositoryLockVerifier",
    "TwoStageInternalLockBundle",
    "create_internal_locked_protocol_release",
    "create_internal_repository_lock_receipt",
]
