from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash

from .dataset import (
    ObservationDataset,
    _freeze_mapping,
)


EXTERNAL_EVIDENCE_ORIGIN = "external_observational"
EXTERNAL_CLAIM_SCOPE = "external_observational_predictive_only"
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExternalValidationError(ValueError):
    """External predictive-validation protocol or evidence is invalid."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ExternalValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ExternalValidationError(f"{label} must be a sha256 content hash")
    return value


def _external_source(dataset: ObservationDataset) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not isinstance(dataset.source, Mapping):
        raise ExternalValidationError("external observation dataset source must be a mapping")
    source = dataset.source
    provenance = dataset.provenance
    source_kind = source.get("kind")
    if source_kind == "synthetic_fixture" or provenance.get("synthetic_non_empirical") is True:
        raise ExternalValidationError("synthetic evidence cannot enter external validation")
    if source_kind != EXTERNAL_EVIDENCE_ORIGIN:
        raise ExternalValidationError("external evidence requires positive source origin")
    if provenance.get("external_observational") is not True:
        raise ExternalValidationError("external evidence requires positive provenance origin")
    return source, provenance


def _partition_assignment_hash(dataset: ObservationDataset) -> str:
    assignments: list[tuple[str, str]] = []
    for partition in dataset.partitions:
        if partition.records:
            assignments.extend(
                (partition.role.value, record.id)
                for record in partition.records
            )
        else:
            assignments.extend(
                (partition.role.value, observation_id)
                for case in partition.cases
                for observation_id in case.observation_ids
            )
    return stable_content_hash(tuple(sorted(assignments)))


@dataclass(frozen=True)
class ExternalEvidenceDeclaration:
    name: str
    version: str
    dataset_hash: str
    source_snapshot_hash: str
    source_reference: str
    source_revision: Mapping[str, object]
    transform_identity: Mapping[str, object]
    record_namespace: str
    partition_assignment_hash: str
    evidence_origin: str = EXTERNAL_EVIDENCE_ORIGIN
    claim_scope: str = EXTERNAL_CLAIM_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="external evidence name"))
        object.__setattr__(self, "version", _text(self.version, label="external evidence version"))
        object.__setattr__(
            self,
            "dataset_hash",
            _hash(self.dataset_hash, label="external evidence dataset hash"),
        )
        object.__setattr__(
            self,
            "source_snapshot_hash",
            _hash(self.source_snapshot_hash, label="external source snapshot hash"),
        )
        object.__setattr__(
            self,
            "source_reference",
            _text(self.source_reference, label="external source reference"),
        )
        object.__setattr__(
            self,
            "source_revision",
            _freeze_mapping(self.source_revision, label="external source revision"),
        )
        object.__setattr__(
            self,
            "transform_identity",
            _freeze_mapping(self.transform_identity, label="external transform identity"),
        )
        object.__setattr__(
            self,
            "record_namespace",
            _text(self.record_namespace, label="external record namespace"),
        )
        object.__setattr__(
            self,
            "partition_assignment_hash",
            _hash(
                self.partition_assignment_hash,
                label="external partition assignment hash",
            ),
        )
        if self.evidence_origin != EXTERNAL_EVIDENCE_ORIGIN:
            raise ExternalValidationError("external evidence origin is fixed")
        if self.claim_scope != EXTERNAL_CLAIM_SCOPE:
            raise ExternalValidationError("external evidence claim scope is fixed")

    @classmethod
    def from_dataset(
        cls,
        dataset: ObservationDataset,
        *,
        name: str,
        version: str,
        source_snapshot_hash: str,
        source_reference: str,
        source_revision: Mapping[str, object],
        transform_identity: Mapping[str, object],
        record_namespace: str,
    ) -> "ExternalEvidenceDeclaration":
        if not isinstance(dataset, ObservationDataset):
            raise TypeError("external evidence requires ObservationDataset")
        _external_source(dataset)
        return cls(
            name=name,
            version=version,
            dataset_hash=dataset.content_hash,
            source_snapshot_hash=source_snapshot_hash,
            source_reference=source_reference,
            source_revision=source_revision,
            transform_identity=transform_identity,
            record_namespace=record_namespace,
            partition_assignment_hash=_partition_assignment_hash(dataset),
        )

    def require_matches(
        self,
        dataset: ObservationDataset,
        *,
        source_snapshot_hash: str | None = None,
        source_reference: str | None = None,
        source_revision: Mapping[str, object] | None = None,
        transform_identity: Mapping[str, object] | None = None,
        record_namespace: str | None = None,
    ) -> None:
        if not isinstance(dataset, ObservationDataset):
            raise TypeError("external evidence match requires ObservationDataset")
        _external_source(dataset)
        if dataset.content_hash != self.dataset_hash:
            raise ExternalValidationError("external evidence dataset identity changed")
        if _partition_assignment_hash(dataset) != self.partition_assignment_hash:
            raise ExternalValidationError("external evidence partition assignment changed")
        if (
            source_snapshot_hash is not None
            and _hash(source_snapshot_hash, label="external source snapshot hash")
            != self.source_snapshot_hash
        ):
            raise ExternalValidationError("external source snapshot changed")
        if (
            source_reference is not None
            and _text(source_reference, label="external source reference")
            != self.source_reference
        ):
            raise ExternalValidationError("external source reference changed")
        if source_revision is not None:
            frozen = _freeze_mapping(source_revision, label="external source revision")
            if frozen != self.source_revision:
                raise ExternalValidationError("external source revision changed")
        if transform_identity is not None:
            frozen = _freeze_mapping(
                transform_identity,
                label="external transform identity",
            )
            if frozen != self.transform_identity:
                raise ExternalValidationError("external transform identity changed")
        if (
            record_namespace is not None
            and _text(record_namespace, label="external record namespace")
            != self.record_namespace
        ):
            raise ExternalValidationError("external record namespace changed")

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "dataset_hash": self.dataset_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "source_reference": self.source_reference,
            "source_revision": self.source_revision,
            "transform_identity": self.transform_identity,
            "record_namespace": self.record_namespace,
            "partition_assignment_hash": self.partition_assignment_hash,
            "evidence_origin": self.evidence_origin,
            "claim_scope": self.claim_scope,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


__all__ = [
    "EXTERNAL_CLAIM_SCOPE",
    "EXTERNAL_EVIDENCE_ORIGIN",
    "ExternalEvidenceDeclaration",
    "ExternalValidationError",
]
