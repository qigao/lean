from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import re
from typing import Generic, TypeVar

from narrative_dynamics.contracts import stable_content_hash


REPORT_ARTIFACT_SCHEMA_VERSION = 1
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPORT_EXCLUDED_FIELDS = frozenset({"manifest", "artifact"})
T = TypeVar("T")


def _validated_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _report_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_report_value(value: object, *, active: set[int]) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _canonical_report_value(value.value, active=active)

    tracked = is_dataclass(value) or isinstance(
        value,
        (Mapping, list, tuple, set, frozenset),
    )
    marker = id(value)
    if tracked:
        if marker in active:
            raise ValueError("aggregate report payload cannot contain a cycle")
        active.add(marker)

    try:
        if is_dataclass(value) and not isinstance(value, type):
            payload: dict[str, object] = {}
            for report_field in fields(value):
                if report_field.name in _REPORT_EXCLUDED_FIELDS:
                    continue
                if report_field.metadata.get("report_artifact_exclude", False):
                    continue
                payload[report_field.name] = _canonical_report_value(
                    getattr(value, report_field.name),
                    active=active,
                )
            return {
                "dataclass": _report_type(value),
                "fields": payload,
            }

        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(
                        "aggregate report mappings require non-empty string keys"
                    )
                result[key] = _canonical_report_value(item, active=active)
            return result

        if isinstance(value, (list, tuple)):
            return tuple(
                _canonical_report_value(item, active=active) for item in value
            )

        if isinstance(value, (set, frozenset)):
            canonical = tuple(
                _canonical_report_value(item, active=active) for item in value
            )
            return tuple(sorted(canonical, key=stable_content_hash))

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return tuple(
                _canonical_report_value(item, active=active) for item in value
            )
    finally:
        if tracked:
            active.remove(marker)

    raise TypeError(
        "aggregate report payload contains an unsupported value "
        f"of type {type(value).__name__}"
    )


def _report_manifest_hash(report: object) -> str:
    manifest = getattr(report, "manifest", None)
    content_hash = getattr(manifest, "content_hash", None)
    return _validated_hash(content_hash, label="aggregate report manifest hash")


def canonical_report_payload(report: object) -> Mapping[str, object]:
    """Return the canonical semantic payload of a dataclass report.

    The existing manifest and any already attached artifact are deliberately
    excluded. The artifact binds the payload to the manifest separately, which
    avoids a circular manifest/artifact dependency.
    """

    if not is_dataclass(report) or isinstance(report, type):
        raise TypeError("aggregate report artifacts require a dataclass report")
    canonical = _canonical_report_value(report, active=set())
    if not isinstance(canonical, Mapping):
        raise RuntimeError("canonical report payload must be a mapping")
    return canonical


@dataclass(frozen=True)
class AggregateReportArtifact:
    """Content identity linking one aggregate report payload to its manifest."""

    report_type: str
    manifest_hash: str
    payload_hash: str
    schema_version: int = REPORT_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.report_type, str) or not self.report_type.strip():
            raise ValueError("aggregate report type must be a non-empty string")
        if self.report_type != self.report_type.strip():
            raise ValueError("aggregate report type cannot contain surrounding whitespace")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError(
                "aggregate report artifact schema version must be positive"
            )
        object.__setattr__(
            self,
            "manifest_hash",
            _validated_hash(
                self.manifest_hash,
                label="aggregate report manifest hash",
            ),
        )
        object.__setattr__(
            self,
            "payload_hash",
            _validated_hash(
                self.payload_hash,
                label="aggregate report payload hash",
            ),
        )

    @classmethod
    def from_report(cls, report: object) -> "AggregateReportArtifact":
        payload = canonical_report_payload(report)
        return cls(
            report_type=_report_type(report),
            manifest_hash=_report_manifest_hash(report),
            payload_hash=stable_content_hash(payload),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "manifest_hash": self.manifest_hash,
            "payload_hash": self.payload_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._identity_payload())

    def manifest_identity(self) -> dict[str, object]:
        return {**self._identity_payload(), "content_hash": self.content_hash}

    def matches(self, report: object) -> bool:
        try:
            expected = type(self).from_report(report)
        except (TypeError, ValueError):
            return False
        return self == expected


@dataclass(frozen=True)
class AttestedReport(Generic[T]):
    """Immutable envelope retaining an aggregate report and its payload identity."""

    report: T
    artifact: AggregateReportArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, AggregateReportArtifact):
            raise TypeError("attested report artifact must be AggregateReportArtifact")
        if not self.artifact.matches(self.report):
            raise ValueError("aggregate report artifact does not match report payload")

    @property
    def content_hash(self) -> str:
        return self.artifact.content_hash

    def require_integrity(self) -> T:
        if not self.artifact.matches(self.report):
            raise RuntimeError("aggregate report artifact integrity check failed")
        return self.report


def attest_report(report: T) -> AttestedReport[T]:
    return AttestedReport(
        report=report,
        artifact=AggregateReportArtifact.from_report(report),
    )


__all__ = [
    "REPORT_ARTIFACT_SCHEMA_VERSION",
    "AggregateReportArtifact",
    "AttestedReport",
    "attest_report",
    "canonical_report_payload",
]
