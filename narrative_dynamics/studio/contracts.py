"""Immutable, path-free contracts for the World Studio draft authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import TypeAlias

from narrative_dynamics.abm.scenario_package_contracts import (
    SCENARIO_DOCUMENT_SCHEMA,
    ScenarioDocumentRole,
)
from narrative_dynamics.contracts import stable_content_hash


SCENARIO_DRAFT_SNAPSHOT_SCHEMA = "narrative-dynamics.scenario-draft-snapshot/v1"
SCENARIO_DRAFT_OPERATION_SCHEMA = "narrative-dynamics.scenario-draft-operation/v1"
SCENARIO_DIAGNOSTIC_REPORT_SCHEMA = (
    "narrative-dynamics.scenario-diagnostic-report/v1"
)

DEFAULT_JSON_DOCUMENT_BYTES = 1_048_576
DEFAULT_TILED_MAP_BYTES = 16_777_216
DEFAULT_OPERATION_PAYLOAD_BYTES = 16_777_216
DEFAULT_ACCEPTED_OPERATION_JOURNAL = 10_000

_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_MESSAGE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_JSON_POINTER_ESCAPE = re.compile(r"~(?:0|1)")
_MAX_JSON_DEPTH = 128

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class DraftOperationKind(str, Enum):
    REPLACE_DOCUMENT = "replace_document"
    SET_VALUE = "set_value"
    INSERT_VALUE = "insert_value"
    REMOVE_VALUE = "remove_value"
    MOVE_VALUE = "move_value"
    SET_LAYOUT = "set_layout"


class ScenarioDiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


def _non_empty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _safe_identity(value: object, *, label: str) -> str:
    text = _non_empty_text(value, label=label)
    if (
        _IDENTITY.fullmatch(text) is None
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
    ):
        raise ValueError(f"{label} must be a path-free identity")
    return text


def _optional_identity(value: object, *, label: str) -> str | None:
    return None if value is None else _safe_identity(value, label=label)


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _bounded_positive_int(value: object, *, label: str) -> int:
    return _positive_int(value, label=label)


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact sha256 content hash")
    return value


def _optional_content_hash(value: object, *, label: str) -> str | None:
    return None if value is None else _content_hash(value, label=label)


def _pointer(value: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"{label} must be an RFC 6901 JSON pointer")
    if value and not value.startswith("/"):
        raise ValueError(f"{label} must be an RFC 6901 JSON pointer")
    index = 0
    while index < len(value):
        if value[index] == "~":
            match = _JSON_POINTER_ESCAPE.match(value, index)
            if match is None:
                raise ValueError(f"{label} must be an RFC 6901 JSON pointer")
            index = match.end()
        else:
            index += 1
    return value


def _freeze_json(value: object, *, label: str, depth: int = 0) -> JsonValue:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{label} exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} object keys must be strings")
            frozen[key] = _freeze_json(
                item, label=f"{label}.{key}", depth=depth + 1
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, label=f"{label}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} must contain JSON values")


def thaw_json(value: JsonValue) -> object:
    """Return an ordinary detached JSON tree from a frozen contract value."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    frozen = _freeze_json(value, label="canonical JSON")
    return json.dumps(
        thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def scenario_document_raw_value(role: str, value: object) -> object:
    """Reconstruct the canonical package file JSON for a draft document."""

    if role == ScenarioDocumentRole.PHYSICAL_MAP.value:
        return thaw_json(_freeze_json(value, label="scenario draft document value"))
    return {
        "schema": SCENARIO_DOCUMENT_SCHEMA,
        "value": thaw_json(_freeze_json(value, label="scenario draft document value")),
    }


@dataclass(frozen=True)
class ScenarioWorkspaceLimits:
    json_document_bytes: int = DEFAULT_JSON_DOCUMENT_BYTES
    tiled_map_bytes: int = DEFAULT_TILED_MAP_BYTES
    operation_payload_bytes: int = DEFAULT_OPERATION_PAYLOAD_BYTES
    accepted_operation_journal: int = DEFAULT_ACCEPTED_OPERATION_JOURNAL

    def __post_init__(self) -> None:
        for name in (
            "json_document_bytes",
            "tiled_map_bytes",
            "operation_payload_bytes",
            "accepted_operation_journal",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_positive_int(getattr(self, name), label=name.replace("_", " ")),
            )


@dataclass(frozen=True)
class ScenarioDraftDocument:
    role: str
    logical_id: str | None
    value: JsonValue
    content_hash: str

    def __post_init__(self) -> None:
        role = _safe_identity(self.role, label="scenario draft document role")
        try:
            parsed_role = ScenarioDocumentRole(role)
        except ValueError:
            raise ValueError("scenario draft document role must be supported") from None
        logical_id = _optional_identity(
            self.logical_id, label="scenario draft document logical ID"
        )
        if parsed_role is ScenarioDocumentRole.AGENT:
            if logical_id is None:
                raise ValueError("scenario draft agent document requires a logical ID")
        elif logical_id is not None:
            raise ValueError("scenario draft singleton document cannot have a logical ID")
        frozen = _freeze_json(self.value, label="scenario draft document value")
        if not isinstance(frozen, Mapping):
            raise TypeError("scenario draft document value must be a JSON object")
        content_hash = _content_hash(
            self.content_hash, label="scenario draft document content hash"
        )
        expected_hash = canonical_json_hash(
            scenario_document_raw_value(parsed_role.value, frozen)
        )
        if content_hash != expected_hash:
            raise ValueError("scenario draft document content hash does not match its value")
        object.__setattr__(self, "role", parsed_role.value)
        object.__setattr__(self, "logical_id", logical_id)
        object.__setattr__(self, "value", frozen)
        object.__setattr__(self, "content_hash", content_hash)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "logical_id": self.logical_id,
            "value": thaw_json(self.value),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ScenarioDiagnostic:
    severity: ScenarioDiagnosticSeverity
    code: str
    document_role: str
    logical_id: str | None
    pointer: str
    related_ids: tuple[str, ...] = ()
    message_key: str | None = None

    def __post_init__(self) -> None:
        try:
            severity = (
                self.severity
                if isinstance(self.severity, ScenarioDiagnosticSeverity)
                else ScenarioDiagnosticSeverity(self.severity)
            )
        except (TypeError, ValueError):
            raise ValueError("scenario diagnostic severity must be supported") from None
        code = _non_empty_text(self.code, label="scenario diagnostic code")
        if _MESSAGE_KEY.fullmatch(code) is None:
            raise ValueError("scenario diagnostic code must be a stable message key")
        role = _safe_identity(self.document_role, label="scenario diagnostic role")
        logical_id = _optional_identity(
            self.logical_id, label="scenario diagnostic logical ID"
        )
        pointer = _pointer(
            self.pointer, label="scenario diagnostic pointer", allow_empty=True
        )
        related = tuple(
            _safe_identity(item, label="scenario diagnostic related ID")
            for item in self.related_ids
        )
        if len(set(related)) != len(related):
            raise ValueError("scenario diagnostic related IDs must be unique")
        message_key = self.message_key if self.message_key is not None else code
        if not isinstance(message_key, str) or _MESSAGE_KEY.fullmatch(message_key) is None:
            raise ValueError("scenario diagnostic message key must be stable")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "document_role", role)
        object.__setattr__(self, "logical_id", logical_id)
        object.__setattr__(self, "pointer", pointer)
        object.__setattr__(self, "related_ids", tuple(sorted(related)))
        object.__setattr__(self, "message_key", message_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "document_role": self.document_role,
            "logical_id": self.logical_id,
            "pointer": self.pointer,
            "related_ids": list(self.related_ids),
            "message_key": self.message_key,
        }


def _diagnostic_key(
    diagnostic: ScenarioDiagnostic,
) -> tuple[str, str, str, str, str, tuple[str, ...]]:
    return (
        diagnostic.document_role,
        diagnostic.logical_id or "",
        diagnostic.pointer,
        diagnostic.code,
        diagnostic.severity.value,
        diagnostic.related_ids,
    )


@dataclass(frozen=True)
class ScenarioDiagnosticReport:
    project_id: str
    revision: int
    diagnostics: tuple[ScenarioDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        project_id = _safe_identity(self.project_id, label="scenario project ID")
        revision = _positive_int(self.revision, label="scenario diagnostic revision")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, ScenarioDiagnostic) for item in diagnostics):
            raise TypeError("scenario diagnostics must be ScenarioDiagnostic values")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "diagnostics", tuple(sorted(diagnostics, key=_diagnostic_key)))

    @property
    def has_errors(self) -> bool:
        return any(
            item.severity is ScenarioDiagnosticSeverity.ERROR
            for item in self.diagnostics
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": SCENARIO_DIAGNOSTIC_REPORT_SCHEMA,
            "project_id": self.project_id,
            "revision": self.revision,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._identity_payload())

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "content_hash": self.content_hash}


def _document_key(document: ScenarioDraftDocument) -> tuple[str, str]:
    return (document.role, document.logical_id or "")


@dataclass(frozen=True)
class ScenarioDraftSnapshot:
    project_id: str
    revision: int
    documents: tuple[ScenarioDraftDocument, ...]
    layout: JsonValue
    diagnostic_report_hash: str
    compiled_scenario_hash: str | None
    scenario_id: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        project_id = _safe_identity(self.project_id, label="scenario project ID")
        revision = _positive_int(self.revision, label="scenario draft revision")
        documents = tuple(self.documents)
        if any(not isinstance(item, ScenarioDraftDocument) for item in documents):
            raise TypeError("scenario draft documents must be ScenarioDraftDocument values")
        if len({_document_key(item) for item in documents}) != len(documents):
            raise ValueError("scenario draft document identities must be unique")
        layout = _freeze_json(self.layout, label="scenario draft layout")
        if not isinstance(layout, Mapping):
            raise TypeError("scenario draft layout must be a JSON object")
        scenario_id = _optional_identity(self.scenario_id, label="scenario source ID")
        version = None
        if self.version is not None:
            version = _non_empty_text(self.version, label="scenario source version")
        if (scenario_id is None) != (version is None):
            raise ValueError("scenario source ID and version must be present together")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "documents", tuple(sorted(documents, key=_document_key)))
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "diagnostic_report_hash",
            _content_hash(
                self.diagnostic_report_hash,
                label="scenario diagnostic report hash",
            ),
        )
        object.__setattr__(
            self,
            "compiled_scenario_hash",
            _optional_content_hash(
                self.compiled_scenario_hash,
                label="compiled scenario hash",
            ),
        )
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "version", version)

    @property
    def document_semantic_hash(self) -> str:
        return stable_content_hash(
            {
                "scenario_id": self.scenario_id,
                "version": self.version,
                "documents": [
                    {
                        "role": item.role,
                        "logical_id": item.logical_id,
                        "content_hash": item.content_hash,
                    }
                    for item in self.documents
                ],
            }
        )

    @property
    def layout_hash(self) -> str:
        return stable_content_hash(thaw_json(self.layout))

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": SCENARIO_DRAFT_SNAPSHOT_SCHEMA,
            "project_id": self.project_id,
            "revision": self.revision,
            "scenario_id": self.scenario_id,
            "version": self.version,
            "documents": [item.to_dict() for item in self.documents],
            "document_semantic_hash": self.document_semantic_hash,
            "layout": thaw_json(self.layout),
            "layout_hash": self.layout_hash,
            "diagnostic_report_hash": self.diagnostic_report_hash,
            "compiled_scenario_hash": self.compiled_scenario_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._identity_payload())

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "content_hash": self.content_hash}


@dataclass(frozen=True)
class ScenarioDraftOperation:
    operation_id: str
    idempotency_key: str
    project_id: str
    expected_revision: int
    expected_snapshot_hash: str
    document_role: str
    logical_id: str | None
    kind: DraftOperationKind
    pointer: str = ""
    value: JsonValue = None
    from_pointer: str | None = None

    def __post_init__(self) -> None:
        operation_id = _safe_identity(self.operation_id, label="draft operation ID")
        idempotency_key = _safe_identity(
            self.idempotency_key, label="draft operation idempotency key"
        )
        project_id = _safe_identity(self.project_id, label="scenario project ID")
        revision = _positive_int(self.expected_revision, label="expected revision")
        expected_hash = _content_hash(
            self.expected_snapshot_hash, label="expected snapshot hash"
        )
        try:
            kind = self.kind if isinstance(self.kind, DraftOperationKind) else DraftOperationKind(self.kind)
        except (TypeError, ValueError):
            raise ValueError("draft operation kind must be supported") from None
        role = _safe_identity(self.document_role, label="draft operation document role")
        logical_id = _optional_identity(
            self.logical_id, label="draft operation logical ID"
        )
        if kind is DraftOperationKind.SET_LAYOUT:
            if role != "layout" or logical_id is not None:
                raise ValueError("layout operations cannot address domain JSON")
        else:
            try:
                parsed_role = ScenarioDocumentRole(role)
            except ValueError:
                raise ValueError("draft operation document role must be supported") from None
            if parsed_role is ScenarioDocumentRole.AGENT:
                if logical_id is None:
                    raise ValueError("draft agent operation requires a logical ID")
            elif logical_id is not None:
                raise ValueError("draft singleton operation cannot have a logical ID")
        pointer = _pointer(
            self.pointer,
            label="draft operation pointer",
            allow_empty=kind in {DraftOperationKind.REPLACE_DOCUMENT, DraftOperationKind.SET_LAYOUT},
        )
        frozen_value = _freeze_json(self.value, label="draft operation value")
        from_pointer = self.from_pointer
        if kind is DraftOperationKind.MOVE_VALUE:
            if from_pointer is None:
                raise ValueError("move operation requires a source pointer")
            from_pointer = _pointer(from_pointer, label="draft operation source pointer")
        elif from_pointer is not None:
            raise ValueError("only move operations accept a source pointer")
        if kind is DraftOperationKind.REMOVE_VALUE and self.value is not None:
            raise ValueError("remove operation cannot contain a value")
        if kind is DraftOperationKind.MOVE_VALUE and self.value is not None:
            raise ValueError("move operation cannot contain a value")
        payload = {
            "pointer": pointer,
            "from_pointer": from_pointer,
            "value": thaw_json(frozen_value),
        }
        if len(canonical_json_bytes(payload)) > DEFAULT_OPERATION_PAYLOAD_BYTES:
            raise ValueError("draft operation payload exceeds the permitted size")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "idempotency_key", idempotency_key)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "expected_revision", revision)
        object.__setattr__(self, "expected_snapshot_hash", expected_hash)
        object.__setattr__(self, "document_role", role)
        object.__setattr__(self, "logical_id", logical_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "pointer", pointer)
        object.__setattr__(self, "value", frozen_value)
        object.__setattr__(self, "from_pointer", from_pointer)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SCENARIO_DRAFT_OPERATION_SCHEMA,
            "operation_id": self.operation_id,
            "idempotency_key": self.idempotency_key,
            "project_id": self.project_id,
            "expected_revision": self.expected_revision,
            "expected_snapshot_hash": self.expected_snapshot_hash,
            "document_role": self.document_role,
            "logical_id": self.logical_id,
            "kind": self.kind.value,
        }
        if self.pointer or self.kind not in {DraftOperationKind.REPLACE_DOCUMENT}:
            payload["pointer"] = self.pointer
        if self.kind in {
            DraftOperationKind.REPLACE_DOCUMENT,
            DraftOperationKind.SET_VALUE,
            DraftOperationKind.INSERT_VALUE,
            DraftOperationKind.SET_LAYOUT,
        }:
            payload["value"] = thaw_json(self.value)
        if self.from_pointer is not None:
            payload["from_pointer"] = self.from_pointer
        return payload

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ScenarioDraftOperationResult:
    operation_hash: str
    prior_revision: int
    next_snapshot: ScenarioDraftSnapshot
    diagnostic_report: ScenarioDiagnosticReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_hash",
            _content_hash(self.operation_hash, label="draft operation hash"),
        )
        object.__setattr__(
            self,
            "prior_revision",
            _positive_int(self.prior_revision, label="prior draft revision"),
        )
        if not isinstance(self.next_snapshot, ScenarioDraftSnapshot):
            raise TypeError("next snapshot must be a ScenarioDraftSnapshot")
        if not isinstance(self.diagnostic_report, ScenarioDiagnosticReport):
            raise TypeError("diagnostic report must be a ScenarioDiagnosticReport")
        if self.next_snapshot.revision != self.prior_revision + 1:
            raise ValueError("draft operation result revision must increase by one")
        if self.next_snapshot.project_id != self.diagnostic_report.project_id:
            raise ValueError("draft operation result project identities must match")
        if self.next_snapshot.revision != self.diagnostic_report.revision:
            raise ValueError("draft operation result revisions must match")
        if self.next_snapshot.diagnostic_report_hash != self.diagnostic_report.content_hash:
            raise ValueError("draft operation result diagnostic hash must match")

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_hash": self.operation_hash,
            "prior_revision": self.prior_revision,
            "next_snapshot": self.next_snapshot.to_dict(),
            "diagnostic_report": self.diagnostic_report.to_dict(),
        }


@dataclass(frozen=True)
class ScenarioProjectExport:
    project_id: str
    revision: int
    snapshot_hash: str
    target_name: str
    package_hash: str
    compiled_scenario_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id", _safe_identity(self.project_id, label="scenario project ID")
        )
        object.__setattr__(
            self, "revision", _positive_int(self.revision, label="scenario export revision")
        )
        object.__setattr__(
            self,
            "snapshot_hash",
            _content_hash(self.snapshot_hash, label="scenario export snapshot hash"),
        )
        target_name = _safe_identity(self.target_name, label="scenario export target")
        if ":" in target_name:
            raise ValueError("scenario export target must be a path-safe single name")
        object.__setattr__(self, "target_name", target_name)
        object.__setattr__(
            self,
            "package_hash",
            _content_hash(self.package_hash, label="scenario export package hash"),
        )
        object.__setattr__(
            self,
            "compiled_scenario_hash",
            _content_hash(
                self.compiled_scenario_hash,
                label="scenario export compiled hash",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "revision": self.revision,
            "snapshot_hash": self.snapshot_hash,
            "target_name": self.target_name,
            "package_hash": self.package_hash,
            "compiled_scenario_hash": self.compiled_scenario_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self._identity_payload())

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "content_hash": self.content_hash}


__all__ = (
    "DEFAULT_ACCEPTED_OPERATION_JOURNAL",
    "DEFAULT_JSON_DOCUMENT_BYTES",
    "DEFAULT_OPERATION_PAYLOAD_BYTES",
    "DEFAULT_TILED_MAP_BYTES",
    "DraftOperationKind",
    "JsonValue",
    "SCENARIO_DIAGNOSTIC_REPORT_SCHEMA",
    "SCENARIO_DRAFT_OPERATION_SCHEMA",
    "SCENARIO_DRAFT_SNAPSHOT_SCHEMA",
    "ScenarioDiagnostic",
    "ScenarioDiagnosticReport",
    "ScenarioDiagnosticSeverity",
    "ScenarioDraftDocument",
    "ScenarioDraftOperation",
    "ScenarioDraftOperationResult",
    "ScenarioDraftSnapshot",
    "ScenarioProjectExport",
    "ScenarioWorkspaceLimits",
    "canonical_json_bytes",
    "canonical_json_hash",
    "scenario_document_raw_value",
    "thaw_json",
)
