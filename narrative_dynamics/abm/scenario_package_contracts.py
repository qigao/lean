"""Immutable, path-free contracts for situated scenario package sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash


SCENARIO_PACKAGE_SCHEMA = "narrative-dynamics.scenario-package/v1"
SCENARIO_DOCUMENT_SCHEMA = "narrative-dynamics.scenario-document/v1"
_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class ScenarioDocumentRole(str, Enum):
    PHYSICAL_WORLD = "physical.world"
    PHYSICAL_PERCEPTION = "physical.perception"
    PHYSICAL_INITIAL_STATE = "physical.initial_state"
    PHYSICAL_MAP = "physical.map"
    SOCIAL_INSTITUTIONS = "social.institutions"
    SOCIAL_RELATIONSHIPS = "social.relationships"
    SOCIAL_NORMS = "social.norms"
    AGENT = "agent"
    STORY_OUTLINE = "story.outline"
    STORY_INTERVENTIONS = "story.interventions"
    KNOWLEDGE_CATALOG = "knowledge.catalog"
    KNOWLEDGE_ACCESS = "knowledge.access"
    ASSET_CATALOG = "asset.catalog"
    RUN = "run"


_REQUIRED_SINGLETON_ROLES = frozenset({
    ScenarioDocumentRole.PHYSICAL_WORLD,
    ScenarioDocumentRole.PHYSICAL_PERCEPTION,
    ScenarioDocumentRole.PHYSICAL_INITIAL_STATE,
    ScenarioDocumentRole.SOCIAL_RELATIONSHIPS,
    ScenarioDocumentRole.STORY_OUTLINE,
    ScenarioDocumentRole.KNOWLEDGE_CATALOG,
    ScenarioDocumentRole.KNOWLEDGE_ACCESS,
    ScenarioDocumentRole.ASSET_CATALOG,
    ScenarioDocumentRole.RUN,
})
_OPTIONAL_SINGLETON_ROLES = frozenset({
    ScenarioDocumentRole.PHYSICAL_MAP,
    ScenarioDocumentRole.SOCIAL_INSTITUTIONS,
    ScenarioDocumentRole.SOCIAL_NORMS,
    ScenarioDocumentRole.STORY_INTERVENTIONS,
})
_SINGLETON_ROLES = _REQUIRED_SINGLETON_ROLES | _OPTIONAL_SINGLETON_ROLES


def _non_empty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact sha256 content hash")
    return value


def _freeze_json_value(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            frozen[key] = _freeze_json_value(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} must contain JSON values")


def _freeze_json_mapping(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    frozen = _freeze_json_value(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _canonical_locator_key(locator: "ScenarioDocumentLocator") -> tuple[str, str]:
    return (locator.role.value, locator.logical_id)


def _validate_document_roles(roles: list[ScenarioDocumentRole]) -> None:
    if any(roles.count(role) != 1 for role in _SINGLETON_ROLES if role in roles):
        raise ValueError("scenario package singleton document roles must occur exactly once")
    missing = _REQUIRED_SINGLETON_ROLES - set(roles)
    if missing:
        raise ValueError("scenario package is missing required document roles")
    if ScenarioDocumentRole.AGENT not in roles:
        raise ValueError("scenario package requires at least one agent document")


@dataclass(frozen=True)
class ScenarioDocumentLocator:
    role: ScenarioDocumentRole
    logical_id: str
    relative_path: str
    expected_hash: str

    def __post_init__(self) -> None:
        try:
            role = self.role if isinstance(self.role, ScenarioDocumentRole) else ScenarioDocumentRole(self.role)
        except (TypeError, ValueError) as error:
            raise ValueError("scenario document role must be supported") from error
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "logical_id", _non_empty_text(self.logical_id, label="scenario document logical id"))
        object.__setattr__(self, "relative_path", _non_empty_text(self.relative_path, label="scenario document relative path"))
        object.__setattr__(self, "expected_hash", _content_hash(self.expected_hash, label="scenario document expected hash"))

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role.value,
            "logical_id": self.logical_id,
            "relative_path": self.relative_path,
            "expected_hash": self.expected_hash,
        }


@dataclass(frozen=True)
class ScenarioPackageManifest:
    scenario_id: str
    version: str
    documents: tuple[ScenarioDocumentLocator, ...]
    schema: str = SCENARIO_PACKAGE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _non_empty_text(self.scenario_id, label="scenario id"))
        object.__setattr__(self, "version", _non_empty_text(self.version, label="scenario version"))
        if self.schema != SCENARIO_PACKAGE_SCHEMA:
            raise ValueError("scenario package manifest schema is not supported")
        documents = tuple(self.documents)
        if any(not isinstance(locator, ScenarioDocumentLocator) for locator in documents):
            raise TypeError("scenario package documents must be ScenarioDocumentLocator values")
        if len({_canonical_locator_key(locator) for locator in documents}) != len(documents):
            raise ValueError("scenario package document role and logical id pairs must be unique")
        documents = tuple(sorted(documents, key=_canonical_locator_key))
        _validate_document_roles([locator.role for locator in documents])
        object.__setattr__(self, "documents", documents)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "scenario_id": self.scenario_id,
            "version": self.version,
            "documents": [locator.to_dict() for locator in self.documents],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash({
            "schema": self.schema,
            "scenario_id": self.scenario_id,
            "version": self.version,
            "documents": [
                {
                    "role": locator.role.value,
                    "logical_id": locator.logical_id,
                    "expected_hash": locator.expected_hash,
                }
                for locator in self.documents
            ],
        })


@dataclass(frozen=True)
class ScenarioSourceDocument:
    role: ScenarioDocumentRole
    logical_id: str
    schema: str
    value: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            role = self.role if isinstance(self.role, ScenarioDocumentRole) else ScenarioDocumentRole(self.role)
        except (TypeError, ValueError) as error:
            raise ValueError("scenario source document role must be supported") from error
        if self.schema != SCENARIO_DOCUMENT_SCHEMA:
            raise ValueError("scenario source document schema is not supported")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "logical_id", _non_empty_text(self.logical_id, label="scenario source document logical id"))
        object.__setattr__(self, "value", _freeze_json_mapping(self.value, label="scenario source document value"))

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "logical_id": self.logical_id,
            "schema": self.schema,
            "value": self.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validate_optional_fallbacks(documents: tuple[ScenarioSourceDocument, ...]) -> None:
    present = {document.role for document in documents}
    missing = _OPTIONAL_SINGLETON_ROLES - present
    if not missing:
        return
    run = next(document for document in documents if document.role is ScenarioDocumentRole.RUN)
    fallbacks = run.value.get("fallbacks")
    if not isinstance(fallbacks, Mapping):
        raise ValueError("scenario run document must declare fallbacks for missing optional documents")
    for role in missing:
        fallback = fallbacks.get(role.value)
        if not isinstance(fallback, str) or not fallback.strip():
            raise ValueError("scenario run document must declare a fallback for each missing optional document")


@dataclass(frozen=True)
class ScenarioPackageSource:
    scenario_id: str
    version: str
    documents: tuple[ScenarioSourceDocument, ...]
    manifest_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _non_empty_text(self.scenario_id, label="scenario source id"))
        object.__setattr__(self, "version", _non_empty_text(self.version, label="scenario source version"))
        documents = tuple(self.documents)
        if any(not isinstance(document, ScenarioSourceDocument) for document in documents):
            raise TypeError("scenario source documents must be ScenarioSourceDocument values")
        if len({_canonical_locator_key(document) for document in documents}) != len(documents):
            raise ValueError("scenario source document role and logical id pairs must be unique")
        _validate_document_roles([document.role for document in documents])
        _validate_optional_fallbacks(documents)
        object.__setattr__(self, "documents", tuple(sorted(documents, key=_canonical_locator_key)))
        object.__setattr__(self, "manifest_hash", _content_hash(self.manifest_hash, label="scenario source manifest hash"))

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "version": self.version,
            "documents": [document.to_dict() for document in self.documents],
            "manifest_hash": self.manifest_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
