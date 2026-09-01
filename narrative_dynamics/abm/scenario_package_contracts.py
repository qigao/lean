"""Immutable, path-free contracts for situated scenario package sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.abm.scenario_authoring_contracts import (
    ScenarioAssetCatalog,
    ScenarioKnowledgeCatalog,
    ScenarioRunPolicy,
    ScenarioSocialWorld,
    ScenarioStoryPlan,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedCognitiveState,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_contracts import validate_situated_state
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedSocialMemoryState,
    validate_situated_social_memory_state,
)
from narrative_dynamics.abm.situated_spatial_map_contracts import SituatedSpatialMap
from narrative_dynamics.abm.situated_story import SituatedStory
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


def _contract_hash_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _contract_hash_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _contract_hash_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_contract_hash_value(item) for item in value]
    return value


def _contract_content_hash(value: object) -> str:
    return stable_content_hash(_contract_hash_value(value))


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


@dataclass(frozen=True)
class CompiledSituatedScenario:
    """One complete, immutable checkpoint compiled from a logical source package."""

    scenario_id: str
    version: str
    package_hash: str
    source_document_hashes: tuple[tuple[str, str, str], ...]
    runtime_model: SituatedNetworkRuntimeModel
    spatial_map: SituatedSpatialMap
    initial_story: SituatedStory
    initial_cognitive_state: SituatedCognitiveState
    initial_social_state: SituatedSocialMemoryState
    social_world: ScenarioSocialWorld
    story_plan: ScenarioStoryPlan
    knowledge_catalog: ScenarioKnowledgeCatalog
    asset_catalog: ScenarioAssetCatalog
    run_policy: ScenarioRunPolicy

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_id",
            _non_empty_text(self.scenario_id, label="compiled scenario id"),
        )
        object.__setattr__(
            self,
            "version",
            _non_empty_text(self.version, label="compiled scenario version"),
        )
        object.__setattr__(
            self,
            "package_hash",
            _content_hash(self.package_hash, label="compiled scenario package hash"),
        )
        if not isinstance(self.source_document_hashes, tuple):
            raise TypeError("compiled scenario source document hashes must be a tuple")
        source_hashes = []
        source_keys = set()
        for item in self.source_document_hashes:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("compiled scenario source document hashes must be triples")
            role, logical_id, content_hash = item
            role = _non_empty_text(role, label="compiled source document role")
            logical_id = _non_empty_text(
                logical_id,
                label="compiled source document logical id",
            )
            content_hash = _content_hash(
                content_hash,
                label="compiled source document hash",
            )
            key = (role, logical_id)
            if key in source_keys:
                raise ValueError(
                    "compiled source document role and logical id pairs must be unique"
                )
            source_keys.add(key)
            source_hashes.append((role, logical_id, content_hash))
        object.__setattr__(
            self,
            "source_document_hashes",
            tuple(sorted(source_hashes, key=lambda item: (item[0], item[1]))),
        )

        for name, expected in (
            ("runtime_model", SituatedNetworkRuntimeModel),
            ("spatial_map", SituatedSpatialMap),
            ("initial_story", SituatedStory),
            ("initial_cognitive_state", SituatedCognitiveState),
            ("initial_social_state", SituatedSocialMemoryState),
            ("social_world", ScenarioSocialWorld),
            ("story_plan", ScenarioStoryPlan),
            ("knowledge_catalog", ScenarioKnowledgeCatalog),
            ("asset_catalog", ScenarioAssetCatalog),
            ("run_policy", ScenarioRunPolicy),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"compiled scenario {name} must be {expected.__name__}")

        cognition = self.runtime_model.percept_memory_model.cognitive_model
        perception = self.runtime_model.percept_memory_model.perception_model
        world = cognition.world_model
        if self.spatial_map.world_model != world:
            raise ValueError("compiled scenario spatial map must bind the exact world")
        if self.initial_story.perception_model != perception:
            raise ValueError(
                "compiled scenario initial story must bind the exact perception model"
            )
        if self.initial_story.rounds:
            raise ValueError("compiled scenario initial story must be at round zero")
        validate_situated_state(world, self.initial_story.initial_state)
        validate_situated_cognitive_state(
            cognition,
            self.initial_story,
            self.initial_cognitive_state,
        )
        validate_situated_social_memory_state(
            self.runtime_model.social_memory_model,
            self.initial_cognitive_state,
            self.initial_social_state,
        )
        if not self.initial_cognitive_state.checkpoint:
            raise ValueError("compiled scenario initial cognition must be a checkpoint")
        if not self.initial_social_state.checkpoint:
            raise ValueError("compiled scenario initial social state must be a checkpoint")
        if self.story_plan.mode is not self.run_policy.mode:
            raise ValueError("compiled scenario execution modes must match")

        authored_affinities = {
            (item.source_agent_id, item.target_agent_id): item.strength
            for item in self.social_world.relationships
        }
        initial_relationships = {
            (item.observer_agent_id, item.source_agent_id): item
            for item in self.initial_social_state.relationships
        }
        if set(authored_affinities) != set(initial_relationships):
            raise ValueError(
                "compiled scenario social definitions must bind the exact initial relationships"
            )
        initial_trust = self.runtime_model.social_memory_model.policy.initial_source_trust
        if any(
            item.trust != initial_trust
            or item.affinity != authored_affinities[pair]
            for pair, item in initial_relationships.items()
        ):
            raise ValueError(
                "compiled scenario social definitions must bind exact relationship seeds"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "version": self.version,
            "package_hash": self.package_hash,
            "source_document_hashes": [
                list(item) for item in self.source_document_hashes
            ],
            "runtime_model_hash": self.runtime_model.content_hash,
            "spatial_map_hash": self.spatial_map.content_hash,
            "initial_story_hash": self.initial_story.content_hash,
            "initial_cognitive_state_hash": self.initial_cognitive_state.content_hash,
            "initial_social_state_hash": self.initial_social_state.content_hash,
            "social_world_hash": _contract_content_hash(self.social_world),
            "story_plan_hash": _contract_content_hash(self.story_plan),
            "knowledge_catalog_hash": _contract_content_hash(self.knowledge_catalog),
            "asset_catalog_hash": _contract_content_hash(self.asset_catalog),
            "run_policy_hash": _contract_content_hash(self.run_policy),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
