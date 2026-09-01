"""Immutable, path-free contracts for situated scenario package sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.abm.scenario_authoring_contracts import (
    ScenarioAssetCatalog,
    ScenarioKnowledgeCatalog,
    ScenarioNormEffect,
    ScenarioPredicate,
    ScenarioPredicateKind,
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
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedSocialMemoryState,
    SituatedSourceRelationship,
    initialize_situated_social_memory,
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
    return (locator.role.value, locator.path)


def _canonical_document_key(document: "ScenarioSourceDocument") -> tuple[str, str]:
    return (document.role.value, document.logical_id)


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


def _semantic_package_hash(
    scenario_id: str,
    version: str,
    document_hashes: tuple[tuple[str, str, str], ...],
) -> str:
    return stable_content_hash(
        {
            "scenario_id": scenario_id,
            "version": version,
            "documents": [
                {
                    "role": role,
                    "logical_id": logical_id,
                    "content_hash": content_hash,
                }
                for role, logical_id, content_hash in document_hashes
            ],
        }
    )


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
    path: str
    sha256: str

    def __post_init__(self) -> None:
        try:
            role = self.role if isinstance(self.role, ScenarioDocumentRole) else ScenarioDocumentRole(self.role)
        except (TypeError, ValueError):
            raise ValueError("scenario document role must be supported") from None
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "path", _non_empty_text(self.path, label="scenario document path"))
        object.__setattr__(self, "sha256", _content_hash(self.sha256, label="scenario document sha256"))

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role.value,
            "path": self.path,
            "sha256": self.sha256,
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
        if len({locator.path for locator in documents}) != len(documents):
            raise ValueError("scenario package document paths must be unique")
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
                    "sha256": locator.sha256,
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
    raw_content_hash: str = field(compare=False)

    def __post_init__(self) -> None:
        try:
            role = self.role if isinstance(self.role, ScenarioDocumentRole) else ScenarioDocumentRole(self.role)
        except (TypeError, ValueError):
            raise ValueError("scenario source document role must be supported") from None
        if self.schema != SCENARIO_DOCUMENT_SCHEMA:
            raise ValueError("scenario source document schema is not supported")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "logical_id", _non_empty_text(self.logical_id, label="scenario source document logical id"))
        object.__setattr__(self, "value", _freeze_json_mapping(self.value, label="scenario source document value"))
        object.__setattr__(self, "raw_content_hash", _content_hash(
            self.raw_content_hash,
            label="scenario source document raw content hash",
        ))

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
    raw_manifest_hash: str = field(compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _non_empty_text(self.scenario_id, label="scenario source id"))
        object.__setattr__(self, "version", _non_empty_text(self.version, label="scenario source version"))
        documents = tuple(self.documents)
        if any(not isinstance(document, ScenarioSourceDocument) for document in documents):
            raise TypeError("scenario source documents must be ScenarioSourceDocument values")
        if len({_canonical_document_key(document) for document in documents}) != len(documents):
            raise ValueError("scenario source document role and logical id pairs must be unique")
        _validate_document_roles([document.role for document in documents])
        _validate_optional_fallbacks(documents)
        object.__setattr__(self, "documents", tuple(sorted(documents, key=_canonical_document_key)))
        object.__setattr__(self, "raw_manifest_hash", _content_hash(
            self.raw_manifest_hash,
            label="scenario source raw manifest hash",
        ))

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "version": self.version,
            "documents": [document.to_dict() for document in self.documents],
        }

    @property
    def raw_document_hashes(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (document.role.value, document.logical_id, document.raw_content_hash)
            for document in self.documents
        )

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _normalize_agent_body_roles(
    value: object,
    agent_ids: set[str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise TypeError("compiled scenario agent body roles must be a tuple")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("compiled scenario agent body roles must be pairs")
        agent_id = _non_empty_text(item[0], label="compiled body role agent ID")
        role_id = _non_empty_text(item[1], label="compiled body role ID")
        result.append((agent_id, role_id))
    if len({item[0] for item in result}) != len(result) or {
        item[0] for item in result
    } != agent_ids:
        raise ValueError("compiled scenario body role roster must cover exact agents")
    return tuple(sorted(result))


def _normalize_agent_knowledge_grants(
    value: object,
    agent_ids: set[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, tuple):
        raise TypeError("compiled scenario agent knowledge grants must be a tuple")
    result: list[tuple[str, tuple[str, ...]]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("compiled scenario agent knowledge grants must be pairs")
        agent_id = _non_empty_text(
            item[0],
            label="compiled knowledge grant agent ID",
        )
        resources = item[1]
        if not isinstance(resources, tuple):
            raise TypeError("compiled direct knowledge resource IDs must be a tuple")
        normalized = tuple(
            _non_empty_text(
                resource_id,
                label="compiled direct knowledge resource ID",
            )
            for resource_id in resources
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("compiled direct knowledge resource IDs must be unique")
        result.append((agent_id, tuple(sorted(normalized))))
    if len({item[0] for item in result}) != len(result) or {
        item[0] for item in result
    } != agent_ids:
        raise ValueError("compiled scenario knowledge grant roster must cover exact agents")
    return tuple(sorted(result))


def _validate_story_predicate_references(
    predicate: ScenarioPredicate,
    *,
    agent_ids: set[str],
    place_ids: set[str],
    passage_ids: set[str],
    object_ids: set[str],
    hypotheses_by_agent: Mapping[str, set[str]],
    topic_ids: set[str],
) -> None:
    kind = predicate.kind
    if kind is ScenarioPredicateKind.AGENT_AT:
        valid = predicate.subject_id in agent_ids and predicate.object_id in place_ids
    elif kind is ScenarioPredicateKind.PASSAGE_OPEN:
        valid = predicate.subject_id in passage_ids
    elif kind is ScenarioPredicateKind.OBJECT_AT:
        valid = predicate.subject_id in object_ids and predicate.object_id in place_ids
    elif kind is ScenarioPredicateKind.AGENT_HOLDS:
        valid = predicate.subject_id in agent_ids and predicate.object_id in object_ids
    elif kind is ScenarioPredicateKind.BELIEF_AT_LEAST:
        valid = (
            predicate.subject_id in agent_ids
            and predicate.object_id in hypotheses_by_agent[predicate.subject_id]
        )
    elif kind is ScenarioPredicateKind.CLAIM_STATUS:
        valid = predicate.subject_id in agent_ids and predicate.object_id in topic_ids
    else:
        valid = (
            predicate.subject_id in agent_ids
            and predicate.object_id in agent_ids
        )
    if not valid:
        raise ValueError("compiled scenario story predicate references must be known")


def _subject_is_known(
    scope: str,
    subject_id: str | None,
    *,
    agent_ids: set[str],
    role_ids: set[str],
    institution_ids: set[str],
) -> bool:
    return (
        (scope == "public" and subject_id is None)
        or (scope == "agent" and subject_id in agent_ids)
        or (scope == "role" and subject_id in role_ids)
        or (scope == "institution" and subject_id in institution_ids)
    )


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
    agent_body_roles: tuple[tuple[str, str], ...]
    agent_knowledge_grants: tuple[tuple[str, tuple[str, ...]], ...]
    intervention_kinds: tuple[str, ...]
    raw_manifest_hash: str = field(compare=False)
    raw_source_document_hashes: tuple[tuple[str, str, str], ...] = field(
        compare=False
    )

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
        source_roles = []
        for item in self.source_document_hashes:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("compiled scenario source document hashes must be triples")
            role, logical_id, content_hash = item
            try:
                document_role = ScenarioDocumentRole(role)
            except (TypeError, ValueError):
                raise ValueError(
                    "compiled source document role must be supported"
                ) from None
            role = document_role.value
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
            source_roles.append(document_role)
            source_hashes.append((role, logical_id, content_hash))
        _validate_document_roles(source_roles)
        object.__setattr__(
            self,
            "source_document_hashes",
            tuple(sorted(source_hashes, key=lambda item: (item[0], item[1]))),
        )
        if self.package_hash != _semantic_package_hash(
            self.scenario_id,
            self.version,
            self.source_document_hashes,
        ):
            raise ValueError(
                "compiled scenario package hash must be the exact semantic package hash"
            )
        object.__setattr__(
            self,
            "raw_manifest_hash",
            _content_hash(
                self.raw_manifest_hash,
                label="compiled scenario raw manifest hash",
            ),
        )
        if not isinstance(self.raw_source_document_hashes, tuple):
            raise TypeError("compiled raw source document hashes must be a tuple")
        raw_hashes: list[tuple[str, str, str]] = []
        raw_keys: set[tuple[str, str]] = set()
        for item in self.raw_source_document_hashes:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("compiled raw source document hashes must be triples")
            role, logical_id, content_hash = item
            try:
                normalized_role = ScenarioDocumentRole(role).value
            except (TypeError, ValueError):
                raise ValueError("compiled raw source document role must be supported") from None
            normalized_logical_id = _non_empty_text(
                logical_id,
                label="compiled raw source document logical ID",
            )
            key = (normalized_role, normalized_logical_id)
            if key in raw_keys:
                raise ValueError("compiled raw source document identities must be unique")
            raw_keys.add(key)
            raw_hashes.append(
                (
                    normalized_role,
                    normalized_logical_id,
                    _content_hash(
                        content_hash,
                        label="compiled raw source document hash",
                    ),
                )
            )
        if raw_keys != source_keys:
            raise ValueError(
                "compiled raw provenance must cover exact source document identities"
            )
        object.__setattr__(
            self,
            "raw_source_document_hashes",
            tuple(sorted(raw_hashes)),
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
        expected_cognitive_state = initialize_situated_percept_memory_cognition(
            self.runtime_model.percept_memory_model,
            self.initial_story,
        )
        if self.initial_cognitive_state != expected_cognitive_state:
            raise ValueError(
                "compiled scenario cognition must equal the exact public initializer checkpoint"
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

        agent_ids = {item.agent_id for item in world.agents}
        place_ids = {item.place_id for item in world.places}
        passage_ids = {item.passage_id for item in world.passages}
        object_ids = {item.object_id for item in world.objects}
        institution_ids = {
            item.institution_id for item in self.social_world.institutions
        }
        memberships = self.social_world.memberships
        for membership in memberships:
            if membership.agent_id not in agent_ids:
                raise ValueError("compiled scenario membership agent must be known")
            if membership.institution_id not in institution_ids:
                raise ValueError("compiled scenario membership institution must be known")
        if memberships and {item.agent_id for item in memberships} != agent_ids:
            raise ValueError("compiled scenario membership roster must cover exact agents")
        role_ids = {item.role_id for item in memberships}
        roles_by_agent: dict[str, set[str]] = {
            agent_id: set() for agent_id in agent_ids
        }
        institutions_by_agent: dict[str, set[str]] = {
            agent_id: set() for agent_id in agent_ids
        }
        for membership in memberships:
            roles_by_agent[membership.agent_id].add(membership.role_id)
            institutions_by_agent[membership.agent_id].add(
                membership.institution_id
            )

        body_roles = _normalize_agent_body_roles(self.agent_body_roles, agent_ids)
        expected_world_roles = {
            item.agent_id: item.role for item in world.agents
        }
        for agent_id, role_id in body_roles:
            if role_id != expected_world_roles[agent_id] or (
                memberships and role_id not in roles_by_agent[agent_id]
            ):
                raise ValueError(
                    "compiled scenario body role must bind the exact world membership"
                )
        object.__setattr__(self, "agent_body_roles", body_roles)

        for relationship in self.social_world.relationships:
            if (
                relationship.source_agent_id not in agent_ids
                or relationship.target_agent_id not in agent_ids
            ):
                raise ValueError(
                    "compiled scenario typed relationship agents must be known"
                )
        relationship_types = {
            item.relationship_type for item in self.social_world.relationships
        }
        if set(self.social_world.registered_relationship_types) != relationship_types:
            raise ValueError(
                "compiled scenario relationship catalog must equal typed relationships"
            )

        seed_by_pair = {
            (item.observer_agent_id, item.source_agent_id): item
            for item in self.social_world.relationship_seeds
        }
        expected_pairs = {
            (observer, source)
            for observer in agent_ids
            for source in agent_ids
            if observer != source
        }
        if set(seed_by_pair) != expected_pairs:
            raise ValueError(
                "compiled scenario relationship seeds must cover exact directed agent pairs"
            )

        action_ids = {
            action.action_id
            for agent in cognition.agents
            for action in agent.actions
        }
        if set(self.social_world.registered_action_ids) != action_ids:
            raise ValueError(
                "compiled scenario action catalog must equal cognitive actions"
            )
        knowledge_resources = {
            item.resource_id: item for item in self.knowledge_catalog.resources
        }
        asset_resources = {
            item.resource_id: item for item in self.asset_catalog.resources
        }
        all_resources = {**knowledge_resources, **asset_resources}
        if len(all_resources) != len(knowledge_resources) + len(asset_resources):
            raise ValueError("compiled scenario resource IDs must be globally unique")
        if set(self.social_world.registered_resource_ids) != set(all_resources):
            raise ValueError(
                "compiled scenario resource catalog must equal retained resources"
            )

        def validate_catalog(catalog: ScenarioKnowledgeCatalog | ScenarioAssetCatalog) -> None:
            resources = {item.resource_id: item for item in catalog.resources}
            for resource in catalog.resources:
                for entitlement in resource.entitlements:
                    if not _subject_is_known(
                        entitlement.subject_scope,
                        entitlement.subject_id,
                        agent_ids=agent_ids,
                        role_ids=role_ids,
                        institution_ids=institution_ids,
                    ):
                        raise ValueError(
                            "compiled scenario resource entitlement subject must be known"
                        )
            for grant in catalog.grants:
                if not _subject_is_known(
                    grant.subject_scope,
                    grant.subject_id,
                    agent_ids=agent_ids,
                    role_ids=role_ids,
                    institution_ids=institution_ids,
                ):
                    raise ValueError(
                        "compiled scenario catalog grant subject must be known"
                    )
                for resource_id in grant.resource_ids:
                    if resource_id not in resources:
                        raise ValueError(
                            "compiled scenario catalog grant resource must be known"
                        )
                    allowed = {
                        (item.subject_scope, item.subject_id)
                        for item in resources[resource_id].entitlements
                    }
                    if (
                        (grant.subject_scope, grant.subject_id) not in allowed
                        and ("public", None) not in allowed
                    ):
                        raise ValueError(
                            "compiled scenario catalog grant must be entitled"
                        )

        validate_catalog(self.knowledge_catalog)
        validate_catalog(self.asset_catalog)

        agent_grants = _normalize_agent_knowledge_grants(
            self.agent_knowledge_grants,
            agent_ids,
        )
        for agent_id, resource_ids in agent_grants:
            for resource_id in resource_ids:
                if resource_id not in knowledge_resources:
                    raise ValueError(
                        "compiled scenario direct knowledge grant resource must be known"
                    )
                allowed = {
                    (item.subject_scope, item.subject_id)
                    for item in knowledge_resources[resource_id].entitlements
                }
                if not (
                    ("public", None) in allowed
                    or ("agent", agent_id) in allowed
                    or any(("role", role_id) in allowed for role_id in roles_by_agent[agent_id])
                    or any(
                        ("institution", institution_id) in allowed
                        for institution_id in institutions_by_agent[agent_id]
                    )
                ):
                    raise ValueError(
                        "compiled scenario direct knowledge grant must be entitled"
                    )
        object.__setattr__(self, "agent_knowledge_grants", agent_grants)

        known_target_scopes = (
            agent_ids
            | place_ids
            | passage_ids
            | object_ids
            | institution_ids
            | role_ids
            | relationship_types
            | set(all_resources)
        )
        for norm in self.social_world.norms:
            active_field = (
                "action_id"
                if norm.effect in {
                    ScenarioNormEffect.ALLOW_ACTION,
                    ScenarioNormEffect.DENY_ACTION,
                }
                else "resource_id"
                if norm.effect is ScenarioNormEffect.REQUIRE_KNOWLEDGE_GRANT
                else "relationship_type"
                if norm.effect is ScenarioNormEffect.REQUIRE_RELATIONSHIP
                else None
            )
            if any(
                field_name != active_field and getattr(norm, field_name) is not None
                for field_name in ("action_id", "resource_id", "relationship_type")
            ):
                raise ValueError("compiled scenario norm must retain its exact tagged union")
            if norm.subject_role_id not in role_ids:
                raise ValueError("compiled scenario norm subject role must be known")
            if norm.action_id is not None and norm.action_id not in action_ids:
                raise ValueError("compiled scenario norm action must be known")
            if norm.resource_id is not None and norm.resource_id not in all_resources:
                raise ValueError("compiled scenario norm resource must be known")
            if (
                norm.relationship_type is not None
                and norm.relationship_type not in relationship_types
            ):
                raise ValueError("compiled scenario norm relationship type must be known")
            if (
                norm.target_scope_id is not None
                and norm.target_scope_id not in known_target_scopes
            ):
                raise ValueError("compiled scenario norm target scope must be known")

        if not isinstance(self.intervention_kinds, tuple):
            raise TypeError("compiled scenario intervention kinds must be a tuple")
        intervention_kinds = tuple(
            _non_empty_text(item, label="compiled intervention kind")
            for item in self.intervention_kinds
        )
        if len(set(intervention_kinds)) != len(intervention_kinds):
            raise ValueError("compiled scenario intervention kinds must be unique")
        intervention_kinds = tuple(sorted(intervention_kinds))
        object.__setattr__(self, "intervention_kinds", intervention_kinds)

        hypotheses_by_agent = {
            agent.agent_id: {item.hypothesis_id for item in agent.hypotheses}
            for agent in cognition.agents
        }
        topic_ids = {
            item.topic_id for item in self.runtime_model.social_memory_model.topics
        }
        predicates: list[ScenarioPredicate] = list(
            self.story_plan.continuity_predicates
            + self.story_plan.terminal_predicates
        )
        for scene in self.story_plan.scenes:
            if not set(scene.place_ids).issubset(place_ids):
                raise ValueError("compiled scenario story place must be known")
            if not set(scene.participant_agent_ids).issubset(agent_ids):
                raise ValueError("compiled scenario story participant agent must be known")
            if not set(scene.allowed_intervention_kinds).issubset(intervention_kinds):
                raise ValueError(
                    "compiled scenario story intervention kind must be registered"
                )
            predicates.extend(scene.preconditions)
            predicates.extend(scene.exit_predicates)
        for predicate in predicates:
            _validate_story_predicate_references(
                predicate,
                agent_ids=agent_ids,
                place_ids=place_ids,
                passage_ids=passage_ids,
                object_ids=object_ids,
                hypotheses_by_agent=hypotheses_by_agent,
                topic_ids=topic_ids,
            )

        base_social_state = initialize_situated_social_memory(
            self.runtime_model.social_memory_model,
            expected_cognitive_state,
        )
        expected_social_state = SituatedSocialMemoryState(
            base_social_state.model_id,
            base_social_state.model_hash,
            base_social_state.round_index,
            base_social_state.parent_state_hash,
            base_social_state.cognitive_state_hash,
            tuple(
                SituatedSourceRelationship(
                    item.observer_agent_id,
                    item.source_agent_id,
                    seed_by_pair[
                        (item.observer_agent_id, item.source_agent_id)
                    ].trust,
                    seed_by_pair[
                        (item.observer_agent_id, item.source_agent_id)
                    ].affinity,
                    item.confirmation_count,
                    item.contradiction_count,
                )
                for item in base_social_state.relationships
            ),
            base_social_state.claims,
            base_social_state.processed_evidence_ids,
            base_social_state.checkpoint,
        )
        if self.initial_social_state != expected_social_state:
            raise ValueError(
                "compiled scenario social state must equal the exact public initializer checkpoint"
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
            "agent_body_roles": [list(item) for item in self.agent_body_roles],
            "agent_knowledge_grants": [
                [agent_id, list(resource_ids)]
                for agent_id, resource_ids in self.agent_knowledge_grants
            ],
            "intervention_kinds": list(self.intervention_kinds),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
