"""Frozen, declarative authoring contracts for situated scenarios.

These values deliberately validate local structure only.  Resolving their stable
IDs against a physical scenario package is the compiler's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from narrative_dynamics.abm.situated_social_memory_contracts import SituatedClaimStatus


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_URI_LENGTH = 2048
_MAX_METADATA_LENGTH = 256
_MAX_DESCRIPTIVE_TEXT_LENGTH = 2048
_OUTPUT_KINDS = frozenset({
    "state.delta",
    "event.objective",
    "percept.private",
    "agent.decision",
    "memory.update",
    "social.update",
    "network.metrics",
    "story.progress",
    "narrative.scene",
    "blender.delta",
    "command.result",
    "diagnostic",
})


def _text(value: object, label: str, *, maximum: int = _MAX_METADATA_LENGTH) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be bounded non-empty text")
    return value


def _optional_text(value: object, label: str, *, maximum: int = _MAX_METADATA_LENGTH) -> str | None:
    if value is None:
        return None
    return _text(value, label, maximum=maximum)


def _identifier_tuple(values: object, label: str, *, ordered: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a tuple of IDs")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be a tuple of IDs") from error
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{label} must contain non-empty IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} IDs must be unique")
    return result if ordered else tuple(sorted(result))


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _content_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact sha256 content hash")
    return value


def _as_enum(value: object, enum_type: type[Enum], label: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)  # type: ignore[call-arg]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not supported") from error


def _acyclic(edges: tuple[tuple[str, str], ...], label: str) -> None:
    children: dict[str, set[str]] = {}
    for source, target in edges:
        children.setdefault(source, set()).add(target)
        children.setdefault(target, set())
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"{label} contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in children[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in children:
        visit(node)


class ScenarioExecutionMode(str, Enum):
    AUTHORED = "authored"
    HYBRID = "hybrid"
    SANDBOX = "sandbox"


@dataclass(frozen=True)
class ScenarioInstitution:
    institution_id: str
    institution_kind: str
    parent_institution_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "institution_id", _text(self.institution_id, "institution ID"))
        object.__setattr__(self, "institution_kind", _text(self.institution_kind, "institution kind"))
        object.__setattr__(self, "parent_institution_id", _optional_text(
            self.parent_institution_id, "institution parent ID"))
        if self.parent_institution_id == self.institution_id:
            raise ValueError("institution containment contains a cycle")


@dataclass(frozen=True)
class ScenarioMembership:
    agent_id: str
    institution_id: str
    role_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, "membership agent ID"))
        object.__setattr__(self, "institution_id", _text(self.institution_id, "membership institution ID"))
        object.__setattr__(self, "role_id", _text(self.role_id, "membership role ID"))


@dataclass(frozen=True)
class ScenarioRelationship:
    source_agent_id: str
    target_agent_id: str
    relationship_type: str
    strength: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_agent_id", _text(self.source_agent_id, "relationship source agent ID"))
        object.__setattr__(self, "target_agent_id", _text(self.target_agent_id, "relationship target agent ID"))
        object.__setattr__(self, "relationship_type", _text(self.relationship_type, "relationship type"))
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise ValueError("relationship strength must be finite and within [-1, 1]")
        strength = float(self.strength)
        if not math.isfinite(strength) or not -1 <= strength <= 1:
            raise ValueError("relationship strength must be finite and within [-1, 1]")
        object.__setattr__(self, "strength", strength)


class ScenarioNormEffect(str, Enum):
    ALLOW_ACTION = "allow_action"
    DENY_ACTION = "deny_action"
    REQUIRE_KNOWLEDGE_GRANT = "require_knowledge_grant"
    REQUIRE_RELATIONSHIP = "require_relationship"
    DESCRIPTIVE = "descriptive"


@dataclass(frozen=True)
class ScenarioNorm:
    norm_id: str
    subject_role_id: str
    effect: ScenarioNormEffect
    priority: int
    action_id: str | None = None
    resource_id: str | None = None
    relationship_type: str | None = None
    target_scope_id: str | None = None
    descriptive_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "norm_id", _text(self.norm_id, "norm ID"))
        object.__setattr__(self, "subject_role_id", _text(self.subject_role_id, "norm subject role ID"))
        effect = _as_enum(self.effect, ScenarioNormEffect, "norm effect")
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "priority", _positive_int(self.priority, "norm priority"))
        object.__setattr__(self, "action_id", _optional_text(self.action_id, "norm action ID"))
        object.__setattr__(self, "resource_id", _optional_text(self.resource_id, "norm resource ID"))
        object.__setattr__(self, "relationship_type", _optional_text(
            self.relationship_type, "norm relationship type"))
        object.__setattr__(self, "target_scope_id", _optional_text(
            self.target_scope_id, "norm target scope ID"))
        object.__setattr__(self, "descriptive_text", _optional_text(
            self.descriptive_text, "norm descriptive text", maximum=_MAX_DESCRIPTIVE_TEXT_LENGTH))
        if effect in (ScenarioNormEffect.ALLOW_ACTION, ScenarioNormEffect.DENY_ACTION) and self.action_id is None:
            raise ValueError("action norm requires an action ID")
        if effect is ScenarioNormEffect.REQUIRE_KNOWLEDGE_GRANT and self.resource_id is None:
            raise ValueError("knowledge-grant norm requires a resource ID")
        if effect is ScenarioNormEffect.REQUIRE_RELATIONSHIP and self.relationship_type is None:
            raise ValueError("relationship norm requires a relationship type")
        if effect is ScenarioNormEffect.DESCRIPTIVE and self.descriptive_text is None:
            raise ValueError("descriptive norm requires bounded text")

    @property
    def enforcement_hook(self) -> str | None:
        return None if self.effect is ScenarioNormEffect.DESCRIPTIVE else self.effect.value


@dataclass(frozen=True)
class ScenarioSocialWorld:
    institutions: tuple[ScenarioInstitution, ...]
    memberships: tuple[ScenarioMembership, ...]
    relationships: tuple[ScenarioRelationship, ...]
    norms: tuple[ScenarioNorm, ...]
    registered_action_ids: tuple[str, ...] = ()
    registered_resource_ids: tuple[str, ...] = ()
    registered_relationship_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        institutions = tuple(self.institutions)
        memberships = tuple(self.memberships)
        relationships = tuple(self.relationships)
        norms = tuple(self.norms)
        if any(not isinstance(value, ScenarioInstitution) for value in institutions):
            raise TypeError("institutions must be ScenarioInstitution values")
        if any(not isinstance(value, ScenarioMembership) for value in memberships):
            raise TypeError("memberships must be ScenarioMembership values")
        if any(not isinstance(value, ScenarioRelationship) for value in relationships):
            raise TypeError("relationships must be ScenarioRelationship values")
        if any(not isinstance(value, ScenarioNorm) for value in norms):
            raise TypeError("norms must be ScenarioNorm values")
        if len({item.institution_id for item in institutions}) != len(institutions):
            raise ValueError("institution IDs must be unique")
        institution_ids = {item.institution_id for item in institutions}
        if any(item.parent_institution_id not in institution_ids for item in institutions if item.parent_institution_id):
            raise ValueError("institution parent ID must name an institution")
        _acyclic(tuple(
            (item.institution_id, item.parent_institution_id)
            for item in institutions if item.parent_institution_id is not None
        ), "institution containment")
        membership_keys = {(item.agent_id, item.institution_id, item.role_id) for item in memberships}
        if len(membership_keys) != len(memberships):
            raise ValueError("membership IDs must be unique")
        relationship_keys = {
            (item.source_agent_id, item.target_agent_id, item.relationship_type)
            for item in relationships
        }
        if len(relationship_keys) != len(relationships):
            raise ValueError("relationship edges must be unique")
        if len({item.norm_id for item in norms}) != len(norms):
            raise ValueError("norm IDs must be unique")
        action_ids = _identifier_tuple(self.registered_action_ids, "registered action", ordered=False)
        resource_ids = _identifier_tuple(self.registered_resource_ids, "registered resource", ordered=False)
        relationship_types = _identifier_tuple(
            self.registered_relationship_types, "registered relationship type", ordered=False)
        for norm in norms:
            if norm.effect in (ScenarioNormEffect.ALLOW_ACTION, ScenarioNormEffect.DENY_ACTION) and norm.action_id not in action_ids:
                raise ValueError("norm action ID must be a registered action")
            if norm.effect is ScenarioNormEffect.REQUIRE_KNOWLEDGE_GRANT and norm.resource_id not in resource_ids:
                raise ValueError("norm resource ID must be a registered resource")
            if norm.effect is ScenarioNormEffect.REQUIRE_RELATIONSHIP and norm.relationship_type not in relationship_types:
                raise ValueError("norm relationship type must be registered")
        object.__setattr__(self, "institutions", tuple(sorted(institutions, key=lambda item: item.institution_id)))
        object.__setattr__(self, "memberships", tuple(sorted(memberships, key=lambda item: (
            item.agent_id, item.institution_id, item.role_id))))
        object.__setattr__(self, "relationships", tuple(sorted(relationships, key=lambda item: (
            item.source_agent_id, item.target_agent_id, item.relationship_type))))
        object.__setattr__(self, "norms", tuple(sorted(norms, key=lambda item: item.norm_id)))
        object.__setattr__(self, "registered_action_ids", action_ids)
        object.__setattr__(self, "registered_resource_ids", resource_ids)
        object.__setattr__(self, "registered_relationship_types", relationship_types)


class ScenarioPredicateKind(str, Enum):
    AGENT_AT = "agent_at"
    PASSAGE_OPEN = "passage_open"
    OBJECT_AT = "object_at"
    AGENT_HOLDS = "agent_holds"
    BELIEF_AT_LEAST = "belief_at_least"
    CLAIM_STATUS = "claim_status"
    RELATIONSHIP_AT_LEAST = "relationship_at_least"


@dataclass(frozen=True)
class ScenarioPredicate:
    kind: ScenarioPredicateKind
    subject_id: str
    object_id: str | None
    value: object

    def __post_init__(self) -> None:
        kind = _as_enum(self.kind, ScenarioPredicateKind, "predicate kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "subject_id", _text(self.subject_id, "predicate subject ID"))
        object.__setattr__(self, "object_id", _optional_text(self.object_id, "predicate object ID"))
        if kind in (ScenarioPredicateKind.AGENT_AT, ScenarioPredicateKind.OBJECT_AT, ScenarioPredicateKind.AGENT_HOLDS):
            if self.object_id is None or self.value is not None:
                raise ValueError(f"{kind.value} predicate requires object ID and no value")
        elif kind is ScenarioPredicateKind.PASSAGE_OPEN:
            if self.object_id is not None or not isinstance(self.value, bool):
                raise ValueError("passage_open predicate value must be boolean")
        elif kind is ScenarioPredicateKind.BELIEF_AT_LEAST:
            self._numeric_value(-math.inf, math.inf)
            if self.object_id is None:
                raise ValueError("belief_at_least predicate requires object ID")
        elif kind is ScenarioPredicateKind.CLAIM_STATUS:
            if self.object_id is None:
                raise ValueError("claim_status predicate requires object ID and status value")
            object.__setattr__(self, "value", _as_enum(
                self.value, SituatedClaimStatus, "claim_status predicate status"))
        elif kind is ScenarioPredicateKind.RELATIONSHIP_AT_LEAST:
            if self.object_id is None:
                raise ValueError("relationship_at_least predicate requires object ID")
            self._numeric_value(-1, 1)

    def _numeric_value(self, minimum: float, maximum: float) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError(f"{self.kind.value} predicate value must be numeric")
        value = float(self.value)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{self.kind.value} predicate value is out of range")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class ScenarioSceneDependency:
    predecessor_scene_id: str
    successor_scene_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessor_scene_id", _text(
            self.predecessor_scene_id, "dependency predecessor scene ID"))
        object.__setattr__(self, "successor_scene_id", _text(
            self.successor_scene_id, "dependency successor scene ID"))
        if self.predecessor_scene_id == self.successor_scene_id:
            raise ValueError("story dependency contains a cycle")


@dataclass(frozen=True)
class ScenarioSceneContract:
    scene_id: str
    place_ids: tuple[str, ...]
    participant_agent_ids: tuple[str, ...]
    preconditions: tuple[ScenarioPredicate, ...]
    exit_predicates: tuple[ScenarioPredicate, ...]
    allowed_intervention_kinds: tuple[str, ...]
    desired_outcome_ids: tuple[str, ...]
    maximum_rounds: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _text(self.scene_id, "scene ID"))
        object.__setattr__(self, "place_ids", _identifier_tuple(self.place_ids, "scene place"))
        object.__setattr__(self, "participant_agent_ids", _identifier_tuple(
            self.participant_agent_ids, "scene participant agent"))
        preconditions = tuple(self.preconditions)
        exit_predicates = tuple(self.exit_predicates)
        if any(not isinstance(value, ScenarioPredicate) for value in preconditions + exit_predicates):
            raise TypeError("scene predicates must be ScenarioPredicate values")
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "exit_predicates", exit_predicates)
        object.__setattr__(self, "allowed_intervention_kinds", _identifier_tuple(
            self.allowed_intervention_kinds, "scene intervention kind"))
        object.__setattr__(self, "desired_outcome_ids", _identifier_tuple(
            self.desired_outcome_ids, "scene desired outcome"))
        object.__setattr__(self, "maximum_rounds", _positive_int(self.maximum_rounds, "scene maximum rounds"))


@dataclass(frozen=True)
class ScenarioStoryAct:
    act_id: str
    scene_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "act_id", _text(self.act_id, "story act ID"))
        object.__setattr__(self, "scene_ids", _identifier_tuple(
            self.scene_ids, "story act scene", ordered=True))


@dataclass(frozen=True)
class ScenarioStoryPlan:
    plan_id: str
    version: str
    mode: ScenarioExecutionMode
    acts: tuple[ScenarioStoryAct, ...]
    scenes: tuple[ScenarioSceneContract, ...]
    dependencies: tuple[ScenarioSceneDependency, ...]
    continuity_predicates: tuple[ScenarioPredicate, ...] = ()
    terminal_predicates: tuple[ScenarioPredicate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "story plan ID"))
        object.__setattr__(self, "version", _text(self.version, "story plan version"))
        object.__setattr__(self, "mode", _as_enum(self.mode, ScenarioExecutionMode, "story execution mode"))
        acts = tuple(self.acts)
        scenes = tuple(self.scenes)
        dependencies = tuple(self.dependencies)
        continuity = tuple(self.continuity_predicates)
        terminal = tuple(self.terminal_predicates)
        if any(not isinstance(value, ScenarioStoryAct) for value in acts):
            raise TypeError("story acts must be ScenarioStoryAct values")
        if any(not isinstance(value, ScenarioSceneContract) for value in scenes):
            raise TypeError("story scenes must be ScenarioSceneContract values")
        if any(not isinstance(value, ScenarioSceneDependency) for value in dependencies):
            raise TypeError("story dependencies must be ScenarioSceneDependency values")
        if any(not isinstance(value, ScenarioPredicate) for value in continuity + terminal):
            raise TypeError("story predicates must be ScenarioPredicate values")
        if len({item.act_id for item in acts}) != len(acts):
            raise ValueError("story act IDs must be unique")
        scene_ids = {item.scene_id for item in scenes}
        if len(scene_ids) != len(scenes):
            raise ValueError("story scene IDs must be unique")
        act_scene_ids = tuple(scene_id for act in acts for scene_id in act.scene_ids)
        if len(set(act_scene_ids)) != len(act_scene_ids):
            raise ValueError("story scene may belong to only one act")
        if set(act_scene_ids) != scene_ids:
            raise ValueError("story acts must contain every scene exactly once")
        if any(
            dependency.predecessor_scene_id not in scene_ids
            or dependency.successor_scene_id not in scene_ids
            for dependency in dependencies
        ):
            raise ValueError("story dependency must name known scenes")
        dependency_keys = {
            (item.predecessor_scene_id, item.successor_scene_id) for item in dependencies
        }
        if len(dependency_keys) != len(dependencies):
            raise ValueError("story dependencies must be unique")
        _acyclic(tuple((item.predecessor_scene_id, item.successor_scene_id) for item in dependencies), "story dependencies")
        if self.mode is ScenarioExecutionMode.AUTHORED and not scenes:
            raise ValueError("authored story plan requires a scene")
        object.__setattr__(self, "acts", acts)
        object.__setattr__(self, "scenes", tuple(sorted(scenes, key=lambda item: item.scene_id)))
        object.__setattr__(self, "dependencies", tuple(sorted(dependencies, key=lambda item: (
            item.predecessor_scene_id, item.successor_scene_id))))
        object.__setattr__(self, "continuity_predicates", tuple(sorted(continuity, key=_predicate_key)))
        object.__setattr__(self, "terminal_predicates", tuple(sorted(terminal, key=_predicate_key)))


class ScenarioResourceKind(str, Enum):
    DOCUMENT = "document"
    DATASET = "dataset"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MODEL_3D = "model_3d"


def _predicate_key(predicate: ScenarioPredicate) -> tuple[str, str, str, str, str]:
    return (
        predicate.kind.value,
        predicate.subject_id,
        predicate.object_id or "",
        type(predicate.value).__name__,
        repr(predicate.value),
    )


def _resource_values(resource_id: object, kind: object, content_hash: object, uri: object, media_type: object) -> tuple[str, ScenarioResourceKind, str, str, str]:
    return (
        _text(resource_id, "resource ID"),
        _as_enum(kind, ScenarioResourceKind, "resource kind"),  # type: ignore[return-value]
        _content_hash(content_hash, "resource content hash"),
        _text(uri, "resource URI", maximum=_MAX_URI_LENGTH),
        _text(media_type, "resource media type"),
    )


@dataclass(frozen=True)
class ScenarioKnowledgeResource:
    resource_id: str
    kind: ScenarioResourceKind
    content_hash: str
    uri: str
    media_type: str
    language: str
    version: str
    authority: str
    license_tag: str
    concept_ids: tuple[str, ...] = ()
    index_id: str | None = None

    def __post_init__(self) -> None:
        values = _resource_values(self.resource_id, self.kind, self.content_hash, self.uri, self.media_type)
        for field, value in zip(("resource_id", "kind", "content_hash", "uri", "media_type"), values):
            object.__setattr__(self, field, value)
        object.__setattr__(self, "language", _text(self.language, "knowledge language"))
        object.__setattr__(self, "version", _text(self.version, "knowledge version"))
        object.__setattr__(self, "authority", _text(self.authority, "knowledge authority"))
        object.__setattr__(self, "license_tag", _text(self.license_tag, "knowledge license tag"))
        object.__setattr__(self, "concept_ids", _identifier_tuple(self.concept_ids, "knowledge concept"))
        object.__setattr__(self, "index_id", _optional_text(self.index_id, "knowledge index ID"))


@dataclass(frozen=True)
class ScenarioAssetResource:
    resource_id: str
    kind: ScenarioResourceKind
    content_hash: str
    uri: str
    media_type: str
    authority: str
    license_tag: str
    dimensions: tuple[float, ...] = ()
    unit: str | None = None
    format: str | None = None
    collection_id: str | None = None
    rig_metadata_ids: tuple[str, ...] = ()
    collision_metadata_ids: tuple[str, ...] = ()
    preview_hash: str | None = None

    def __post_init__(self) -> None:
        values = _resource_values(self.resource_id, self.kind, self.content_hash, self.uri, self.media_type)
        for field, value in zip(("resource_id", "kind", "content_hash", "uri", "media_type"), values):
            object.__setattr__(self, field, value)
        object.__setattr__(self, "authority", _text(self.authority, "asset authority"))
        object.__setattr__(self, "license_tag", _text(self.license_tag, "asset license tag"))
        try:
            dimensions = tuple(self.dimensions)
        except TypeError as error:
            raise TypeError("asset dimensions must be a tuple") from error
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in dimensions):
            raise ValueError("asset dimensions must be finite positive numbers")
        object.__setattr__(self, "dimensions", tuple(float(value) for value in dimensions))
        object.__setattr__(self, "unit", _optional_text(self.unit, "asset unit"))
        object.__setattr__(self, "format", _optional_text(self.format, "asset format"))
        object.__setattr__(self, "collection_id", _optional_text(self.collection_id, "asset collection ID"))
        object.__setattr__(self, "rig_metadata_ids", _identifier_tuple(self.rig_metadata_ids, "asset rig metadata"))
        object.__setattr__(self, "collision_metadata_ids", _identifier_tuple(
            self.collision_metadata_ids, "asset collision metadata"))
        if self.preview_hash is not None:
            object.__setattr__(self, "preview_hash", _content_hash(self.preview_hash, "asset preview hash"))


@dataclass(frozen=True)
class ScenarioResourceGrant:
    subject_scope: str
    subject_id: str | None
    resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        scope = _text(self.subject_scope, "grant subject scope")
        if scope not in {"agent", "role", "institution", "public"}:
            raise ValueError("grant subject scope is not supported")
        subject_id = _optional_text(self.subject_id, "grant subject ID")
        if (scope == "public") != (subject_id is None):
            raise ValueError("public grants have no subject ID and scoped grants require one")
        object.__setattr__(self, "subject_scope", scope)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "resource_ids", _identifier_tuple(self.resource_ids, "grant resource"))
        if not self.resource_ids:
            raise ValueError("grant must name at least one resource")


def _catalog_values(resources: object, grants: object, resource_type: type[object], label: str) -> tuple[tuple[object, ...], tuple[ScenarioResourceGrant, ...]]:
    resources_tuple = tuple(resources)  # type: ignore[arg-type]
    grants_tuple = tuple(grants)  # type: ignore[arg-type]
    if any(not isinstance(item, resource_type) for item in resources_tuple):
        raise TypeError(f"{label} resources have an unsupported type")
    if any(not isinstance(item, ScenarioResourceGrant) for item in grants_tuple):
        raise TypeError(f"{label} grants must be ScenarioResourceGrant values")
    ids = tuple(item.resource_id for item in resources_tuple)  # type: ignore[union-attr]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} resource IDs must be unique")
    known = set(ids)
    if any(resource_id not in known for grant in grants_tuple for resource_id in grant.resource_ids):
        raise ValueError(f"{label} grants must name known resource IDs")
    grant_keys = {(grant.subject_scope, grant.subject_id) for grant in grants_tuple}
    if len(grant_keys) != len(grants_tuple):
        raise ValueError(f"{label} grants must be unique")
    return tuple(sorted(resources_tuple, key=lambda item: item.resource_id)), tuple(sorted(
        grants_tuple, key=lambda item: (item.subject_scope, item.subject_id or "", item.resource_ids)))


@dataclass(frozen=True)
class ScenarioKnowledgeCatalog:
    resources: tuple[ScenarioKnowledgeResource, ...]
    grants: tuple[ScenarioResourceGrant, ...]

    def __post_init__(self) -> None:
        resources, grants = _catalog_values(self.resources, self.grants, ScenarioKnowledgeResource, "knowledge catalog")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "grants", grants)


@dataclass(frozen=True)
class ScenarioAssetCatalog:
    resources: tuple[ScenarioAssetResource, ...]
    grants: tuple[ScenarioResourceGrant, ...] = ()

    def __post_init__(self) -> None:
        resources, grants = _catalog_values(self.resources, self.grants, ScenarioAssetResource, "asset catalog")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "grants", grants)


@dataclass(frozen=True)
class ScenarioRunPolicy:
    mode: ScenarioExecutionMode
    maximum_rounds: int
    checkpoint_interval: int
    maximum_output_records: int
    allowed_output_kinds: tuple[str, ...]
    public_journal: bool
    blender_mode: str
    deterministic_seed: int | None = None
    maximum_resource_bytes: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _as_enum(self.mode, ScenarioExecutionMode, "execution mode"))
        object.__setattr__(self, "maximum_rounds", _positive_int(self.maximum_rounds, "maximum rounds"))
        object.__setattr__(self, "checkpoint_interval", _positive_int(self.checkpoint_interval, "checkpoint interval"))
        object.__setattr__(self, "maximum_output_records", _positive_int(
            self.maximum_output_records, "maximum output records"))
        output_kinds = _identifier_tuple(self.allowed_output_kinds, "allowed output kind")
        if not output_kinds:
            raise ValueError("allowed output kinds must not be empty")
        if any(kind not in _OUTPUT_KINDS for kind in output_kinds):
            raise ValueError("allowed output kind is not supported")
        object.__setattr__(self, "allowed_output_kinds", output_kinds)
        if not isinstance(self.public_journal, bool):
            raise ValueError("public journal policy must be boolean")
        if self.blender_mode not in {"none", "final_blend", "live_mirror"}:
            raise ValueError("Blender mode is not supported")
        object.__setattr__(self, "blender_mode", self.blender_mode)
        if self.deterministic_seed is not None and (isinstance(self.deterministic_seed, bool) or not isinstance(self.deterministic_seed, int)):
            raise ValueError("deterministic seed must be an integer")
        object.__setattr__(self, "maximum_resource_bytes", _positive_int(
            self.maximum_resource_bytes, "maximum resource bytes"))


__all__ = (
    "ScenarioExecutionMode",
    "ScenarioInstitution",
    "ScenarioMembership",
    "ScenarioRelationship",
    "ScenarioNormEffect",
    "ScenarioNorm",
    "ScenarioSocialWorld",
    "ScenarioPredicateKind",
    "ScenarioPredicate",
    "ScenarioSceneDependency",
    "ScenarioSceneContract",
    "ScenarioStoryAct",
    "ScenarioStoryPlan",
    "ScenarioResourceKind",
    "ScenarioKnowledgeResource",
    "ScenarioKnowledgeCatalog",
    "ScenarioAssetResource",
    "ScenarioAssetCatalog",
    "ScenarioResourceGrant",
    "ScenarioRunPolicy",
)
