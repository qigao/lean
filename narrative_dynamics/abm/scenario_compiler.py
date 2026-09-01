"""Pure compilation and initialization of declarative V21.1 scenario packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import sqlite3

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from narrative_dynamics.abm.scenario_authoring_contracts import (
    ScenarioAssetCatalog,
    ScenarioAssetResource,
    ScenarioExecutionMode,
    ScenarioInstitution,
    ScenarioKnowledgeCatalog,
    ScenarioKnowledgeResource,
    ScenarioMembership,
    ScenarioNorm,
    ScenarioNormEffect,
    ScenarioPredicate,
    ScenarioPredicateKind,
    ScenarioRelationship,
    ScenarioRelationshipSeed,
    ScenarioResourceEntitlement,
    ScenarioResourceGrant,
    ScenarioResourceKind,
    ScenarioRunPolicy,
    ScenarioSceneContract,
    ScenarioSceneDependency,
    ScenarioSocialWorld,
    ScenarioStoryAct,
    ScenarioStoryPlan,
)
from narrative_dynamics.abm.scenario_package_contracts import (
    CompiledSituatedScenario,
    ScenarioDocumentRole,
    ScenarioPackageSource,
    ScenarioSourceDocument,
)
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedActionSpec,
    SituatedAgentCognitiveModel,
    SituatedCognitiveState,
    SituatedCognitiveModel,
    SituatedGoalReward,
    SituatedGoalSpec,
    SituatedHypothesis,
    SituatedHypothesisTransition,
    SituatedObservationLikelihood,
    SituatedObservationRule,
    SituatedObservationSymbol,
)
from narrative_dynamics.abm.situated_contracts import (
    AgentBodyState,
    EmbodiedAgentSpec,
    EvidenceFact,
    PassageState,
    PassageSpec,
    PlaceSpec,
    SituatedWorldState,
    SituatedWorldModel,
    WorldObjectState,
    WorldObjectSpec,
    validate_situated_state,
)
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
    SituatedMemoryCognitiveModel,
    SituatedMemoryRecallCue,
)
from narrative_dynamics.abm.situated_memory_contracts import (
    MemoryChannelPolicy,
    SituatedMemoryPolicy,
)
from narrative_dynamics.abm.situated_network import initialize_situated_network_runtime
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRuntimeModel,
    SituatedNetworkRuntimeState,
)
from narrative_dynamics.abm.situated_percept_memory import (
    hash_situated_percept_memory_store,
    initialize_situated_percept_memory,
    situated_percept_memory_schema_snapshot,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    SituatedPerceptMemoryFidelityPolicy,
    SituatedPerceptMemoryPolicy,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedAgentPerceptionProfile,
    SituatedEdgeActivation,
    SituatedEventSignalProfile,
    SituatedPerceptFidelity,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimTopic,
    SituatedSocialMemoryModel,
    SituatedSocialMemoryPolicy,
    SituatedSocialMemoryState,
    SituatedSourceRelationship,
    initialize_situated_social_memory,
    validate_situated_social_memory_state,
)
from narrative_dynamics.abm.situated_spatial_map_contracts import (
    SituatedSpatialMap,
)
from narrative_dynamics.abm.situated_spatial_map import (
    auto_layout_situated_spatial_map,
    compile_tiled_situated_spatial_map,
)
from narrative_dynamics.abm.situated_story import SituatedStory, initialize_situated_story


_CANONICAL_EMPTY_PERCEPT_MEMORY_STORE_HASH = stable_content_hash(
    {
        "store": "situated-percept-memory",
        "metadata": [{"key": "schema_version", "value": "1"}],
        "records": [],
    }
)


class ScenarioCompilationError(ValueError):
    """A value-free, path-free compiler diagnostic at one logical source pointer."""

    def __init__(self, document_role: str, json_pointer: str, code: str) -> None:
        self.document_role = document_role
        self.json_pointer = json_pointer
        self.code = code
        super().__init__(f"{document_role} {json_pointer}: {code}")


@dataclass(frozen=True)
class _InitialObjectLocation:
    object_id: str
    location_kind: str
    location_id: str


@dataclass(frozen=True)
class _ValidatedInitialStateSource:
    agent_places: tuple[tuple[str, str], ...]
    object_locations: tuple[_InitialObjectLocation, ...]
    passage_states: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class _CompiledScenarioComponents:
    scenario_id: str
    version: str
    package_hash: str
    source_document_hashes: tuple[tuple[str, str, str], ...]
    runtime_model: SituatedNetworkRuntimeModel
    spatial_map: SituatedSpatialMap
    social_world: ScenarioSocialWorld
    story_plan: ScenarioStoryPlan
    knowledge_catalog: ScenarioKnowledgeCatalog
    asset_catalog: ScenarioAssetCatalog
    run_policy: ScenarioRunPolicy
    initial_state_source: _ValidatedInitialStateSource
    intervention_kinds: tuple[str, ...]
    agent_body_roles: tuple[tuple[str, str], ...]
    agent_knowledge_grants: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def content_hash(self) -> str:
        return stable_content_hash(
            {
                "scenario_id": self.scenario_id,
                "version": self.version,
                "package_hash": self.package_hash,
                "source_document_hashes": [list(item) for item in self.source_document_hashes],
                "runtime_model_hash": self.runtime_model.content_hash,
                "spatial_map_hash": self.spatial_map.content_hash,
            }
        )


def _role(document: ScenarioSourceDocument) -> str:
    if document.role is ScenarioDocumentRole.AGENT:
        return f"agent:{document.logical_id}"
    return document.role.value


def _error(document: ScenarioSourceDocument, pointer: str, code: str) -> ScenarioCompilationError:
    return ScenarioCompilationError(_role(document), pointer, code)


def _singleton(source: ScenarioPackageSource, role: ScenarioDocumentRole) -> ScenarioSourceDocument:
    return next(document for document in source.documents if document.role is role)


def _optional_singleton(
    source: ScenarioPackageSource,
    role: ScenarioDocumentRole,
) -> ScenarioSourceDocument | None:
    return next((document for document in source.documents if document.role is role), None)


def _agents(source: ScenarioPackageSource) -> tuple[ScenarioSourceDocument, ...]:
    return tuple(document for document in source.documents if document.role is ScenarioDocumentRole.AGENT)


_SEMANTICALLY_ORDERED_ARRAY_PATHS: Mapping[
    ScenarioDocumentRole,
    frozenset[tuple[str, ...]],
] = {
    ScenarioDocumentRole.AGENT: frozenset({("cognition", "action_schedule")}),
    ScenarioDocumentRole.STORY_OUTLINE: frozenset(
        {
            ("acts",),
            ("acts", "*", "scene_ids"),
        }
    ),
    ScenarioDocumentRole.ASSET_CATALOG: frozenset(
        {("resources", "*", "dimensions")}
    ),
}


_SEMANTIC_INTEGER_PATHS = {
    ScenarioDocumentRole.AGENT: frozenset({
        ("body", "inventory_capacity"),
        ("memory", "recall", "cues", "*", "limit"),
        ("memory", "recall", "max_memories_per_round"),
    }),
    ScenarioDocumentRole.SOCIAL_RELATIONSHIPS: frozenset({
        ("policy", "max_unresolved_age_rounds"),
        ("policy", "max_active_claims"),
    }),
    ScenarioDocumentRole.SOCIAL_NORMS: frozenset({
        ("norms", "*", "priority"),
    }),
    ScenarioDocumentRole.STORY_OUTLINE: frozenset({
        ("scenes", "*", "maximum_rounds"),
    }),
    ScenarioDocumentRole.RUN: frozenset({
        ("deterministic_seed",),
        ("maximum_rounds",),
        ("checkpoint_interval",),
        ("maximum_output_records",),
        ("maximum_resource_bytes",),
    }),
}


def _semantic_path_matches(
    path: tuple[str, ...],
    pattern: tuple[str, ...],
) -> bool:
    return len(path) == len(pattern) and all(
        expected == "*" or expected == actual
        for actual, expected in zip(path, pattern)
    )


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _semantic_source_value(
    value: object,
    *,
    document: ScenarioSourceDocument,
    path: tuple[str, ...] = (),
) -> object:
    """Normalize unordered authored arrays while retaining true sequence semantics."""

    role = document.role
    if isinstance(value, Mapping):
        return {
            key: _semantic_source_value(
                item,
                document=document,
                path=path + (key,),
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        normalized = tuple(
            _semantic_source_value(
                item,
                document=document,
                path=path + (str(index),),
            )
            for index, item in enumerate(value)
        )
        ordered_paths = _SEMANTICALLY_ORDERED_ARRAY_PATHS.get(role, frozenset())
        if any(_semantic_path_matches(path, item) for item in ordered_paths):
            return normalized
        return tuple(sorted(normalized, key=stable_content_hash))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        integer_paths = _SEMANTIC_INTEGER_PATHS.get(role, frozenset())
        if any(_semantic_path_matches(path, item) for item in integer_paths):
            return value
        try:
            result = float(value)
        except (OverflowError, ValueError):
            pointer = "".join(f"/{_pointer_token(item)}" for item in path)
            raise _error(document, pointer, "invalid_value") from None
        if not math.isfinite(result):
            pointer = "".join(f"/{_pointer_token(item)}" for item in path)
            raise _error(document, pointer, "invalid_value")
        if result == 0.0:
            result = 0.0
        return result
    return value


def _semantic_document_hash(document: ScenarioSourceDocument) -> str:
    return stable_content_hash(
        {
            "role": document.role.value,
            "logical_id": document.logical_id,
            "schema": document.schema,
            "value": _semantic_source_value(document.value, document=document),
        }
    )


def _semantic_package_identity(
    source: ScenarioPackageSource,
) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    document_hashes = tuple(
        sorted(
            (
                document.role.value,
                document.logical_id,
                _semantic_document_hash(document),
            )
            for document in source.documents
        )
    )
    package_hash = stable_content_hash(
        {
            "scenario_id": source.scenario_id,
            "version": source.version,
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
    return package_hash, document_hashes


def _object(
    document: ScenarioSourceDocument,
    value: object,
    pointer: str,
    keys: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(document, pointer, "invalid_type")
    if set(value) != keys:
        raise _error(document, pointer, "unsupported_shape")
    return value


def _array(document: ScenarioSourceDocument, value: object, pointer: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise _error(document, pointer, "invalid_type")
    return value


def _text(document: ScenarioSourceDocument, value: object, pointer: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(document, pointer, "invalid_type")
    return value


def _optional_text(document: ScenarioSourceDocument, value: object, pointer: str) -> str | None:
    return None if value is None else _text(document, value, pointer)


def _boolean(document: ScenarioSourceDocument, value: object, pointer: str) -> bool:
    if not isinstance(value, bool):
        raise _error(document, pointer, "invalid_type")
    return value


def _number(document: ScenarioSourceDocument, value: object, pointer: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(document, pointer, "invalid_type")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise _error(document, pointer, "invalid_value") from None
    if not math.isfinite(result):
        raise _error(document, pointer, "invalid_value")
    return 0.0 if result == 0.0 else result


def _integer(document: ScenarioSourceDocument, value: object, pointer: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(document, pointer, "invalid_type")
    return value


def _enum(
    document: ScenarioSourceDocument,
    value: object,
    pointer: str,
    enum_type: type[Enum],
) -> Enum:
    if not isinstance(value, str):
        raise _error(document, pointer, "invalid_type")
    try:
        return enum_type(value)
    except ValueError:
        raise _error(document, pointer, "invalid_value") from None


def _texts(document: ScenarioSourceDocument, value: object, pointer: str) -> tuple[str, ...]:
    values = _array(document, value, pointer)
    return tuple(
        _text(document, item, f"{pointer}/{index}")
        for index, item in enumerate(values)
    )


def _enums(
    document: ScenarioSourceDocument,
    value: object,
    pointer: str,
    enum_type: type[Enum],
) -> tuple[Enum, ...]:
    values = _array(document, value, pointer)
    return tuple(
        _enum(document, item, f"{pointer}/{index}", enum_type)
        for index, item in enumerate(values)
    )


def _construct(document: ScenarioSourceDocument, pointer: str, factory):
    try:
        return factory()
    except ScenarioCompilationError:
        raise
    except (ArithmeticError, RecursionError, TypeError, ValueError):
        raise _error(document, pointer, "contract_violation") from None


_AGENT_KEYS = {
    "agent_id",
    "body",
    "perception",
    "cognition",
    "memory",
    "social",
    "knowledge_grants",
}


def _agent_value(document: ScenarioSourceDocument) -> Mapping[str, object]:
    value = _object(document, document.value, "", _AGENT_KEYS)
    agent_id = _text(document, value["agent_id"], "/agent_id")
    if agent_id != document.logical_id:
        raise _error(document, "/agent_id", "identity_mismatch")
    return value


def _compile_world(source: ScenarioPackageSource) -> SituatedWorldModel:
    document = _singleton(source, ScenarioDocumentRole.PHYSICAL_WORLD)
    value = _object(
        document,
        document.value,
        "",
        {"model_id", "version", "places", "passages", "objects"},
    )
    places = []
    for index, raw in enumerate(_array(document, value["places"], "/places")):
        pointer = f"/places/{index}"
        item = _object(document, raw, pointer, {"place_id", "label"})
        places.append(
            _construct(
                document,
                pointer,
                lambda item=item: PlaceSpec(
                    _text(document, item["place_id"], f"{pointer}/place_id"),
                    _text(document, item["label"], f"{pointer}/label"),
                ),
            )
        )
    place_ids = {item.place_id for item in places}

    passages = []
    for index, raw in enumerate(_array(document, value["passages"], "/passages")):
        pointer = f"/passages/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"passage_id", "source_place_id", "target_place_id", "initially_open"},
        )
        source_id = _text(document, item["source_place_id"], f"{pointer}/source_place_id")
        target_id = _text(document, item["target_place_id"], f"{pointer}/target_place_id")
        if source_id not in place_ids:
            raise _error(document, f"{pointer}/source_place_id", "unknown_reference")
        if target_id not in place_ids:
            raise _error(document, f"{pointer}/target_place_id", "unknown_reference")
        passages.append(
            _construct(
                document,
                pointer,
                lambda item=item, source_id=source_id, target_id=target_id: PassageSpec(
                    _text(document, item["passage_id"], f"{pointer}/passage_id"),
                    source_id,
                    target_id,
                    _boolean(document, item["initially_open"], f"{pointer}/initially_open"),
                ),
            )
        )

    objects = []
    for index, raw in enumerate(_array(document, value["objects"], "/objects")):
        pointer = f"/objects/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"object_id", "kind", "initial_place_id", "portable", "evidence"},
        )
        initial_place = _text(document, item["initial_place_id"], f"{pointer}/initial_place_id")
        if initial_place not in place_ids:
            raise _error(document, f"{pointer}/initial_place_id", "unknown_reference")
        evidence = []
        for fact_index, fact_raw in enumerate(_array(document, item["evidence"], f"{pointer}/evidence")):
            fact_pointer = f"{pointer}/evidence/{fact_index}"
            fact = _object(document, fact_raw, fact_pointer, {"name", "value"})
            evidence.append(
                _construct(
                    document,
                    fact_pointer,
                    lambda fact=fact: EvidenceFact(
                        _text(document, fact["name"], f"{fact_pointer}/name"),
                        _text(document, fact["value"], f"{fact_pointer}/value"),
                    ),
                )
            )
        objects.append(
            _construct(
                document,
                pointer,
                lambda item=item, initial_place=initial_place, evidence=evidence: WorldObjectSpec(
                    _text(document, item["object_id"], f"{pointer}/object_id"),
                    _text(document, item["kind"], f"{pointer}/kind"),
                    initial_place,
                    _boolean(document, item["portable"], f"{pointer}/portable"),
                    tuple(evidence),
                ),
            )
        )

    agents = []
    for agent_document in _agents(source):
        agent_value = _agent_value(agent_document)
        body = _object(
            agent_document,
            agent_value["body"],
            "/body",
            {"role", "initial_place", "inventory_capacity"},
        )
        initial_place = _text(agent_document, body["initial_place"], "/body/initial_place")
        if initial_place not in place_ids:
            raise _error(agent_document, "/body/initial_place", "unknown_reference")
        agents.append(
            _construct(
                agent_document,
                "/body",
                lambda agent_document=agent_document, body=body, initial_place=initial_place: EmbodiedAgentSpec(
                    agent_document.logical_id,
                    _text(agent_document, body["role"], "/body/role"),
                    initial_place,
                    _integer(agent_document, body["inventory_capacity"], "/body/inventory_capacity"),
                ),
            )
        )

    return _construct(
        document,
        "",
        lambda: SituatedWorldModel(
            _text(document, value["model_id"], "/model_id"),
            _text(document, value["version"], "/version"),
            tuple(places),
            tuple(passages),
            tuple(agents),
            tuple(objects),
        ),
    )


def _compile_perception(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
) -> SituatedPerceptionModel:
    document = _singleton(source, ScenarioDocumentRole.PHYSICAL_PERCEPTION)
    value = _object(
        document,
        document.value,
        "",
        {"model_id", "version", "edges", "signal_profiles"},
    )
    place_ids = {item.place_id for item in world.places}
    passage_ids = {item.passage_id for item in world.passages}
    edges = []
    for index, raw in enumerate(_array(document, value["edges"], "/edges")):
        pointer = f"/edges/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {
                "edge_id",
                "layer",
                "source_place_id",
                "target_place_id",
                "cost",
                "activation",
                "passage_id",
            },
        )
        source_id = _text(document, item["source_place_id"], f"{pointer}/source_place_id")
        target_id = _text(document, item["target_place_id"], f"{pointer}/target_place_id")
        passage_id = _optional_text(document, item["passage_id"], f"{pointer}/passage_id")
        if source_id not in place_ids:
            raise _error(document, f"{pointer}/source_place_id", "unknown_reference")
        if target_id not in place_ids:
            raise _error(document, f"{pointer}/target_place_id", "unknown_reference")
        if passage_id is not None and passage_id not in passage_ids:
            raise _error(document, f"{pointer}/passage_id", "unknown_reference")
        edges.append(
            _construct(
                document,
                pointer,
                lambda item=item, source_id=source_id, target_id=target_id, passage_id=passage_id: SituatedPerceptionEdge(
                    _text(document, item["edge_id"], f"{pointer}/edge_id"),
                    _enum(document, item["layer"], f"{pointer}/layer", SituatedPerceptionLayer),
                    source_id,
                    target_id,
                    _number(document, item["cost"], f"{pointer}/cost"),
                    _enum(document, item["activation"], f"{pointer}/activation", SituatedEdgeActivation),
                    passage_id,
                ),
            )
        )

    profiles = []
    for agent_document in _agents(source):
        agent_value = _agent_value(agent_document)
        profile = _object(
            agent_document,
            agent_value["perception"],
            "/perception",
            {"max_visual_cost", "minimum_detectable_sound", "minimum_clear_sound"},
        )
        profiles.append(
            _construct(
                agent_document,
                "/perception",
                lambda agent_document=agent_document, profile=profile: SituatedAgentPerceptionProfile(
                    agent_document.logical_id,
                    _number(agent_document, profile["max_visual_cost"], "/perception/max_visual_cost"),
                    _number(agent_document, profile["minimum_detectable_sound"], "/perception/minimum_detectable_sound"),
                    _number(agent_document, profile["minimum_clear_sound"], "/perception/minimum_clear_sound"),
                ),
            )
        )

    signals = []
    for index, raw in enumerate(_array(document, value["signal_profiles"], "/signal_profiles")):
        pointer = f"/signal_profiles/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"kind", "visually_observable", "auditory_intensity"},
        )
        intensity = (
            None
            if item["auditory_intensity"] is None
            else _number(document, item["auditory_intensity"], f"{pointer}/auditory_intensity")
        )
        signals.append(
            _construct(
                document,
                pointer,
                lambda item=item, intensity=intensity: SituatedEventSignalProfile(
                    _enum(document, item["kind"], f"{pointer}/kind", SituatedActionKind),
                    _boolean(document, item["visually_observable"], f"{pointer}/visually_observable"),
                    intensity,
                ),
            )
        )
    return _construct(
        document,
        "",
        lambda: SituatedPerceptionModel(
            _text(document, value["model_id"], "/model_id"),
            _text(document, value["version"], "/version"),
            world,
            tuple(edges),
            tuple(profiles),
            tuple(signals),
        ),
    )


def _compile_cognitive_agent(
    document: ScenarioSourceDocument,
    value: Mapping[str, object],
    world: SituatedWorldModel,
) -> SituatedAgentCognitiveModel:
    cognition = _object(
        document,
        value["cognition"],
        "/cognition",
        {
            "hypotheses",
            "prior_belief",
            "observation_symbols",
            "observation_rules",
            "likelihoods",
            "actions",
            "action_schedule",
            "transitions",
            "goals",
            "rewards",
            "discount",
            "beta",
        },
    )
    hypotheses = []
    for index, raw in enumerate(_array(document, cognition["hypotheses"], "/cognition/hypotheses")):
        pointer = f"/cognition/hypotheses/{index}"
        item = _object(document, raw, pointer, {"hypothesis_id", "description"})
        hypotheses.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedHypothesis(
                    _text(document, item["hypothesis_id"], f"{pointer}/hypothesis_id"),
                    _text(document, item["description"], f"{pointer}/description"),
                ),
            )
        )

    belief_raw = cognition["prior_belief"]
    if not isinstance(belief_raw, Mapping):
        raise _error(document, "/cognition/prior_belief", "invalid_type")
    belief: dict[str, float] = {}
    for hypothesis_id, probability in belief_raw.items():
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise _error(document, "/cognition/prior_belief", "invalid_type")
        belief[hypothesis_id] = _number(
            document,
            probability,
            f"/cognition/prior_belief/{_pointer_token(hypothesis_id)}",
        )
    prior_belief = _construct(
        document,
        "/cognition/prior_belief",
        lambda: PlanningBeliefState(belief),
    )

    symbols = []
    for index, raw in enumerate(
        _array(document, cognition["observation_symbols"], "/cognition/observation_symbols")
    ):
        pointer = f"/cognition/observation_symbols/{index}"
        item = _object(document, raw, pointer, {"symbol_id", "description"})
        symbols.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedObservationSymbol(
                    _text(document, item["symbol_id"], f"{pointer}/symbol_id"),
                    _text(document, item["description"], f"{pointer}/description"),
                ),
            )
        )

    rules = []
    for index, raw in enumerate(
        _array(document, cognition["observation_rules"], "/cognition/observation_rules")
    ):
        pointer = f"/cognition/observation_rules/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {
                "rule_id",
                "symbol_id",
                "likelihood_action_id",
                "event_kind",
                "outcome",
                "detail_name",
                "detail_value",
            },
        )
        rules.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedObservationRule(
                    _text(document, item["rule_id"], f"{pointer}/rule_id"),
                    _text(document, item["symbol_id"], f"{pointer}/symbol_id"),
                    _text(document, item["likelihood_action_id"], f"{pointer}/likelihood_action_id"),
                    _enum(document, item["event_kind"], f"{pointer}/event_kind", SituatedActionKind),
                    _optional_text(document, item["outcome"], f"{pointer}/outcome"),
                    _optional_text(document, item["detail_name"], f"{pointer}/detail_name"),
                    _optional_text(document, item["detail_value"], f"{pointer}/detail_value"),
                ),
            )
        )

    likelihoods = []
    for index, raw in enumerate(_array(document, cognition["likelihoods"], "/cognition/likelihoods")):
        pointer = f"/cognition/likelihoods/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"action_id", "hypothesis_id", "symbol_id", "probability"},
        )
        likelihoods.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedObservationLikelihood(
                    _text(document, item["action_id"], f"{pointer}/action_id"),
                    _text(document, item["hypothesis_id"], f"{pointer}/hypothesis_id"),
                    _text(document, item["symbol_id"], f"{pointer}/symbol_id"),
                    _number(document, item["probability"], f"{pointer}/probability"),
                ),
            )
        )

    place_ids = {item.place_id for item in world.places}
    passage_ids = {item.passage_id for item in world.passages}
    object_ids = {item.object_id for item in world.objects}
    actions = []
    for index, raw in enumerate(_array(document, cognition["actions"], "/cognition/actions")):
        pointer = f"/cognition/actions/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {
                "action_id",
                "kind",
                "target_id",
                "message",
                "required_place_ids",
                "repeatable",
                "source_event_kinds",
            },
        )
        kind = _enum(document, item["kind"], f"{pointer}/kind", SituatedActionKind)
        target_id = _optional_text(document, item["target_id"], f"{pointer}/target_id")
        if kind is SituatedActionKind.MOVE and target_id not in passage_ids:
            raise _error(document, f"{pointer}/target_id", "unknown_reference")
        if kind in {SituatedActionKind.INSPECT, SituatedActionKind.TAKE, SituatedActionKind.DROP} and target_id not in object_ids:
            raise _error(document, f"{pointer}/target_id", "unknown_reference")
        required_places = _texts(document, item["required_place_ids"], f"{pointer}/required_place_ids")
        for place_index, place_id in enumerate(required_places):
            if place_id not in place_ids:
                raise _error(document, f"{pointer}/required_place_ids/{place_index}", "unknown_reference")
        actions.append(
            _construct(
                document,
                pointer,
                lambda item=item, kind=kind, target_id=target_id, required_places=required_places: SituatedActionSpec(
                    _text(document, item["action_id"], f"{pointer}/action_id"),
                    kind,
                    target_id,
                    _optional_text(document, item["message"], f"{pointer}/message"),
                    required_places,
                    _boolean(document, item["repeatable"], f"{pointer}/repeatable"),
                    _enums(
                        document,
                        item["source_event_kinds"],
                        f"{pointer}/source_event_kinds",
                        SituatedActionKind,
                    ),
                ),
            )
        )

    schedule = tuple(
        _texts(document, row, f"/cognition/action_schedule/{index}")
        for index, row in enumerate(
            _array(document, cognition["action_schedule"], "/cognition/action_schedule")
        )
    )

    transitions = []
    for index, raw in enumerate(_array(document, cognition["transitions"], "/cognition/transitions")):
        pointer = f"/cognition/transitions/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"action_id", "prior_hypothesis_id", "next_hypothesis_id", "probability"},
        )
        transitions.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedHypothesisTransition(
                    _text(document, item["action_id"], f"{pointer}/action_id"),
                    _text(document, item["prior_hypothesis_id"], f"{pointer}/prior_hypothesis_id"),
                    _text(document, item["next_hypothesis_id"], f"{pointer}/next_hypothesis_id"),
                    _number(document, item["probability"], f"{pointer}/probability"),
                ),
            )
        )

    goals = []
    for index, raw in enumerate(_array(document, cognition["goals"], "/cognition/goals")):
        pointer = f"/cognition/goals/{index}"
        item = _object(document, raw, pointer, {"goal_id", "description", "weight"})
        goals.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedGoalSpec(
                    _text(document, item["goal_id"], f"{pointer}/goal_id"),
                    _text(document, item["description"], f"{pointer}/description"),
                    _number(document, item["weight"], f"{pointer}/weight"),
                ),
            )
        )

    rewards = []
    for index, raw in enumerate(_array(document, cognition["rewards"], "/cognition/rewards")):
        pointer = f"/cognition/rewards/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"goal_id", "hypothesis_id", "action_id", "value"},
        )
        rewards.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedGoalReward(
                    _text(document, item["goal_id"], f"{pointer}/goal_id"),
                    _text(document, item["hypothesis_id"], f"{pointer}/hypothesis_id"),
                    _text(document, item["action_id"], f"{pointer}/action_id"),
                    _number(document, item["value"], f"{pointer}/value"),
                ),
            )
        )

    return _construct(
        document,
        "/cognition",
        lambda: SituatedAgentCognitiveModel(
            document.logical_id,
            tuple(hypotheses),
            prior_belief,
            tuple(symbols),
            tuple(rules),
            tuple(likelihoods),
            tuple(actions),
            schedule,
            tuple(transitions),
            tuple(goals),
            tuple(rewards),
            _number(document, cognition["discount"], "/cognition/discount"),
            _number(document, cognition["beta"], "/cognition/beta"),
        ),
    )


def _compile_cognition(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
) -> SituatedCognitiveModel:
    agents = tuple(
        _compile_cognitive_agent(document, _agent_value(document), world)
        for document in _agents(source)
    )
    document = _singleton(source, ScenarioDocumentRole.PHYSICAL_WORLD)
    return _construct(
        document,
        "",
        lambda: SituatedCognitiveModel(
            f"{source.scenario_id}-cognition",
            source.version,
            world,
            agents,
        ),
    )


def _memory_value(
    document: ScenarioSourceDocument,
    agent_value: Mapping[str, object],
) -> Mapping[str, object]:
    return _object(
        document,
        agent_value["memory"],
        "/memory",
        {"percept_policy", "channel_policy", "recall"},
    )


def _percept_memory_policy(
    document: ScenarioSourceDocument,
    memory: Mapping[str, object],
) -> SituatedPerceptMemoryPolicy:
    policy = _object(
        document,
        memory["percept_policy"],
        "/memory/percept_policy",
        {"policy_id", "version", "fidelities"},
    )
    fidelities = []
    for index, raw in enumerate(
        _array(document, policy["fidelities"], "/memory/percept_policy/fidelities")
    ):
        pointer = f"/memory/percept_policy/fidelities/{index}"
        item = _object(document, raw, pointer, {"fidelity", "confidence", "salience"})
        fidelities.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedPerceptMemoryFidelityPolicy(
                    _enum(document, item["fidelity"], f"{pointer}/fidelity", SituatedPerceptFidelity),
                    _number(document, item["confidence"], f"{pointer}/confidence"),
                    _number(document, item["salience"], f"{pointer}/salience"),
                ),
            )
        )
    return _construct(
        document,
        "/memory/percept_policy",
        lambda: SituatedPerceptMemoryPolicy(
            _text(document, policy["policy_id"], "/memory/percept_policy/policy_id"),
            _text(document, policy["version"], "/memory/percept_policy/version"),
            tuple(fidelities),
        ),
    )


def _channel_memory_policy(
    document: ScenarioSourceDocument,
    memory: Mapping[str, object],
) -> SituatedMemoryPolicy:
    policy = _object(
        document,
        memory["channel_policy"],
        "/memory/channel_policy",
        {"policy_id", "version", "channels"},
    )
    channels = []
    for index, raw in enumerate(
        _array(document, policy["channels"], "/memory/channel_policy/channels")
    ):
        pointer = f"/memory/channel_policy/channels/{index}"
        item = _object(document, raw, pointer, {"channel", "confidence", "salience"})
        channels.append(
            _construct(
                document,
                pointer,
                lambda item=item: MemoryChannelPolicy(
                    _enum(document, item["channel"], f"{pointer}/channel", ObservationChannel),
                    _number(document, item["confidence"], f"{pointer}/confidence"),
                    _number(document, item["salience"], f"{pointer}/salience"),
                ),
            )
        )
    return _construct(
        document,
        "/memory/channel_policy",
        lambda: SituatedMemoryPolicy(
            _text(document, policy["policy_id"], "/memory/channel_policy/policy_id"),
            _text(document, policy["version"], "/memory/channel_policy/version"),
            tuple(channels),
        ),
    )


def _recall_policy(
    document: ScenarioSourceDocument,
    memory: Mapping[str, object],
    place_ids: set[str],
) -> SituatedAgentRecallPolicy:
    recall = _object(
        document,
        memory["recall"],
        "/memory/recall",
        {"max_memories_per_round", "cues"},
    )
    cues = []
    for index, raw in enumerate(_array(document, recall["cues"], "/memory/recall/cues")):
        pointer = f"/memory/recall/cues/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {
                "cue_id",
                "text",
                "required_place_ids",
                "event_kinds",
                "channels",
                "min_confidence",
                "limit",
            },
        )
        required_places = _texts(document, item["required_place_ids"], f"{pointer}/required_place_ids")
        for place_index, place_id in enumerate(required_places):
            if place_id not in place_ids:
                raise _error(document, f"{pointer}/required_place_ids/{place_index}", "unknown_reference")
        cues.append(
            _construct(
                document,
                pointer,
                lambda item=item, required_places=required_places: SituatedMemoryRecallCue(
                    _text(document, item["cue_id"], f"{pointer}/cue_id"),
                    _text(document, item["text"], f"{pointer}/text"),
                    required_places,
                    _enums(document, item["event_kinds"], f"{pointer}/event_kinds", SituatedActionKind),
                    _enums(document, item["channels"], f"{pointer}/channels", ObservationChannel),
                    _number(document, item["min_confidence"], f"{pointer}/min_confidence"),
                    _integer(document, item["limit"], f"{pointer}/limit"),
                ),
            )
        )
    return _construct(
        document,
        "/memory/recall",
        lambda: SituatedAgentRecallPolicy(
            document.logical_id,
            tuple(cues),
            _integer(
                document,
                recall["max_memories_per_round"],
                "/memory/recall/max_memories_per_round",
            ),
        ),
    )


def _run_value(source: ScenarioPackageSource) -> tuple[ScenarioSourceDocument, Mapping[str, object]]:
    document = _singleton(source, ScenarioDocumentRole.RUN)
    value = _object(
        document,
        document.value,
        "",
        {
            "mode",
            "maximum_rounds",
            "checkpoint_interval",
            "maximum_output_records",
            "allowed_output_kinds",
            "public_journal",
            "blender_mode",
            "deterministic_seed",
            "maximum_resource_bytes",
            "runtime_model",
            "fallbacks",
        },
    )
    fallbacks = value["fallbacks"]
    if not isinstance(fallbacks, Mapping):
        raise _error(document, "/fallbacks", "invalid_type")
    supported = {
        ScenarioDocumentRole.PHYSICAL_MAP.value: "auto_grid",
        ScenarioDocumentRole.SOCIAL_INSTITUTIONS.value: "empty",
        ScenarioDocumentRole.SOCIAL_NORMS.value: "empty",
        ScenarioDocumentRole.STORY_INTERVENTIONS.value: "none",
    }
    unknown = sorted(set(fallbacks) - set(supported))
    if unknown:
        raise _error(
            document,
            f"/fallbacks/{_pointer_token(unknown[0])}",
            "unsupported_shape",
        )
    for role, fallback in fallbacks.items():
        if fallback != supported[role]:
            raise _error(
                document,
                f"/fallbacks/{_pointer_token(role)}",
                "unsupported_fallback",
            )
    return document, value


def _runtime_config(source: ScenarioPackageSource) -> tuple[ScenarioSourceDocument, Mapping[str, object]]:
    document, run = _run_value(source)
    return document, _object(
        document,
        run["runtime_model"],
        "/runtime_model",
        {
            "model_id",
            "version",
            "percept_memory_model_id",
            "tracked_hypothesis_id",
            "adoption_threshold",
            "relationship_trust_threshold",
        },
    )


def _compile_situated_recall_policies(
    source: ScenarioPackageSource,
    cognition: SituatedCognitiveModel,
) -> tuple[SituatedAgentRecallPolicy, ...]:
    place_ids = {item.place_id for item in cognition.world_model.places}
    return tuple(
        _recall_policy(
            document,
            _memory_value(document, _agent_value(document)),
            place_ids,
        )
        for document in _agents(source)
    )


def _compile_percept_memory(
    source: ScenarioPackageSource,
    perception: SituatedPerceptionModel,
    cognition: SituatedCognitiveModel,
) -> SituatedPerceptMemoryCognitiveModel:
    policies = []
    for document in _agents(source):
        policies.append(
            (
                document,
                _percept_memory_policy(
                    document,
                    _memory_value(document, _agent_value(document)),
                ),
            )
        )
    policy = policies[0][1]
    for document, candidate in policies[1:]:
        if candidate != policy:
            raise _error(document, "/memory/percept_policy", "inconsistent_definition")
    run_document, runtime = _runtime_config(source)
    return _construct(
        run_document,
        "/runtime_model/percept_memory_model_id",
        lambda: SituatedPerceptMemoryCognitiveModel(
            _text(
                run_document,
                runtime["percept_memory_model_id"],
                "/runtime_model/percept_memory_model_id",
            ),
            source.version,
            perception,
            cognition,
            policy,
            _compile_situated_recall_policies(source, cognition),
        ),
    )


def _agent_topics(
    document: ScenarioSourceDocument,
    agent_value: Mapping[str, object],
) -> tuple[SituatedClaimTopic, ...]:
    social = _object(document, agent_value["social"], "/social", {"topics"})
    topics = []
    for index, raw in enumerate(_array(document, social["topics"], "/social/topics")):
        pointer = f"/social/topics/{index}"
        item = _object(document, raw, pointer, {"topic_id", "symbol_ids"})
        topics.append(
            _construct(
                document,
                pointer,
                lambda item=item: SituatedClaimTopic(
                    _text(document, item["topic_id"], f"{pointer}/topic_id"),
                    _texts(document, item["symbol_ids"], f"{pointer}/symbol_ids"),
                ),
            )
        )
    return tuple(topics)


def _social_relationship_value(
    source: ScenarioPackageSource,
) -> tuple[ScenarioSourceDocument, Mapping[str, object]]:
    document = _singleton(source, ScenarioDocumentRole.SOCIAL_RELATIONSHIPS)
    return document, _object(
        document,
        document.value,
        "",
        {
            "model_id",
            "version",
            "memory_cognitive_model_id",
            "relationships",
            "runtime_seeds",
            "policy",
        },
    )


def _compile_relationship_seeds(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
    social_memory: SituatedSocialMemoryModel,
) -> tuple[ScenarioRelationshipSeed, ...]:
    if not isinstance(social_memory, SituatedSocialMemoryModel):
        raise ScenarioCompilationError("social.relationships", "/runtime_seeds", "invalid_type")
    document, value = _social_relationship_value(source)
    agent_ids = {item.agent_id for item in world.agents}
    expected_pairs = {
        (observer, other)
        for observer in agent_ids
        for other in agent_ids
        if observer != other
    }
    seeds: list[ScenarioRelationshipSeed] = []
    pairs: list[tuple[str, str]] = []
    for index, raw in enumerate(
        _array(document, value["runtime_seeds"], "/runtime_seeds")
    ):
        pointer = f"/runtime_seeds/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"observer_agent_id", "source_agent_id", "trust", "affinity"},
        )
        observer = _text(
            document,
            item["observer_agent_id"],
            f"{pointer}/observer_agent_id",
        )
        other = _text(
            document,
            item["source_agent_id"],
            f"{pointer}/source_agent_id",
        )
        if observer not in agent_ids:
            raise _error(document, f"{pointer}/observer_agent_id", "unknown_reference")
        if other not in agent_ids:
            raise _error(document, f"{pointer}/source_agent_id", "unknown_reference")
        pairs.append((observer, other))
        seeds.append(
            _construct(
                document,
                pointer,
                lambda observer=observer, other=other, item=item: ScenarioRelationshipSeed(
                    observer,
                    other,
                    _number(document, item["trust"], f"{pointer}/trust"),
                    _number(document, item["affinity"], f"{pointer}/affinity"),
                ),
            )
        )
    if set(pairs) != expected_pairs or len(pairs) != len(expected_pairs):
        raise _error(document, "/runtime_seeds", "roster_mismatch")
    return tuple(
        sorted(
            seeds,
            key=lambda item: (item.observer_agent_id, item.source_agent_id),
        )
    )


def _compile_social_memory(
    source: ScenarioPackageSource,
    cognition: SituatedCognitiveModel,
) -> SituatedSocialMemoryModel:
    channel_policies = []
    topic_sets = []
    cognitive_by_agent = {item.agent_id: item for item in cognition.agents}
    for document in _agents(source):
        agent_value = _agent_value(document)
        memory = _memory_value(document, agent_value)
        channel_policies.append((document, _channel_memory_policy(document, memory)))
        topics = _agent_topics(document, agent_value)
        available_symbols = {
            item.symbol_id
            for item in cognitive_by_agent[document.logical_id].observation_symbols
        }
        social = _object(document, agent_value["social"], "/social", {"topics"})
        for topic_index, raw in enumerate(_array(document, social["topics"], "/social/topics")):
            topic = _object(
                document,
                raw,
                f"/social/topics/{topic_index}",
                {"topic_id", "symbol_ids"},
            )
            for symbol_index, symbol_id in enumerate(
                _texts(
                    document,
                    topic["symbol_ids"],
                    f"/social/topics/{topic_index}/symbol_ids",
                )
            ):
                if symbol_id not in available_symbols:
                    raise _error(
                        document,
                        f"/social/topics/{topic_index}/symbol_ids/{symbol_index}",
                        "unknown_reference",
                    )
        topic_sets.append((document, topics))
    channel_policy = channel_policies[0][1]
    for document, candidate in channel_policies[1:]:
        if candidate != channel_policy:
            raise _error(document, "/memory/channel_policy", "inconsistent_definition")
    topics = topic_sets[0][1]
    for document, candidate in topic_sets[1:]:
        if candidate != topics:
            raise _error(document, "/social/topics", "roster_mismatch")

    document, value = _social_relationship_value(source)
    memory_cognitive = _construct(
        document,
        "/memory_cognitive_model_id",
        lambda: SituatedMemoryCognitiveModel(
            _text(
                document,
                value["memory_cognitive_model_id"],
                "/memory_cognitive_model_id",
            ),
            source.version,
            cognition,
            channel_policy,
            _compile_situated_recall_policies(source, cognition),
        ),
    )
    policy_raw = _object(
        document,
        value["policy"],
        "/policy",
        {
            "initial_source_trust",
            "confirmation_rate",
            "contradiction_rate",
            "confirmation_affinity_delta",
            "contradiction_affinity_delta",
            "max_unresolved_age_rounds",
            "max_active_claims",
        },
    )
    policy = _construct(
        document,
        "/policy",
        lambda: SituatedSocialMemoryPolicy(
            _number(document, policy_raw["initial_source_trust"], "/policy/initial_source_trust"),
            _number(document, policy_raw["confirmation_rate"], "/policy/confirmation_rate"),
            _number(document, policy_raw["contradiction_rate"], "/policy/contradiction_rate"),
            _number(
                document,
                policy_raw["confirmation_affinity_delta"],
                "/policy/confirmation_affinity_delta",
            ),
            _number(
                document,
                policy_raw["contradiction_affinity_delta"],
                "/policy/contradiction_affinity_delta",
            ),
            _integer(
                document,
                policy_raw["max_unresolved_age_rounds"],
                "/policy/max_unresolved_age_rounds",
            ),
            _integer(document, policy_raw["max_active_claims"], "/policy/max_active_claims"),
        ),
    )
    return _construct(
        document,
        "",
        lambda: SituatedSocialMemoryModel(
            _text(document, value["model_id"], "/model_id"),
            _text(document, value["version"], "/version"),
            memory_cognitive,
            topics,
            policy,
        ),
    )


def _compile_runtime_model(
    source: ScenarioPackageSource,
    percept_memory: SituatedPerceptMemoryCognitiveModel,
    social_memory: SituatedSocialMemoryModel,
) -> SituatedNetworkRuntimeModel:
    document, value = _runtime_config(source)
    tracked = _text(
        document,
        value["tracked_hypothesis_id"],
        "/runtime_model/tracked_hypothesis_id",
    )
    if any(
        tracked not in {item.hypothesis_id for item in agent.hypotheses}
        for agent in percept_memory.cognitive_model.agents
    ):
        raise _error(
            document,
            "/runtime_model/tracked_hypothesis_id",
            "unknown_reference",
        )
    return _construct(
        document,
        "/runtime_model",
        lambda: SituatedNetworkRuntimeModel(
            _text(document, value["model_id"], "/runtime_model/model_id"),
            _text(document, value["version"], "/runtime_model/version"),
            percept_memory,
            social_memory,
            tracked,
            _number(document, value["adoption_threshold"], "/runtime_model/adoption_threshold"),
            _number(
                document,
                value["relationship_trust_threshold"],
                "/runtime_model/relationship_trust_threshold",
            ),
        ),
    )


def _fallback(source: ScenarioPackageSource, role: ScenarioDocumentRole) -> str:
    document, run = _run_value(source)
    fallbacks = run["fallbacks"]
    assert isinstance(fallbacks, Mapping)
    value = fallbacks.get(role.value)
    if not isinstance(value, str) or not value.strip():
        raise _error(document, f"/fallbacks/{role.value}", "invalid_value")
    return value


def _auto_spatial_map(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
) -> SituatedSpatialMap:
    run_document = _singleton(source, ScenarioDocumentRole.RUN)
    if _fallback(source, ScenarioDocumentRole.PHYSICAL_MAP) != "auto_grid":
        raise _error(run_document, "/fallbacks/physical.map", "unsupported_fallback")
    return _construct(
        run_document,
        "/fallbacks/physical.map",
        lambda: auto_layout_situated_spatial_map(world),
    )


def _compile_spatial_map(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
) -> SituatedSpatialMap:
    document = _optional_singleton(source, ScenarioDocumentRole.PHYSICAL_MAP)
    if document is None:
        return _auto_spatial_map(source, world)
    return _construct(
        document,
        "",
        lambda: compile_tiled_situated_spatial_map(document.value, world),
    )


def _grant(
    document: ScenarioSourceDocument,
    raw: object,
    pointer: str,
) -> ScenarioResourceGrant:
    item = _object(
        document,
        raw,
        pointer,
        {"subject_scope", "subject_id", "resource_ids"},
    )
    return _construct(
        document,
        pointer,
        lambda: ScenarioResourceGrant(
            _text(document, item["subject_scope"], f"{pointer}/subject_scope"),
            _optional_text(document, item["subject_id"], f"{pointer}/subject_id"),
            _texts(document, item["resource_ids"], f"{pointer}/resource_ids"),
        ),
    )


def _entitlement(
    document: ScenarioSourceDocument,
    raw: object,
    pointer: str,
) -> ScenarioResourceEntitlement:
    item = _object(
        document,
        raw,
        pointer,
        {"subject_scope", "subject_id"},
    )
    return _construct(
        document,
        pointer,
        lambda: ScenarioResourceEntitlement(
            _text(document, item["subject_scope"], f"{pointer}/subject_scope"),
            _optional_text(document, item["subject_id"], f"{pointer}/subject_id"),
        ),
    )


def _entitlements(
    document: ScenarioSourceDocument,
    value: object,
    pointer: str,
) -> tuple[ScenarioResourceEntitlement, ...]:
    return tuple(
        _entitlement(document, item, f"{pointer}/{index}")
        for index, item in enumerate(_array(document, value, pointer))
    )


def _resource_grant_is_entitled(
    resource: ScenarioKnowledgeResource | ScenarioAssetResource,
    scope: str,
    subject_id: str | None,
) -> bool:
    subjects = {
        (item.subject_scope, item.subject_id)
        for item in resource.entitlements
    }
    return (scope, subject_id) in subjects or ("public", None) in subjects


def _validate_grant_subject_reference(
    source: ScenarioPackageSource,
    document: ScenarioSourceDocument,
    pointer: str,
    scope: str,
    subject_id: str | None,
) -> None:
    agent_ids = {item.logical_id for item in _agents(source)}
    institution_ids: set[str] = set()
    role_ids: set[str] = set()
    institution_document = _optional_singleton(
        source,
        ScenarioDocumentRole.SOCIAL_INSTITUTIONS,
    )
    if institution_document is not None:
        value = _object(
            institution_document,
            institution_document.value,
            "",
            {"institutions", "memberships"},
        )
        for index, raw in enumerate(
            _array(institution_document, value["institutions"], "/institutions")
        ):
            item = _object(
                institution_document,
                raw,
                f"/institutions/{index}",
                {"institution_id", "institution_kind", "parent_institution_id"},
            )
            institution_ids.add(
                _text(
                    institution_document,
                    item["institution_id"],
                    f"/institutions/{index}/institution_id",
                )
            )
        for index, raw in enumerate(
            _array(institution_document, value["memberships"], "/memberships")
        ):
            item = _object(
                institution_document,
                raw,
                f"/memberships/{index}",
                {"agent_id", "institution_id", "role_id"},
            )
            role_ids.add(
                _text(
                    institution_document,
                    item["role_id"],
                    f"/memberships/{index}/role_id",
                )
            )
    known = (
        scope == "public"
        or (scope == "agent" and subject_id in agent_ids)
        or (scope == "role" and subject_id in role_ids)
        or (scope == "institution" and subject_id in institution_ids)
    )
    if scope in {"agent", "role", "institution"} and not known:
        raise _error(document, f"{pointer}/subject_id", "unknown_reference")


def _compile_knowledge_catalog(source: ScenarioPackageSource) -> ScenarioKnowledgeCatalog:
    catalog_document = _singleton(source, ScenarioDocumentRole.KNOWLEDGE_CATALOG)
    catalog = _object(catalog_document, catalog_document.value, "", {"resources"})
    resources = []
    for index, raw in enumerate(_array(catalog_document, catalog["resources"], "/resources")):
        pointer = f"/resources/{index}"
        item = _object(
            catalog_document,
            raw,
            pointer,
            {
                "resource_id",
                "kind",
                "content_hash",
                "uri",
                "media_type",
                "language",
                "version",
                "authority",
                "license_tag",
                "entitlements",
                "concept_ids",
                "index_id",
            },
        )
        resources.append(
            _construct(
                catalog_document,
                pointer,
                lambda item=item: ScenarioKnowledgeResource(
                    _text(catalog_document, item["resource_id"], f"{pointer}/resource_id"),
                    _enum(catalog_document, item["kind"], f"{pointer}/kind", ScenarioResourceKind),
                    _text(catalog_document, item["content_hash"], f"{pointer}/content_hash"),
                    _text(catalog_document, item["uri"], f"{pointer}/uri"),
                    _text(catalog_document, item["media_type"], f"{pointer}/media_type"),
                    _text(catalog_document, item["language"], f"{pointer}/language"),
                    _text(catalog_document, item["version"], f"{pointer}/version"),
                    _text(catalog_document, item["authority"], f"{pointer}/authority"),
                    _text(catalog_document, item["license_tag"], f"{pointer}/license_tag"),
                    _entitlements(
                        catalog_document,
                        item["entitlements"],
                        f"{pointer}/entitlements",
                    ),
                    _texts(catalog_document, item["concept_ids"], f"{pointer}/concept_ids"),
                    _optional_text(catalog_document, item["index_id"], f"{pointer}/index_id"),
                ),
            )
        )
    access_document = _singleton(source, ScenarioDocumentRole.KNOWLEDGE_ACCESS)
    access = _object(access_document, access_document.value, "", {"grants"})
    known_resource_ids = {item.resource_id for item in resources}
    resources_by_id = {item.resource_id: item for item in resources}
    grants = []
    for index, raw in enumerate(_array(access_document, access["grants"], "/grants")):
        pointer = f"/grants/{index}"
        item = _object(
            access_document,
            raw,
            pointer,
            {"subject_scope", "subject_id", "resource_ids"},
        )
        scope = _text(access_document, item["subject_scope"], f"{pointer}/subject_scope")
        subject_id = _optional_text(
            access_document,
            item["subject_id"],
            f"{pointer}/subject_id",
        )
        _validate_grant_subject_reference(
            source,
            access_document,
            pointer,
            scope,
            subject_id,
        )
        for resource_index, resource_id in enumerate(
            _texts(access_document, item["resource_ids"], f"{pointer}/resource_ids")
        ):
            if resource_id not in known_resource_ids:
                raise _error(
                    access_document,
                    f"{pointer}/resource_ids/{resource_index}",
                    "unknown_reference",
                )
            if not _resource_grant_is_entitled(
                resources_by_id[resource_id],
                scope,
                subject_id,
            ):
                raise _error(
                    access_document,
                    f"{pointer}/resource_ids/{resource_index}",
                    "grant_not_entitled",
                )
        grants.append(_grant(access_document, raw, pointer))
    return _construct(
        access_document,
        "",
        lambda: ScenarioKnowledgeCatalog(tuple(resources), tuple(grants)),
    )


def _compile_asset_catalog(source: ScenarioPackageSource) -> ScenarioAssetCatalog:
    document = _singleton(source, ScenarioDocumentRole.ASSET_CATALOG)
    value = _object(document, document.value, "", {"resources", "grants"})
    resources = []
    for index, raw in enumerate(_array(document, value["resources"], "/resources")):
        pointer = f"/resources/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {
                "resource_id",
                "kind",
                "content_hash",
                "uri",
                "media_type",
                "authority",
                "license_tag",
                "entitlements",
                "dimensions",
                "unit",
                "format",
                "collection_id",
                "rig_metadata_ids",
                "collision_metadata_ids",
                "preview_hash",
            },
        )
        dimensions = tuple(
            _number(document, dimension, f"{pointer}/dimensions/{dimension_index}")
            for dimension_index, dimension in enumerate(
                _array(document, item["dimensions"], f"{pointer}/dimensions")
            )
        )
        preview_hash = _optional_text(document, item["preview_hash"], f"{pointer}/preview_hash")
        resources.append(
            _construct(
                document,
                pointer,
                lambda item=item, dimensions=dimensions, preview_hash=preview_hash: ScenarioAssetResource(
                    _text(document, item["resource_id"], f"{pointer}/resource_id"),
                    _enum(document, item["kind"], f"{pointer}/kind", ScenarioResourceKind),
                    _text(document, item["content_hash"], f"{pointer}/content_hash"),
                    _text(document, item["uri"], f"{pointer}/uri"),
                    _text(document, item["media_type"], f"{pointer}/media_type"),
                    _text(document, item["authority"], f"{pointer}/authority"),
                    _text(document, item["license_tag"], f"{pointer}/license_tag"),
                    _entitlements(
                        document,
                        item["entitlements"],
                        f"{pointer}/entitlements",
                    ),
                    dimensions,
                    _optional_text(document, item["unit"], f"{pointer}/unit"),
                    _optional_text(document, item["format"], f"{pointer}/format"),
                    _optional_text(document, item["collection_id"], f"{pointer}/collection_id"),
                    _texts(document, item["rig_metadata_ids"], f"{pointer}/rig_metadata_ids"),
                    _texts(
                        document,
                        item["collision_metadata_ids"],
                        f"{pointer}/collision_metadata_ids",
                    ),
                    preview_hash,
                ),
            )
        )
    known_resource_ids = {item.resource_id for item in resources}
    resources_by_id = {item.resource_id: item for item in resources}
    grants = []
    for index, raw in enumerate(_array(document, value["grants"], "/grants")):
        pointer = f"/grants/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"subject_scope", "subject_id", "resource_ids"},
        )
        scope = _text(document, item["subject_scope"], f"{pointer}/subject_scope")
        subject_id = _optional_text(
            document,
            item["subject_id"],
            f"{pointer}/subject_id",
        )
        _validate_grant_subject_reference(
            source,
            document,
            pointer,
            scope,
            subject_id,
        )
        for resource_index, resource_id in enumerate(
            _texts(document, item["resource_ids"], f"{pointer}/resource_ids")
        ):
            if resource_id not in known_resource_ids:
                raise _error(
                    document,
                    f"{pointer}/resource_ids/{resource_index}",
                    "unknown_reference",
                )
            if not _resource_grant_is_entitled(
                resources_by_id[resource_id],
                scope,
                subject_id,
            ):
                raise _error(
                    document,
                    f"{pointer}/resource_ids/{resource_index}",
                    "grant_not_entitled",
                )
        grants.append(_grant(document, raw, pointer))
    return _construct(
        document,
        "",
        lambda: ScenarioAssetCatalog(tuple(resources), tuple(grants)),
    )


def _compile_social_world(
    source: ScenarioPackageSource,
    runtime_model: SituatedNetworkRuntimeModel,
    knowledge_catalog: ScenarioKnowledgeCatalog,
    asset_catalog: ScenarioAssetCatalog,
) -> ScenarioSocialWorld:
    world = runtime_model.percept_memory_model.cognitive_model.world_model
    agent_ids = {item.agent_id for item in world.agents}
    institution_document = _optional_singleton(source, ScenarioDocumentRole.SOCIAL_INSTITUTIONS)
    institutions: list[ScenarioInstitution] = []
    memberships: list[ScenarioMembership] = []
    if institution_document is None:
        if _fallback(source, ScenarioDocumentRole.SOCIAL_INSTITUTIONS) != "empty":
            run_document = _singleton(source, ScenarioDocumentRole.RUN)
            raise _error(run_document, "/fallbacks/social.institutions", "unsupported_fallback")
    else:
        value = _object(
            institution_document,
            institution_document.value,
            "",
            {"institutions", "memberships"},
        )
        for index, raw in enumerate(
            _array(institution_document, value["institutions"], "/institutions")
        ):
            pointer = f"/institutions/{index}"
            item = _object(
                institution_document,
                raw,
                pointer,
                {"institution_id", "institution_kind", "parent_institution_id"},
            )
            institutions.append(
                _construct(
                    institution_document,
                    pointer,
                    lambda item=item: ScenarioInstitution(
                        _text(institution_document, item["institution_id"], f"{pointer}/institution_id"),
                        _text(institution_document, item["institution_kind"], f"{pointer}/institution_kind"),
                        _optional_text(
                            institution_document,
                            item["parent_institution_id"],
                            f"{pointer}/parent_institution_id",
                        ),
                    ),
                )
            )
        institution_ids = {item.institution_id for item in institutions}
        for index, institution in enumerate(institutions):
            if (
                institution.parent_institution_id is not None
                and institution.parent_institution_id not in institution_ids
            ):
                raise _error(
                    institution_document,
                    f"/institutions/{index}/parent_institution_id",
                    "unknown_reference",
                )
        for index, raw in enumerate(
            _array(institution_document, value["memberships"], "/memberships")
        ):
            pointer = f"/memberships/{index}"
            item = _object(
                institution_document,
                raw,
                pointer,
                {"agent_id", "institution_id", "role_id"},
            )
            agent_id = _text(
                institution_document,
                item["agent_id"],
                f"{pointer}/agent_id",
            )
            institution_id = _text(
                institution_document,
                item["institution_id"],
                f"{pointer}/institution_id",
            )
            role_id = _text(
                institution_document,
                item["role_id"],
                f"{pointer}/role_id",
            )
            if agent_id not in agent_ids:
                raise _error(
                    institution_document,
                    f"{pointer}/agent_id",
                    "unknown_reference",
                )
            if institution_id not in institution_ids:
                raise _error(
                    institution_document,
                    f"{pointer}/institution_id",
                    "unknown_reference",
                )
            memberships.append(
                _construct(
                    institution_document,
                    pointer,
                    lambda agent_id=agent_id, institution_id=institution_id, role_id=role_id: ScenarioMembership(
                        agent_id,
                        institution_id,
                        role_id,
                    ),
                )
            )

    relationship_document, relationship_value = _social_relationship_value(source)
    relationships: list[ScenarioRelationship] = []
    for index, raw in enumerate(
        _array(relationship_document, relationship_value["relationships"], "/relationships")
    ):
        pointer = f"/relationships/{index}"
        item = _object(
            relationship_document,
            raw,
            pointer,
            {"source_agent_id", "target_agent_id", "relationship_type", "strength"},
        )
        source_agent_id = _text(
            relationship_document,
            item["source_agent_id"],
            f"{pointer}/source_agent_id",
        )
        target_agent_id = _text(
            relationship_document,
            item["target_agent_id"],
            f"{pointer}/target_agent_id",
        )
        if source_agent_id not in agent_ids:
            raise _error(
                relationship_document,
                f"{pointer}/source_agent_id",
                "unknown_reference",
            )
        if target_agent_id not in agent_ids:
            raise _error(
                relationship_document,
                f"{pointer}/target_agent_id",
                "unknown_reference",
            )
        relationships.append(
            _construct(
                relationship_document,
                pointer,
                lambda item=item, source_agent_id=source_agent_id, target_agent_id=target_agent_id: ScenarioRelationship(
                    source_agent_id,
                    target_agent_id,
                    _text(
                        relationship_document,
                        item["relationship_type"],
                        f"{pointer}/relationship_type",
                    ),
                    _number(relationship_document, item["strength"], f"{pointer}/strength"),
                ),
            )
        )

    action_ids = tuple(sorted({
        action.action_id
        for agent in runtime_model.percept_memory_model.cognitive_model.agents
        for action in agent.actions
    }))
    resource_ids = tuple(
        item.resource_id
        for item in knowledge_catalog.resources + asset_catalog.resources
    )
    relationship_types = tuple(sorted({item.relationship_type for item in relationships}))
    roles = {item.role_id for item in memberships}
    known_target_scopes = (
        {item.agent_id for item in world.agents}
        | {item.place_id for item in world.places}
        | {item.passage_id for item in world.passages}
        | {item.object_id for item in world.objects}
        | {item.institution_id for item in institutions}
        | set(resource_ids)
        | set(roles)
        | set(relationship_types)
    )

    norm_document = _optional_singleton(source, ScenarioDocumentRole.SOCIAL_NORMS)
    norms: list[ScenarioNorm] = []
    if norm_document is None:
        if _fallback(source, ScenarioDocumentRole.SOCIAL_NORMS) != "empty":
            run_document = _singleton(source, ScenarioDocumentRole.RUN)
            raise _error(run_document, "/fallbacks/social.norms", "unsupported_fallback")
    else:
        norm_value = _object(norm_document, norm_document.value, "", {"norms"})
        for index, raw in enumerate(_array(norm_document, norm_value["norms"], "/norms")):
            pointer = f"/norms/{index}"
            item = _object(
                norm_document,
                raw,
                pointer,
                {
                    "norm_id",
                    "subject_role_id",
                    "effect",
                    "priority",
                    "action_id",
                    "resource_id",
                    "relationship_type",
                    "target_scope_id",
                    "descriptive_text",
                },
            )
            subject_role = _text(
                norm_document,
                item["subject_role_id"],
                f"{pointer}/subject_role_id",
            )
            if subject_role not in roles:
                raise _error(norm_document, f"{pointer}/subject_role_id", "unknown_reference")
            effect = _enum(norm_document, item["effect"], f"{pointer}/effect", ScenarioNormEffect)
            action_id = _optional_text(norm_document, item["action_id"], f"{pointer}/action_id")
            resource_id = _optional_text(norm_document, item["resource_id"], f"{pointer}/resource_id")
            relationship_type = _optional_text(
                norm_document,
                item["relationship_type"],
                f"{pointer}/relationship_type",
            )
            active_reference = (
                "action_id"
                if effect in {ScenarioNormEffect.ALLOW_ACTION, ScenarioNormEffect.DENY_ACTION}
                else "resource_id"
                if effect is ScenarioNormEffect.REQUIRE_KNOWLEDGE_GRANT
                else "relationship_type"
                if effect is ScenarioNormEffect.REQUIRE_RELATIONSHIP
                else None
            )
            for field, reference in (
                ("action_id", action_id),
                ("resource_id", resource_id),
                ("relationship_type", relationship_type),
            ):
                if field != active_reference and reference is not None:
                    raise _error(
                        norm_document,
                        f"{pointer}/{field}",
                        "invalid_value",
                    )
            target_scope_id = _optional_text(
                norm_document,
                item["target_scope_id"],
                f"{pointer}/target_scope_id",
            )
            if effect in {ScenarioNormEffect.ALLOW_ACTION, ScenarioNormEffect.DENY_ACTION} and action_id not in action_ids:
                raise _error(norm_document, f"{pointer}/action_id", "unsupported_norm_effect")
            if effect is ScenarioNormEffect.REQUIRE_KNOWLEDGE_GRANT and resource_id not in resource_ids:
                raise _error(norm_document, f"{pointer}/resource_id", "unsupported_norm_effect")
            if effect is ScenarioNormEffect.REQUIRE_RELATIONSHIP and relationship_type not in relationship_types:
                raise _error(
                    norm_document,
                    f"{pointer}/relationship_type",
                    "unsupported_norm_effect",
                )
            if target_scope_id is not None and target_scope_id not in known_target_scopes:
                raise _error(
                    norm_document,
                    f"{pointer}/target_scope_id",
                    "unknown_reference",
                )
            norms.append(
                _construct(
                    norm_document,
                    pointer,
                    lambda item=item, subject_role=subject_role, effect=effect, action_id=action_id, resource_id=resource_id, relationship_type=relationship_type, target_scope_id=target_scope_id: ScenarioNorm(
                        _text(norm_document, item["norm_id"], f"{pointer}/norm_id"),
                        subject_role,
                        effect,
                        _integer(norm_document, item["priority"], f"{pointer}/priority"),
                        action_id,
                        resource_id,
                        relationship_type,
                        target_scope_id,
                        _optional_text(
                            norm_document,
                            item["descriptive_text"],
                            f"{pointer}/descriptive_text",
                        ),
                    ),
                )
            )

    owner = institution_document or relationship_document
    return _construct(
        owner,
        "",
        lambda: ScenarioSocialWorld(
            tuple(institutions),
            tuple(memberships),
            tuple(relationships),
            tuple(norms),
            action_ids,
            resource_ids,
            relationship_types,
            _compile_relationship_seeds(
                source,
                world,
                runtime_model.social_memory_model,
            ),
        ),
    )


def _compile_predicate(
    document: ScenarioSourceDocument,
    raw: object,
    pointer: str,
    world: SituatedWorldModel,
    cognition: SituatedCognitiveModel,
    topic_ids: set[str],
) -> ScenarioPredicate:
    item = _object(document, raw, pointer, {"kind", "subject_id", "object_id", "value"})
    kind = _enum(document, item["kind"], f"{pointer}/kind", ScenarioPredicateKind)
    subject_id = _text(document, item["subject_id"], f"{pointer}/subject_id")
    object_id = _optional_text(document, item["object_id"], f"{pointer}/object_id")
    agent_ids = {item.agent_id for item in world.agents}
    place_ids = {item.place_id for item in world.places}
    passage_ids = {item.passage_id for item in world.passages}
    object_ids = {item.object_id for item in world.objects}
    hypotheses_by_agent = {
        agent.agent_id: {item.hypothesis_id for item in agent.hypotheses}
        for agent in cognition.agents
    }
    if kind is ScenarioPredicateKind.AGENT_AT:
        if subject_id not in agent_ids:
            raise _error(document, f"{pointer}/subject_id", "unknown_reference")
        if object_id not in place_ids:
            raise _error(document, f"{pointer}/object_id", "unknown_reference")
    elif kind is ScenarioPredicateKind.PASSAGE_OPEN:
        if subject_id not in passage_ids:
            raise _error(document, f"{pointer}/subject_id", "unknown_reference")
    elif kind is ScenarioPredicateKind.OBJECT_AT:
        if subject_id not in object_ids:
            raise _error(document, f"{pointer}/subject_id", "unknown_reference")
        if object_id not in place_ids:
            raise _error(document, f"{pointer}/object_id", "unknown_reference")
    elif kind is ScenarioPredicateKind.AGENT_HOLDS:
        if subject_id not in agent_ids:
            raise _error(document, f"{pointer}/subject_id", "unknown_reference")
        if object_id not in object_ids:
            raise _error(document, f"{pointer}/object_id", "unknown_reference")
    elif kind is ScenarioPredicateKind.BELIEF_AT_LEAST:
        if subject_id not in agent_ids:
            raise _error(document, f"{pointer}/subject_id", "unknown_reference")
        if object_id not in hypotheses_by_agent[subject_id]:
            raise _error(document, f"{pointer}/object_id", "unknown_reference")
    elif kind is ScenarioPredicateKind.CLAIM_STATUS:
        if subject_id not in agent_ids:
            raise _error(document, f"{pointer}/subject_id", "unknown_reference")
        if object_id not in topic_ids:
            raise _error(document, f"{pointer}/object_id", "unknown_reference")
    elif kind is ScenarioPredicateKind.RELATIONSHIP_AT_LEAST:
        if subject_id not in agent_ids:
            raise _error(document, f"{pointer}/subject_id", "unknown_reference")
        if object_id not in agent_ids:
            raise _error(document, f"{pointer}/object_id", "unknown_reference")
    predicate_value = item["value"]
    if kind in {
        ScenarioPredicateKind.BELIEF_AT_LEAST,
        ScenarioPredicateKind.RELATIONSHIP_AT_LEAST,
    }:
        predicate_value = _number(
            document,
            predicate_value,
            f"{pointer}/value",
        )
    return _construct(
        document,
        pointer,
        lambda: ScenarioPredicate(kind, subject_id, object_id, predicate_value),
    )


def _compile_story_plan(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
    cognition: SituatedCognitiveModel,
    social_memory: SituatedSocialMemoryModel,
) -> ScenarioStoryPlan:
    document = _singleton(source, ScenarioDocumentRole.STORY_OUTLINE)
    value = _object(
        document,
        document.value,
        "",
        {
            "plan_id",
            "version",
            "mode",
            "acts",
            "scenes",
            "dependencies",
            "continuity_predicates",
            "terminal_predicates",
        },
    )
    acts = []
    for index, raw in enumerate(_array(document, value["acts"], "/acts")):
        pointer = f"/acts/{index}"
        item = _object(document, raw, pointer, {"act_id", "scene_ids"})
        acts.append(
            _construct(
                document,
                pointer,
                lambda item=item: ScenarioStoryAct(
                    _text(document, item["act_id"], f"{pointer}/act_id"),
                    _texts(document, item["scene_ids"], f"{pointer}/scene_ids"),
                ),
            )
        )
    place_ids = {item.place_id for item in world.places}
    agent_ids = {item.agent_id for item in world.agents}
    topic_ids = {item.topic_id for item in social_memory.topics}
    scenes = []
    for index, raw in enumerate(_array(document, value["scenes"], "/scenes")):
        pointer = f"/scenes/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {
                "scene_id",
                "place_ids",
                "participant_agent_ids",
                "preconditions",
                "exit_predicates",
                "allowed_intervention_kinds",
                "desired_outcome_ids",
                "maximum_rounds",
            },
        )
        scene_places = _texts(document, item["place_ids"], f"{pointer}/place_ids")
        for place_index, place_id in enumerate(scene_places):
            if place_id not in place_ids:
                raise _error(document, f"{pointer}/place_ids/{place_index}", "unknown_reference")
        participants = _texts(
            document,
            item["participant_agent_ids"],
            f"{pointer}/participant_agent_ids",
        )
        for agent_index, agent_id in enumerate(participants):
            if agent_id not in agent_ids:
                raise _error(
                    document,
                    f"{pointer}/participant_agent_ids/{agent_index}",
                    "unknown_reference",
                )
        preconditions = tuple(
            _compile_predicate(
                document,
                predicate,
                f"{pointer}/preconditions/{predicate_index}",
                world,
                cognition,
                topic_ids,
            )
            for predicate_index, predicate in enumerate(
                _array(document, item["preconditions"], f"{pointer}/preconditions")
            )
        )
        exits = tuple(
            _compile_predicate(
                document,
                predicate,
                f"{pointer}/exit_predicates/{predicate_index}",
                world,
                cognition,
                topic_ids,
            )
            for predicate_index, predicate in enumerate(
                _array(document, item["exit_predicates"], f"{pointer}/exit_predicates")
            )
        )
        scenes.append(
            _construct(
                document,
                pointer,
                lambda item=item, scene_places=scene_places, participants=participants, preconditions=preconditions, exits=exits: ScenarioSceneContract(
                    _text(document, item["scene_id"], f"{pointer}/scene_id"),
                    scene_places,
                    participants,
                    preconditions,
                    exits,
                    _texts(
                        document,
                        item["allowed_intervention_kinds"],
                        f"{pointer}/allowed_intervention_kinds",
                    ),
                    _texts(
                        document,
                        item["desired_outcome_ids"],
                        f"{pointer}/desired_outcome_ids",
                    ),
                    _integer(document, item["maximum_rounds"], f"{pointer}/maximum_rounds"),
                ),
            )
        )
    dependencies = []
    scene_ids = {item.scene_id for item in scenes}
    for act_index, raw in enumerate(_array(document, value["acts"], "/acts")):
        act = _object(document, raw, f"/acts/{act_index}", {"act_id", "scene_ids"})
        for scene_index, scene_id in enumerate(
            _texts(document, act["scene_ids"], f"/acts/{act_index}/scene_ids")
        ):
            if scene_id not in scene_ids:
                raise _error(
                    document,
                    f"/acts/{act_index}/scene_ids/{scene_index}",
                    "unknown_reference",
                )
    for index, raw in enumerate(_array(document, value["dependencies"], "/dependencies")):
        pointer = f"/dependencies/{index}"
        item = _object(
            document,
            raw,
            pointer,
            {"predecessor_scene_id", "successor_scene_id"},
        )
        predecessor = _text(
            document,
            item["predecessor_scene_id"],
            f"{pointer}/predecessor_scene_id",
        )
        successor = _text(
            document,
            item["successor_scene_id"],
            f"{pointer}/successor_scene_id",
        )
        if predecessor not in scene_ids:
            raise _error(document, f"{pointer}/predecessor_scene_id", "unknown_reference")
        if successor not in scene_ids:
            raise _error(document, f"{pointer}/successor_scene_id", "unknown_reference")
        dependencies.append(
            _construct(
                document,
                pointer,
                lambda predecessor=predecessor, successor=successor: ScenarioSceneDependency(
                    predecessor,
                    successor,
                ),
            )
        )
    continuity = tuple(
        _compile_predicate(
            document,
            raw,
            f"/continuity_predicates/{index}",
            world,
            cognition,
            topic_ids,
        )
        for index, raw in enumerate(
            _array(document, value["continuity_predicates"], "/continuity_predicates")
        )
    )
    terminal = tuple(
        _compile_predicate(
            document,
            raw,
            f"/terminal_predicates/{index}",
            world,
            cognition,
            topic_ids,
        )
        for index, raw in enumerate(
            _array(document, value["terminal_predicates"], "/terminal_predicates")
        )
    )
    return _construct(
        document,
        "",
        lambda: ScenarioStoryPlan(
            _text(document, value["plan_id"], "/plan_id"),
            _text(document, value["version"], "/version"),
            _enum(document, value["mode"], "/mode", ScenarioExecutionMode),
            tuple(acts),
            tuple(scenes),
            tuple(dependencies),
            continuity,
            terminal,
        ),
    )


def _compile_run_policy(source: ScenarioPackageSource) -> ScenarioRunPolicy:
    document, value = _run_value(source)
    seed = (
        None
        if value["deterministic_seed"] is None
        else _integer(document, value["deterministic_seed"], "/deterministic_seed")
    )
    return _construct(
        document,
        "",
        lambda: ScenarioRunPolicy(
            _enum(document, value["mode"], "/mode", ScenarioExecutionMode),
            _integer(document, value["maximum_rounds"], "/maximum_rounds"),
            _integer(document, value["checkpoint_interval"], "/checkpoint_interval"),
            _integer(
                document,
                value["maximum_output_records"],
                "/maximum_output_records",
            ),
            _texts(document, value["allowed_output_kinds"], "/allowed_output_kinds"),
            _boolean(document, value["public_journal"], "/public_journal"),
            _text(document, value["blender_mode"], "/blender_mode"),
            seed,
            _integer(
                document,
                value["maximum_resource_bytes"],
                "/maximum_resource_bytes",
            ),
        ),
    )


def _compile_intervention_kinds(source: ScenarioPackageSource) -> tuple[str, ...]:
    document = _optional_singleton(source, ScenarioDocumentRole.STORY_INTERVENTIONS)
    if document is None:
        if _fallback(source, ScenarioDocumentRole.STORY_INTERVENTIONS) != "none":
            run_document = _singleton(source, ScenarioDocumentRole.RUN)
            raise _error(run_document, "/fallbacks/story.interventions", "unsupported_fallback")
        return ()
    value = _object(document, document.value, "", {"intervention_kinds"})
    kinds = _texts(document, value["intervention_kinds"], "/intervention_kinds")
    if len(set(kinds)) != len(kinds):
        raise _error(document, "/intervention_kinds", "contract_violation")
    return tuple(sorted(kinds))


def _compile_initial_state_source(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
) -> _ValidatedInitialStateSource:
    document = _singleton(source, ScenarioDocumentRole.PHYSICAL_INITIAL_STATE)
    value = _object(document, document.value, "", {"agents", "objects", "passages"})
    world_agent_ids = {item.agent_id for item in world.agents}
    world_object_ids = {item.object_id for item in world.objects}
    world_passage_ids = {item.passage_id for item in world.passages}
    place_ids = {item.place_id for item in world.places}

    agents = []
    for index, raw in enumerate(_array(document, value["agents"], "/agents")):
        pointer = f"/agents/{index}"
        item = _object(document, raw, pointer, {"agent_id", "place_id"})
        agent_id = _text(document, item["agent_id"], f"{pointer}/agent_id")
        place_id = _text(document, item["place_id"], f"{pointer}/place_id")
        if agent_id not in world_agent_ids:
            raise _error(document, f"{pointer}/agent_id", "unknown_reference")
        if place_id not in place_ids:
            raise _error(document, f"{pointer}/place_id", "unknown_reference")
        agents.append((agent_id, place_id))
    if {item[0] for item in agents} != world_agent_ids or len(agents) != len(world_agent_ids):
        raise _error(document, "/agents", "roster_mismatch")

    objects = []
    for index, raw in enumerate(_array(document, value["objects"], "/objects")):
        pointer = f"/objects/{index}"
        item = _object(document, raw, pointer, {"object_id", "place_id", "holder_agent_id"})
        object_id = _text(document, item["object_id"], f"{pointer}/object_id")
        place_id = _optional_text(document, item["place_id"], f"{pointer}/place_id")
        holder_id = _optional_text(
            document,
            item["holder_agent_id"],
            f"{pointer}/holder_agent_id",
        )
        if object_id not in world_object_ids:
            raise _error(document, f"{pointer}/object_id", "unknown_reference")
        if (place_id is None) == (holder_id is None):
            raise _error(document, pointer, "contract_violation")
        if place_id is not None:
            if place_id not in place_ids:
                raise _error(document, f"{pointer}/place_id", "unknown_reference")
            location_kind = "place"
            location_id = place_id
        else:
            assert holder_id is not None
            if holder_id not in world_agent_ids:
                raise _error(document, f"{pointer}/holder_agent_id", "unknown_reference")
            location_kind = "agent"
            location_id = holder_id
        objects.append(_InitialObjectLocation(object_id, location_kind, location_id))
    if {item.object_id for item in objects} != world_object_ids or len(objects) != len(world_object_ids):
        raise _error(document, "/objects", "roster_mismatch")

    passages = []
    for index, raw in enumerate(_array(document, value["passages"], "/passages")):
        pointer = f"/passages/{index}"
        item = _object(document, raw, pointer, {"passage_id", "open"})
        passage_id = _text(document, item["passage_id"], f"{pointer}/passage_id")
        if passage_id not in world_passage_ids:
            raise _error(document, f"{pointer}/passage_id", "unknown_reference")
        passages.append((passage_id, _boolean(document, item["open"], f"{pointer}/open")))
    if {item[0] for item in passages} != world_passage_ids or len(passages) != len(world_passage_ids):
        raise _error(document, "/passages", "roster_mismatch")
    return _ValidatedInitialStateSource(
        tuple(sorted(agents)),
        tuple(sorted(objects, key=lambda item: item.object_id)),
        tuple(sorted(passages)),
    )


def _validate_social_references(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
    social_world: ScenarioSocialWorld,
) -> tuple[tuple[str, str], ...]:
    agent_ids = {item.agent_id for item in world.agents}
    institution_ids = {item.institution_id for item in social_world.institutions}
    body_roles: list[tuple[str, str]] = []
    if social_world.memberships:
        document = _singleton(source, ScenarioDocumentRole.SOCIAL_INSTITUTIONS)
        for index, membership in enumerate(social_world.memberships):
            if membership.agent_id not in agent_ids:
                raise _error(document, f"/memberships/{index}/agent_id", "unknown_reference")
            if membership.institution_id not in institution_ids:
                raise _error(
                    document,
                    f"/memberships/{index}/institution_id",
                    "unknown_reference",
                )
        if {item.agent_id for item in social_world.memberships} != agent_ids:
            raise _error(document, "/memberships", "roster_mismatch")
        roles_by_agent: dict[str, set[str]] = {agent_id: set() for agent_id in agent_ids}
        for membership in social_world.memberships:
            roles_by_agent[membership.agent_id].add(membership.role_id)
        for agent_document in _agents(source):
            agent_value = _agent_value(agent_document)
            body = _object(
                agent_document,
                agent_value["body"],
                "/body",
                {"role", "initial_place", "inventory_capacity"},
            )
            body_role = _text(agent_document, body["role"], "/body/role")
            if body_role not in roles_by_agent[agent_document.logical_id]:
                raise _error(agent_document, "/body/role", "unknown_reference")
            body_roles.append((agent_document.logical_id, body_role))
    else:
        for agent_document in _agents(source):
            agent_value = _agent_value(agent_document)
            body = _object(
                agent_document,
                agent_value["body"],
                "/body",
                {"role", "initial_place", "inventory_capacity"},
            )
            body_roles.append(
                (
                    agent_document.logical_id,
                    _text(agent_document, body["role"], "/body/role"),
                )
            )
    return tuple(sorted(body_roles))


def _validate_grant_references(
    source: ScenarioPackageSource,
    world: SituatedWorldModel,
    social_world: ScenarioSocialWorld,
    knowledge_catalog: ScenarioKnowledgeCatalog,
    asset_catalog: ScenarioAssetCatalog,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    agent_ids = {item.agent_id for item in world.agents}
    role_ids = {item.role_id for item in social_world.memberships}
    institution_ids = {item.institution_id for item in social_world.institutions}

    def validate_authored(document: ScenarioSourceDocument, root_keys: set[str]) -> None:
        value = _object(document, document.value, "", root_keys)
        for index, raw in enumerate(_array(document, value["grants"], "/grants")):
            pointer = f"/grants/{index}"
            grant = _object(
                document,
                raw,
                pointer,
                {"subject_scope", "subject_id", "resource_ids"},
            )
            scope = _text(document, grant["subject_scope"], f"{pointer}/subject_scope")
            subject_id = _optional_text(
                document,
                grant["subject_id"],
                f"{pointer}/subject_id",
            )
            if scope == "agent" and subject_id not in agent_ids:
                raise _error(document, f"/grants/{index}/subject_id", "unknown_reference")
            if scope == "role" and subject_id not in role_ids:
                raise _error(document, f"/grants/{index}/subject_id", "unknown_reference")
            if scope == "institution" and subject_id not in institution_ids:
                raise _error(document, f"/grants/{index}/subject_id", "unknown_reference")

    knowledge_document = _singleton(source, ScenarioDocumentRole.KNOWLEDGE_ACCESS)
    asset_document = _singleton(source, ScenarioDocumentRole.ASSET_CATALOG)
    validate_authored(knowledge_document, {"grants"})
    validate_authored(asset_document, {"resources", "grants"})
    resources_by_id = {
        resource.resource_id: resource for resource in knowledge_catalog.resources
    }
    roles_by_agent: dict[str, set[str]] = {agent_id: set() for agent_id in agent_ids}
    institutions_by_agent: dict[str, set[str]] = {
        agent_id: set() for agent_id in agent_ids
    }
    for membership in social_world.memberships:
        roles_by_agent[membership.agent_id].add(membership.role_id)
        institutions_by_agent[membership.agent_id].add(membership.institution_id)
    retained: list[tuple[str, tuple[str, ...]]] = []
    for agent_document in _agents(source):
        agent_value = _agent_value(agent_document)
        authored = _texts(
            agent_document,
            agent_value["knowledge_grants"],
            "/knowledge_grants",
        )
        for index, resource_id in enumerate(authored):
            if resource_id not in resources_by_id:
                raise _error(
                    agent_document,
                    f"/knowledge_grants/{index}",
                    "unknown_reference",
                )
            entitlements = {
                (item.subject_scope, item.subject_id)
                for item in resources_by_id[resource_id].entitlements
            }
            authorized = (
                ("public", None) in entitlements
                or ("agent", agent_document.logical_id) in entitlements
                or any(
                    ("role", role_id) in entitlements
                    for role_id in roles_by_agent[agent_document.logical_id]
                )
                or any(
                    ("institution", institution_id) in entitlements
                    for institution_id in institutions_by_agent[agent_document.logical_id]
                )
            )
            if not authorized:
                raise _error(
                    agent_document,
                    f"/knowledge_grants/{index}",
                    "grant_not_entitled",
                )
        retained.append((agent_document.logical_id, tuple(sorted(authored))))
    return tuple(sorted(retained))


def _validate_interventions(
    source: ScenarioPackageSource,
    intervention_kinds: tuple[str, ...],
) -> None:
    known = set(intervention_kinds)
    document = _singleton(source, ScenarioDocumentRole.STORY_OUTLINE)
    value = _object(
        document,
        document.value,
        "",
        {
            "plan_id",
            "version",
            "mode",
            "acts",
            "scenes",
            "dependencies",
            "continuity_predicates",
            "terminal_predicates",
        },
    )
    for scene_index, raw in enumerate(_array(document, value["scenes"], "/scenes")):
        scene = _object(
            document,
            raw,
            f"/scenes/{scene_index}",
            {
                "scene_id",
                "place_ids",
                "participant_agent_ids",
                "preconditions",
                "exit_predicates",
                "allowed_intervention_kinds",
                "desired_outcome_ids",
                "maximum_rounds",
            },
        )
        for kind_index, kind in enumerate(
            _texts(
                document,
                scene["allowed_intervention_kinds"],
                f"/scenes/{scene_index}/allowed_intervention_kinds",
            )
        ):
            if kind not in known:
                raise _error(
                    document,
                    f"/scenes/{scene_index}/allowed_intervention_kinds/{kind_index}",
                    "unknown_reference",
                )


def _validate_perception_signal_coverage(
    source: ScenarioPackageSource,
    runtime_model: SituatedNetworkRuntimeModel,
) -> None:
    cognition = runtime_model.percept_memory_model.cognitive_model
    required = {action.kind for agent in cognition.agents for action in agent.actions}
    actual = {
        signal.kind
        for signal in runtime_model.percept_memory_model.perception_model.signal_profiles
    }
    if not required.issubset(actual):
        document = _singleton(source, ScenarioDocumentRole.PHYSICAL_PERCEPTION)
        raise _error(document, "/signal_profiles", "roster_mismatch")


def _compile_situated_scenario_components(
    source: ScenarioPackageSource,
) -> _CompiledScenarioComponents:
    if not isinstance(source, ScenarioPackageSource):
        raise ScenarioCompilationError("package", "", "invalid_type")
    world = _compile_world(source)
    perception = _compile_perception(source, world)
    cognition = _compile_cognition(source, world)
    percept_memory = _compile_percept_memory(source, perception, cognition)
    social_memory = _compile_social_memory(source, cognition)
    runtime_model = _compile_runtime_model(source, percept_memory, social_memory)
    spatial_map = _compile_spatial_map(source, world)
    knowledge_catalog = _compile_knowledge_catalog(source)
    asset_catalog = _compile_asset_catalog(source)
    social_world = _compile_social_world(
        source,
        runtime_model,
        knowledge_catalog,
        asset_catalog,
    )
    story_plan = _compile_story_plan(source, world, cognition, social_memory)
    run_policy = _compile_run_policy(source)
    if run_policy.mode is not story_plan.mode:
        run_document = _singleton(source, ScenarioDocumentRole.RUN)
        raise _error(run_document, "/mode", "execution_mode_mismatch")
    initial_state_source = _compile_initial_state_source(source, world)
    intervention_kinds = _compile_intervention_kinds(source)
    agent_body_roles = _validate_social_references(source, world, social_world)
    agent_knowledge_grants = _validate_grant_references(
        source,
        world,
        social_world,
        knowledge_catalog,
        asset_catalog,
    )
    _validate_interventions(source, intervention_kinds)
    _validate_perception_signal_coverage(source, runtime_model)
    package_hash, source_hashes = _semantic_package_identity(source)
    return _CompiledScenarioComponents(
        source.scenario_id,
        source.version,
        package_hash,
        source_hashes,
        runtime_model,
        spatial_map,
        social_world,
        story_plan,
        knowledge_catalog,
        asset_catalog,
        run_policy,
        initial_state_source,
        intervention_kinds,
        agent_body_roles,
        agent_knowledge_grants,
    )


def _compile_initial_states(
    source: ScenarioPackageSource,
    components: _CompiledScenarioComponents,
) -> tuple[SituatedStory, SituatedCognitiveState, SituatedSocialMemoryState]:
    document = _singleton(source, ScenarioDocumentRole.PHYSICAL_INITIAL_STATE)
    world = components.runtime_model.percept_memory_model.cognitive_model.world_model
    perception = components.runtime_model.percept_memory_model.perception_model
    initial = components.initial_state_source
    try:
        world_state = SituatedWorldState(
            world.model_id,
            world.content_hash,
            0,
            None,
            tuple(
                AgentBodyState(agent_id, place_id)
                for agent_id, place_id in initial.agent_places
            ),
            tuple(
                WorldObjectState(
                    item.object_id,
                    item.location_id if item.location_kind == "place" else None,
                    item.location_id if item.location_kind == "agent" else None,
                )
                for item in initial.object_locations
            ),
            tuple(
                PassageState(passage_id, open_)
                for passage_id, open_ in initial.passage_states
            ),
        )
        validate_situated_state(world, world_state)
        story = initialize_situated_story(
            world,
            world_state,
            perception_model=perception,
        )
        cognitive_state = initialize_situated_percept_memory_cognition(
            components.runtime_model.percept_memory_model,
            story,
        )
        base_social_state = initialize_situated_social_memory(
            components.runtime_model.social_memory_model,
            cognitive_state,
        )
        social_state = SituatedSocialMemoryState(
            base_social_state.model_id,
            base_social_state.model_hash,
            base_social_state.round_index,
            base_social_state.parent_state_hash,
            base_social_state.cognitive_state_hash,
            tuple(
                SituatedSourceRelationship(
                    item.observer_agent_id,
                    item.source_agent_id,
                    item.trust,
                    item.affinity,
                    0,
                    0,
                )
                for item in components.social_world.relationship_seeds
            ),
            base_social_state.claims,
            base_social_state.processed_evidence_ids,
            base_social_state.checkpoint,
        )
        validate_situated_social_memory_state(
            components.runtime_model.social_memory_model,
            cognitive_state,
            social_state,
        )
    except (ArithmeticError, RecursionError, TypeError, ValueError):
        raise _error(document, "", "contract_violation") from None
    return story, cognitive_state, social_state


def compile_situated_scenario_package(
    source: ScenarioPackageSource,
) -> CompiledSituatedScenario:
    """Compile one source package into a complete path-independent checkpoint."""

    components = _compile_situated_scenario_components(source)
    initial_story, initial_cognitive_state, initial_social_state = _compile_initial_states(
        source,
        components,
    )
    try:
        return CompiledSituatedScenario(
            components.scenario_id,
            components.version,
            components.package_hash,
            components.source_document_hashes,
            components.runtime_model,
            components.spatial_map,
            initial_story,
            initial_cognitive_state,
            initial_social_state,
            components.social_world,
            components.story_plan,
            components.knowledge_catalog,
            components.asset_catalog,
            components.run_policy,
            components.agent_body_roles,
            components.agent_knowledge_grants,
            components.intervention_kinds,
            source.raw_manifest_hash,
            source.raw_document_hashes,
        )
    except (ArithmeticError, RecursionError, TypeError, ValueError):
        raise ScenarioCompilationError("package", "", "contract_violation") from None


def initialize_compiled_scenario(
    database_path: str | Path,
    scenario: CompiledSituatedScenario,
) -> SituatedNetworkRuntimeState:
    """Create the percept-memory store and bind the exact compiled V19 checkpoint."""

    if not isinstance(scenario, CompiledSituatedScenario):
        raise TypeError("compiled scenario initialization requires CompiledSituatedScenario")
    _preflight_compiled_scenario_database(database_path)
    index_report = initialize_situated_percept_memory(database_path)
    if index_report.record_count != 0:
        raise ValueError(
            "compiled scenario initialization requires an empty percept-memory store"
        )
    if (
        hash_situated_percept_memory_store(database_path)
        != _CANONICAL_EMPTY_PERCEPT_MEMORY_STORE_HASH
    ):
        raise ValueError(
            "compiled scenario initialization requires the canonical empty percept-memory store"
        )
    return initialize_situated_network_runtime(
        database_path,
        scenario.runtime_model,
        scenario.initial_story,
        scenario.initial_cognitive_state,
        scenario.initial_social_state,
    )


def _preflight_compiled_scenario_database(database_path: str | Path) -> None:
    if isinstance(database_path, Path):
        path = database_path
    elif isinstance(database_path, str) and database_path.strip():
        path = Path(database_path)
    else:
        raise ValueError("compiled scenario database path must be non-empty")
    folded = str(path).casefold().replace(" ", "")
    if str(path) == ":memory:" or (
        folded.startswith("file:") and "mode=memory" in folded
    ):
        raise ValueError("compiled scenario initialization requires a file-backed database")
    if not path.exists():
        return

    connection = None
    try:
        connection = sqlite3.connect(
            path.resolve(strict=True).as_uri() + "?mode=ro",
            uri=True,
        )
        connection.execute("PRAGMA query_only = ON")
        schema_objects = _sqlite_schema_snapshot(connection)
        if not schema_objects:
            raise ValueError(
                "compiled scenario initialization rejected a noncanonical database schema"
            )
        if schema_objects != situated_percept_memory_schema_snapshot():
            raise ValueError(
                "compiled scenario initialization rejected a noncanonical database schema"
            )
        metadata = tuple(
            connection.execute(
                "SELECT key, value FROM percept_memory_metadata ORDER BY key"
            ).fetchall()
        )
        record_count = connection.execute(
            "SELECT COUNT(*) FROM percept_memory_records"
        ).fetchone()[0]
        if metadata != (("schema_version", "1"),) or record_count != 0:
            raise ValueError(
                "compiled scenario initialization requires the canonical empty percept-memory store"
            )
    except ValueError:
        raise
    except (OSError, sqlite3.Error):
        raise ValueError(
            "compiled scenario initialization rejected a noncanonical database schema"
        ) from None
    finally:
        if connection is not None:
            connection.close()


def _canonicalize_sqlite_schema_sql(sql: str | None) -> str | None:
    if sql is None:
        return None
    return sql.replace("\r\n", "\n").strip()


def _sqlite_schema_snapshot(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str | None], ...]:
    return tuple(
        (object_type, name, table_name, _canonicalize_sqlite_schema_sql(sql))
        for object_type, name, table_name, sql in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
    )


__all__ = (
    "ScenarioCompilationError",
    "compile_situated_scenario_package",
    "initialize_compiled_scenario",
)
