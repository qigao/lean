"""Private retrieval and provider-neutral compilation for V16 grounding."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Protocol

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_grounding_contracts import (
    SituatedGroundedClaim,
    SituatedGroundingEvidence,
    SituatedGroundingEvidenceKind,
    SituatedGroundingModality,
    SituatedGroundingPolarity,
    SituatedGroundingPrompt,
    SituatedGroundingProviderIdentity,
    SituatedGroundingRequest,
    SituatedGroundingRetrievalPlan,
    SituatedGroundingTemporalScope,
    SituatedSemanticGroundingArtifact,
    SituatedSemanticGroundingModel,
)
from narrative_dynamics.abm.situated_percept_memory import (
    search_situated_percept_memories,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    SituatedPerceptMemoryQuery,
    SituatedPerceptMemoryRecord,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptFidelity,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedSocialEvidence,
    SituatedSocialEvidenceKind,
)


class SituatedStructuredLanguageProvider(Protocol):
    """Minimal provider seam; adapters may call any local or remote LLM."""

    identity: SituatedGroundingProviderIdentity

    def complete_json(
        self,
        *,
        task: str,
        payload: Mapping[str, object],
    ) -> str: ...


def _reject_constant(value: str) -> object:
    raise ValueError(f"provider JSON constant {value} is not allowed")


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"provider JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_mapping(response: object) -> dict[str, object]:
    if not isinstance(response, str):
        raise TypeError("structured language provider must return JSON text")
    if len(response.encode("utf-8")) > 65_536:
        raise ValueError("structured language provider response is too large")
    try:
        value = json.loads(
            response,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError("structured language provider returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("structured language provider JSON must be an object")
    return value


def _provider_response(
    provider: SituatedStructuredLanguageProvider,
    *,
    task: str,
    payload: Mapping[str, object],
) -> str:
    identity = _provider_identity(provider)
    complete = getattr(provider, "complete_json", None)
    if not callable(complete):
        raise TypeError("structured language provider requires complete_json")
    return complete(task=task, payload=payload)


def _provider_identity(
    provider: SituatedStructuredLanguageProvider,
) -> SituatedGroundingProviderIdentity:
    identity = getattr(provider, "identity", None)
    if not isinstance(identity, SituatedGroundingProviderIdentity):
        raise TypeError("structured language provider requires a grounding provider identity")
    return identity


_RETRIEVAL_KEYS = frozenset({
    "terms",
    "actor_agent_id",
    "place_id",
    "fidelities",
    "event_kinds",
    "channels",
    "min_round",
    "max_round",
})


def _enum_values(value: object, enum_type: type, *, label: str) -> tuple:
    if not isinstance(value, list):
        raise ValueError(f"grounding retrieval {label} must be a JSON array")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"grounding retrieval {label} must contain strings")
        try:
            result.append(enum_type(item))
        except ValueError as error:
            raise ValueError(f"grounding retrieval {label} contains an unknown value") from error
    return tuple(result)


def parse_situated_grounding_retrieval_plan(
    model: SituatedSemanticGroundingModel,
    request: SituatedGroundingRequest,
    response: str,
) -> SituatedGroundingRetrievalPlan:
    """Parse a provider proposal without permitting row IDs or open filters."""

    if not isinstance(model, SituatedSemanticGroundingModel):
        raise TypeError("grounding retrieval plan requires a semantic grounding model")
    if not isinstance(request, SituatedGroundingRequest):
        raise TypeError("grounding retrieval plan requires a grounding request")
    value = _strict_json_mapping(response)
    if set(value) != _RETRIEVAL_KEYS:
        raise ValueError("grounding retrieval plan must use the exact schema")
    terms = value["terms"]
    if not isinstance(terms, list) or any(not isinstance(item, str) for item in terms):
        raise ValueError("grounding retrieval terms must be a JSON string array")
    actor = value["actor_agent_id"]
    place = value["place_id"]
    for candidate, label in ((actor, "actor"), (place, "place")):
        if candidate is not None and not isinstance(candidate, str):
            raise ValueError(f"grounding retrieval {label} must be a string or null")
    world = model.percept_memory_model.cognitive_model.world_model
    if actor is not None and actor not in {item.agent_id for item in world.agents}:
        raise ValueError("grounding retrieval actor must be a declared world agent")
    if place is not None and place not in {item.place_id for item in world.places}:
        raise ValueError("grounding retrieval place must be a declared world place")
    for key in ("min_round", "max_round"):
        candidate = value[key]
        if candidate is not None and (
            not isinstance(candidate, int) or isinstance(candidate, bool)
        ):
            raise ValueError(f"grounding retrieval {key.replace('_', ' ')} must be an integer or null")
    return SituatedGroundingRetrievalPlan(
        tuple(terms),
        actor,
        place,
        _enum_values(value["fidelities"], SituatedPerceptFidelity, label="fidelities"),
        _enum_values(value["event_kinds"], SituatedActionKind, label="event kinds"),
        _enum_values(value["channels"], ObservationChannel, label="channels"),
        value["min_round"],
        value["max_round"],
    )


def _retrieval_planner_payload(
    model: SituatedSemanticGroundingModel,
    request: SituatedGroundingRequest,
) -> dict[str, object]:
    world = model.percept_memory_model.cognitive_model.world_model
    return {
        "schema_version": "situated-memory-retrieval-plan-v1",
        "request_id": request.request_id,
        "observer_agent_id": request.agent_id,
        "query": request.retrieval_query,
        "maximum_terms": 4,
        "allowed_actor_agent_ids": [item.agent_id for item in world.agents],
        "allowed_place_ids": [item.place_id for item in world.places],
        "allowed_fidelities": [item.value for item in SituatedPerceptFidelity],
        "allowed_event_kinds": [item.value for item in SituatedActionKind],
        "allowed_channels": [item.value for item in ObservationChannel],
        "output_schema": {
            "terms": ["literal FTS5 phrase"],
            "actor_agent_id": "allowed agent id or null",
            "place_id": "allowed place id or null",
            "fidelities": ["allowed fidelity"],
            "event_kinds": ["allowed event kind"],
            "channels": ["allowed observation channel"],
            "min_round": "positive integer or null",
            "max_round": "positive integer or null",
        },
    }


def _memory_evidence(memory: SituatedPerceptMemoryRecord) -> SituatedGroundingEvidence:
    return SituatedGroundingEvidence(
        memory.memory_id,
        SituatedGroundingEvidenceKind.MEMORY,
        memory.agent_id,
        memory.source_event_id,
        memory.source_event_hash,
        memory.round_index,
        memory.channels,
        memory.fidelity,
        memory.summary,
        memory.actor_agent_id,
        memory.kind,
        memory.place_id,
        memory.outcome,
        memory.details,
        memory.memory_id,
    )


def _validate_memory_binding(
    model: SituatedSemanticGroundingModel,
    memory: SituatedPerceptMemoryRecord,
) -> None:
    private_model = model.percept_memory_model
    if (
        memory.perception_model_id != private_model.perception_model.model_id
        or memory.perception_model_hash != private_model.perception_model.content_hash
        or memory.story_model_id != private_model.cognitive_model.world_model.model_id
        or memory.story_model_hash != private_model.cognitive_model.world_model.content_hash
    ):
        raise ValueError("grounding memory must bind the exact perception and world models")


def build_situated_grounding_prompt(
    database_path: str | Path,
    model: SituatedSemanticGroundingModel,
    request: SituatedGroundingRequest,
    *,
    retrieval_planner: SituatedStructuredLanguageProvider | None = None,
) -> SituatedGroundingPrompt:
    """Build one bounded prompt from only the requester's sanitized memory rows."""

    if not isinstance(model, SituatedSemanticGroundingModel):
        raise TypeError("grounding prompt construction requires a semantic grounding model")
    if not isinstance(request, SituatedGroundingRequest):
        raise TypeError("grounding prompt construction requires a grounding request")
    if request.maximum_memories > model.maximum_evidence_items:
        raise ValueError("grounding request exceeds model evidence budget")
    world = model.percept_memory_model.cognitive_model.world_model
    if request.agent_id not in {item.agent_id for item in world.agents}:
        raise ValueError("grounding requester must be a declared world agent")
    primary_hits = search_situated_percept_memories(
        database_path,
        SituatedPerceptMemoryQuery(
            request.agent_id,
            story_model_hash=world.content_hash,
            included_memory_ids=(request.primary_memory_id,),
            limit=1,
        ),
    )
    if len(primary_hits) != 1:
        raise ValueError("grounding primary memory must be active and belong to the requesting agent")
    primary = primary_hits[0].memory
    _validate_memory_binding(model, primary)

    if retrieval_planner is None:
        plan = SituatedGroundingRetrievalPlan((request.retrieval_query,))
        retrieval_provider = None
        retrieval_response_hash = None
    else:
        retrieval_provider = _provider_identity(retrieval_planner)
        response = _provider_response(
            retrieval_planner,
            task="situated_memory_retrieval_plan_v1",
            payload=_retrieval_planner_payload(model, request),
        )
        plan = parse_situated_grounding_retrieval_plan(model, request, response)
        retrieval_response_hash = stable_content_hash({"provider_response": response})

    memories = {primary.memory_id: primary}
    lexical_ranks: dict[str, float | None] = {primary.memory_id: None}
    for term in plan.terms:
        hits = search_situated_percept_memories(
            database_path,
            SituatedPerceptMemoryQuery(
                request.agent_id,
                text=term,
                actor_agent_id=plan.actor_agent_id,
                place_id=plan.place_id,
                fidelities=plan.fidelities,
                event_kinds=plan.event_kinds,
                channels=plan.channels,
                min_round=plan.min_round,
                max_round=plan.max_round,
                story_model_hash=world.content_hash,
                limit=request.maximum_memories,
            ),
        )
        for hit in hits:
            _validate_memory_binding(model, hit.memory)
            memories.setdefault(hit.memory.memory_id, hit.memory)
            prior_rank = lexical_ranks.get(hit.memory.memory_id)
            if prior_rank is None or (
                hit.lexical_rank is not None and hit.lexical_rank < prior_rank
            ):
                lexical_ranks[hit.memory.memory_id] = hit.lexical_rank

    other = sorted(
        (item for key, item in memories.items() if key != primary.memory_id),
        key=lambda item: (
            float("inf") if lexical_ranks.get(item.memory_id) is None else lexical_ranks[item.memory_id],
            -item.salience,
            -item.round_index,
            item.memory_id,
        ),
    )
    selected = (primary,) + tuple(other[: request.maximum_memories - 1])
    return SituatedGroundingPrompt(
        model,
        request,
        plan,
        primary.memory_id,
        tuple(_memory_evidence(item) for item in selected),
        retrieval_provider,
        retrieval_response_hash,
    )


_CLAIM_KEYS = frozenset({
    "subject_id",
    "predicate_id",
    "value_id",
    "polarity",
    "modality",
    "temporal_scope",
    "source_agent_id",
    "confidence",
    "evidence_ids",
})


def _claim_from_json(value: object) -> SituatedGroundedClaim:
    if not isinstance(value, dict) or set(value) != _CLAIM_KEYS:
        raise ValueError("grounded claim must use the exact schema")
    for key in ("subject_id", "predicate_id", "value_id"):
        if not isinstance(value[key], str):
            raise ValueError(f"grounded claim {key.replace('_', ' ')} must be a string")
    source = value["source_agent_id"]
    if source is not None and not isinstance(source, str):
        raise ValueError("grounded claim source agent id must be a string or null")
    evidence_ids = value["evidence_ids"]
    if not isinstance(evidence_ids, list) or any(not isinstance(item, str) for item in evidence_ids):
        raise ValueError("grounded claim evidence ids must be a JSON string array")
    try:
        polarity = SituatedGroundingPolarity(value["polarity"])
        modality = SituatedGroundingModality(value["modality"])
        temporal_scope = SituatedGroundingTemporalScope(value["temporal_scope"])
    except (TypeError, ValueError) as error:
        raise ValueError("grounded claim enum value is unknown") from error
    return SituatedGroundedClaim(
        value["subject_id"],
        value["predicate_id"],
        value["value_id"],
        polarity,
        modality,
        temporal_scope,
        source,
        value["confidence"],
        tuple(evidence_ids),
    )


def _validate_claim(
    model: SituatedSemanticGroundingModel,
    observer_agent_id: str,
    evidence_by_id: Mapping[str, SituatedGroundingEvidence],
    claim: SituatedGroundedClaim,
) -> None:
    if claim.subject_id not in model.subject_ids:
        raise ValueError("grounded claim subject is outside the finite vocabulary")
    predicate = model.predicate(claim.predicate_id)
    if predicate is None:
        raise ValueError("grounded claim predicate is outside the finite vocabulary")
    if claim.subject_id not in predicate.subject_ids:
        raise ValueError("grounded claim subject is outside its predicate vocabulary")
    if claim.value_id not in predicate.value_ids:
        raise ValueError("grounded claim value is outside its predicate vocabulary")
    if claim.temporal_scope not in predicate.allowed_temporal_scopes:
        raise ValueError("grounded claim temporal scope is unsupported by its predicate")
    world_agents = {
        item.agent_id
        for item in model.percept_memory_model.cognitive_model.world_model.agents
    }
    if claim.source_agent_id is not None and claim.source_agent_id not in world_agents:
        raise ValueError("grounded claim source is outside the world agent vocabulary")
    try:
        cited = tuple(evidence_by_id[item] for item in claim.evidence_ids)
    except KeyError as error:
        raise ValueError("grounded claim evidence is not privately available") from error
    if any(item.agent_id != observer_agent_id for item in cited):
        raise ValueError("grounded claim evidence is not private to the observer")
    fidelity_rank = {
        SituatedPerceptFidelity.DETECTED: 0,
        SituatedPerceptFidelity.IDENTIFIED: 1,
        SituatedPerceptFidelity.EXACT: 2,
    }
    if any(
        fidelity_rank[item.fidelity]
        < fidelity_rank[predicate.minimum_evidence_fidelity]
        for item in cited
    ):
        raise ValueError("grounded claim evidence fidelity is below its predicate requirement")
    if claim.source_agent_id is not None and not any(
        item.actor_agent_id == claim.source_agent_id for item in cited
    ):
        raise ValueError("grounded claim source is unsupported by cited evidence")


def validate_situated_semantic_grounding_artifact(
    model: SituatedSemanticGroundingModel,
    artifact: SituatedSemanticGroundingArtifact,
) -> None:
    """Validate an accepted artifact without invoking or trusting its provider."""

    if not isinstance(model, SituatedSemanticGroundingModel):
        raise TypeError("semantic grounding validation requires a grounding model")
    if not isinstance(artifact, SituatedSemanticGroundingArtifact):
        raise TypeError("semantic grounding validation requires a grounding artifact")
    if artifact.model_id != model.model_id or artifact.model_hash != model.content_hash:
        raise ValueError("semantic grounding artifact must bind the exact model")
    world_agents = {
        item.agent_id
        for item in model.percept_memory_model.cognitive_model.world_model.agents
    }
    if artifact.observer_agent_id not in world_agents:
        raise ValueError("semantic grounding artifact observer must be a world agent")
    if len(artifact.evidence) > model.maximum_evidence_items:
        raise ValueError("semantic grounding artifact exceeds the evidence budget")
    if len(artifact.claims) > model.maximum_claims:
        raise ValueError("semantic grounding artifact exceeds the claim budget")
    evidence_by_id = {item.evidence_id: item for item in artifact.evidence}
    for claim in artifact.claims:
        _validate_claim(model, artifact.observer_agent_id, evidence_by_id, claim)


def compile_situated_semantic_grounding(
    prompt: SituatedGroundingPrompt,
    provider: SituatedStructuredLanguageProvider,
) -> SituatedSemanticGroundingArtifact:
    """Compile one provider proposal into an accepted, replayable artifact."""

    if not isinstance(prompt, SituatedGroundingPrompt):
        raise TypeError("semantic grounding compilation requires a grounding prompt")
    provider_identity = _provider_identity(provider)
    response = _provider_response(
        provider,
        task="situated_semantic_grounding_v1",
        payload=prompt.to_provider_payload(),
    )
    value = _strict_json_mapping(response)
    if set(value) != {"claims"}:
        raise ValueError("semantic grounding response must use the exact schema")
    values = value["claims"]
    if not isinstance(values, list):
        raise ValueError("semantic grounding claims must be a JSON array")
    if len(values) > prompt.model.maximum_claims:
        raise ValueError("semantic grounding response exceeds the claim budget")
    claims = tuple(_claim_from_json(item) for item in values)
    artifact = SituatedSemanticGroundingArtifact(
        prompt.model.model_id,
        prompt.model.content_hash,
        prompt.request.request_id,
        prompt.request.agent_id,
        prompt.primary_evidence_id,
        provider_identity,
        prompt.content_hash,
        prompt.schema_hash,
        prompt.prompt_template_hash,
        prompt.private_context_hash,
        stable_content_hash({"provider_response": response}),
        "accepted",
        prompt.evidence,
        claims,
        prompt.retrieval_provider,
        prompt.retrieval_response_hash,
    )
    validate_situated_semantic_grounding_artifact(prompt.model, artifact)
    return artifact


def replay_situated_semantic_grounding(
    model: SituatedSemanticGroundingModel,
    artifact: SituatedSemanticGroundingArtifact,
) -> SituatedSemanticGroundingArtifact:
    """Replay means deterministic validation of the accepted payload, with no call."""

    validate_situated_semantic_grounding_artifact(model, artifact)
    return artifact


def _social_anchor(
    observer_agent_id: str,
    source_agent_id: str | None,
    evidence: SituatedGroundingEvidence,
) -> SituatedSocialEvidenceKind | None:
    if evidence.fidelity is not SituatedPerceptFidelity.EXACT:
        return None
    if source_agent_id is not None:
        if (
            source_agent_id == observer_agent_id
            or evidence.event_kind is not SituatedActionKind.TELL
            or evidence.actor_agent_id != source_agent_id
            or not any(item.name == "message" for item in evidence.details)
        ):
            return None
        return SituatedSocialEvidenceKind.TESTIMONY
    if (
        evidence.event_kind is SituatedActionKind.INSPECT
        and evidence.actor_agent_id == observer_agent_id
    ):
        return SituatedSocialEvidenceKind.VERIFICATION
    return None


def grounded_claims_to_situated_social_evidence(
    model: SituatedSemanticGroundingModel,
    artifact: SituatedSemanticGroundingArtifact,
) -> tuple[SituatedSocialEvidence, ...]:
    """Bridge only definite positive topic claims into the existing V14 input."""

    validate_situated_semantic_grounding_artifact(model, artifact)
    evidence_by_id = {item.evidence_id: item for item in artifact.evidence}
    candidates: dict[
        tuple[str, SituatedSocialEvidenceKind, str, str | None, str],
        tuple[str, SituatedGroundingEvidence],
    ] = {}
    for claim in artifact.claims:
        predicate = model.predicate(claim.predicate_id)
        if (
            predicate is None
            or predicate.social_topic_id is None
            or claim.subject_id != predicate.social_subject_id
            or predicate.minimum_evidence_fidelity is not SituatedPerceptFidelity.EXACT
            or claim.polarity is not SituatedGroundingPolarity.AFFIRMED
            or claim.modality is not SituatedGroundingModality.ASSERTED
            or claim.temporal_scope is not SituatedGroundingTemporalScope.PRESENT
        ):
            continue
        for evidence_id in claim.evidence_ids:
            source = evidence_by_id[evidence_id]
            kind = _social_anchor(
                artifact.observer_agent_id,
                claim.source_agent_id,
                source,
            )
            if kind is None:
                continue
            key = (
                artifact.observer_agent_id,
                kind,
                source.source_event_id,
                claim.source_agent_id,
                predicate.social_topic_id,
            )
            prior = candidates.get(key)
            if prior is not None and prior[0] != claim.value_id:
                raise ValueError("conflicting grounded claims share one social evidence event")
            candidates[key] = (claim.value_id, source)
    result = []
    for key, (symbol_id, source) in sorted(
        candidates.items(),
        key=lambda item: (
            item[0][2],
            item[0][0],
            item[0][1].value,
            "" if item[0][3] is None else item[0][3],
            item[0][4],
        ),
    ):
        observer, kind, event_id, source_agent, topic_id = key
        identity = stable_content_hash({
            "observer_agent_id": observer,
            "kind": kind.value,
            "source_event_id": event_id,
            "source_agent_id": source_agent,
            "topic_id": topic_id,
        })
        result.append(SituatedSocialEvidence(
            identity,
            kind,
            observer,
            topic_id,
            symbol_id,
            source.round_index,
            event_id,
            source_agent,
            source.memory_id,
        ))
    return tuple(sorted(result, key=lambda item: (item.round_index, item.evidence_id)))


__all__ = (
    "SituatedStructuredLanguageProvider",
    "parse_situated_grounding_retrieval_plan",
    "build_situated_grounding_prompt",
    "validate_situated_semantic_grounding_artifact",
    "compile_situated_semantic_grounding",
    "replay_situated_semantic_grounding",
    "grounded_claims_to_situated_social_evidence",
)
