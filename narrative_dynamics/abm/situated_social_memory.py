"""Pure deterministic transition for V14 situated social memory."""

from __future__ import annotations

from dataclasses import dataclass, replace

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_cognition_contracts import SituatedCognitiveState
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedConsolidatedClaim,
    SituatedSocialEvidence,
    SituatedSocialEvidenceKind,
    SituatedSocialMemoryModel,
    SituatedSocialMemoryState,
    SituatedSourceRelationship,
    validate_situated_social_memory_state,
)


@dataclass(frozen=True)
class SituatedSocialMemoryUpdate:
    prior_state: SituatedSocialMemoryState
    admitted_evidence: tuple[SituatedSocialEvidence, ...]
    next_state: SituatedSocialMemoryState

    def __post_init__(self) -> None:
        if not isinstance(self.prior_state, SituatedSocialMemoryState) or not isinstance(self.next_state, SituatedSocialMemoryState):
            raise TypeError("situated social update requires social memory states")
        if not isinstance(self.admitted_evidence, tuple) or any(
            not isinstance(item, SituatedSocialEvidence) for item in self.admitted_evidence
        ):
            raise TypeError("situated social update evidence must be a tuple")
        ids = tuple(item.evidence_id for item in self.admitted_evidence)
        if len(set(ids)) != len(ids):
            raise ValueError("situated social update evidence ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_state_hash": self.prior_state.content_hash,
            "admitted_evidence": [item.to_dict() for item in self.admitted_evidence],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _canonical_evidence(
    model: SituatedSocialMemoryModel,
    next_cognitive_state: SituatedCognitiveState,
    evidence: tuple[SituatedSocialEvidence, ...],
) -> tuple[SituatedSocialEvidence, ...]:
    if not isinstance(evidence, tuple) or any(not isinstance(item, SituatedSocialEvidence) for item in evidence):
        raise TypeError("situated social evidence must be a tuple")
    by_id = {}
    for item in evidence:
        existing = by_id.get(item.evidence_id)
        if existing is not None and existing != item:
            raise ValueError("situated social evidence id has conflicting payloads")
        by_id[item.evidence_id] = item
    agent_ids = {item.agent_id for item in model.memory_cognitive_model.cognitive_model.agents}
    topic_by_id = {item.topic_id: item for item in model.topics}
    for item in evidence:
        if item.observer_agent_id not in agent_ids or (
            item.source_agent_id is not None and item.source_agent_id not in agent_ids
        ):
            raise ValueError("situated social evidence agents must belong to the model")
        topic = topic_by_id.get(item.topic_id)
        if topic is None or item.symbol_id not in topic.symbol_ids:
            raise ValueError("situated social evidence must use a declared topic symbol")
        if item.round_index > next_cognitive_state.round_index:
            raise ValueError("situated social evidence cannot come from a future round")
    return tuple(sorted(by_id.values(), key=lambda item: (item.round_index, item.evidence_id)))


def _claim_id(item: SituatedSocialEvidence) -> str:
    return stable_content_hash({
        "observer_agent_id": item.observer_agent_id,
        "source_agent_id": item.source_agent_id,
        "topic_id": item.topic_id,
        "first_evidence_id": item.evidence_id,
    })


def _consolidate_testimony(
    claims: list[SituatedConsolidatedClaim],
    item: SituatedSocialEvidence,
) -> None:
    related = [
        claim
        for claim in claims
        if claim.observer_agent_id == item.observer_agent_id
        and claim.source_agent_id == item.source_agent_id
        and claim.topic_id == item.topic_id
    ]
    active = [
        claim
        for claim in related
        if claim.status is SituatedClaimStatus.ACTIVE
    ]
    merge_candidates = []
    for claim in related:
        if (
            claim.symbol_id != item.symbol_id
            or claim.status not in {
                SituatedClaimStatus.ACTIVE,
                SituatedClaimStatus.SUPERSEDED,
            }
        ):
            continue
        interval_start = min(item.round_index, claim.first_round)
        interval_end = max(item.round_index, claim.last_round)
        crosses_revision = any(
            other.symbol_id != item.symbol_id
            and interval_start < other.first_round <= interval_end
            for other in related
        )
        if not crosses_revision:
            distance = min(
                abs(item.round_index - claim.first_round),
                abs(item.round_index - claim.last_round),
            )
            merge_candidates.append((distance, claim.first_round, claim.claim_id, claim))
    same = None if not merge_candidates else min(merge_candidates)[-1]
    if same is not None:
        claims[claims.index(same)] = replace(
            same,
            event_ids=tuple(sorted(same.event_ids + (item.event_id,))),
            memory_ids=tuple(sorted(
                same.memory_ids + (() if item.memory_id is None else (item.memory_id,))
            )),
            first_round=min(same.first_round, item.round_index),
            last_round=max(same.last_round, item.round_index),
            support_count=same.support_count + 1,
        )
        return
    if related and item.round_index < max(claim.last_round for claim in related):
        claims.append(_new_claim(item, status=SituatedClaimStatus.SUPERSEDED))
        return
    for claim in active:
        claims[claims.index(claim)] = replace(claim, status=SituatedClaimStatus.SUPERSEDED)
    claims.append(_new_claim(item, status=SituatedClaimStatus.ACTIVE))


def _new_claim(
    item: SituatedSocialEvidence,
    *,
    status: SituatedClaimStatus,
) -> SituatedConsolidatedClaim:
    return SituatedConsolidatedClaim(
        _claim_id(item),
        item.observer_agent_id,
        item.source_agent_id or "",
        item.topic_id,
        item.symbol_id,
        (item.event_id,),
        () if item.memory_id is None else (item.memory_id,),
        item.round_index,
        item.round_index,
        1,
        status,
    )


def _learn_relationship(
    relationship: SituatedSourceRelationship,
    *,
    confirmed: bool,
    model: SituatedSocialMemoryModel,
) -> SituatedSourceRelationship:
    policy = model.policy
    if confirmed:
        return replace(
            relationship,
            trust=relationship.trust + policy.confirmation_rate * (1.0 - relationship.trust),
            affinity=min(1.0, relationship.affinity + policy.confirmation_affinity_delta),
            confirmation_count=relationship.confirmation_count + 1,
        )
    return replace(
        relationship,
        trust=relationship.trust * (1.0 - policy.contradiction_rate),
        affinity=max(-1.0, relationship.affinity - policy.contradiction_affinity_delta),
        contradiction_count=relationship.contradiction_count + 1,
    )


def _apply_verification(
    model: SituatedSocialMemoryModel,
    claims: list[SituatedConsolidatedClaim],
    relationships: dict[tuple[str, str], SituatedSourceRelationship],
    item: SituatedSocialEvidence,
) -> None:
    eligible = [
        claim
        for claim in claims
        if claim.status is SituatedClaimStatus.ACTIVE
        and claim.observer_agent_id == item.observer_agent_id
        and claim.topic_id == item.topic_id
        and claim.first_round <= item.round_index
    ]
    for claim in eligible:
        confirmed = claim.symbol_id == item.symbol_id
        claims[claims.index(claim)] = replace(
            claim,
            status=(SituatedClaimStatus.CONFIRMED if confirmed else SituatedClaimStatus.CONTRADICTED),
        )
        key = (claim.observer_agent_id, claim.source_agent_id)
        relationships[key] = _learn_relationship(
            relationships[key],
            confirmed=confirmed,
            model=model,
        )


def _forget_claims(
    model: SituatedSocialMemoryModel,
    claims: list[SituatedConsolidatedClaim],
    *,
    round_index: int,
) -> None:
    maximum_age = model.policy.max_unresolved_age_rounds
    for claim in tuple(claims):
        if claim.status is SituatedClaimStatus.ACTIVE and round_index - claim.last_round > maximum_age:
            claims[claims.index(claim)] = replace(claim, status=SituatedClaimStatus.FORGOTTEN)
    observers = {item.observer_agent_id for item in claims}
    for observer in observers:
        active = sorted(
            (
                item for item in claims
                if item.observer_agent_id == observer and item.status is SituatedClaimStatus.ACTIVE
            ),
            key=lambda item: (item.last_round, item.claim_id),
        )
        overflow = len(active) - model.policy.max_active_claims
        for claim in active[:max(overflow, 0)]:
            claims[claims.index(claim)] = replace(claim, status=SituatedClaimStatus.FORGOTTEN)


def advance_situated_social_memory(
    model: SituatedSocialMemoryModel,
    prior_cognitive_state: SituatedCognitiveState,
    state: SituatedSocialMemoryState,
    next_cognitive_state: SituatedCognitiveState,
    evidence: tuple[SituatedSocialEvidence, ...],
) -> SituatedSocialMemoryUpdate:
    """Consolidate claims, learn relationships, and forget unresolved claims."""

    validate_situated_social_memory_state(model, prior_cognitive_state, state)
    cognition = model.memory_cognitive_model.cognitive_model
    if (
        next_cognitive_state.model_id != cognition.model_id
        or next_cognitive_state.model_hash != cognition.content_hash
    ):
        raise ValueError("situated social transition must bind the exact next cognition")
    same_cognition = next_cognitive_state == prior_cognitive_state
    exact_cognitive_child = (
        next_cognitive_state.round_index == prior_cognitive_state.round_index + 1
        and next_cognitive_state.parent_state_hash
        == prior_cognitive_state.content_hash
    )
    if not same_cognition and not exact_cognitive_child:
        raise ValueError(
            "situated social transition requires the same cognition or its exact cognitive child"
        )
    canonical_evidence = _canonical_evidence(model, next_cognitive_state, evidence)

    processed = set(state.processed_evidence_ids)
    admitted = tuple(sorted(
        (item for item in canonical_evidence if item.evidence_id not in processed),
        key=lambda item: (item.round_index, item.evidence_id),
    ))
    if not admitted and next_cognitive_state == prior_cognitive_state:
        return SituatedSocialMemoryUpdate(state, (), state)

    claims = list(state.claims)
    relationships = {
        (item.observer_agent_id, item.source_agent_id): item
        for item in state.relationships
    }
    evidence_rounds = sorted({item.round_index for item in admitted})
    for evidence_round in evidence_rounds:
        same_round = tuple(item for item in admitted if item.round_index == evidence_round)
        for item in same_round:
            if item.kind is SituatedSocialEvidenceKind.TESTIMONY:
                _consolidate_testimony(claims, item)
        for item in same_round:
            if item.kind is SituatedSocialEvidenceKind.VERIFICATION:
                _apply_verification(model, claims, relationships, item)
    _forget_claims(model, claims, round_index=next_cognitive_state.round_index)
    processed.update(item.evidence_id for item in admitted)

    next_state = SituatedSocialMemoryState(
        model.model_id,
        model.content_hash,
        next_cognitive_state.round_index,
        state.content_hash,
        next_cognitive_state.content_hash,
        tuple(relationships.values()),
        tuple(claims),
        tuple(processed),
    )
    validate_situated_social_memory_state(model, next_cognitive_state, next_state)
    return SituatedSocialMemoryUpdate(state, admitted, next_state)


__all__ = (
    "SituatedSocialMemoryUpdate",
    "advance_situated_social_memory",
)
