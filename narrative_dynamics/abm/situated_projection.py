"""Deterministic POV-safe projection of situated stories into narrative cuts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionKind, SituatedWorldEvent
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPercept,
    SituatedPerceptFidelity,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    SituatedPerceptSocialCognitiveTrajectory,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedSocialEvidenceKind,
)
from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeAuthority,
    NarrativeBeat,
    NarrativeBeatPhase,
    NarrativeBeatKind,
    NarrativeCut,
    NarrativeEntitlement,
    NarrativeEntitlementScope,
    NarrativeFact,
    NarrativeProjection,
    NarrativeProjectionPolicy,
    NarrativeScene,
    NarrativeSupportRef,
    NarrativeTemporalOrder,
)
from narrative_dynamics.abm.situated_story import SituatedStory, objective_timeline


_INFORMATION_ACTIONS = {SituatedActionKind.INSPECT, SituatedActionKind.TELL}
_PHASE_ORDER = {
    NarrativeBeatPhase.MEMORY_RECALL: 0,
    NarrativeBeatPhase.BELIEF: 1,
    NarrativeBeatPhase.DECISION: 2,
    NarrativeBeatPhase.WORLD: 3,
    NarrativeBeatPhase.SOCIAL: 4,
}


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}:{stable_content_hash(value).removeprefix('sha256:')}"


def _text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _fact_items(items: tuple[tuple[str, object], ...]) -> tuple[NarrativeFact, ...]:
    facts: list[NarrativeFact] = []
    used: set[str] = set()
    for key, value in items:
        candidate = key
        suffix = 2
        while candidate in used:
            candidate = f"{key}:{suffix}"
            suffix += 1
        used.add(candidate)
        facts.append(NarrativeFact(candidate, _text(value)))
    return tuple(facts)


def _event_facts(event: SituatedWorldEvent) -> tuple[NarrativeFact, ...]:
    fields: list[tuple[str, object]] = [
        ("actor_agent_id", event.actor_agent_id),
        ("kind", event.kind.value),
        ("place_id", event.place_id),
        ("success", event.success),
        ("outcome", event.outcome),
    ]
    if event.target_id is not None:
        fields.append(("target_id", event.target_id))
    fields.extend((item.name, item.value) for item in event.details)
    return _fact_items(tuple(fields))


def _percept_facts(percept: SituatedPercept) -> tuple[NarrativeFact, ...]:
    fields: list[tuple[str, object]] = [
        ("percept_id", percept.percept_id),
        ("round_index", percept.round_index),
        ("agent_id", percept.agent_id),
        ("source_event_id", percept.source_event_id),
        ("channels", ",".join(item.value for item in percept.channels)),
        ("fidelity", percept.fidelity.value),
    ]
    for key, value in (
        ("actor_agent_id", percept.actor_agent_id),
        ("kind", None if percept.kind is None else percept.kind.value),
        ("place_id", percept.place_id),
        ("outcome", percept.outcome),
    ):
        if value is not None:
            fields.append((key, value))
    fields.extend((item.name, item.value) for item in percept.details)
    return _fact_items(tuple(fields))


@dataclass(frozen=True)
class _Candidate:
    source_event_id: str | None
    round_index: int
    sequence: int
    place_id: str | None
    active_pov_agent_id: str | None
    agent_ids: tuple[str, ...]
    base_kind: NarrativeBeatKind
    support: NarrativeSupportRef
    entitlement: NarrativeEntitlement
    direct_cause_event_ids: tuple[str, ...]
    magnitude: float = 1.0
    phase: NarrativeBeatPhase = NarrativeBeatPhase.WORLD
    additional_supports: tuple[NarrativeSupportRef, ...] = ()
    event_origin: bool = False


def _entitlement(
    *,
    scope: NarrativeEntitlementScope,
    owner_agent_id: str | None,
    round_index: int,
    facts: tuple[NarrativeFact, ...],
    support: NarrativeSupportRef,
    additional_supports: tuple[NarrativeSupportRef, ...] = (),
) -> NarrativeEntitlement:
    supports = (support,) + additional_supports
    entitlement_id = _stable_id(
        "entitlement",
        {
            "scope": scope.value,
            "owner_agent_id": owner_agent_id,
            "round_index": round_index,
            "facts": [item.to_dict() for item in facts],
            "support": [item.to_dict() for item in supports],
        },
    )
    return NarrativeEntitlement(
        entitlement_id,
        scope,
        owner_agent_id,
        round_index,
        facts,
        supports,
    )


def _objective_candidates(story: SituatedStory) -> tuple[_Candidate, ...]:
    candidates = []
    for event in objective_timeline(story):
        support = NarrativeSupportRef("world_event", event.event_id, event.content_hash)
        facts = _event_facts(event)
        candidates.append(
            _Candidate(
                event.event_id,
                event.round_index,
                event.sequence,
                event.place_id,
                None,
                (event.actor_agent_id,),
                NarrativeBeatKind.INFORMATION
                if event.kind in _INFORMATION_ACTIONS
                else NarrativeBeatKind.PHYSICAL,
                support,
                _entitlement(
                    scope=NarrativeEntitlementScope.OBJECTIVE,
                    owner_agent_id=None,
                    round_index=event.round_index,
                    facts=facts,
                    support=support,
                ),
                event.cause_event_ids,
                event_origin=True,
            )
        )
    return tuple(candidates)


def _private_candidates(
    story: SituatedStory, policy: NarrativeProjectionPolicy
) -> tuple[_Candidate, ...]:
    model = story.perception_model
    if model is None:
        raise ValueError("limited and multi-POV projection requires a perception model")
    world_agents = {item.agent_id for item in model.world_model.agents}
    unknown = set(policy.pov_agent_ids).difference(world_agents)
    if unknown:
        raise ValueError("narrative POV agent must belong to the situated story")

    candidates = []
    for agent_id in policy.pov_agent_ids:
        per_round_sequence: dict[int, int] = {}
        for percept in perceptual_timeline(model, story, agent_id):
            sequence = per_round_sequence.get(percept.round_index, 0) + 1
            per_round_sequence[percept.round_index] = sequence
            support = NarrativeSupportRef(
                "percept", percept.percept_id, percept.content_hash
            )
            facts = _percept_facts(percept)
            disclosed_agents = {agent_id}
            if percept.actor_agent_id is not None:
                disclosed_agents.add(percept.actor_agent_id)
            candidates.append(
                _Candidate(
                    percept.source_event_id,
                    percept.round_index,
                    sequence,
                    percept.place_id,
                    agent_id,
                    tuple(disclosed_agents),
                    NarrativeBeatKind.INFORMATION
                    if percept.kind is None or percept.kind in _INFORMATION_ACTIONS
                    else NarrativeBeatKind.PHYSICAL,
                    support,
                    _entitlement(
                        scope=NarrativeEntitlementScope.PRIVATE,
                        owner_agent_id=agent_id,
                        round_index=percept.round_index,
                        facts=facts,
                        support=support,
                    ),
                    (),
                    event_origin=True,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.round_index,
                item.sequence,
                item.active_pov_agent_id or "",
                item.support.artifact_id,
            ),
        )
    )


def _float_fact(value: float) -> str:
    return repr(float(value))


def _private_artifact_candidate(
    *,
    source_event_id: str | None,
    round_index: int,
    sequence: int,
    place_id: str | None,
    owner_agent_id: str,
    agent_ids: tuple[str, ...],
    kind: NarrativeBeatKind,
    artifact_kind: str,
    artifact_id: str,
    artifact_hash: str,
    facts: tuple[tuple[str, object], ...],
    magnitude: float,
    phase: NarrativeBeatPhase,
    additional_supports: tuple[NarrativeSupportRef, ...] = (),
) -> _Candidate:
    support = NarrativeSupportRef(artifact_kind, artifact_id, artifact_hash)
    fact_items = _fact_items(facts)
    return _Candidate(
        source_event_id,
        round_index,
        sequence,
        place_id,
        owner_agent_id,
        agent_ids,
        kind,
        support,
        _entitlement(
            scope=NarrativeEntitlementScope.PRIVATE,
            owner_agent_id=owner_agent_id,
            round_index=round_index,
            facts=fact_items,
            support=support,
            additional_supports=additional_supports,
        ),
        (),
        magnitude,
        phase,
        additional_supports,
    )


def _unique_evidence(items):
    return items[0] if len(items) == 1 else None


def _claim_status_only_change(prior, claim) -> bool:
    return (
        claim.claim_id == prior.claim_id
        and claim.observer_agent_id == prior.observer_agent_id
        and claim.source_agent_id == prior.source_agent_id
        and claim.topic_id == prior.topic_id
        and claim.symbol_id == prior.symbol_id
        and claim.event_ids == prior.event_ids
        and claim.memory_ids == prior.memory_ids
        and claim.first_round == prior.first_round
        and claim.last_round == prior.last_round
        and claim.support_count == prior.support_count
        and claim.status is not prior.status
    )


def _claim_support_only_change(prior, claim, trigger) -> bool:
    expected_memories = prior.memory_ids + (() if trigger.memory_id is None else (
        trigger.memory_id,
    ))
    return (
        claim.claim_id == prior.claim_id
        and claim.observer_agent_id == prior.observer_agent_id
        and claim.source_agent_id == prior.source_agent_id
        and claim.topic_id == prior.topic_id
        and claim.symbol_id == prior.symbol_id
        and claim.event_ids == tuple(sorted(prior.event_ids + (trigger.event_id,)))
        and claim.memory_ids == tuple(sorted(expected_memories))
        and claim.first_round == min(prior.first_round, trigger.round_index)
        and claim.last_round == max(prior.last_round, trigger.round_index)
        and claim.support_count == prior.support_count + 1
        and claim.status is prior.status
    )


def _claim_trigger_evidence(prior, claim, evidence):
    related_testimony = tuple(
        item
        for item in evidence
        if item.kind is SituatedSocialEvidenceKind.TESTIMONY
        and item.observer_agent_id == claim.observer_agent_id
        and item.source_agent_id == claim.source_agent_id
        and item.topic_id == claim.topic_id
    )
    if prior is None:
        if claim.status is not SituatedClaimStatus.ACTIVE:
            return None
        trigger = _unique_evidence(related_testimony)
        if trigger is None or trigger.symbol_id != claim.symbol_id:
            return None
        expected_memories = () if trigger.memory_id is None else (trigger.memory_id,)
        if not (
            claim.event_ids == (trigger.event_id,)
            and claim.memory_ids == expected_memories
            and claim.first_round == trigger.round_index
            and claim.last_round == trigger.round_index
            and claim.support_count == 1
        ):
            return None
        return trigger

    if claim.status is SituatedClaimStatus.FORGOTTEN:
        return None

    same_symbol = tuple(
        item
        for item in related_testimony
        if item.symbol_id == claim.symbol_id
    )
    support_trigger = _unique_evidence(same_symbol)
    if (
        support_trigger is not None
        and _claim_support_only_change(prior, claim, support_trigger)
    ):
        return support_trigger

    if not _claim_status_only_change(prior, claim):
        return None

    if prior.status is SituatedClaimStatus.ACTIVE and claim.status in {
        SituatedClaimStatus.CONFIRMED,
        SituatedClaimStatus.CONTRADICTED,
    }:
        return _unique_evidence(tuple(
            item
            for item in evidence
            if item.kind is SituatedSocialEvidenceKind.VERIFICATION
            and item.observer_agent_id == claim.observer_agent_id
            and item.topic_id == claim.topic_id
            and (
                (claim.status is SituatedClaimStatus.CONFIRMED and item.symbol_id == claim.symbol_id)
                or (
                    claim.status is SituatedClaimStatus.CONTRADICTED
                    and item.symbol_id != claim.symbol_id
                )
            )
        ))
    if (
        prior.status is SituatedClaimStatus.ACTIVE
        and claim.status is SituatedClaimStatus.SUPERSEDED
    ):
        return _unique_evidence(tuple(
            item
            for item in related_testimony
            if item.symbol_id != claim.symbol_id
        ))
    return None


def _relationship_trigger_evidence(
    prior_claims,
    next_claims,
    relationship,
    evidence,
):
    next_by_id = {item.claim_id: item for item in next_claims}
    affected = tuple(
        (prior, next_by_id.get(prior.claim_id))
        for prior in prior_claims
        if prior.observer_agent_id == relationship.observer_agent_id
        and prior.source_agent_id == relationship.source_agent_id
        and prior.status is SituatedClaimStatus.ACTIVE
        and next_by_id.get(prior.claim_id) is not None
        and next_by_id[prior.claim_id].status
        in {SituatedClaimStatus.CONFIRMED, SituatedClaimStatus.CONTRADICTED}
    )
    if len(affected) != 1:
        return None
    prior, next_claim = affected[0]
    assert next_claim is not None
    trigger = _claim_trigger_evidence(prior, next_claim, evidence)
    if (
        trigger is None
        or trigger.kind is not SituatedSocialEvidenceKind.VERIFICATION
    ):
        return None
    return trigger


def _belief_facts(decision) -> tuple[tuple[str, object], ...]:
    facts: list[tuple[str, object]] = [
        ("decision.agent_id", decision.agent_id),
        ("decision.round_index", decision.round_index),
    ]
    for hypothesis_id in decision.prior_belief.probabilities:
        facts.extend((
            (
                f"decision.prior_belief.probabilities.{hypothesis_id}",
                _float_fact(decision.prior_belief.probabilities[hypothesis_id]),
            ),
            (
                f"decision.posterior_belief.probabilities.{hypothesis_id}",
                _float_fact(decision.posterior_belief.probabilities[hypothesis_id]),
            ),
        ))
    return tuple(facts)


def _mind_support(mind, artifact_id: str) -> NarrativeSupportRef:
    return NarrativeSupportRef(
        "cognitive_mind_state",
        artifact_id,
        mind.content_hash,
    )


def _exact_self_percept(
    percepts: tuple[SituatedPercept, ...],
    decision,
) -> SituatedPercept | None:
    matches = tuple(
        percept
        for percept in percepts
        if percept.round_index == decision.round_index
        and percept.agent_id == decision.agent_id
        and percept.actor_agent_id == decision.agent_id
        and percept.kind is decision.intent.kind
        and percept.fidelity is SituatedPerceptFidelity.EXACT
    )
    return matches[0] if len(matches) == 1 else None


def _trajectory_candidates(
    story: SituatedStory,
    policy: NarrativeProjectionPolicy,
    trajectory: SituatedPerceptSocialCognitiveTrajectory,
) -> tuple[_Candidate, ...]:
    authorized = (
        {decision.agent_id for item in trajectory.rounds for decision in item.decisions}
        if policy.authority is NarrativeAuthority.OBJECTIVE
        else set(policy.pov_agent_ids)
    )
    story_event_ids = {item.event_id for item in objective_timeline(story)}
    private_percepts = (
        {}
        if policy.authority is NarrativeAuthority.OBJECTIVE
        else {
            owner: perceptual_timeline(story.perception_model, story, owner)
            for owner in authorized
        }
    )
    candidates: list[_Candidate] = []
    previous_decisions = {}

    for round_result in trajectory.rounds:
        prior_state = round_result.cognitive_round.cognitive_round.prior_state
        prior_minds = {item.agent_id: item for item in prior_state.minds}
        next_minds = {
            item.agent_id: item for item in round_result.next_cognitive_state.minds
        }
        decision_events = (
            {
                event.action_id: event
                for event in round_result.next_story.rounds[-1].events
            }
            if policy.authority is NarrativeAuthority.OBJECTIVE
            else {}
        )
        sequence = 0

        for decision in round_result.decisions:
            owner = decision.agent_id
            prior_decision = previous_decisions.get(owner)
            previous_decisions[owner] = decision
            if owner not in authorized:
                continue
            sequence += 1
            prior_mind = prior_minds[owner]
            metadata_supports = (
                _mind_support(
                    prior_mind,
                    f"{owner}:{prior_state.round_index}:prior",
                ),
            )
            if policy.authority is NarrativeAuthority.OBJECTIVE:
                event = decision_events.get(decision.intent.action_id)
                if event is None:
                    raise ValueError(
                        "trajectory decision must resolve to an accepted story event"
                    )
                source_event_id = event.event_id
                beat_round_index = decision.round_index
                beat_sequence = event.sequence
                metadata_supports += (
                    NarrativeSupportRef(
                        "world_event", event.event_id, event.content_hash
                    ),
                )
            else:
                percept = _exact_self_percept(
                    private_percepts.get(owner, ()), decision
                )
                source_event_id = (
                    None if percept is None else percept.source_event_id
                )
                beat_round_index = (
                    decision.round_index
                    if percept is None
                    else percept.round_index
                )
                beat_sequence = sequence
                if percept is not None:
                    metadata_supports += (
                        NarrativeSupportRef(
                            "percept", percept.percept_id, percept.content_hash
                        ),
                    )
            if policy.include_beliefs and decision.prior_belief != decision.posterior_belief:
                magnitude = 0.5 * math.fsum(
                    abs(
                        decision.posterior_belief.probabilities[key]
                        - decision.prior_belief.probabilities[key]
                    )
                    for key in decision.prior_belief.probabilities
                )
                candidates.append(_private_artifact_candidate(
                    source_event_id=source_event_id,
                    round_index=beat_round_index,
                    sequence=beat_sequence,
                    place_id=prior_mind.own_place_id,
                    owner_agent_id=owner,
                    agent_ids=(owner,),
                    kind=NarrativeBeatKind.BELIEF_SHIFT,
                    artifact_kind="cognitive_decision",
                    artifact_id=f"{owner}:{decision.round_index}",
                    artifact_hash=decision.content_hash,
                    facts=_belief_facts(decision),
                    magnitude=magnitude,
                    phase=NarrativeBeatPhase.BELIEF,
                    additional_supports=metadata_supports,
                ))
            if (
                prior_decision is not None
                and prior_decision.selected_action_id != decision.selected_action_id
            ):
                sequence += 1
                reversal_supports = metadata_supports + (
                    NarrativeSupportRef(
                        "cognitive_decision",
                        f"{prior_decision.agent_id}:{prior_decision.round_index}",
                        prior_decision.content_hash,
                    ),
                )
                candidates.append(_private_artifact_candidate(
                    source_event_id=source_event_id,
                    round_index=beat_round_index,
                    sequence=(event.sequence if policy.authority is NarrativeAuthority.OBJECTIVE else sequence),
                    place_id=prior_mind.own_place_id,
                    owner_agent_id=owner,
                    agent_ids=(owner,),
                    kind=NarrativeBeatKind.ACTION_REVERSAL,
                    artifact_kind="cognitive_decision",
                    artifact_id=f"{owner}:{decision.round_index}",
                    artifact_hash=decision.content_hash,
                    facts=(
                        ("decision.agent_id", owner),
                        ("decision.round_index", decision.round_index),
                        (
                            "decision.prior_selected_action_id",
                            prior_decision.selected_action_id,
                        ),
                        ("decision.selected_action_id", decision.selected_action_id),
                    ),
                    magnitude=1.0,
                    phase=NarrativeBeatPhase.DECISION,
                    additional_supports=reversal_supports,
                ))

        if policy.include_memories:
            for recall in round_result.recalls:
                owner = recall.prior_mind.agent_id
                if owner not in authorized:
                    continue
                for admission in recall.admissions:
                    if admission.consolidated:
                        continue
                    if admission.event_id not in story_event_ids:
                        raise ValueError(
                            "trajectory recall must resolve to an accepted story event"
                        )
                    sequence += 1
                    round_index = round_result.next_cognitive_state.round_index
                    candidates.append(_private_artifact_candidate(
                        source_event_id=admission.event_id,
                        round_index=round_index,
                        sequence=sequence,
                        place_id=recall.prior_mind.own_place_id,
                        owner_agent_id=owner,
                        agent_ids=(owner,),
                        kind=NarrativeBeatKind.MEMORY_RECALL,
                        artifact_kind="memory_recall_admission",
                        artifact_id=(
                            f"{owner}:{round_result.next_cognitive_state.round_index}:"
                            f"{admission.memory_id}"
                        ),
                        artifact_hash=admission.content_hash,
                        facts=(
                            ("admission.agent_id", admission.agent_id),
                            ("admission.cue_id", admission.cue_id),
                            ("admission.memory_id", admission.memory_id),
                            ("admission.event_id", admission.event_id),
                            ("admission.evidence_weight", _float_fact(admission.evidence_weight)),
                        ),
                        magnitude=admission.evidence_weight,
                        phase=NarrativeBeatPhase.MEMORY_RECALL,
                        additional_supports=(
                            _mind_support(
                                recall.prior_mind,
                                f"{owner}:{round_index}:recall-prior",
                            ),
                        ),
                    ))

        social_update = round_result.social_update
        for item in social_update.admitted_evidence:
            if item.event_id not in story_event_ids:
                raise ValueError(
                    "trajectory social evidence event must belong to the accepted story"
                )
        update_support = NarrativeSupportRef(
            "social_memory_update",
            f"{trajectory.social_model_id}:{social_update.next_state.round_index}",
            social_update.content_hash,
        )
        prior_claims = {item.claim_id: item for item in social_update.prior_state.claims}
        if policy.include_claim_revisions:
            for claim in social_update.next_state.claims:
                prior = prior_claims.get(claim.claim_id)
                if claim == prior or claim.observer_agent_id not in authorized:
                    continue
                trigger = _claim_trigger_evidence(
                    prior,
                    claim,
                    social_update.admitted_evidence,
                )
                additional_supports = (update_support,) + (() if trigger is None else (
                    NarrativeSupportRef(
                        "social_evidence",
                        trigger.evidence_id,
                        trigger.content_hash,
                    ),
                ))
                next_mind = next_minds[claim.observer_agent_id]
                additional_supports += (
                    _mind_support(
                        next_mind,
                        f"{claim.observer_agent_id}:"
                        f"{social_update.next_state.round_index}:next",
                    ),
                )
                sequence += 1
                candidates.append(_private_artifact_candidate(
                    source_event_id=None if trigger is None else trigger.event_id,
                    round_index=social_update.next_state.round_index,
                    sequence=sequence,
                    place_id=next_mind.own_place_id,
                    owner_agent_id=claim.observer_agent_id,
                    agent_ids=(claim.observer_agent_id, claim.source_agent_id),
                    kind=NarrativeBeatKind.CLAIM_REVISION,
                    artifact_kind="consolidated_claim",
                    artifact_id=claim.claim_id,
                    artifact_hash=claim.content_hash,
                    facts=(
                        ("claim.claim_id", claim.claim_id),
                        ("claim.observer_agent_id", claim.observer_agent_id),
                        ("claim.source_agent_id", claim.source_agent_id),
                        ("claim.topic_id", claim.topic_id),
                        ("claim.symbol_id", claim.symbol_id),
                        ("claim.prior_status", "none" if prior is None else prior.status.value),
                        ("claim.status", claim.status.value),
                        ("claim.support_count", claim.support_count),
                    ),
                    magnitude=1.0,
                    phase=NarrativeBeatPhase.SOCIAL,
                    additional_supports=additional_supports,
                ))

        prior_relationships = {
            (item.observer_agent_id, item.source_agent_id): item
            for item in social_update.prior_state.relationships
        }
        if policy.include_relationship_changes:
            for relationship in social_update.next_state.relationships:
                owner = relationship.observer_agent_id
                prior = prior_relationships[
                    (relationship.observer_agent_id, relationship.source_agent_id)
                ]
                if relationship == prior or owner not in authorized:
                    continue
                trigger = _relationship_trigger_evidence(
                    social_update.prior_state.claims,
                    social_update.next_state.claims,
                    relationship,
                    social_update.admitted_evidence,
                )
                additional_supports = (update_support,) + (() if trigger is None else (
                    NarrativeSupportRef(
                        "social_evidence",
                        trigger.evidence_id,
                        trigger.content_hash,
                    ),
                ))
                next_mind = next_minds[owner]
                additional_supports += (
                    _mind_support(
                        next_mind,
                        f"{owner}:{social_update.next_state.round_index}:next",
                    ),
                )
                sequence += 1
                trust_delta = relationship.trust - prior.trust
                affinity_delta = relationship.affinity - prior.affinity
                candidates.append(_private_artifact_candidate(
                    source_event_id=None if trigger is None else trigger.event_id,
                    round_index=social_update.next_state.round_index,
                    sequence=sequence,
                    place_id=next_mind.own_place_id,
                    owner_agent_id=owner,
                    agent_ids=(owner, relationship.source_agent_id),
                    kind=NarrativeBeatKind.RELATIONSHIP_CHANGE,
                    artifact_kind="source_relationship",
                    artifact_id=(
                        f"{owner}:{relationship.source_agent_id}:"
                        f"{social_update.next_state.round_index}"
                    ),
                    artifact_hash=relationship.content_hash,
                    facts=(
                        ("relationship.observer_agent_id", owner),
                        ("relationship.source_agent_id", relationship.source_agent_id),
                        ("relationship.prior_trust", _float_fact(prior.trust)),
                        ("relationship.trust", _float_fact(relationship.trust)),
                        ("relationship.trust_delta", _float_fact(trust_delta)),
                        ("relationship.prior_affinity", _float_fact(prior.affinity)),
                        ("relationship.affinity", _float_fact(relationship.affinity)),
                        ("relationship.affinity_delta", _float_fact(affinity_delta)),
                    ),
                    magnitude=abs(trust_delta) + abs(affinity_delta),
                    phase=NarrativeBeatPhase.SOCIAL,
                    additional_supports=additional_supports,
                ))

    return tuple(candidates)


def _candidate_key(
    candidate: _Candidate,
) -> tuple[object, ...]:
    return (
        candidate.source_event_id,
        candidate.round_index,
        candidate.sequence,
        candidate.place_id,
        candidate.active_pov_agent_id,
        candidate.agent_ids,
        candidate.base_kind.value,
        (
            candidate.support.artifact_kind,
            candidate.support.artifact_id,
            candidate.support.artifact_hash,
        ),
        tuple(
            (support.artifact_kind, support.artifact_id, support.artifact_hash)
            for support in candidate.additional_supports
        ),
        candidate.entitlement.content_hash,
        candidate.direct_cause_event_ids,
        repr(candidate.magnitude),
        candidate.phase.value,
        candidate.event_origin,
    )


def _candidate_beat_id(candidate: _Candidate) -> str:
    return _stable_id(
        "beat",
        {"candidate_identity": _candidate_key(candidate)},
    )


def _beat(
    candidate: _Candidate,
    policy: NarrativeProjectionPolicy,
    cause_beat_ids: tuple[str, ...],
    effective_kind: NarrativeBeatKind,
) -> NarrativeBeat:
    beat_id = _candidate_beat_id(candidate)
    return NarrativeBeat(
        beat_id,
        effective_kind,
        candidate.round_index,
        candidate.sequence,
        candidate.place_id,
        candidate.active_pov_agent_id,
        candidate.agent_ids,
        policy.weight_for(effective_kind) * candidate.magnitude,
        (candidate.support,) + candidate.additional_supports,
        (candidate.entitlement.entitlement_id,),
        cause_beat_ids,
        source_event_id=candidate.source_event_id,
        phase=candidate.phase,
    )


def _chronological_key(candidate: _Candidate) -> tuple[object, ...]:
    return (
        candidate.round_index,
        _PHASE_ORDER[candidate.phase],
        candidate.sequence,
        candidate.active_pov_agent_id or "",
        _candidate_beat_id(candidate),
    )


def _lane(candidate: _Candidate) -> str | None:
    return candidate.active_pov_agent_id


def _authorized_causes(
    candidate: _Candidate,
    by_lane_and_event: dict[
        tuple[str | None, str | None], tuple[_Candidate, ...]
    ],
) -> tuple[_Candidate, ...]:
    causes = tuple(
        cause
        for source_event_id in candidate.direct_cause_event_ids
        for cause in by_lane_and_event.get((_lane(candidate), source_event_id), ())
    )
    return tuple(sorted(causes, key=_chronological_key))


def _effective_kind(
    candidate: _Candidate,
) -> NarrativeBeatKind:
    if candidate.direct_cause_event_ids:
        return NarrativeBeatKind.CAUSAL_PAYOFF
    return candidate.base_kind


def _select_candidates(
    candidates: tuple[_Candidate, ...], policy: NarrativeProjectionPolicy
) -> tuple[_Candidate, ...]:
    if not candidates:
        return ()
    by_lane_and_event: dict[
        tuple[str | None, str | None], tuple[_Candidate, ...]
    ] = {}
    for candidate in candidates:
        key = (_lane(candidate), candidate.source_event_id)
        by_lane_and_event[key] = by_lane_and_event.get(key, ()) + (candidate,)

    by_lane: dict[str | None, list[_Candidate]] = {}
    for candidate in candidates:
        by_lane.setdefault(_lane(candidate), []).append(candidate)
    selected: set[tuple[object, ...]] = set()
    for lane_candidates in by_lane.values():
        ordered = sorted(lane_candidates, key=_chronological_key)
        for candidate in ordered:
            score = (
                policy.weight_for(_effective_kind(candidate))
                * candidate.magnitude
            )
            if score >= policy.minimum_salience:
                selected.add(_candidate_key(candidate))
        event_candidates = [item for item in ordered if item.event_origin]
        if not event_candidates:
            continue
        selected.add(_candidate_key(event_candidates[0]))
        selected.add(_candidate_key(event_candidates[-1]))
        maximum_gap = policy.maximum_event_omission_gap
        if maximum_gap:
            selected_positions = [
                index
                for index, candidate in enumerate(event_candidates)
                if _candidate_key(candidate) in selected
            ]
            anchors: list[int] = [selected_positions[0]]
            for right in selected_positions[1:]:
                left = anchors[-1]
                while right - left > maximum_gap:
                    left += maximum_gap
                    selected.add(_candidate_key(event_candidates[left]))
                    anchors.append(left)
                anchors.append(right)
    changed = True
    while changed:
        changed = False
        selected_candidates = sorted(
            (item for item in candidates if _candidate_key(item) in selected),
            key=_chronological_key,
        )
        for candidate in selected_candidates:
            causes = _authorized_causes(candidate, by_lane_and_event)
            required = math.ceil(len(causes) * policy.required_causal_coverage)
            for cause in causes[:required]:
                key = _candidate_key(cause)
                if key not in selected:
                    selected.add(key)
                    changed = True
    return tuple(item for item in candidates if _candidate_key(item) in selected)


def _order_candidates(
    story: SituatedStory,
    candidates: tuple[_Candidate, ...],
    policy: NarrativeProjectionPolicy,
) -> tuple[_Candidate, ...]:
    chronological = tuple(sorted(candidates, key=_chronological_key))
    if policy.temporal_order is NarrativeTemporalOrder.CHRONOLOGICAL:
        return chronological
    story_event_ids = {item.event_id for item in objective_timeline(story)}
    if any(item not in story_event_ids for item in policy.authored_event_order):
        raise ValueError("authored event order ids must belong to the accepted story")
    selected_event_ids = {
        item.source_event_id
        for item in candidates
        if item.source_event_id is not None
    }
    if (
        len(policy.authored_event_order) != len(selected_event_ids)
        or set(policy.authored_event_order) != selected_event_ids
    ):
        raise ValueError(
            "authored event order must be an exact permutation of selected source events"
        )
    position = {
        event_id: index for index, event_id in enumerate(policy.authored_event_order)
    }
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                position.get(item.source_event_id, len(position)),
                _chronological_key(item),
            ),
        )
    )


def _make_beats(
    candidates: tuple[_Candidate, ...], policy: NarrativeProjectionPolicy
) -> tuple[NarrativeBeat, ...]:
    by_lane_and_event: dict[
        tuple[str | None, str | None], tuple[_Candidate, ...]
    ] = {}
    for candidate in candidates:
        key = (_lane(candidate), candidate.source_event_id)
        by_lane_and_event[key] = by_lane_and_event.get(key, ()) + (candidate,)
    selected_keys = {_candidate_key(item) for item in candidates}
    beats = []
    for candidate in candidates:
        authorized_causes = _authorized_causes(candidate, by_lane_and_event)
        selected_causes = tuple(
            cause
            for cause in authorized_causes
            if _candidate_key(cause) in selected_keys
        )
        beats.append(
            _beat(
                candidate,
                policy,
                tuple(_candidate_beat_id(item) for item in selected_causes),
                _effective_kind(candidate),
            )
        )
    return tuple(beats)


def _scene(
    scene_beats: tuple[NarrativeBeat, ...],
) -> NarrativeScene:
    beat_ids = tuple(item.beat_id for item in scene_beats)
    return NarrativeScene(
        _stable_id(
            "scene",
            {
                "beat_ids": list(beat_ids),
                "place_id": scene_beats[0].place_id,
                "active_pov_agent_id": scene_beats[0].active_pov_agent_id,
            },
        ),
        beat_ids,
        min(item.round_index for item in scene_beats),
        max(item.round_index for item in scene_beats),
        scene_beats[0].place_id,
        scene_beats[0].active_pov_agent_id,
    )


def _group_scenes(
    beats: tuple[NarrativeBeat, ...], policy: NarrativeProjectionPolicy
) -> tuple[NarrativeScene, ...]:
    groups: list[list[NarrativeBeat]] = []
    for beat in beats:
        if not groups:
            groups.append([beat])
            continue
        current = groups[-1]
        previous = current[-1]
        continues = (
            beat.place_id is not None
            and previous.place_id is not None
            and beat.place_id == previous.place_id
            and beat.active_pov_agent_id == previous.active_pov_agent_id
            and abs(beat.round_index - previous.round_index) <= policy.scene_round_gap
            and len(current) < policy.maximum_scene_beats
        )
        if continues:
            current.append(beat)
        else:
            groups.append([beat])
    return tuple(_scene(tuple(group)) for group in groups)


def project_situated_narrative(
    story: SituatedStory,
    policy: NarrativeProjectionPolicy,
    *,
    trajectory: SituatedPerceptSocialCognitiveTrajectory | None = None,
) -> NarrativeProjection:
    """Project an accepted story without expanding its information authority."""

    if not isinstance(story, SituatedStory):
        raise TypeError("situated narrative projection requires a SituatedStory")
    if not isinstance(policy, NarrativeProjectionPolicy):
        raise TypeError("situated narrative projection requires a NarrativeProjectionPolicy")
    if trajectory is not None and not isinstance(
        trajectory, SituatedPerceptSocialCognitiveTrajectory
    ):
        raise TypeError(
            "situated narrative trajectory must be a "
            "SituatedPerceptSocialCognitiveTrajectory"
        )
    if trajectory is not None and trajectory.final_story != story:
        raise ValueError("situated narrative trajectory final story must equal story")

    story_candidates = (
        _objective_candidates(story)
        if policy.authority is NarrativeAuthority.OBJECTIVE
        else _private_candidates(story, policy)
    )
    candidates = story_candidates + (
        () if trajectory is None else _trajectory_candidates(story, policy, trajectory)
    )
    selected = _select_candidates(candidates, policy)
    ordered = _order_candidates(story, selected, policy)
    beats = _make_beats(ordered, policy)
    scenes = _group_scenes(beats, policy)
    entitlement_ids = {
        entitlement_id for beat in beats for entitlement_id in beat.entitlement_ids
    }
    entitlements = tuple(
        item.entitlement
        for item in ordered
        if item.entitlement.entitlement_id in entitlement_ids
    )
    cut = NarrativeCut(
        _stable_id(
            "cut",
            {
                "policy_hash": policy.content_hash,
                "scene_ids": [item.scene_id for item in scenes],
                "entitlement_ids": [item.entitlement_id for item in entitlements],
            },
        ),
        tuple(item.scene_id for item in scenes),
        entitlements,
    )
    return NarrativeProjection(
        story.content_hash,
        None if trajectory is None else trajectory.content_hash,
        policy,
        beats,
        scenes,
        cut,
    )


__all__ = ("project_situated_narrative",)
