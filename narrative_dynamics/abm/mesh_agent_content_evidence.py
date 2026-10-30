"""R2.M3: source-authenticated *in-process* evidence transitions, read-only.

A TELL event, private percept, cognitive belief revision, admitted social
evidence, claim revision and directed trust update are DISTINCT facts. Reuse
the existing V19 perception/social transition implementations and never
construct a competing social memory, graph, or shared-content owner.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionKind, resolve_situated_round
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedCognitiveState, validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    SituatedPerceptSocialCognitiveRoundResult, _social_evidence,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptFidelity,
)
from narrative_dynamics.abm.situated_social_memory import (
    advance_situated_social_memory,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedSocialEvidenceKind, SituatedSocialMemoryModel,
    SituatedSocialMemoryState, validate_situated_social_memory_state,
)
from narrative_dynamics.abm.situated_story import SituatedStory

_SCHEMA = "v19-agent-content-evidence-trace-v1"
_MAX_ROUNDS = 20
_MAX_EVENTS = 160
_MAX_AGENTS = 24
_MAX_EVIDENCE = 128


@dataclass(frozen=True)
class AgentContentEvidenceTrace:
    """Immutable, privacy-redacted canonical source evidence report."""

    prior_story_hash: str
    next_social_state_hash: str
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("Agent content report requires immutable canonical JSON")
        payload = json.loads(self._canonical_json)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _SCHEMA
            or payload.get("causal_attribution") != "UNATTRIBUTED"
            or payload.get("prior_story_hash") != self.prior_story_hash
            or payload.get("next_social_state_hash") != self.next_social_state_hash
        ):
            raise ValueError("Agent content evidence report must bind exact source and scope")

    def to_dict(self) -> dict[str, object]:
        # Never expose the stored immutable source through mutable return values.
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _belief_hashes(state: SituatedCognitiveState) -> dict[str, str]:
    return {mind.agent_id: stable_content_hash(mind.belief.to_dict()) for mind in state.minds}


def _claim_index(state: SituatedSocialMemoryState) -> dict[str, object]:
    return {claim.claim_id: claim for claim in state.claims}


def _trust_index(state: SituatedSocialMemoryState) -> dict[tuple[str, str], object]:
    return {(rel.observer_agent_id, rel.source_agent_id): rel for rel in state.relationships}


def observe_agent_content_evidence(
    memory_model: SituatedPerceptMemoryCognitiveModel,
    social_model: SituatedSocialMemoryModel,
    prior_story: SituatedStory,
    prior_cognition: SituatedCognitiveState,
    prior_social: SituatedSocialMemoryState,
    result: SituatedPerceptSocialCognitiveRoundResult,
) -> AgentContentEvidenceTrace:
    """Verify a *real* V19 percept/social round then report redacted changes.

    All admitted evidence must be exactly reproducible from actor-specific,
    source-backed V19 percepts and the original social evidence learner.
    This does not re-run SQLite-based cognitive decisions; the supplied
    cognitive round is checked against source lineage and its resolved
    world action, and its belief revisions are reported, not independently
    proved or attributed to a particular message.
    """
    if not isinstance(memory_model, SituatedPerceptMemoryCognitiveModel):
        raise TypeError("content observation requires the V19 percept memory model")
    if not isinstance(social_model, SituatedSocialMemoryModel):
        raise TypeError("content observation requires authoritative social memory model")
    if not isinstance(prior_story, SituatedStory):
        raise TypeError("content observation requires SituatedStory")
    if not isinstance(prior_cognition, SituatedCognitiveState):
        raise TypeError("content observation requires prior SituatedCognitiveState")
    if not isinstance(prior_social, SituatedSocialMemoryState):
        raise TypeError("content observation requires prior SituatedSocialMemoryState")
    if not isinstance(result, SituatedPerceptSocialCognitiveRoundResult):
        raise TypeError("content observation requires a real V19 social-cognitive round result")
    if social_model.memory_cognitive_model.cognitive_model != memory_model.cognitive_model:
        raise ValueError("content observation social and percept models must share cognition")
    if prior_story.perception_model != memory_model.perception_model:
        raise ValueError("content observation requires the same source perception model")
    if result.percept_memory_model_hash != memory_model.content_hash or (
        result.social_model_hash != social_model.content_hash
    ):
        raise ValueError("content observation transition must bind exact model hashes")
    validate_situated_cognitive_state(
        memory_model.cognitive_model, prior_story, prior_cognition,
    )
    validate_situated_social_memory_state(
        social_model, prior_cognition, prior_social,
    )
    cognitive = result.cognitive_round
    if (
        cognitive.cognitive_round.prior_state != prior_cognition
        or cognitive.cognitive_round.prior_story_hash != prior_story.content_hash
        or result.social_update.prior_state != prior_social
    ):
        raise ValueError("content observation has stale V19 cognition, story or social source")
    next_story = result.next_story
    if (
        next_story.model_hash != prior_story.model_hash
        or next_story.initial_state != prior_story.initial_state
        or next_story.perception_model != prior_story.perception_model
        or len(next_story.rounds) != len(prior_story.rounds) + 1
        or next_story.rounds[:-1] != prior_story.rounds
    ):
        raise ValueError("content observation must extend the exact V19 source story")
    if len(prior_story.rounds) > _MAX_ROUNDS or len(next_story.rounds[-1].events) > _MAX_AGENTS:
        raise ValueError("content observation exceeds bounded V19 story rounds")
    actors = tuple(sorted(a.agent_id for a in memory_model.cognitive_model.world_model.agents))
    if len(actors) > _MAX_AGENTS or len(prior_story.rounds) * len(actors) > _MAX_EVENTS:
        raise ValueError("content observation exceeds bounded Agent/story budget")
    actual_world_round = resolve_situated_round(
        memory_model.cognitive_model.world_model,
        prior_story.current_state,
        tuple(decision.intent for decision in result.decisions),
    )
    if actual_world_round != next_story.rounds[-1]:
        raise ValueError("content observation claimed a non-reproducible V19 world event")
    validate_situated_cognitive_state(
        memory_model.cognitive_model, next_story, result.next_cognitive_state,
    )
    validate_situated_social_memory_state(
        social_model, result.next_cognitive_state, result.next_social_state,
    )

    # Reuse the original exact-only percept evidence generator and original
    # claims/trust owner. No second claims algorithm or graph construction.
    candidates = _social_evidence(social_model, prior_story, cognitive)
    reproduced = advance_situated_social_memory(
        social_model, prior_cognition, prior_social,
        result.next_cognitive_state, candidates,
    )
    if reproduced != result.social_update:
        raise ValueError("content observation social claims or trust are not source-reproducible")
    evidence = result.social_update.admitted_evidence
    if len(evidence) > _MAX_EVIDENCE:
        raise ValueError("content observation has excessive admitted evidence")

    events = {
        event.event_id: event
        for round_result in prior_story.rounds
        for event in round_result.events
    }
    private_percepts = {
        agent_id: {percept.percept_id: percept for percept in
                   perceptual_timeline(memory_model.perception_model, prior_story, agent_id)}
        for agent_id in actors
    }
    # Privacy-safe per-event count: a DETECTED percept does NOT reveal a
    # message, topic symbol, actor or memory body.
    tell_events: list[dict[str, object]] = []
    for event in sorted(events.values(), key=lambda e: (e.round_index, e.sequence)):
        if event.kind is not SituatedActionKind.TELL or not event.success:
            continue
        observed = Counter(
            percept.fidelity.value
            for agent_id in actors if agent_id != event.actor_agent_id
            for percept in private_percepts[agent_id].values()
            if percept.source_event_id == event.event_id
        )
        tell_events.append({
            "event_id": event.event_id,
            "event_hash": event.content_hash,
            "round_index": event.round_index,
            "actor_agent_id": event.actor_agent_id,
            "other_agent_perception_by_fidelity": dict(sorted(observed.items())),
        })

    admitted: list[dict[str, object]] = []
    for item in evidence:
        percept = private_percepts.get(item.observer_agent_id, {}).get(item.memory_id)
        event = events.get(item.event_id)
        if (
            percept is None or event is None
            or percept.source_event_id != item.event_id
            or percept.source_event_hash != event.content_hash
            or percept.fidelity is not SituatedPerceptFidelity.EXACT
            or (
                item.kind is SituatedSocialEvidenceKind.TESTIMONY
                and (event.kind is not SituatedActionKind.TELL
                     or not event.success
                     or event.actor_agent_id != item.source_agent_id
                     or item.observer_agent_id == item.source_agent_id)
            )
            or (
                item.kind is SituatedSocialEvidenceKind.VERIFICATION
                and (event.kind is not SituatedActionKind.INSPECT
                     or not event.success
                     or event.actor_agent_id != item.observer_agent_id
                     or item.source_agent_id is not None)
            )
        ):
            raise ValueError("admitted social evidence has no exact private source percept")
        admitted.append({
            "evidence_id": item.evidence_id,
            "evidence_hash": item.content_hash,
            "event_id": item.event_id,
            "percept_id": item.memory_id,
            "kind": item.kind.value,
            "observer_agent_id": item.observer_agent_id,
            "source_agent_id": item.source_agent_id,
            "round_index": item.round_index,
        })

    old_claims, new_claims = _claim_index(prior_social), _claim_index(result.next_social_state)
    claims = [
        {
            "claim_id": k,
            "observer_agent_id": c.observer_agent_id if c else old_claims[k].observer_agent_id,
            "source_agent_id": c.source_agent_id if c else old_claims[k].source_agent_id,
            "before_hash": old_claims[k].content_hash if k in old_claims else None,
            "after_hash": new_claims[k].content_hash if k in new_claims else None,
            "status_before": old_claims[k].status.value if k in old_claims else None,
            "status_after": new_claims[k].status.value if k in new_claims else None,
        }
        for k in sorted(old_claims.keys() | new_claims.keys())
        for c in (new_claims.get(k),)
        if old_claims.get(k) != new_claims.get(k)
    ]
    old_trust, new_trust = _trust_index(prior_social), _trust_index(result.next_social_state)
    if old_trust.keys() != new_trust.keys():
        raise ValueError("content transition cannot invent or remove social relationship owners")
    trust_changes = [
        {
            "observer_agent_id": obs,
            "source_agent_id": src,
            "trust_before": old_trust[(obs, src)].trust,
            "trust_after": new_trust[(obs, src)].trust,
            "relation_before_hash": old_trust[(obs, src)].content_hash,
            "relation_after_hash": new_trust[(obs, src)].content_hash,
        }
        for obs, src in sorted(old_trust)
        if old_trust[(obs, src)] != new_trust[(obs, src)]
    ]
    prior_beliefs = _belief_hashes(prior_cognition)
    next_beliefs = _belief_hashes(result.next_cognitive_state)
    if prior_beliefs.keys() != next_beliefs.keys():
        raise ValueError("content transition cognitive Agent roster changed")
    beliefs = [
        {"agent_id": agent_id, "before_hash": prior_beliefs[agent_id],
         "after_hash": next_beliefs[agent_id]}
        for agent_id in sorted(prior_beliefs)
        if prior_beliefs[agent_id] != next_beliefs[agent_id]
    ]
    data = {
        "schema": _SCHEMA,
        "scope": "finite exact V19 private percept/social evidence; not external historical proof",
        "causal_attribution": "UNATTRIBUTED",
        "prior_story_hash": prior_story.content_hash,
        "next_story_hash": next_story.content_hash,
        "prior_cognitive_state_hash": prior_cognition.content_hash,
        "next_cognitive_state_hash": result.next_cognitive_state.content_hash,
        "prior_social_state_hash": prior_social.content_hash,
        "next_social_state_hash": result.next_social_state.content_hash,
        "transition_hash": result.content_hash,
        "source_tell_events": tell_events,
        "admitted_evidence": admitted,
        "claim_revisions": claims,
        "belief_revisions": beliefs,
        "directed_trust_revisions": trust_changes,
        "counts": {
            "actual_tell_events": len(tell_events),
            "admitted_testimony": sum(x.kind is SituatedSocialEvidenceKind.TESTIMONY for x in evidence),
            "admitted_verification": sum(x.kind is SituatedSocialEvidenceKind.VERIFICATION for x in evidence),
            "claim_revisions": len(claims),
            "belief_revisions": len(beliefs),
            "trust_revisions": len(trust_changes),
        },
    }
    return AgentContentEvidenceTrace(
        prior_story.content_hash,
        result.next_social_state.content_hash,
        json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
    )
