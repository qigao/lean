"""Deterministic POV-safe projection of situated stories into narrative cuts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionKind, SituatedWorldEvent
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_perception_contracts import SituatedPercept
from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeAuthority,
    NarrativeBeat,
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
_UNDISCLOSED_PLACE = "undisclosed"


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
    source_event_id: str
    round_index: int
    sequence: int
    place_id: str
    active_pov_agent_id: str | None
    agent_ids: tuple[str, ...]
    base_kind: NarrativeBeatKind
    support: NarrativeSupportRef
    entitlement: NarrativeEntitlement
    direct_cause_event_ids: tuple[str, ...]


def _entitlement(
    *,
    scope: NarrativeEntitlementScope,
    owner_agent_id: str | None,
    round_index: int,
    facts: tuple[NarrativeFact, ...],
    support: NarrativeSupportRef,
) -> NarrativeEntitlement:
    entitlement_id = _stable_id(
        "entitlement",
        {
            "scope": scope.value,
            "owner_agent_id": owner_agent_id,
            "round_index": round_index,
            "facts": [item.to_dict() for item in facts],
            "support": support.to_dict(),
        },
    )
    return NarrativeEntitlement(
        entitlement_id,
        scope,
        owner_agent_id,
        round_index,
        facts,
        (support,),
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
    # Causal selection reads only the accepted event graph. No world-event field is
    # copied into a private fact, support reference, beat place, actor, kind, or outcome.
    cause_ids_by_event = {
        event.event_id: event.cause_event_ids for event in objective_timeline(story)
    }
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
                    percept.place_id or _UNDISCLOSED_PLACE,
                    agent_id,
                    tuple(disclosed_agents),
                    NarrativeBeatKind.INFORMATION
                    if percept.kind in _INFORMATION_ACTIONS
                    else NarrativeBeatKind.PHYSICAL,
                    support,
                    _entitlement(
                        scope=NarrativeEntitlementScope.PRIVATE,
                        owner_agent_id=agent_id,
                        round_index=percept.round_index,
                        facts=facts,
                        support=support,
                    ),
                    cause_ids_by_event[percept.source_event_id],
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


def _candidate_key(candidate: _Candidate) -> tuple[str, str | None, str]:
    return (
        candidate.source_event_id,
        candidate.active_pov_agent_id,
        candidate.support.artifact_id,
    )


def _candidate_beat_id(candidate: _Candidate) -> str:
    return _stable_id(
        "beat",
        {
            "source_event_id": candidate.source_event_id,
            "active_pov_agent_id": candidate.active_pov_agent_id,
            "support": candidate.support.to_dict(),
        },
    )


def _beat(
    candidate: _Candidate,
    policy: NarrativeProjectionPolicy,
    cause_beat_ids: tuple[str, ...],
) -> NarrativeBeat:
    beat_id = _candidate_beat_id(candidate)
    kind = (
        NarrativeBeatKind.CAUSAL_PAYOFF
        if cause_beat_ids
        else candidate.base_kind
    )
    return NarrativeBeat(
        beat_id,
        kind,
        candidate.round_index,
        candidate.sequence,
        candidate.place_id,
        candidate.active_pov_agent_id,
        candidate.agent_ids,
        policy.weight_for(candidate.base_kind),
        (candidate.support,),
        (candidate.entitlement.entitlement_id,),
        cause_beat_ids,
        source_event_id=candidate.source_event_id,
    )


def _chronological_key(candidate: _Candidate) -> tuple[object, ...]:
    return (
        candidate.round_index,
        candidate.sequence,
        candidate.active_pov_agent_id or "",
        _candidate_beat_id(candidate),
    )


def _lane(candidate: _Candidate) -> str | None:
    return candidate.active_pov_agent_id


def _authorized_causes(
    candidate: _Candidate,
    by_lane_and_event: dict[tuple[str | None, str], tuple[_Candidate, ...]],
) -> tuple[_Candidate, ...]:
    causes = tuple(
        cause
        for source_event_id in candidate.direct_cause_event_ids
        for cause in by_lane_and_event.get((_lane(candidate), source_event_id), ())
    )
    return tuple(sorted(causes, key=_chronological_key))


def _select_candidates(
    candidates: tuple[_Candidate, ...], policy: NarrativeProjectionPolicy
) -> tuple[_Candidate, ...]:
    if not candidates:
        return ()
    by_lane: dict[str | None, list[_Candidate]] = {}
    for candidate in candidates:
        by_lane.setdefault(_lane(candidate), []).append(candidate)
    selected: set[tuple[str, str | None, str]] = set()
    for lane_candidates in by_lane.values():
        ordered = sorted(lane_candidates, key=_chronological_key)
        for candidate in ordered:
            score = policy.weight_for(candidate.base_kind) * 1.0
            if score >= policy.minimum_salience:
                selected.add(_candidate_key(candidate))
        selected.add(_candidate_key(ordered[0]))
        selected.add(_candidate_key(ordered[-1]))
        maximum_gap = policy.maximum_event_omission_gap
        if maximum_gap:
            selected_positions = [
                index
                for index, candidate in enumerate(ordered)
                if _candidate_key(candidate) in selected
            ]
            anchors: list[int] = [selected_positions[0]]
            for right in selected_positions[1:]:
                left = anchors[-1]
                while right - left > maximum_gap:
                    left += maximum_gap
                    selected.add(_candidate_key(ordered[left]))
                    anchors.append(left)
                anchors.append(right)

    by_lane_and_event: dict[tuple[str | None, str], tuple[_Candidate, ...]] = {}
    for candidate in candidates:
        key = (_lane(candidate), candidate.source_event_id)
        by_lane_and_event[key] = by_lane_and_event.get(key, ()) + (candidate,)
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
    selected_event_ids = {item.source_event_id for item in candidates}
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
            key=lambda item: (position[item.source_event_id], _chronological_key(item)),
        )
    )


def _make_beats(
    candidates: tuple[_Candidate, ...], policy: NarrativeProjectionPolicy
) -> tuple[NarrativeBeat, ...]:
    by_lane_and_event: dict[tuple[str | None, str], tuple[_Candidate, ...]] = {}
    for candidate in candidates:
        key = (_lane(candidate), candidate.source_event_id)
        by_lane_and_event[key] = by_lane_and_event.get(key, ()) + (candidate,)
    selected_keys = {_candidate_key(item) for item in candidates}
    beats = []
    for candidate in candidates:
        causes = tuple(
            cause
            for cause in _authorized_causes(candidate, by_lane_and_event)
            if _candidate_key(cause) in selected_keys
        )
        beats.append(
            _beat(
                candidate,
                policy,
                tuple(_candidate_beat_id(item) for item in causes),
            )
        )
    return tuple(beats)


def _keep_causal_pairs_adjacent(
    beats: tuple[NarrativeBeat, ...], policy: NarrativeProjectionPolicy
) -> tuple[NarrativeBeat, ...]:
    if policy.temporal_order is NarrativeTemporalOrder.AUTHORED:
        return beats
    ordered = list(beats)
    for payoff in tuple(beats):
        if not payoff.cause_beat_ids:
            continue
        causes = [item for item in ordered if item.beat_id in payoff.cause_beat_ids]
        if not causes:
            continue
        ordered = [item for item in ordered if item not in causes]
        payoff_index = ordered.index(payoff)
        ordered[payoff_index:payoff_index] = causes
    return tuple(ordered)


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
            beat.place_id == previous.place_id
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
    trajectory: object | None = None,
) -> NarrativeProjection:
    """Project an accepted story without expanding its information authority."""

    if not isinstance(story, SituatedStory):
        raise TypeError("situated narrative projection requires a SituatedStory")
    if not isinstance(policy, NarrativeProjectionPolicy):
        raise TypeError("situated narrative projection requires a NarrativeProjectionPolicy")
    if trajectory is not None:
        raise ValueError("story-only narrative projection does not accept a trajectory")

    candidates = (
        _objective_candidates(story)
        if policy.authority is NarrativeAuthority.OBJECTIVE
        else _private_candidates(story, policy)
    )
    selected = _select_candidates(candidates, policy)
    ordered = _order_candidates(story, selected, policy)
    beats = _keep_causal_pairs_adjacent(_make_beats(ordered, policy), policy)
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
        None,
        policy,
        beats,
        scenes,
        cut,
    )


__all__ = ("project_situated_narrative",)
