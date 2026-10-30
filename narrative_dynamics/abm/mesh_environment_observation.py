"""R2.M2: immutable V19 environment-action evidence over actual story rounds.

This is a read-only, payload-redacted observation of the existing
SituatedStory and V19/V24 source. All action effects are independently
re-executed through the ONE authoritative situated round resolver; no
second world/graph engine or external historical attestation is claimed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_evolution_observation import (
    NetworkLayerView, compare_network_layers,
)
from narrative_dynamics.abm.situated import (
    SituatedActionKind, resolve_situated_round,
)
from narrative_dynamics.abm.situated_perception import can_situated_agents_interact

_SCHEMA = "v19-environment-action-trace-v1"
_MAX_ROUNDS = 12
_MAX_WORLD_OBJECTS = 64
_MAX_EVENTS = 128

_EFFECT_KINDS = frozenset((
    SituatedActionKind.MOVE, SituatedActionKind.TAKE, SituatedActionKind.DROP,
))


@dataclass(frozen=True)
class EnvironmentActionTrace:
    """Immutable canonical report; caller edits cannot forge source evidence."""

    prior_source_hash: str
    next_source_hash: str
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("environment trace must contain immutable canonical JSON")
        data = json.loads(self._canonical_json)
        if (
            type(data) is not dict
            or data.get("schema") != _SCHEMA
            or data.get("causal_attribution") != "UNATTRIBUTED"
            or data.get("story_lineage") != "STORY_EXTENSION"
            or data.get("prior_source_hash") != self.prior_source_hash
            or data.get("next_source_hash") != self.next_source_hash
        ):
            raise ValueError("environment trace must bind exact sources and event scope")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _interaction_edges(view: NetworkLayerView, state: object) -> list[list[str]]:
    perception = view.model.percept_memory_model.perception_model
    agents = tuple(sorted(body.agent_id for body in state.agents))
    return [
        [source, observer]
        for source in agents
        for observer in agents
        if source != observer and can_situated_agents_interact(
            perception, state, source, observer,
        )
    ]


def _location(place_id: str | None, holder_agent_id: str | None) -> dict[str, object]:
    return {"place_id": place_id, "holder_agent_id": holder_agent_id}


def observe_environment_action_trace(
    before: NetworkLayerView,
    after: NetworkLayerView,
) -> EnvironmentActionTrace:
    """Require a real V19 story extension and re-execute its world effects.

    Different initial conditions are *counterfactuals*, not an executable
    environment history. All per-round MOVE/TAKE/DROP effects must match
    successful, typed V19 source events. Failed actions cannot alter their
    owner; real TELL after a move remains a separate content/transmission
    event and its message is never serialized.
    """
    observed = compare_network_layers(before, after)
    report = observed.to_dict()
    if observed.story_lineage != "STORY_EXTENSION":
        raise ValueError("environment action trace requires an exact real V19 story extension")
    initial_round = len(before.story.rounds)
    incoming = after.story.rounds[initial_round:]
    if not incoming or len(incoming) > _MAX_ROUNDS:
        raise ValueError("environment trace requires 1 to 12 new real story rounds")
    if sum(len(r.events) for r in incoming) > _MAX_EVENTS:
        raise ValueError("environment trace exceeds bounded source event count")
    world_model = before.model.percept_memory_model.cognitive_model.world_model
    if len(world_model.objects) > _MAX_WORLD_OBJECTS:
        raise ValueError("environment trace exceeds bounded world object count")
    if before.story.current_state != incoming[0].prior_state:
        raise ValueError("environment trace source does not match the first prior world")

    rounds: list[dict[str, object]] = []
    prior_world = before.story.current_state
    for item in incoming:
        if item.prior_state != prior_world:
            raise ValueError("environment trace has a disconnected V19 story state")
        # The V19 story constructor validates event identity/observation
        # integrity, but the whole deterministic action effect must also be
        # independently reproduced to exclude false MOVE/TAKE/DROP outcomes.
        actual = resolve_situated_round(world_model, prior_world, item.intents)
        if actual != item:
            raise ValueError("environment trace recorded effects differ from authoritative V19 execution")
        old_agents = {x.agent_id: x.place_id for x in item.prior_state.agents}
        new_agents = {x.agent_id: x.place_id for x in item.next_state.agents}
        old_objects = {x.object_id: x for x in item.prior_state.objects}
        new_objects = {x.object_id: x for x in item.next_state.objects}
        changed_agents = {k for k in old_agents if old_agents[k] != new_agents[k]}
        changed_objects = {k for k in old_objects if old_objects[k] != new_objects[k]}
        matched_agents: set[str] = set()
        matched_objects: set[str] = set()
        effects: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        actual_tell: list[str] = []
        boundary_events: list[str] = []
        for event in item.events:
            if event.kind is SituatedActionKind.TELL and event.success:
                actual_tell.append(event.event_id)
            if event.kind in (SituatedActionKind.OPEN_PASSAGE, SituatedActionKind.CLOSE_PASSAGE):
                if event.success:
                    boundary_events.append(event.event_id)
            if event.kind in _EFFECT_KINDS and not event.success:
                rejected.append({
                    "event_id": event.event_id,
                    "event_hash": event.content_hash,
                    "kind": event.kind.value,
                    "actor_agent_id": event.actor_agent_id,
                    "outcome": event.outcome,
                })
            if not event.success or event.kind not in _EFFECT_KINDS:
                continue
            if event.kind is SituatedActionKind.MOVE:
                key = event.actor_agent_id
                if key not in changed_agents or key in matched_agents:
                    raise ValueError("environment MOVE has no unique sourced Agent displacement")
                matched_agents.add(key)
                effects.append({
                    "event_id": event.event_id,
                    "event_hash": event.content_hash,
                    "round_index": event.round_index,
                    "kind": event.kind.value,
                    "actor_agent_id": key,
                    "passage_id": event.target_id,
                    "before_place": old_agents[key],
                    "after_place": new_agents[key],
                })
            else:
                key = event.target_id
                if key not in changed_objects or key in matched_objects:
                    raise ValueError("environment object transfer lacks unique source event")
                matched_objects.add(key)
                before_obj, after_obj = old_objects[key], new_objects[key]
                effects.append({
                    "event_id": event.event_id,
                    "event_hash": event.content_hash,
                    "round_index": event.round_index,
                    "kind": event.kind.value,
                    "actor_agent_id": event.actor_agent_id,
                    "object_id": key,
                    "before": _location(before_obj.place_id, before_obj.holder_agent_id),
                    "after": _location(after_obj.place_id, after_obj.holder_agent_id),
                })
        if changed_agents != matched_agents or changed_objects != matched_objects:
            raise ValueError("environment owner changed without successful MOVE/TAKE/DROP event")
        if before.model.percept_memory_model.perception_model.world_model.content_hash != world_model.content_hash:
            raise ValueError("environment observation perception world model changed")
        channels = Counter(obs.channel.value for obs in item.observations)
        rounds.append({
            "round_index": item.next_state.round_index,
            "prior_world_hash": item.prior_state.content_hash,
            "next_world_hash": item.next_state.content_hash,
            "story_round_hash": item.content_hash,
            "sourced_effects": effects,
            "failed_environment_actions": rejected,
            "successful_tell_event_ids": actual_tell,
            "successful_boundary_event_ids": boundary_events,
            "observation_channels": dict(sorted(channels.items())),
            "direct_interaction_before": _interaction_edges(before, item.prior_state),
            "direct_interaction_after": _interaction_edges(after, item.next_state),
        })
        prior_world = item.next_state
    if prior_world != after.story.current_state:
        raise ValueError("environment trace final world does not match V19 source")

    data = {
        "schema": _SCHEMA,
        "scope": "re-executable V19 world effects only, not authenticated historical causation",
        "causal_attribution": "UNATTRIBUTED",
        "story_lineage": "STORY_EXTENSION",
        "prior_source_hash": before.mesh.source_snapshot.content_hash,
        "next_source_hash": after.mesh.source_snapshot.content_hash,
        "prior_story_hash": before.story.content_hash,
        "next_story_hash": after.story.content_hash,
        "actual_source_rounds": rounds,
        "selected_relation_graphs": report["selected_relation_graphs"],
        "physical_boundary_changes": report["boundary"]["physical_passages"],
        "environment_changes": report["environment"],
        "multi_agent_content": {
            "successful_tell_event_ids": report["interaction_content"]["new_successful_tell_event_ids"],
            "changed_claim_count": len(report["interaction_content"]["claim_changes"]),
            "changed_trust_pair_count": len(report["interaction_content"]["social_trust_changes"]),
        },
    }
    return EnvironmentActionTrace(
        before.mesh.source_snapshot.content_hash,
        after.mesh.source_snapshot.content_hash,
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )
