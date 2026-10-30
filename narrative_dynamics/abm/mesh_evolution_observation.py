"""R2.M0 read-only observations of V19 boundary, environment and content deltas.

All state is held by the existing V19 world/story/cognitive/social owners.
V24 MeshSnapshot is a source-backed, relation-selected derived view. This
module compares those immutable inputs; it does NOT admit an event, infer
a cause, reveal message payloads or own a second graph.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_contracts import (
    MeshBridge, MeshRelation, MeshSnapshot, MeshSociety, source_relation_edges,
)
from narrative_dynamics.abm.mesh_topology_events import observe_topology
from narrative_dynamics.abm.situated_cognition_contracts import SituatedCognitiveState
from narrative_dynamics.abm.situated_mesh import mesh_edges
from narrative_dynamics.abm.situated_network import project_situated_network_snapshot
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_social_memory_contracts import SituatedSocialMemoryState
from narrative_dynamics.abm.situated_story import SituatedStory

_SCHEMA = "v19-network-three-driver-observation-v1"
_MAX_AGENTS = 24
_MAX_SOCIETIES = 12
_MAX_OBSERVED_EVENTS = 128


@dataclass(frozen=True)
class NetworkLayerView:
    """Immutable references only; each source is checked against V19 projector."""

    model: SituatedNetworkRuntimeModel
    story: SituatedStory
    cognition: SituatedCognitiveState
    social: SituatedSocialMemoryState
    mesh: MeshSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.model, SituatedNetworkRuntimeModel):
            raise TypeError("network view requires authoritative V19 runtime model")
        if not isinstance(self.story, SituatedStory):
            raise TypeError("network view requires a V19 SituatedStory")
        if not isinstance(self.cognition, SituatedCognitiveState):
            raise TypeError("network view requires a V19 cognitive state")
        if not isinstance(self.social, SituatedSocialMemoryState):
            raise TypeError("network view requires a V19 social-memory state")
        if not isinstance(self.mesh, MeshSnapshot):
            raise TypeError("network view requires V24 MeshSnapshot")
        source = project_situated_network_snapshot(
            self.model, self.story, self.cognition, self.social,
        )
        if self.mesh.source_snapshot != source:
            raise ValueError("network view does not match the authoritative V19 source")
        if (len(source.nodes) > _MAX_AGENTS or len(self.mesh.societies) > _MAX_SOCIETIES):
            raise ValueError("R2 finite network observation budget exceeded")

    def identities(self) -> dict[str, object]:
        return {
            "model_hash": self.model.content_hash,
            "round_index": self.story.current_state.round_index,
            "world_state_hash": self.story.current_state.content_hash,
            "story_hash": self.story.content_hash,
            "cognitive_state_hash": self.cognition.content_hash,
            "social_state_hash": self.social.content_hash,
            "source_snapshot_hash": self.mesh.source_snapshot.content_hash,
            "mesh_hash": self.mesh.content_hash,
        }


def observe_network_layers(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
    cognition: SituatedCognitiveState,
    social: SituatedSocialMemoryState,
    societies: tuple[MeshSociety, ...],
    bridges: tuple[MeshBridge, ...] = (),
) -> NetworkLayerView:
    """Construct a read-only view from the existing authoritative V19 states."""
    source = project_situated_network_snapshot(model, story, cognition, social)
    return NetworkLayerView(
        model, story, cognition, social, MeshSnapshot(source, societies, bridges),
    )


@dataclass(frozen=True)
class NetworkChangeReport:
    """Versioned, payload-redacted result; never an editable source owner."""

    before_source_hash: str
    after_source_hash: str
    story_lineage: str
    _data: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self._data)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _story_lineage(previous: SituatedStory, current: SituatedStory) -> str:
    if previous == current:
        return "SAME_STORY"
    if previous.initial_state != current.initial_state:
        return "DIFFERENT_INITIAL_CONDITIONS"
    if (
        len(current.rounds) >= len(previous.rounds)
        and current.rounds[:len(previous.rounds)] == previous.rounds
    ):
        return "STORY_EXTENSION"
    return "DIVERGENT_STORY_BRANCH"


def _ordered_pairs(before: set[tuple[str, str]], after: set[tuple[str, str]]) -> dict[str, object]:
    return {
        "added": [list(e) for e in sorted(after - before)],
        "removed": [list(e) for e in sorted(before - after)],
    }


def _membership(view: NetworkLayerView) -> dict[str, tuple[str, ...]]:
    return {s.society_id: s.member_agent_ids for s in view.mesh.societies}


def _bridge_identity(b: MeshBridge) -> tuple[str, str, str, str, str]:
    return (b.relation.value, b.source_society_id, b.target_society_id,
            b.source_agent_id, b.target_agent_id)


def _bridge_changes(before: NetworkLayerView, after: NetworkLayerView) -> dict[str, object]:
    old = {_bridge_identity(b): b for b in before.mesh.bridges}
    new = {_bridge_identity(b): b for b in after.mesh.bridges}
    return {
        "added": [new[k].to_dict() for k in sorted(new.keys() - old.keys())],
        "removed": [old[k].to_dict() for k in sorted(old.keys() - new.keys())],
    }


def _claims(view: NetworkLayerView) -> dict[str, str]:
    return {c.claim_id: c.content_hash for c in view.social.claims}


def _changes_by_id(before: dict[str, str], after: dict[str, str], id_field: str) -> list[dict[str, object]]:
    return [
        {id_field: key, "before_hash": before.get(key), "after_hash": after.get(key)}
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    ]


def compare_network_layers(before: NetworkLayerView, after: NetworkLayerView) -> NetworkChangeReport:
    """Report facts from the existing V19 owners and V24 projections.

    STORY_EXTENSION identifies a real common event-chain prefix; it does NOT
    attribute any downstream change to one cause. Other comparisons may be
    counterfactual snapshots, same-story checkpoints or diverging branches.
    """
    if not isinstance(before, NetworkLayerView) or not isinstance(after, NetworkLayerView):
        raise TypeError("R2 comparison requires typed authoritative NetworkLayerView inputs")
    # Revalidate against the source even if callers bypass dataclass freezing.
    before.__post_init__()
    after.__post_init__()
    if before.model.content_hash != after.model.content_hash:
        raise ValueError("R2 cannot compare incompatible V19 runtime models")
    previous_ids = tuple(x.agent_id for x in before.mesh.source_snapshot.nodes)
    current_ids = tuple(x.agent_id for x in after.mesh.source_snapshot.nodes)
    if previous_ids != current_ids:
        raise ValueError("R2 comparison needs the exact same Agent roster")
    if after.story.current_state.round_index < before.story.current_state.round_index:
        raise ValueError("R2 comparison must not reverse V19 logical round order")

    old_world, new_world = before.story.current_state, after.story.current_state
    old_passages = {p.passage_id: p.open for p in old_world.passages}
    new_passages = {p.passage_id: p.open for p in new_world.passages}
    if old_passages.keys() != new_passages.keys():
        raise ValueError("R2 physical boundary must keep the same world passage roster")
    physical = [
        {"passage_id": key, "before_open": old_passages[key], "after_open": new_passages[key]}
        for key in sorted(old_passages)
        if old_passages[key] != new_passages[key]
    ]

    old_members, new_members = _membership(before), _membership(after)
    membership = [
        {"society_id": key, "before_members": list(old_members.get(key, ())),
         "after_members": list(new_members.get(key, ()))}
        for key in sorted(old_members.keys() | new_members.keys())
        if old_members.get(key) != new_members.get(key)
    ]
    bridges = _bridge_changes(before, after)

    old_places = {p.agent_id: p.place_id for p in old_world.agents}
    new_places = {p.agent_id: p.place_id for p in new_world.agents}
    if old_places.keys() != new_places.keys():
        raise ValueError("R2 physical environment needs an identical Agent roster")
    movements = [
        {"agent_id": key, "before_place": old_places[key], "after_place": new_places[key]}
        for key in sorted(old_places)
        if old_places[key] != new_places[key]
    ]
    old_objects = {o.object_id: o.content_hash for o in old_world.objects}
    new_objects = {o.object_id: o.content_hash for o in new_world.objects}
    object_changes = _changes_by_id(old_objects, new_objects, "object_id")

    lineage = _story_lineage(before.story, after.story)
    new_rounds = after.story.rounds[len(before.story.rounds):] if lineage == "STORY_EXTENSION" else ()
    if sum(len(round_result.events) for round_result in new_rounds) > _MAX_OBSERVED_EVENTS:
        raise ValueError("R2 observed story-event budget exceeded")
    actual_events = [
        {"event_id": event.event_id, "round_index": event.round_index,
         "kind": event.kind.value, "success": event.success}
        for round_result in new_rounds for event in round_result.events
    ]
    observation_counts = Counter(
        obs.channel.value for round_result in new_rounds
        for obs in round_result.observations
    )
    tell_events = [
        item["event_id"] for item in actual_events
        if item["kind"] == "tell" and item["success"] is True
    ]
    old_minds = {
        m.agent_id: stable_content_hash(m.belief.to_dict())
        for m in before.cognition.minds
    }
    new_minds = {
        m.agent_id: stable_content_hash(m.belief.to_dict())
        for m in after.cognition.minds
    }
    if old_minds.keys() != new_minds.keys():
        raise ValueError("R2 cannot change cognitive Agent ownership")
    belief_changes = _changes_by_id(old_minds, new_minds, "agent_id")
    claim_changes = _changes_by_id(_claims(before), _claims(after), "claim_id")
    old_trust = {
        (s.source_agent_id, s.observer_agent_id): s.trust for s in before.social.relationships
    }
    new_trust = {
        (s.source_agent_id, s.observer_agent_id): s.trust for s in after.social.relationships
    }
    if old_trust.keys() != new_trust.keys():
        raise ValueError("R2 V19 social owner must preserve its directed-pair roster")
    threshold = before.model.relationship_trust_threshold
    trust_changes = [
        {"source_agent_id": source, "observer_agent_id": observer,
         "trust_before": old_trust[(source, observer)],
         "trust_after": new_trust[(source, observer)],
         "effective_before": old_trust[(source, observer)] >= threshold,
         "effective_after": new_trust[(source, observer)] >= threshold}
        for source, observer in sorted(old_trust)
        if old_trust[(source, observer)] != new_trust[(source, observer)]
    ]
    added_evidence_count = len(
        set(after.social.processed_evidence_ids) -
        set(before.social.processed_evidence_ids)
    )
    removed_evidence_count = len(
        set(before.social.processed_evidence_ids) -
        set(after.social.processed_evidence_ids)
    )

    graphs: dict[str, object] = {}
    for relation in MeshRelation:
        old_source = set(source_relation_edges(before.mesh.source_snapshot, relation))
        new_source = set(source_relation_edges(after.mesh.source_snapshot, relation))
        old_projected = set(mesh_edges(before.mesh, relation))
        new_projected = set(mesh_edges(after.mesh, relation))
        old_metrics = observe_topology(before.mesh, relation=relation)
        new_metrics = observe_topology(after.mesh, relation=relation)
        graphs[relation.value] = {
            "source_edges": _ordered_pairs(old_source, new_source),
            "projected_edges": _ordered_pairs(old_projected, new_projected),
            "reachable_pairs_before": old_metrics["reachable_ordered_pair_count"],
            "reachable_pairs_after": new_metrics["reachable_ordered_pair_count"],
            "diameter_before": old_metrics["directed_diameter"],
            "diameter_after": new_metrics["directed_diameter"],
            "strong_component_count_before": len(old_metrics["strong_components"]),
            "strong_component_count_after": len(new_metrics["strong_components"]),
        }
    boundary = {
        "physical_passages": physical,
        "society_memberships": membership,
        "declared_bridges": bridges,
    }
    environment = {
        "agent_movements": movements,
        "world_object_revisions": object_changes,
        "perception_model_hash": before.model.percept_memory_model.perception_model.content_hash,
    }
    content = {
        # Event IDs and channel COUNTS only; no private message or observation payload.
        "new_story_events": actual_events,
        "new_successful_tell_event_ids": tell_events,
        "new_observations_by_channel": dict(sorted(observation_counts.items())),
        "belief_changes": belief_changes,
        "claim_changes": claim_changes,
        "social_trust_changes": trust_changes,
        "new_processed_evidence_count": added_evidence_count,
        "removed_processed_evidence_count": removed_evidence_count,
    }
    observed_axes = {
        "physical_boundary": bool(physical),
        "society_boundary": bool(membership or bridges["added"] or bridges["removed"]),
        "environment": bool(movements or object_changes),
        "multi_agent_content": bool(
            tell_events or claim_changes or belief_changes
            or added_evidence_count or removed_evidence_count
        ),
        "social_trust": bool(trust_changes),
    }
    data = {
        "schema": _SCHEMA,
        "scope": "bounded V19 observed source differences, not causal attribution",
        "causal_attribution": "UNATTRIBUTED",
        "story_lineage": lineage,
        "before": before.identities(),
        "after": after.identities(),
        "observed_axes": observed_axes,
        "boundary": boundary,
        "environment": environment,
        "interaction_content": content,
        "selected_relation_graphs": graphs,
    }
    return NetworkChangeReport(
        before.mesh.source_snapshot.content_hash,
        after.mesh.source_snapshot.content_hash,
        lineage, data,
    )
