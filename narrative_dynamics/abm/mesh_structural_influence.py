"""R2.M5b: task-specific directed structural importance on a real V19/V24 view.

Only the V19 world and social state own events and relations. This module
measures bounded reach and deletion sensitivity on a read-only, selected
V24 source relation. It cannot grant passage authority, generate a message,
or infer actual temporal delivery from a static multi-hop path.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_contracts import (
    MeshRelation, MeshSnapshot, MeshSociety, source_relation_edges,
)
from narrative_dynamics.abm.mesh_queries import (
    find_mesh_path, mesh_path_is_valid,
)
from narrative_dynamics.abm.situated_mesh import mesh_edges
from narrative_dynamics.abm.situated_network import project_situated_network_snapshot
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_story import SituatedStory

_SCHEMA = "r2-m5b-source-directed-key-player-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_AGENTS = 16

_ROOT_FIELDS = frozenset({
    "schema", "source_owner", "source_network_model_hash",
    "source_world_model_hash", "source_snapshot_hash", "source_story_hash",
    "source_cognitive_hash", "source_social_hash", "source_round",
    "source_mesh_hash", "source_relation", "source_relation_semantics",
    "source_agent_roster", "source_roles", "source_directed_edges",
    "source_passage_id", "source_controller_agent_ids",
    "source_latest_tell_events", "source_latest_transmissions",
    "structural_measurements", "bounded_witnesses",
    "highest_out_reach_candidates", "highest_removal_loss_candidates",
    "query_scope", "sample_size", "normalization",
    "cross_mechanism_ranking", "causal_attribution",
    "confidence_interval", "external_validation",
})
_METRIC_FIELDS = frozenset({
    "agent_id", "role", "declared_passage_controller", "out_degree",
    "in_degree", "out_reachable_agents", "in_reachable_agents",
    "removal_lost_ordered_pairs", "removal_eligible_ordered_pairs",
})


def _digest(x: object) -> bool:
    return type(x) is str and _HASH.fullmatch(x) is not None


def _validate_roster_edges(roster: object, edges: object) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    if (type(roster) is not list or not 3 <= len(roster) <= _MAX_AGENTS
            or any(type(x) is not str or not x for x in roster)
            or roster != sorted(set(roster))):
        raise ValueError("bounded structural query requires distinct ordered source Agent roster")
    if type(edges) is not list:
        raise TypeError("structural edges must be an immutable-source directed list")
    pairs = []
    for edge in edges:
        if (type(edge) is not list or len(edge) != 2
                or any(type(x) is not str for x in edge)
                or edge[0] == edge[1]
                or any(x not in roster for x in edge)):
            raise ValueError("structural directed edge must reference real distinct Agent IDs")
        pairs.append((edge[0], edge[1]))
    if pairs != sorted(set(pairs)):
        raise ValueError("source-directed edges must be sorted unique and not synthesized")
    return tuple(roster), tuple(pairs)


def _adjacency(roster: tuple[str, ...], edges: tuple[tuple[str, str], ...]
              ) -> dict[str, tuple[str, ...]]:
    return {agent: tuple(sorted(dst for src, dst in edges if src == agent))
            for agent in roster}


def _shortest(adj: dict[str, tuple[str, ...]], start: str, target: str,
              excluded: str | None = None) -> tuple[str, ...] | None:
    """A bounded query on existing edges, not a source topology mutation."""
    if start == excluded or target == excluded:
        return None
    if start == target:
        return (start,)
    queue = deque([(start,)])
    seen = {start}
    while queue:
        path = queue.popleft()
        for nxt in adj[path[-1]]:
            if nxt == excluded or nxt in seen:
                continue
            item = path + (nxt,)
            if nxt == target:
                return item
            seen.add(nxt)
            queue.append(item)
    return None


def _native_metrics(roster: tuple[str, ...], roles: dict[str, str],
                    edges: tuple[tuple[str, str], ...],
                    controllers: tuple[str, ...]
                    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    adj = _adjacency(roster, edges)
    witnesses = []
    paths = {}
    for source in roster:
        for target in roster:
            if source == target:
                continue
            path = _shortest(adj, source, target)
            if path is not None:
                paths[(source, target)] = path
                witnesses.append({
                    "source_agent_id": source,
                    "target_agent_id": target,
                    "path_agent_ids": list(path),
                    "hop_count": len(path) - 1,
                })
    metrics = []
    for agent in roster:
        remaining = [x for x in roster if x != agent]
        lost = [
            [a, b] for a in remaining for b in remaining
            if a != b and (a, b) in paths
            and _shortest(adj, a, b, excluded=agent) is None
        ]
        metrics.append({
            "agent_id": agent,
            "role": roles[agent],
            "declared_passage_controller": agent in controllers,
            "out_degree": sum(a == agent for a, b in edges),
            "in_degree": sum(b == agent for a, b in edges),
            "out_reachable_agents": sorted(
                b for a, b in paths if a == agent),
            "in_reachable_agents": sorted(
                a for a, b in paths if b == agent),
            "removal_lost_ordered_pairs": lost,
            "removal_eligible_ordered_pairs": (len(roster) - 1) * (len(roster) - 2),
        })
    return metrics, witnesses


@dataclass(frozen=True)
class DirectedSourceInfluenceReport:
    """Immutable source-scoped, exact-relation measurements; no power ranking."""

    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("structural influence evidence must be JSON text")
        d = json.loads(self._canonical_json)
        if (type(d) is not dict or set(d) != _ROOT_FIELDS
                or d["schema"] != _SCHEMA
                or d["source_owner"] != "DERIVED_V24_FROM_V19_NOT_A_GRAPH_OWNER"
                or d["source_relation"] not in {
                    MeshRelation.RELATIONSHIP.value,
                    MeshRelation.DIRECT_INTERACTION.value,
                }
                or d["source_relation_semantics"] !=
                    "SELECTED_DIRECTED_TOPOLOGY_NOT_EVENT_DELIVERY"
                or d["query_scope"] != "EXACT_SOURCE_ROSTER_UNIVERSAL_SOCIETY"
                or d["sample_size"] != 1
                or d["normalization"] != "NONE"
                or d["cross_mechanism_ranking"] != "FORBIDDEN"
                or d["causal_attribution"] != "UNATTRIBUTED"
                or d["confidence_interval"] != "NOT_ESTIMABLE"
                or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
                or type(d["source_round"]) is not int or d["source_round"] < 0
                or type(d["source_latest_tell_events"]) is not int
                or d["source_latest_tell_events"] < 0
                or type(d["source_latest_transmissions"]) is not int
                or d["source_latest_transmissions"] < 0
                or d["source_latest_transmissions"] > (
                    len(d["source_agent_roster"]) - 1
                    ) * d["source_latest_tell_events"]
                or any(not _digest(d[k]) for k in (
                    "source_network_model_hash", "source_world_model_hash",
                    "source_snapshot_hash", "source_story_hash",
                    "source_cognitive_hash", "source_social_hash",
                    "source_mesh_hash",
                ))):
            raise ValueError("structural source report cannot assert unqualified social power")
        roster, edges = _validate_roster_edges(
            d["source_agent_roster"], d["source_directed_edges"]
        )
        roles = d["source_roles"]
        controllers = d["source_controller_agent_ids"]
        if (type(roles) is not dict or set(roles) != set(roster)
                or any(type(role) is not str or not role for role in roles.values())
                or type(d["source_passage_id"]) is not str or not d["source_passage_id"]
                or type(controllers) is not list or not controllers
                or controllers != sorted(set(controllers))
                or any(c not in roster for c in controllers)):
            raise ValueError("structural report must retain source role and controller identity")
        expected_metrics, expected_witnesses = _native_metrics(
            roster, roles, edges, tuple(controllers)
        )
        if (d["structural_measurements"] != expected_metrics
                or d["bounded_witnesses"] != expected_witnesses
                or any(type(m) is not dict or set(m) != _METRIC_FIELDS
                       for m in d["structural_measurements"])):
            raise ValueError("structural report cannot forge native reach or removal-loss")
        highest_reach = max(len(m["out_reachable_agents"]) for m in expected_metrics)
        max_loss = max(len(m["removal_lost_ordered_pairs"]) for m in expected_metrics)
        if (d["highest_out_reach_candidates"] != sorted(
                m["agent_id"] for m in expected_metrics
                if len(m["out_reachable_agents"]) == highest_reach)
                or d["highest_removal_loss_candidates"] != (
                    sorted(m["agent_id"] for m in expected_metrics
                           if len(m["removal_lost_ordered_pairs"]) == max_loss)
                    if max_loss > 0 else [])):
            raise ValueError("key-player candidates must follow registered native task")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_source_structural_influence(
    model: SituatedNetworkRuntimeModel,
    story: SituatedStory,
    cognitive_state,
    social_state,
    *,
    relation: MeshRelation,
    controlled_passage_id: str,
) -> DirectedSourceInfluenceReport:
    """Observe one native source revision, never invent or rewrite its edges."""
    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("structural observer requires native V19 network runtime model")
    if not isinstance(story, SituatedStory):
        raise TypeError("structural observer requires native V19 immutable story")
    if not isinstance(relation, MeshRelation):
        raise TypeError("structural relation must be a typed MeshRelation")
    if relation is MeshRelation.TRANSMISSION:
        raise ValueError("latest-round realized transmissions are not multi-hop temporal paths")
    world = model.percept_memory_model.cognitive_model.world_model
    passage = next((p for p in world.passages if p.passage_id == controlled_passage_id), None)
    if passage is None or not passage.controller_agent_ids:
        raise ValueError("structural comparison requires declared source passage controllers")
    source = project_situated_network_snapshot(
        model, story, cognitive_state, social_state
    )
    roster = tuple(node.agent_id for node in source.nodes)
    if len(roster) < 3 or len(roster) > _MAX_AGENTS:
        raise ValueError("structural qualification uses bounded native Agent roster")
    universe = MeshSnapshot(source, (
        MeshSociety("entire-source-agent-roster", roster),
    ))
    effective = mesh_edges(universe, relation)
    if effective != source_relation_edges(source, relation):
        raise ValueError("universal society cannot invent or drop source edges")
    roles = {n.agent_id: n.role_id for n in source.nodes}
    metrics, witnesses = _native_metrics(
        roster, roles, effective, passage.controller_agent_ids
    )
    for entry in witnesses:
        a, b = entry["source_agent_id"], entry["target_agent_id"]
        path = find_mesh_path(
            universe, a, b, relation=relation, max_hops=len(roster) - 1,
        )
        if (path is None or tuple(entry["path_agent_ids"]) != path.agent_ids
                or not mesh_path_is_valid(
                    universe, path, a, b, relation=relation,
                    max_hops=len(roster) - 1)):
            raise ValueError("bounded structural witness differs from native V24 query")
    max_reach = max(len(x["out_reachable_agents"]) for x in metrics)
    max_loss = max(len(x["removal_lost_ordered_pairs"]) for x in metrics)
    result = {
        "schema": _SCHEMA,
        "source_owner": "DERIVED_V24_FROM_V19_NOT_A_GRAPH_OWNER",
        "source_network_model_hash": model.content_hash,
        "source_world_model_hash": world.content_hash,
        "source_snapshot_hash": source.content_hash,
        "source_story_hash": story.content_hash,
        "source_cognitive_hash": cognitive_state.content_hash,
        "source_social_hash": social_state.content_hash,
        "source_round": source.round_index,
        "source_mesh_hash": universe.content_hash,
        "source_relation": relation.value,
        "source_relation_semantics": "SELECTED_DIRECTED_TOPOLOGY_NOT_EVENT_DELIVERY",
        "source_agent_roster": list(roster),
        "source_roles": roles,
        "source_directed_edges": [list(pair) for pair in effective],
        "source_passage_id": passage.passage_id,
        "source_controller_agent_ids": list(passage.controller_agent_ids),
        "source_latest_tell_events": source.latest_tell_event_count,
        "source_latest_transmissions": len(source.transmissions),
        "structural_measurements": metrics,
        "bounded_witnesses": witnesses,
        "highest_out_reach_candidates": sorted(
            x["agent_id"] for x in metrics
            if len(x["out_reachable_agents"]) == max_reach
        ),
        "highest_removal_loss_candidates": (
            sorted(x["agent_id"] for x in metrics
                   if len(x["removal_lost_ordered_pairs"]) == max_loss)
            if max_loss > 0 else []
        ),
        "query_scope": "EXACT_SOURCE_ROSTER_UNIVERSAL_SOCIETY",
        "sample_size": 1,
        "normalization": "NONE",
        "cross_mechanism_ranking": "FORBIDDEN",
        "causal_attribution": "UNATTRIBUTED",
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
    }
    return DirectedSourceInfluenceReport(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
