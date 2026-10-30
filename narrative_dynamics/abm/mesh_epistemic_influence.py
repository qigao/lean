"""R2.M5c: source-replayed epistemic influence, distinct from authority and reach.

One original V19 world owns passage permissions, TELL/INSPECT events,
private percepts, cognitive beliefs, claims and directional trust. V24
projects read-only selected relations. Never infer enacted future behavior
from an adoption threshold or static graph path.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_agent_content_evidence import observe_agent_content_evidence
from narrative_dynamics.abm.mesh_contracts import MeshRelation, source_relation_edges
from narrative_dynamics.abm.mesh_structural_influence import observe_source_structural_influence
from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_network import project_situated_network_snapshot
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import SituatedStory, replay_situated_story

_SCHEMA = "r2-m5c-source-epistemic-role-authority-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_ACTORS = ("alice", "bob")
_BRANCHES = ("control_wait", "authorized_open")
_ACTION_ORDER = (
    SituatedActionKind.INSPECT,
    None,
    SituatedActionKind.MOVE,
    SituatedActionKind.TELL,
    SituatedActionKind.MOVE,
    SituatedActionKind.INSPECT,
)


def _digest(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _action(story: SituatedStory, name: str):
    return next(event for result in story.rounds for event in result.events
                if event.action_id == name)


def _trust(state, observer: str, source: str) -> float:
    return next(item.trust for item in state.relationships
                if item.observer_agent_id == observer
                and item.source_agent_id == source)


def _validate_schedules(control, intervention) -> None:
    if (type(control) is not tuple or type(intervention) is not tuple
            or any(type(s) is not tuple or len(s) != 1
                   or not isinstance(s[0], SituatedActionIntent)
                   for schedule in (control, intervention) for s in schedule)
            or len(control) != 6 or len(intervention) != 6):
        raise ValueError("M5c needs six frozen single-intent native V19 rounds per arm")
    for index in range(6):
        a, b = control[index][0], intervention[index][0]
        if index != 1 and a != b:
            raise ValueError("M5c paired schedule changed more than passage opening")
        kind = _ACTION_ORDER[index]
        if kind is not None and (a.kind is not kind or b.kind is not kind):
            raise ValueError("M5c action schedule must preserve native INSPECT/MOVE/TELL phases")
    waiting, opening = control[1][0], intervention[1][0]
    if (waiting.kind is not SituatedActionKind.WAIT
            or opening.kind is not SituatedActionKind.OPEN_PASSAGE
            or opening.target_id != "records-open"
            or waiting.agent_id != "alice" or opening.agent_id != "alice"):
        raise ValueError("M5c single intervention must be Alice WAIT vs authorized OPEN")
    first, movement, tell, other_move, verify = (
        control[0][0], control[2][0], control[3][0],
        control[4][0], control[5][0],
    )
    if (first.agent_id != "alice" or first.target_id != "memo"
            or movement.agent_id != "alice" or movement.target_id != "records-open"
            or tell.agent_id != "alice" or not tell.message
            or len(tell.source_event_ids) != 1
            or other_move.agent_id != "bob" or other_move.target_id != "open-records"
            or verify.agent_id != "bob" or verify.target_id != "memo"):
        raise ValueError("M5c must use a real source-linked Alice TELL and Bob private INSPECT")


@dataclass(frozen=True)
class SourceEpistemicInfluenceReport:
    """Strict redacted report of one finite, deterministic paired experiment."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("M5c source report requires canonical JSON text")
        x = json.loads(self._canonical_json)
        keys = {
            "schema", "source_owner", "source_model_hash", "source_world_hash",
            "source_memory_model_hash", "source_social_model_hash",
            "source_perception_model_hash", "initial_world_hash",
            "initial_story_hash", "controller_agent_ids", "resource_controller",
            "structural_source_snapshot_hash", "structural_relation",
            "structural_directed_edges", "structural_out_reach_candidates",
            "structural_disruption_candidates", "structural_event_count",
            "structural_transmission_count", "shared_native_action_ids",
            "tracked_hypothesis_id", "adoption_threshold", "branches",
            "intervention", "source_lineage", "causal_attribution",
            "sample_size_per_arm", "confidence_interval", "external_validation",
            "cross_mechanism_ranking", "post_trust_action_effect",
        }
        if (type(x) is not dict or set(x) != keys
                or x["schema"] != _SCHEMA
                or x["source_owner"] != "V19_SOURCE_COGNITION_AND_SOCIAL_MEMORY_V24_READ_ONLY"
                or x["controller_agent_ids"] != ["alice"]
                or x["resource_controller"] != "alice"
                or x["structural_relation"] != "direct_interaction"
                or x["structural_directed_edges"] != [["bob", "dana"], ["carol", "bob"]]
                or x["structural_out_reach_candidates"] != ["carol"]
                or x["structural_disruption_candidates"] != ["bob"]
                or x["structural_event_count"] != 0
                or x["structural_transmission_count"] != 0
                or x["tracked_hypothesis_id"] != "approved"
                or x["adoption_threshold"] != 0.7
                or x["intervention"] != "ALICE_WAIT_VS_AUTHORIZED_OPEN_ONLY"
                or x["source_lineage"] != "DIVERGENT_STORY_BRANCH"
                or x["causal_attribution"] != "UNATTRIBUTED"
                or x["sample_size_per_arm"] != 1
                or x["confidence_interval"] != "NOT_ESTIMABLE"
                or x["external_validation"] != "NO_EXTERNAL_VALIDATION"
                or x["cross_mechanism_ranking"] != "FORBIDDEN"
                or x["post_trust_action_effect"] != "NOT_ESTABLISHED"
                or type(x["shared_native_action_ids"]) is not list
                or len(x["shared_native_action_ids"]) != 5
                or len(set(x["shared_native_action_ids"])) != 5
                or any(not _digest(x[k]) for k in (
                    "source_model_hash", "source_world_hash",
                    "source_memory_model_hash", "source_social_model_hash",
                    "source_perception_model_hash", "initial_world_hash",
                    "initial_story_hash", "structural_source_snapshot_hash",
                ))
                or type(x["branches"]) is not dict
                or set(x["branches"]) != set(_BRANCHES)):
            raise ValueError("M5c cannot invent social influence, source or causality")
        arm_fields = {
            "story_hash", "final_world_hash", "gate_event_hash",
            "gate_outcome", "gate_success", "move_outcome", "move_success",
            "tell_event_id", "tell_event_hash", "tell_fidelity_bob",
            "bob_inspect_event_hash", "bob_inspect_success",
            "source_network_before_hash", "source_network_after_hash",
            "source_social_before_hash", "source_social_after_hash",
            "content_evidence_trace_hash", "admitted_testimony",
            "admitted_verification", "claim_revisions", "trust_revisions",
            "bob_claim_statuses", "bob_trust_alice_before",
            "bob_trust_alice_after", "reverse_alice_trust_bob_before",
            "reverse_alice_trust_bob_after", "alice_to_bob_relationship_before",
            "alice_to_bob_relationship_after", "bob_to_alice_relationship_before",
            "bob_to_alice_relationship_after", "bob_tracked_belief_before",
            "bob_tracked_belief_after", "bob_above_threshold_before",
            "bob_above_threshold_after",
        }
        wait, open_ = (x["branches"][k] for k in _BRANCHES)
        for arm in (wait, open_):
            if (type(arm) is not dict or set(arm) != arm_fields
                    or any(not _digest(arm[k]) for k in (
                        "story_hash", "final_world_hash", "gate_event_hash",
                        "tell_event_hash", "bob_inspect_event_hash",
                        "source_network_before_hash", "source_network_after_hash",
                        "source_social_before_hash", "source_social_after_hash",
                        "content_evidence_trace_hash",
                    ))
                    or type(arm["tell_event_id"]) is not str
                    or not arm["tell_event_id"]
                    or arm["bob_inspect_success"] is not True
                    or any(type(arm[k]) is not bool for k in (
                        "gate_success", "move_success",
                        "alice_to_bob_relationship_before",
                        "alice_to_bob_relationship_after",
                        "bob_to_alice_relationship_before",
                        "bob_to_alice_relationship_after",
                        "bob_above_threshold_before", "bob_above_threshold_after",
                    ))
                    or any(type(arm[k]) is not int or arm[k] < 0 for k in (
                        "admitted_testimony", "admitted_verification",
                        "claim_revisions", "trust_revisions",
                    ))
                    or type(arm["bob_claim_statuses"]) is not list
                    or any(type(v) is not str or v not in
                           ("active", "confirmed", "contradicted", "superseded", "forgotten")
                           for v in arm["bob_claim_statuses"])
                    or any(type(arm[k]) not in (float, int)
                           or not 0 <= arm[k] <= 1 for k in (
                        "bob_trust_alice_before", "bob_trust_alice_after",
                        "reverse_alice_trust_bob_before",
                        "reverse_alice_trust_bob_after",
                        "bob_tracked_belief_before", "bob_tracked_belief_after",
                    ))
                    or arm["bob_above_threshold_before"] != (
                        arm["bob_tracked_belief_before"] >= 0.7)
                    or arm["bob_above_threshold_after"] != (
                        arm["bob_tracked_belief_after"] >= 0.7)):
                raise ValueError("M5c report has forged directed source/evidence metrics")
        if (wait["gate_outcome"] != "waited" or not wait["gate_success"]
                or open_["gate_outcome"] != "passage_opened" or not open_["gate_success"]
                or wait["move_success"] or wait["move_outcome"] != "passage_closed"
                or not open_["move_success"] or open_["move_outcome"] != "moved"
                or wait["tell_fidelity_bob"] != "detected"
                or open_["tell_fidelity_bob"] != "exact"
                or wait["admitted_testimony"] != 0
                or open_["admitted_testimony"] != 1
                or wait["admitted_verification"] != 1
                or open_["admitted_verification"] != 1
                or wait["claim_revisions"] != 0
                or open_["claim_revisions"] != 1
                or wait["trust_revisions"] != 0
                or open_["trust_revisions"] != 1
                or wait["bob_claim_statuses"] != []
                or open_["bob_claim_statuses"] != ["contradicted"]
                or wait["bob_trust_alice_before"] != 0.5
                or wait["bob_trust_alice_after"] != 0.5
                or open_["bob_trust_alice_before"] != 0.5
                or open_["bob_trust_alice_after"] != 0.3
                or wait["reverse_alice_trust_bob_before"] != 0.5
                or wait["reverse_alice_trust_bob_after"] != 0.5
                or open_["reverse_alice_trust_bob_before"] != 0.5
                or open_["reverse_alice_trust_bob_after"] != 0.5
                or wait["alice_to_bob_relationship_before"] is not True
                or wait["alice_to_bob_relationship_after"] is not True
                or open_["alice_to_bob_relationship_before"] is not True
                or open_["alice_to_bob_relationship_after"] is not False
                or any(arm["bob_to_alice_relationship_before"] is not True or
                       arm["bob_to_alice_relationship_after"] is not True
                       for arm in (wait, open_))
                or wait["tell_event_id"] != open_["tell_event_id"]
                or wait["tell_event_hash"] == open_["tell_event_hash"]
                or wait["story_hash"] == open_["story_hash"]):
            raise ValueError("M5c V19 source branches did not produce the registered outcomes")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_epistemic_key_player_pair(
    model: SituatedNetworkRuntimeModel,
    origin: SituatedStory,
    *,
    control_schedule: tuple[tuple[SituatedActionIntent, ...], ...],
    intervention_schedule: tuple[tuple[SituatedActionIntent, ...], ...],
) -> SourceEpistemicInfluenceReport:
    """Reexecute original V19, with one source-world intervention only."""
    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("M5c needs a real V19 network runtime model")
    if not isinstance(origin, SituatedStory) or origin.rounds:
        raise ValueError("M5c origin must be a pristine immutable V19 Story")
    _validate_schedules(control_schedule, intervention_schedule)
    memory_model = model.percept_memory_model
    social_model = model.social_memory_model
    world = memory_model.cognitive_model.world_model
    if (origin.model_hash != world.content_hash
            or origin.perception_model != memory_model.perception_model):
        raise ValueError("M5c Story must bind exact source world and perception")
    passage = next((p for p in world.passages if p.passage_id == "records-open"), None)
    if passage is None or passage.controller_agent_ids != ("alice",):
        raise ValueError("M5c permission must be source-owned by Alice")
    if next(p.open for p in origin.initial_state.passages if p.passage_id == "records-open"):
        raise ValueError("M5c source passage must start closed")

    origin_cognitive = initialize_situated_percept_memory_cognition(memory_model, origin)
    origin_social = initialize_situated_social_memory(social_model, origin_cognitive)
    structural = observe_source_structural_influence(
        model, origin, origin_cognitive, origin_social,
        relation=MeshRelation.DIRECT_INTERACTION,
        controlled_passage_id="records-open",
    ).to_dict()
    if (structural["source_directed_edges"] != [["bob", "dana"], ["carol", "bob"]]
            or structural["source_latest_tell_events"] != 0):
        raise ValueError("M5c initial key-player topology must be actual source revision")

    shared = tuple(control_schedule[i][0] for i in (0, 2, 3, 4, 5))
    arms: dict[str, dict[str, object]] = {}
    for name, schedule in (
        ("control_wait", control_schedule),
        ("authorized_open", intervention_schedule),
    ):
        story = replay_situated_story(
            world, origin.initial_state, schedule,
            perception_model=memory_model.perception_model,
        )
        if story.initial_state != origin.initial_state:
            raise ValueError("M5c branches must start from identical source initial world")
        actions = [_action(story, intent.action_id) for (intent,) in schedule]
        first, gate, move, tell, bob_move, bob_inspect = actions
        if (not first.success or not bob_move.success or not bob_inspect.success
                or tell.kind is not SituatedActionKind.TELL or not tell.success
                or tell.cause_event_ids != (first.event_id,)):
            raise ValueError("M5c requires real source-linked TELL and own INSPECT")
        percepts = [
            p for p in perceptual_timeline(memory_model.perception_model, story, "bob")
            if p.source_event_id == tell.event_id
        ]
        if len(percepts) != 1 or percepts[0].source_event_hash != tell.content_hash:
            raise ValueError("Bob needs actual story-qualified source percept")
        cognitive = initialize_situated_percept_memory_cognition(memory_model, story)
        social = initialize_situated_social_memory(social_model, cognitive)
        with TemporaryDirectory(prefix="v19-epistemic-keyplayer-" + name + "-") as tmp:
            result = simulate_situated_percept_social_cognitive_round(
                str(Path(tmp) / "memory.sqlite3"),
                memory_model, social_model, story, cognitive, social,
            )
        native_trace = observe_agent_content_evidence(
            memory_model, social_model, story, cognitive, social, result,
        )
        trace = native_trace.to_dict()
        source_before = project_situated_network_snapshot(
            model, story, cognitive, social,
        )
        source_after = project_situated_network_snapshot(
            model, result.next_story,
            result.next_cognitive_state, result.next_social_state,
        )
        relationship_before = set(source_relation_edges(
            source_before, MeshRelation.RELATIONSHIP))
        relationship_after = set(source_relation_edges(
            source_after, MeshRelation.RELATIONSHIP))
        if (relationship_before - relationship_after
                not in (set(), {("alice", "bob")})
                or relationship_after - relationship_before):
            raise ValueError("M5c V19 changed unrelated social source edge")
        old_belief = next(n.tracked_belief_probability for n in source_before.nodes
                          if n.agent_id == "bob")
        new_belief = next(n.tracked_belief_probability for n in source_after.nodes
                          if n.agent_id == "bob")
        arms[name] = {
            "story_hash": story.content_hash,
            "final_world_hash": story.current_state.content_hash,
            "gate_event_hash": gate.content_hash,
            "gate_outcome": gate.outcome,
            "gate_success": gate.success,
            "move_outcome": move.outcome,
            "move_success": move.success,
            "tell_event_id": tell.event_id,
            "tell_event_hash": tell.content_hash,
            "tell_fidelity_bob": percepts[0].fidelity.value,
            "bob_inspect_event_hash": bob_inspect.content_hash,
            "bob_inspect_success": bob_inspect.success,
            "source_network_before_hash": source_before.content_hash,
            "source_network_after_hash": source_after.content_hash,
            "source_social_before_hash": social.content_hash,
            "source_social_after_hash": result.next_social_state.content_hash,
            "content_evidence_trace_hash": native_trace.content_hash,
            "admitted_testimony": trace["counts"]["admitted_testimony"],
            "admitted_verification": trace["counts"]["admitted_verification"],
            "claim_revisions": trace["counts"]["claim_revisions"],
            "trust_revisions": trace["counts"]["trust_revisions"],
            "bob_claim_statuses": sorted(
                c.status.value for c in result.next_social_state.claims
                if c.observer_agent_id == "bob"
            ),
            "bob_trust_alice_before": _trust(social, "bob", "alice"),
            "bob_trust_alice_after": _trust(result.next_social_state, "bob", "alice"),
            "reverse_alice_trust_bob_before": _trust(social, "alice", "bob"),
            "reverse_alice_trust_bob_after": _trust(result.next_social_state, "alice", "bob"),
            "alice_to_bob_relationship_before": ("alice", "bob") in relationship_before,
            "alice_to_bob_relationship_after": ("alice", "bob") in relationship_after,
            "bob_to_alice_relationship_before": ("bob", "alice") in relationship_before,
            "bob_to_alice_relationship_after": ("bob", "alice") in relationship_after,
            "bob_tracked_belief_before": old_belief,
            "bob_tracked_belief_after": new_belief,
            "bob_above_threshold_before": old_belief >= model.adoption_threshold,
            "bob_above_threshold_after": new_belief >= model.adoption_threshold,
        }
    result_data = {
        "schema": _SCHEMA,
        "source_owner": "V19_SOURCE_COGNITION_AND_SOCIAL_MEMORY_V24_READ_ONLY",
        "source_model_hash": model.content_hash,
        "source_world_hash": world.content_hash,
        "source_memory_model_hash": memory_model.content_hash,
        "source_social_model_hash": social_model.content_hash,
        "source_perception_model_hash": memory_model.perception_model.content_hash,
        "initial_world_hash": origin.initial_state.content_hash,
        "initial_story_hash": origin.content_hash,
        "controller_agent_ids": list(passage.controller_agent_ids),
        "resource_controller": "alice",
        "structural_source_snapshot_hash": structural["source_snapshot_hash"],
        "structural_relation": structural["source_relation"],
        "structural_directed_edges": structural["source_directed_edges"],
        "structural_out_reach_candidates": structural["highest_out_reach_candidates"],
        "structural_disruption_candidates": structural["highest_removal_loss_candidates"],
        "structural_event_count": structural["source_latest_tell_events"],
        "structural_transmission_count": structural["source_latest_transmissions"],
        "shared_native_action_ids": [a.action_id for a in shared],
        "tracked_hypothesis_id": model.tracked_hypothesis_id,
        "adoption_threshold": model.adoption_threshold,
        "branches": arms,
        "intervention": "ALICE_WAIT_VS_AUTHORIZED_OPEN_ONLY",
        "source_lineage": "DIVERGENT_STORY_BRANCH",
        "causal_attribution": "UNATTRIBUTED",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "cross_mechanism_ranking": "FORBIDDEN",
        "post_trust_action_effect": "NOT_ESTABLISHED",
    }
    return SourceEpistemicInfluenceReport(
        json.dumps(result_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
