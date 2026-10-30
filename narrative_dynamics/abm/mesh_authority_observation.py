"""R2.M5a: read-only observation of source-owned resource authorization.

World PassageSpec and the original V19 action resolver alone own control
decisions. This observer never grants permissions or synthesizes social edges.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import (
    SituatedWorldModel, initialize_situated_world,
)
from narrative_dynamics.abm.situated_story import (
    explain_situated_event, replay_situated_story,
)
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptionModel,
)

_SCHEMA = "r2-m5a-v19-source-passage-control-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")

_ROOT_KEYS = frozenset({
    "schema", "source_owner", "mechanism", "scope", "source_model_hash",
    "initial_world_hash", "controlled_passage", "controller_agent_ids",
    "controller", "unauthorized_actor", "cross_branch_relation",
    "causal_attribution", "sample_size_per_arm", "confidence_interval",
    "external_validation", "network_centrality", "trust_change", "arms",
})
_ARM_KEYS = frozenset({
    "story_hash", "world_round_hashes", "gate_event_hash",
    "gate_event_actor", "gate_event_outcome", "gate_event_success",
    "move_event_hash", "move_event_outcome", "move_success",
    "tell_event_hash", "bob_tell_fidelity", "gate_open_at_tell",
    "alice_place_at_tell",
})



def _digest(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


@dataclass(frozen=True)
class PassageAuthorityEvidence:
    """Immutable finite source report, not a permission or causal engine."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("source evidence requires immutable JSON")
        d = json.loads(self._canonical_json)
        if (type(d) is not dict or set(d) != _ROOT_KEYS
                or d.get("schema") != _SCHEMA
                or d.get("source_owner") != "V19_SITUATED_WORLD_MODEL_AND_STORY"
                or d.get("mechanism") != "EXPLICIT_PASSAGE_RESOURCE_CONTROL"
                or d.get("causal_attribution") != "UNATTRIBUTED"
                or d.get("cross_branch_relation") != "DIVERGENT_STORY_BRANCH"
                or d.get("sample_size_per_arm") != 1
                or d.get("confidence_interval") != "NOT_ESTIMABLE"
                or d.get("external_validation") != "NO_EXTERNAL_VALIDATION"
                or d.get("network_centrality") != "NOT_MEASURED"
                or d.get("trust_change") != "NOT_MEASURED"
                or any(not _digest(d.get(k)) for k in
                       ("source_model_hash", "initial_world_hash"))
                or d.get("controller") != {"agent_id": "alice", "role": "analyst"}
                or d.get("unauthorized_actor") != {"agent_id": "dana", "role": "manager"}
                or d.get("controlled_passage") != "records-open"
                or d.get("controller_agent_ids") != ["alice"]
                or type(d.get("arms")) is not dict
                or set(d["arms"]) != {
                    "control_wait", "authorized_open", "unauthorized_manager_open",
                }):
            raise ValueError("M5a cannot claim universal social power or causal effect")
        arms = d["arms"]
        for arm in arms.values():
            if (type(arm) is not dict or set(arm) != _ARM_KEYS
                    or any(not _digest(arm.get(k)) for k in (
                        "story_hash", "gate_event_hash", "move_event_hash",
                        "tell_event_hash",
                    ))
                    or type(arm.get("world_round_hashes")) is not list
                    or len(arm["world_round_hashes"]) != 3
                    or any(not _digest(h) for h in arm["world_round_hashes"])
                    or arm.get("gate_event_actor") not in {"alice", "dana"}
                    or arm.get("bob_tell_fidelity") not in {"exact", "detected"}
                    or type(arm.get("gate_open_at_tell")) is not bool
                    or type(arm.get("move_success")) is not bool
                    or type(arm.get("gate_event_success")) is not bool
                    or arm.get("alice_place_at_tell") not in {"records", "open"}):
                raise ValueError("M5a evidence must use exact typed source events")
        control, approved, denied = (
            arms["control_wait"], arms["authorized_open"],
            arms["unauthorized_manager_open"],
        )
        if (control["gate_event_outcome"] != "waited"
                or control["gate_event_actor"] != "alice"
                or not control["gate_event_success"]
                or approved["gate_event_outcome"] != "passage_opened"
                or approved["gate_event_actor"] != "alice"
                or not approved["gate_event_success"]
                or denied["gate_event_outcome"] != "not_authorized"
                or denied["gate_event_actor"] != "dana"
                or denied["gate_event_success"]
                or control["gate_open_at_tell"] or denied["gate_open_at_tell"]
                or not approved["gate_open_at_tell"]
                or control["move_success"] or denied["move_success"]
                or not approved["move_success"]
                or control["bob_tell_fidelity"] != "detected"
                or control["move_event_outcome"] != "passage_closed"
                or denied["move_event_outcome"] != "passage_closed"
                or approved["move_event_outcome"] != "moved"
                or control["tell_event_hash"] != denied["tell_event_hash"]
                or approved["tell_event_hash"] == control["tell_event_hash"]
                or denied["bob_tell_fidelity"] != "detected"
                or approved["bob_tell_fidelity"] != "exact"
                or control["alice_place_at_tell"] != "records"
                or denied["alice_place_at_tell"] != "records"
                or approved["alice_place_at_tell"] != "open"
                or len({arm["story_hash"] for arm in arms.values()}) != 3):
            raise ValueError("M5a reported events contradict the controlled native world")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _event(story, action_id: str):
    return next(e for r in story.rounds for e in r.events if e.action_id == action_id)


def observe_passage_authority(
    world: SituatedWorldModel,
    perception: SituatedPerceptionModel,
) -> PassageAuthorityEvidence:
    """Reexecute the exact V19 world, same immutable genesis in three arms."""
    if not isinstance(world, SituatedWorldModel) or not isinstance(perception, SituatedPerceptionModel):
        raise TypeError("M5a requires the original V19 world and perception models")
    if perception.world_model.content_hash != world.content_hash:
        raise ValueError("perception must bind the exact authorized V19 model")
    policy = next((p for p in world.passages if p.passage_id == "records-open"), None)
    if policy is None or policy.controller_agent_ids != ("alice",):
        raise ValueError("M5a needs a real Alice-controlled passage")
    actors = {a.agent_id: a for a in world.agents}
    if (actors["alice"].role != "analyst" or actors["dana"].role != "manager"
            or actors["alice"].initial_place_id != "records"
            or actors["dana"].initial_place_id != "records"
            or actors["bob"].initial_place_id != "open"):
        raise ValueError("M5a requires co-located controller and higher-role challenger")
    initial = initialize_situated_world(world)
    if next(p.open for p in initial.passages if p.passage_id == "records-open"):
        raise ValueError("source-controlled gate must start closed")
    move = SituatedActionIntent("alice-shared-move", "alice",
                                SituatedActionKind.MOVE, target_id="records-open")
    tell = SituatedActionIntent("alice-shared-tell", "alice",
                                SituatedActionKind.TELL, message="The restructuring is approved.")
    starts = {
        "control_wait": SituatedActionIntent(
            "alice-wait", "alice", SituatedActionKind.WAIT),
        "authorized_open": SituatedActionIntent(
            "alice-open", "alice", SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),
        "unauthorized_manager_open": SituatedActionIntent(
            "dana-open", "dana", SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),
    }
    arms = {}
    for name, start in starts.items():
        story = replay_situated_story(
            world, initial, ((start,), (move,), (tell,)),
            perception_model=perception,
        )
        gate, movement, told = (_event(story, action.action_id)
                                for action in (start, move, tell))
        if told.kind is not SituatedActionKind.TELL or not told.success:
            raise ValueError("source experiment requires actual V19 TELL")
        percepts = [p for p in perceptual_timeline(perception, story, "bob")
                    if p.source_event_id == told.event_id]
        if len(percepts) != 1 or percepts[0].source_event_hash != told.content_hash:
            raise ValueError("Bob percept must bind a real world TELL event")
        if gate.outcome == "not_authorized":
            explanation = explain_situated_event(story, gate.event_id)
            if explanation.success or explanation.outcome != "not_authorized":
                raise ValueError("native world must explain unauthorized gate denial")
        arms[name] = {
            "story_hash": story.content_hash,
            "world_round_hashes": [r.content_hash for r in story.rounds],
            "gate_event_hash": gate.content_hash,
            "gate_event_actor": gate.actor_agent_id,
            "gate_event_outcome": gate.outcome,
            "gate_event_success": gate.success,
            "move_event_hash": movement.content_hash,
            "move_event_outcome": movement.outcome,
            "move_success": movement.success,
            "tell_event_hash": told.content_hash,
            "bob_tell_fidelity": percepts[0].fidelity.value,
            "gate_open_at_tell": next(p.open for p in story.current_state.passages
                                      if p.passage_id == "records-open"),
            "alice_place_at_tell": next(a.place_id for a in story.current_state.agents
                                        if a.agent_id == "alice"),
        }
    result = {
        "schema": _SCHEMA,
        "source_owner": "V19_SITUATED_WORLD_MODEL_AND_STORY",
        "mechanism": "EXPLICIT_PASSAGE_RESOURCE_CONTROL",
        "scope": "finite source model intervention, not a population causal effect",
        "source_model_hash": world.content_hash,
        "initial_world_hash": initial.content_hash,
        "controlled_passage": policy.passage_id,
        "controller_agent_ids": list(policy.controller_agent_ids),
        "controller": {"agent_id": "alice", "role": actors["alice"].role},
        "unauthorized_actor": {"agent_id": "dana", "role": actors["dana"].role},
        "cross_branch_relation": "DIVERGENT_STORY_BRANCH",
        "causal_attribution": "UNATTRIBUTED",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "network_centrality": "NOT_MEASURED",
        "trust_change": "NOT_MEASURED",
        "arms": arms,
    }
    return PassageAuthorityEvidence(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
