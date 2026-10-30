"""Canonical, source-derived V19 physical boundary events and resulting V24 facts."""
from __future__ import annotations

import json

from narrative_dynamics.abm.mesh_contracts import MeshRelation
from narrative_dynamics.abm.situated import SituatedActionKind, resolve_situated_round
from narrative_dynamics.abm.situated_perception import (
    can_situated_agents_interact, derive_situated_perception_reach,
)
from tests.test_network_boundary_events import act, boundary_fixture, build_real_boundary_case

SCHEMA = "v19-physical-boundary-events-v1"


def build_vector_manifest() -> dict[str, object]:
    model, before, after, report = build_real_boundary_case()
    first_model, physical = boundary_fixture()
    initial_hash = physical.content_hash
    opened = resolve_situated_round(first_model.world_model, physical, (
        act("open", "alice", SituatedActionKind.OPEN_PASSAGE, "meeting-door"),
        act("move-too-soon", "carol", SituatedActionKind.MOVE, "meeting-door"),
    ))
    closed = resolve_situated_round(first_model.world_model, opened.next_state, (
        act("close", "carol", SituatedActionKind.CLOSE_PASSAGE, "meeting-door"),
        act("move-while-open", "alice", SituatedActionKind.MOVE, "meeting-door"),
    ))
    opening = next(e for e in opened.events if e.action_id == "open")
    attempted_move = next(e for e in opened.events if e.action_id == "move-too-soon")
    closing = next(e for e in closed.events if e.action_id == "close")
    successful_move = next(e for e in closed.events if e.action_id == "move-while-open")
    if (
        not opening.success or attempted_move.success
        or not closing.success or not successful_move.success
        or not opened.next_state.passages[0].open or closed.next_state.passages[0].open
        or before.social.relationships != after.social.relationships
        or before.mesh.source_snapshot.transmissions != after.mesh.source_snapshot.transmissions
    ):
        raise AssertionError("unreproducible V19 passage boundary semantics")
    original_reach = derive_situated_perception_reach(
        first_model, physical, source_place_id="corridor",
    )
    open_reach = derive_situated_perception_reach(
        first_model, opened.next_state, source_place_id="corridor",
    )
    if (
        can_situated_agents_interact(first_model, physical, "alice", "bob")
        or not can_situated_agents_interact(first_model, opened.next_state, "alice", "bob")
        or ("meeting" in original_reach.visual_costs)
        or open_reach.visual_costs.get("meeting") != 1.0
    ):
        raise AssertionError("V19 passage state did not drive intended perception edge")
    reported = report.to_dict()
    if (
        reported["causal_attribution"] != "UNATTRIBUTED"
        or reported["story_lineage"] != "STORY_EXTENSION"
        or reported["selected_relation_graphs"]["direct_interaction"]
        ["source_edges"]["added"] != [["alice", "bob"]]
        or reported["selected_relation_graphs"]["transmission"]
        ["source_edges"] != {"added": [], "removed": []}
    ):
        raise AssertionError("R2 physical gate observation misreported V19 source")
    return {
        "schema": SCHEMA,
        "notice": ("Real V19 passage action and source projection, not "
                   "authenticated external control or a TELL event"),
        "cases": [
            {
                "name": "directed_passage_snapshot_isolation",
                "prior_world_hash": initial_hash,
                "open_round": {
                    "prior_world_hash": opened.prior_state.content_hash,
                    "next_world_hash": opened.next_state.content_hash,
                    "events": [
                        {"kind": e.kind.value, "success": e.success,
                         "outcome": e.outcome, "event_hash": e.content_hash}
                        for e in opened.events
                    ],
                    "passage_open": opened.next_state.passages[0].open,
                    "initial_auditory_loss": original_reach.auditory_losses["meeting"],
                    "next_auditory_loss": open_reach.auditory_losses["meeting"],
                },
                "close_round": {
                    "prior_world_hash": closed.prior_state.content_hash,
                    "next_world_hash": closed.next_state.content_hash,
                    "events": [
                        {"kind": e.kind.value, "success": e.success,
                         "outcome": e.outcome, "event_hash": e.content_hash}
                        for e in closed.events
                    ],
                    "passage_open": closed.next_state.passages[0].open,
                },
            },
            {
                "name": "real_v19_boundary_to_v24_projection",
                "before_source_hash": before.mesh.source_snapshot.content_hash,
                "after_source_hash": after.mesh.source_snapshot.content_hash,
                "observation_hash": report.content_hash,
                "boundary": reported["boundary"],
                "selected_relation_graphs": reported["selected_relation_graphs"],
                "new_story_events": reported["interaction_content"]["new_story_events"],
                "causal_attribution": reported["causal_attribution"],
            },
        ],
    }


def render_vector_corpus() -> str:
    return json.dumps(build_vector_manifest(), indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


if __name__ == "__main__":
    print(render_vector_corpus(), end="")
