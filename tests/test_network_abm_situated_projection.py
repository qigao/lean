from __future__ import annotations

import json

import pytest

from narrative_dynamics.abm.situated import (
    SituatedActionIntent,
    SituatedActionKind,
    SituatedRoundResult,
    resolve_situated_round,
)
from narrative_dynamics.abm.situated_contracts import (
    EmbodiedAgentSpec,
    PlaceSpec,
    SituatedWorldModel,
    initialize_situated_world,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedAgentPerceptionProfile,
    SituatedEventSignalProfile,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
)
from narrative_dynamics.abm.situated_projection import project_situated_narrative
from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeAuthority,
    NarrativeBeatKind,
    NarrativeEntitlementScope,
    NarrativeProjectionPolicy,
    NarrativeTemporalOrder,
)
from narrative_dynamics.abm.situated_story import (
    advance_situated_story,
    objective_timeline,
    replay_situated_story,
    SituatedStory,
)
from tests.situated_fixtures import office_model


def _perception_model() -> SituatedPerceptionModel:
    world = office_model()
    agents = tuple(item.agent_id for item in world.agents)
    return SituatedPerceptionModel(
        "projection-perception",
        "1.0",
        world,
        (
            SituatedPerceptionEdge(
                "records-open-visual",
                SituatedPerceptionLayer.VISIBILITY,
                "records",
                "open",
                1.0,
            ),
            SituatedPerceptionEdge(
                "records-open-auditory",
                SituatedPerceptionLayer.AUDITORY,
                "records",
                "open",
                0.0,
            ),
        ),
        tuple(
            SituatedAgentPerceptionProfile(agent_id, 2.0, 5.0, 20.0)
            for agent_id in agents
        ),
        (
            SituatedEventSignalProfile(SituatedActionKind.MOVE, True),
            SituatedEventSignalProfile(SituatedActionKind.TELL, False, 10.0),
        ),
    )


def _story(*, second_kind: SituatedActionKind = SituatedActionKind.MOVE):
    perception = _perception_model()
    world = perception.world_model
    first = (SituatedActionIntent("alice-inspects", "alice", SituatedActionKind.INSPECT, "memo"),)
    if second_kind is SituatedActionKind.MOVE:
        second = (SituatedActionIntent("alice-moves", "alice", second_kind, "records-open"),)
    else:
        second = (SituatedActionIntent("alice-tells", "alice", second_kind, message="status update"),)
    return replay_situated_story(
        world,
        initialize_situated_world(world),
        (first, second),
        perception_model=perception,
    )


def objective_office_story():
    return _story()


def private_inspection_then_public_clue_story():
    return _story()


def detected_story():
    perception = _perception_model()
    world = perception.world_model
    prior = initialize_situated_world(world)
    resolved = resolve_situated_round(
        world,
        prior,
        (SituatedActionIntent("alice-tells", "alice", SituatedActionKind.TELL, message="status update"),),
    )
    event = next(item for item in resolved.events if item.actor_agent_id == "alice")
    accepted = SituatedRoundResult(
        resolved.prior_state,
        tuple(item for item in resolved.intents if item.agent_id == "alice"),
        (event,),
        tuple(item for item in resolved.observations if item.event_id == event.event_id),
        resolved.next_state,
    )
    return SituatedStory(
        world.model_id,
        world.content_hash,
        prior,
        (accepted,),
        perception,
    )


def _append_single_alice_tell(
    story: SituatedStory,
    action_id: str,
    *,
    source_event_ids: tuple[str, ...] = (),
) -> tuple[SituatedStory, str]:
    world = story.perception_model.world_model
    resolved = resolve_situated_round(
        world,
        story.current_state,
        (
            SituatedActionIntent(
                action_id,
                "alice",
                SituatedActionKind.TELL,
                message="status update",
                source_event_ids=source_event_ids,
            ),
        ),
    )
    event = next(item for item in resolved.events if item.actor_agent_id == "alice")
    accepted = SituatedRoundResult(
        resolved.prior_state,
        tuple(item for item in resolved.intents if item.agent_id == "alice"),
        (event,),
        tuple(item for item in resolved.observations if item.event_id == event.event_id),
        resolved.next_state,
    )
    return (
        SituatedStory(
            story.model_id,
            story.model_hash,
            story.initial_state,
            story.rounds + (accepted,),
            story.perception_model,
        ),
        event.event_id,
    )


def detected_causal_story():
    perception = _perception_model()
    world = perception.world_model
    story = SituatedStory(
        world.model_id,
        world.content_hash,
        initialize_situated_world(world),
        perception_model=perception,
    )
    story, first_id = _append_single_alice_tell(story, "alice-first")
    story, cause_id = _append_single_alice_tell(story, "alice-cause")
    story, payoff_id = _append_single_alice_tell(
        story,
        "alice-payoff",
        source_event_ids=(cause_id,),
    )
    return story, (first_id, cause_id, payoff_id)


def causal_story(*, private: bool = False):
    perception = _perception_model() if private else None
    world = perception.world_model if perception is not None else office_model()
    story = replay_situated_story(
        world,
        initialize_situated_world(world),
        ((
            SituatedActionIntent("m-inspect", "alice", SituatedActionKind.INSPECT, "memo"),
            SituatedActionIntent("a-bob-waits", "bob", SituatedActionKind.WAIT),
        ),),
        perception_model=perception,
    )
    inspect = next(
        item
        for item in story.rounds[0].events
        if item.actor_agent_id == "alice"
    )
    story = advance_situated_story(
        world,
        story,
        (
            SituatedActionIntent(
                "z-tell",
                "alice",
                SituatedActionKind.TELL,
                message="status update",
                source_event_ids=(inspect.event_id,),
            ),
            SituatedActionIntent("a-bob-waits-again", "bob", SituatedActionKind.WAIT),
        ),
    )
    return story, inspect.event_id


def one_place_story(round_count: int = 4, *, two_agents: bool = False):
    agents = (EmbodiedAgentSpec("alice", "staff", "office"),)
    if two_agents:
        agents += (EmbodiedAgentSpec("bob", "staff", "office"),)
    world = SituatedWorldModel(
        "one-place",
        "1.0",
        (PlaceSpec("office", "Office"),),
        (),
        agents,
    )
    perception = None
    if two_agents:
        perception = SituatedPerceptionModel(
            "one-place-perception",
            "1.0",
            world,
            (),
            tuple(
                SituatedAgentPerceptionProfile(item.agent_id, 0.0, 0.0, 0.0)
                for item in agents
            ),
            (),
        )
    schedule = tuple(
        (SituatedActionIntent(f"wait-{round_index}", "alice", SituatedActionKind.WAIT),)
        for round_index in range(1, round_count + 1)
    )
    return replay_situated_story(
        world,
        initialize_situated_world(world),
        schedule,
        perception_model=perception,
    )


def objective_policy(**kwargs) -> NarrativeProjectionPolicy:
    return NarrativeProjectionPolicy(
        "objective",
        "1.0",
        NarrativeAuthority.OBJECTIVE,
        **kwargs,
    )


def limited_policy(agent_id: str, **kwargs) -> NarrativeProjectionPolicy:
    return NarrativeProjectionPolicy(
        f"limited-{agent_id}",
        "1.0",
        NarrativeAuthority.AGENT_LIMITED,
        pov_agent_ids=(agent_id,),
        **kwargs,
    )


def test_objective_projection_cites_exact_world_events():
    story = objective_office_story()
    projection = project_situated_narrative(story, objective_policy())
    event_hashes = {
        event.content_hash for round_result in story.rounds for event in round_result.events
    }
    cited = {
        ref.artifact_hash
        for beat in projection.beats
        for ref in beat.supporting_artifacts
        if ref.artifact_kind == "world_event"
    }
    assert cited == event_hashes
    assert all(
        item.scope is NarrativeEntitlementScope.OBJECTIVE
        for item in projection.cut.entitlements
    )


def test_bob_limited_projection_does_not_leak_alice_private_inspection():
    projection = project_situated_narrative(
        private_inspection_then_public_clue_story(), limited_policy("bob")
    )
    serialized = json.dumps(projection.to_dict(), sort_keys=True)
    assert "restructuring" not in serialized
    assert "approved" not in serialized
    assert all(
        entitlement.owner_agent_id == "bob"
        for entitlement in projection.cut.entitlements
    )


def test_detected_percept_exposes_signal_without_actor_kind_or_outcome():
    projection = project_situated_narrative(detected_story(), limited_policy("bob"))
    facts = {
        fact.key: fact.value
        for entitlement in projection.cut.entitlements
        for fact in entitlement.facts
    }
    assert facts["fidelity"] == "detected"
    assert "actor_agent_id" not in facts
    assert "kind" not in facts
    assert "outcome" not in facts
    assert tuple(item.kind for item in projection.beats) == (
        NarrativeBeatKind.INFORMATION,
    )


def test_unknown_limited_pov_agent_is_rejected():
    with pytest.raises(ValueError, match="POV agent"):
        project_situated_narrative(objective_office_story(), limited_policy("unknown"))


def test_multi_pov_keeps_each_private_entitlement_with_its_declared_owner():
    story = private_inspection_then_public_clue_story()
    policy = NarrativeProjectionPolicy(
        "alice-and-bob",
        "1.0",
        NarrativeAuthority.MULTI_POV,
        pov_agent_ids=("alice", "bob"),
    )
    projection = project_situated_narrative(story, policy)
    owners = {item.owner_agent_id for item in projection.cut.entitlements}
    assert owners == {"alice", "bob"}
    assert all(
        item.scope is NarrativeEntitlementScope.PRIVATE
        for item in projection.cut.entitlements
    )
    entitlements = {item.entitlement_id: item for item in projection.cut.entitlements}
    assert all(
        all(entitlements[item].owner_agent_id == beat.active_pov_agent_id for item in beat.entitlement_ids)
        for beat in projection.beats
    )


def test_limited_support_is_the_exact_private_percept_not_the_world_event():
    story = private_inspection_then_public_clue_story()
    projection = project_situated_narrative(story, limited_policy("bob"))
    supports = {
        (ref.artifact_kind, ref.artifact_id, ref.artifact_hash)
        for beat in projection.beats
        for ref in beat.supporting_artifacts
    }
    from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline

    expected = {
        ("percept", item.percept_id, item.content_hash)
        for item in perceptual_timeline(story.perception_model, story, "bob")
    }
    assert supports == expected
    assert not any(kind == "world_event" for kind, _, _ in supports)


def test_repeated_projection_is_byte_equivalent_and_content_addressed():
    story = private_inspection_then_public_clue_story()
    policy = limited_policy("bob")
    first = project_situated_narrative(story, policy)
    second = project_situated_narrative(story, policy)
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    assert first.content_hash == second.content_hash


def test_omission_gap_forces_supported_anchors_between_first_and_last_events():
    story = objective_office_story()
    weights = {
        NarrativeBeatKind.PHYSICAL: 0.0,
        NarrativeBeatKind.INFORMATION: 0.0,
    }
    sparse = project_situated_narrative(
        story,
        objective_policy(
            salience_weights=weights,
            minimum_salience=1.0,
            maximum_event_omission_gap=0,
        ),
    )
    anchored = project_situated_narrative(
        story,
        objective_policy(
            salience_weights=weights,
            minimum_salience=1.0,
            maximum_event_omission_gap=2,
        ),
    )
    timeline = objective_timeline(story)
    positions = {event.event_id: index for index, event in enumerate(timeline)}
    sparse_positions = [positions[beat.source_event_id] for beat in sparse.beats]
    anchored_positions = [positions[beat.source_event_id] for beat in anchored.beats]
    assert sparse_positions == [0, len(timeline) - 1]
    assert anchored_positions[0] == 0
    assert anchored_positions[-1] == len(timeline) - 1
    assert all(right - left <= 2 for left, right in zip(anchored_positions, anchored_positions[1:]))


def test_required_causal_coverage_selects_authorized_causes_and_marks_payoff():
    story, inspect_id = causal_story()
    timeline = objective_timeline(story)
    tell = next(item for item in timeline if item.kind is SituatedActionKind.TELL)
    projection = project_situated_narrative(
        story,
        objective_policy(
            salience_weights={
                NarrativeBeatKind.PHYSICAL: 0.0,
                NarrativeBeatKind.INFORMATION: 0.0,
            },
            minimum_salience=1.0,
            required_causal_coverage=1.0,
        ),
    )
    beats = {item.source_event_id: item for item in projection.beats}
    assert inspect_id in beats
    assert tell.event_id in beats
    assert beats[tell.event_id].cause_beat_ids == (beats[inspect_id].beat_id,)
    assert beats[tell.event_id].kind is NarrativeBeatKind.CAUSAL_PAYOFF
    entitlement_by_id = {
        item.entitlement_id: item for item in projection.cut.entitlements
    }
    payoff_facts = {
        fact.key: fact.value
        for entitlement_id in beats[tell.event_id].entitlement_ids
        for fact in entitlement_by_id[entitlement_id].facts
    }
    assert payoff_facts["kind"] == "tell"
    assert payoff_facts["outcome"] == "told"


def test_limited_causal_selection_never_names_an_unperceived_cause():
    story, inspect_id = causal_story(private=True)
    projection = project_situated_narrative(
        story,
        limited_policy("bob", required_causal_coverage=1.0),
    )
    assert inspect_id not in {item.source_event_id for item in projection.beats}
    projected_ids = {item.beat_id for item in projection.beats}
    assert all(set(item.cause_beat_ids) <= projected_ids for item in projection.beats)
    assert all(not item.cause_beat_ids for item in projection.beats)


def test_private_perceived_endpoints_do_not_authorize_objective_causal_edges():
    story, (first_id, cause_id, payoff_id) = detected_causal_story()
    complete = project_situated_narrative(story, limited_policy("bob"))
    assert {item.source_event_id for item in complete.beats} == {
        first_id,
        cause_id,
        payoff_id,
    }
    assert all(not item.cause_beat_ids for item in complete.beats)
    assert all(item.kind is not NarrativeBeatKind.CAUSAL_PAYOFF for item in complete.beats)

    sparse = project_situated_narrative(
        story,
        limited_policy(
            "bob",
            salience_weights={
                NarrativeBeatKind.PHYSICAL: 0.0,
                NarrativeBeatKind.INFORMATION: 0.0,
            },
            minimum_salience=1.0,
            required_causal_coverage=1.0,
        ),
    )
    assert tuple(item.source_event_id for item in sparse.beats) == (
        first_id,
        payoff_id,
    )
    assert cause_id not in {item.source_event_id for item in sparse.beats}


def test_chronological_projection_never_reorders_a_nonadjacent_cause_pair():
    story, _ = causal_story()
    projection = project_situated_narrative(story, objective_policy())
    emitted_positions = [
        (item.round_index, item.sequence) for item in projection.beats
    ]
    assert emitted_positions == sorted(emitted_positions)


def test_authored_event_order_reverses_presentation_without_rewriting_time():
    story = objective_office_story()
    timeline = objective_timeline(story)
    expected_rounds = {item.event_id: item.round_index for item in timeline}
    authored = tuple(item.event_id for item in reversed(timeline))
    projection = project_situated_narrative(
        story,
        objective_policy(
            temporal_order=NarrativeTemporalOrder.AUTHORED,
            authored_event_order=authored,
            maximum_scene_beats=1,
        ),
    )
    beats = {item.beat_id: item for item in projection.beats}
    scenes = {item.scene_id: item for item in projection.scenes}
    presented = tuple(
        beats[beat_id]
        for scene_id in projection.cut.scene_ids
        for beat_id in scenes[scene_id].beat_ids
    )
    assert tuple(item.source_event_id for item in presented) == authored
    assert all(
        item.round_index == expected_rounds[item.source_event_id] for item in presented
    )


def test_authored_event_order_rejects_an_id_outside_the_story():
    story = objective_office_story()
    authored = tuple(item.event_id for item in objective_timeline(story)) + ("missing-event",)
    with pytest.raises(ValueError, match="accepted story"):
        project_situated_narrative(
            story,
            objective_policy(
                temporal_order=NarrativeTemporalOrder.AUTHORED,
                authored_event_order=authored,
            ),
        )


def test_scene_grouping_obeys_continuity_capacity_and_complete_membership():
    story = one_place_story()
    projection = project_situated_narrative(
        story,
        objective_policy(scene_round_gap=1, maximum_scene_beats=3),
    )
    assert tuple(len(item.beat_ids) for item in projection.scenes) == (3, 1)
    assert tuple(
        beat_id for scene in projection.scenes for beat_id in scene.beat_ids
    ) == tuple(item.beat_id for item in projection.beats)
    assert projection.cut.scene_ids == tuple(item.scene_id for item in projection.scenes)


def test_scene_grouping_splits_on_round_gap_and_pov_change():
    story = one_place_story(round_count=3)
    gap_projection = project_situated_narrative(
        story,
        objective_policy(
            salience_weights={NarrativeBeatKind.PHYSICAL: 0.0},
            minimum_salience=1.0,
            required_causal_coverage=0.0,
            scene_round_gap=1,
        ),
    )
    assert tuple(len(item.beat_ids) for item in gap_projection.scenes) == (1, 1)

    multi_story = one_place_story(round_count=1, two_agents=True)
    multi_policy = NarrativeProjectionPolicy(
        "two-pov",
        "1.0",
        NarrativeAuthority.MULTI_POV,
        pov_agent_ids=("alice", "bob"),
        maximum_scene_beats=8,
    )
    multi_projection = project_situated_narrative(multi_story, multi_policy)
    assert len(multi_projection.scenes) == 2
    assert {item.active_pov_agent_id for item in multi_projection.scenes} == {
        "alice",
        "bob",
    }

    moved = private_inspection_then_public_clue_story()
    moved = advance_situated_story(
        moved.perception_model.world_model,
        moved,
        (SituatedActionIntent("alice-waits-open", "alice", SituatedActionKind.WAIT),),
    )
    place_projection = project_situated_narrative(moved, limited_policy("alice"))
    assert place_projection.scenes[0].place_id == "records"
    assert place_projection.scenes[-1].place_id == "open"
    assert len(place_projection.scenes) >= 2
