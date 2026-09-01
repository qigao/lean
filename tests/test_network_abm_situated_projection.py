from __future__ import annotations

from dataclasses import replace
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
from narrative_dynamics.abm.situated_cognition_contracts import SituatedObservationRule
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryRecallCue,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognition,
)
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_projection import project_situated_narrative
from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeAuthority,
    NarrativeBeatPhase,
    NarrativeBeatKind,
    NarrativeEntitlementScope,
    NarrativeProjectionPolicy,
    NarrativeTemporalOrder,
)
from narrative_dynamics.abm.situated_story import (
    advance_situated_story,
    initialize_situated_story,
    objective_timeline,
    replay_situated_story,
    SituatedStory,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedSocialEvidence,
    SituatedSocialEvidenceKind,
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_social_memory import (
    advance_situated_social_memory,
)
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.situated_fixtures import office_model
from tests.test_network_abm_situated_percept_cognition import (
    initial_state as percept_initial_state,
    perception_model as cognitive_perception_model,
)
from tests.test_network_abm_situated_percept_memory_cognition import (
    recall_model,
)
from tests.test_network_abm_situated_percept_social_cognition import (
    bound_social_model,
)


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


def one_place_story(
    round_count: int = 4,
    *,
    two_agents: bool = False,
    place_id: str = "office",
):
    agents = (EmbodiedAgentSpec("alice", "staff", place_id),)
    if two_agents:
        agents += (EmbodiedAgentSpec("bob", "staff", place_id),)
    world = SituatedWorldModel(
        "one-place",
        "1.0",
        (PlaceSpec(place_id, "Office"),),
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


def _trajectory_case(
    tmp_path,
    *,
    late_carol_tell: bool = False,
    conflicting_alice_testimonies: bool = False,
    preload_alice_claim: bool = False,
    force_bob_move: bool = False,
):
    perception = cognitive_perception_model()
    cognition = cognitive_office_model()
    bob = next(item for item in cognition.agents if item.agent_id == "bob")
    bob = replace(
        bob,
        observation_rules=bob.observation_rules
        + (
            SituatedObservationRule(
                "tell-denied",
                "denied",
                "tell",
                SituatedActionKind.TELL,
                "told",
                "message",
                "The restructuring is denied.",
            ),
        ),
        rewards=tuple(
            replace(item, value=10.0)
            if force_bob_move and item.action_id == "move"
            else item
            for item in bob.rewards
        ),
    )
    cognition = replace(
        cognition,
        agents=tuple(
            bob if item.agent_id == "bob" else item for item in cognition.agents
        ),
    )
    story = initialize_situated_story(
        cognition.world_model,
        percept_initial_state(perception, door_open=True),
        perception_model=perception,
    )
    story = advance_situated_story(
        cognition.world_model,
        story,
        (
            SituatedActionIntent(
                "alice-inspect",
                "alice",
                SituatedActionKind.INSPECT,
                "memo",
            ),
        ),
    )
    alice_inspection = story.rounds[-1].events[0]
    story = advance_situated_story(
        cognition.world_model,
        story,
        (
            SituatedActionIntent(
                "alice-move",
                "alice",
                SituatedActionKind.MOVE,
                "records-open",
            ),
        ),
    )
    story = advance_situated_story(
        cognition.world_model,
        story,
        (
            SituatedActionIntent(
                "alice-denied",
                "alice",
                SituatedActionKind.TELL,
                message="The restructuring is denied.",
                source_event_ids=(alice_inspection.event_id,),
            ),
        ),
    )
    story = advance_situated_story(
        cognition.world_model,
        story,
        (
            SituatedActionIntent(
                "bob-move",
                "bob",
                SituatedActionKind.MOVE,
                "open-records",
            ),
        ),
    )
    if not conflicting_alice_testimonies:
        story = advance_situated_story(
            cognition.world_model,
            story,
            (
                SituatedActionIntent(
                    "bob-inspect",
                    "bob",
                    SituatedActionKind.INSPECT,
                    "memo",
                ),
            ),
        )
    else:
        if preload_alice_claim:
            story = advance_situated_story(
                cognition.world_model,
                story,
                (
                    SituatedActionIntent(
                        "alice-denied-again",
                        "alice",
                        SituatedActionKind.TELL,
                        message="The restructuring is denied.",
                    ),
                ),
            )
        story = advance_situated_story(
            cognition.world_model,
            story,
            (
                SituatedActionIntent(
                    "alice-approved",
                    "alice",
                    SituatedActionKind.TELL,
                    message="The restructuring is approved.",
                ),
            ),
        )
    if late_carol_tell:
        story = advance_situated_story(
            cognition.world_model,
            story,
            (
                SituatedActionIntent(
                    "carol-move-records",
                    "carol",
                    SituatedActionKind.MOVE,
                    "open-records",
                ),
            ),
        )
        story = advance_situated_story(
            cognition.world_model,
            story,
            (
                SituatedActionIntent(
                    "carol-approved",
                    "carol",
                    SituatedActionKind.TELL,
                    message="The restructuring is approved.",
                ),
            ),
        )
    cues = (
        SituatedMemoryRecallCue(
            "tell",
            "restructuring",
            event_kinds=(SituatedActionKind.TELL,),
        ),
        SituatedMemoryRecallCue(
            "inspect",
            "restructuring",
            event_kinds=(SituatedActionKind.INSPECT,),
        ),
    )
    memory_model = recall_model(
        {"bob": cues}, cognition=cognition, perception=perception
    )
    social_model = bound_social_model(memory_model)
    cognitive_state = initialize_situated_percept_memory_cognition(
        memory_model, story
    )
    social_state = initialize_situated_social_memory(
        social_model, cognitive_state
    )
    if preload_alice_claim:
        testimony = next(
            item
            for item in perceptual_timeline(perception, story, "bob")
            if item.kind is SituatedActionKind.TELL
            and item.actor_agent_id == "alice"
        )
        evidence = SituatedSocialEvidence(
            f"recalled-percept:{testimony.percept_id}",
            SituatedSocialEvidenceKind.TESTIMONY,
            "bob",
            "restructuring",
            "denied",
            testimony.round_index,
            testimony.source_event_id,
            source_agent_id="alice",
            memory_id=testimony.percept_id,
        )
        social_state = advance_situated_social_memory(
            social_model,
            cognitive_state,
            social_state,
            cognitive_state,
            (evidence,),
        ).next_state
    trajectory = simulate_situated_percept_social_cognition(
        tmp_path / "projection.sqlite3",
        memory_model,
        social_model,
        story,
        cognitive_state,
        social_state,
        round_count=2,
    )
    if not conflicting_alice_testimonies:
        return trajectory

    first_round = trajectory.rounds[0]
    events_by_action = {
        event.action_id: event
        for story_round in trajectory.initial_story.rounds
        for event in story_round.events
    }
    supporting_event = events_by_action[
        "alice-denied-again" if preload_alice_claim else "alice-denied"
    ]
    conflicting_event = events_by_action["alice-approved"]
    aggregate_update = advance_situated_social_memory(
        social_model,
        cognitive_state,
        social_state,
        first_round.next_cognitive_state,
        (
            SituatedSocialEvidence(
                "aggregate-a-support",
                SituatedSocialEvidenceKind.TESTIMONY,
                "bob",
                "restructuring",
                "denied",
                supporting_event.round_index,
                supporting_event.event_id,
                source_agent_id="alice",
            ),
            SituatedSocialEvidence(
                "aggregate-b-conflict",
                SituatedSocialEvidenceKind.TESTIMONY,
                "bob",
                "restructuring",
                "approved",
                conflicting_event.round_index,
                conflicting_event.event_id,
                source_agent_id="alice",
            ),
        ),
    )
    aggregate_round = replace(first_round, social_update=aggregate_update)
    return replace(
        trajectory,
        rounds=(aggregate_round,),
        final_story=aggregate_round.next_story,
        final_cognitive_state=aggregate_round.next_cognitive_state,
        final_social_state=aggregate_round.next_social_state,
    )


@pytest.fixture
def trajectory(tmp_path):
    return _trajectory_case(tmp_path)


@pytest.fixture
def multi_source_trajectory(tmp_path):
    return _trajectory_case(tmp_path, late_carol_tell=True)


@pytest.fixture
def prior_claim_trajectory(tmp_path):
    return _trajectory_case(tmp_path, preload_alice_claim=True)


@pytest.fixture
def moving_social_trajectory(tmp_path):
    return _trajectory_case(
        tmp_path,
        preload_alice_claim=True,
        force_bob_move=True,
    )


@pytest.fixture
def newly_superseded_claim_trajectory(tmp_path):
    return _trajectory_case(tmp_path, conflicting_alice_testimonies=True)


@pytest.fixture
def supported_then_superseded_claim_trajectory(tmp_path):
    return _trajectory_case(
        tmp_path,
        conflicting_alice_testimonies=True,
        preload_alice_claim=True,
    )


def _private_beat_kinds():
    return {
        NarrativeBeatKind.BELIEF_SHIFT,
        NarrativeBeatKind.ACTION_REVERSAL,
        NarrativeBeatKind.MEMORY_RECALL,
        NarrativeBeatKind.CLAIM_REVISION,
        NarrativeBeatKind.RELATIONSHIP_CHANGE,
    }


def test_projection_emits_supported_private_cognitive_and_social_changes(trajectory):
    projection = project_situated_narrative(
        trajectory.final_story,
        limited_policy("bob"),
        trajectory=trajectory,
    )
    kinds = {beat.kind for beat in projection.beats}
    assert NarrativeBeatKind.BELIEF_SHIFT in kinds
    assert NarrativeBeatKind.MEMORY_RECALL in kinds
    assert NarrativeBeatKind.CLAIM_REVISION in kinds
    assert NarrativeBeatKind.RELATIONSHIP_CHANGE in kinds
    assert projection.trajectory_hash == trajectory.content_hash
    assert all(
        item.owner_agent_id == "bob" for item in projection.cut.entitlements
    )


def test_include_flags_remove_private_internal_categories(trajectory):
    policy = replace(
        limited_policy("bob"),
        include_beliefs=False,
        include_memories=False,
        include_claim_revisions=False,
        include_relationship_changes=False,
    )
    projection = project_situated_narrative(
        trajectory.final_story, policy, trajectory=trajectory
    )
    assert not (
        {
            NarrativeBeatKind.BELIEF_SHIFT,
            NarrativeBeatKind.MEMORY_RECALL,
            NarrativeBeatKind.CLAIM_REVISION,
            NarrativeBeatKind.RELATIONSHIP_CHANGE,
        }
        & {beat.kind for beat in projection.beats}
    )


def test_trajectory_private_state_obeys_limited_and_objective_authority(trajectory):
    bob = project_situated_narrative(
        trajectory.final_story, limited_policy("bob"), trajectory=trajectory
    )
    private_ids = {
        entitlement.entitlement_id
        for entitlement in bob.cut.entitlements
        if entitlement.scope is NarrativeEntitlementScope.PRIVATE
    }
    private_beats = [beat for beat in bob.beats if beat.kind in _private_beat_kinds()]
    assert private_beats
    assert all(beat.active_pov_agent_id == "bob" for beat in private_beats)
    assert all(set(beat.entitlement_ids) <= private_ids for beat in private_beats)

    objective = project_situated_narrative(
        trajectory.final_story, objective_policy(), trajectory=trajectory
    )
    objective_private = [
        item
        for item in objective.cut.entitlements
        if item.scope is NarrativeEntitlementScope.PRIVATE
    ]
    assert {item.owner_agent_id for item in objective_private} == {"bob"}
    assert all(item.owner_agent_id is not None for item in objective_private)


def test_trajectory_must_end_at_the_projected_story(trajectory):
    with pytest.raises(ValueError, match="final story"):
        project_situated_narrative(
            objective_office_story(), objective_policy(), trajectory=trajectory
        )


def test_trajectory_salience_and_support_are_exact_source_artifacts(trajectory):
    projection = project_situated_narrative(
        trajectory.final_story, objective_policy(), trajectory=trajectory
    )
    by_kind = {
        kind: [beat for beat in projection.beats if beat.kind is kind]
        for kind in _private_beat_kinds()
    }

    changed_decisions = [
        decision
        for round_result in trajectory.rounds
        for decision in round_result.decisions
        if decision.prior_belief != decision.posterior_belief
    ]
    expected_beliefs = {
        decision.content_hash: 0.5
        * sum(
            abs(
                decision.posterior_belief.probabilities[key]
                - decision.prior_belief.probabilities[key]
            )
            for key in decision.prior_belief.probabilities
        )
        for decision in changed_decisions
    }
    assert {
        beat.supporting_artifacts[0].artifact_hash: beat.salience
        for beat in by_kind[NarrativeBeatKind.BELIEF_SHIFT]
    } == expected_beliefs

    admissions = [
        admission
        for round_result in trajectory.rounds
        for recall in round_result.recalls
        for admission in recall.admissions
        if not admission.consolidated
    ]
    assert {
        next(
            support.artifact_hash
            for support in beat.supporting_artifacts
            if support.artifact_kind == "memory_recall_admission"
        ): beat.salience
        for beat in by_kind[NarrativeBeatKind.MEMORY_RECALL]
    } == {
        admission.content_hash: admission.evidence_weight
        for admission in admissions
    }

    expected_relationships = {}
    expected_claims = {}
    for round_result in trajectory.rounds:
        prior_relationships = {
            (item.observer_agent_id, item.source_agent_id): item
            for item in round_result.social_update.prior_state.relationships
        }
        for item in round_result.social_update.next_state.relationships:
            prior = prior_relationships[(item.observer_agent_id, item.source_agent_id)]
            if item != prior:
                expected_relationships[item.content_hash] = (
                    abs(item.trust - prior.trust)
                    + abs(item.affinity - prior.affinity)
                )
        prior_claims = {
            item.claim_id: item
            for item in round_result.social_update.prior_state.claims
        }
        for item in round_result.social_update.next_state.claims:
            if item != prior_claims.get(item.claim_id):
                expected_claims[item.content_hash] = 1.0

    assert {
        next(
            item.artifact_hash
            for item in beat.supporting_artifacts
            if item.artifact_kind == "source_relationship"
        ): beat.salience
        for beat in by_kind[NarrativeBeatKind.RELATIONSHIP_CHANGE]
    } == expected_relationships
    claim_beats = {
        next(
            item.artifact_hash
            for item in beat.supporting_artifacts
            if item.artifact_kind == "consolidated_claim"
        ): beat.salience
        for beat in by_kind[NarrativeBeatKind.CLAIM_REVISION]
    }
    assert claim_beats == expected_claims
    terminal_hashes = {
        claim.content_hash
        for round_result in trajectory.rounds
        for claim in round_result.social_update.next_state.claims
        if claim.status
        in {
            SituatedClaimStatus.SUPERSEDED,
            SituatedClaimStatus.CONFIRMED,
            SituatedClaimStatus.CONTRADICTED,
            SituatedClaimStatus.FORGOTTEN,
        }
    }
    assert terminal_hashes
    assert all(claim_beats[item] == 1.0 for item in terminal_hashes)

    serialized = json.dumps(projection.to_dict(), sort_keys=True)
    assert "sqlite" not in serialized.lower()


def test_internal_places_and_metadata_cite_complete_trajectory_provenance(trajectory):
    projection = project_situated_narrative(
        trajectory.final_story, objective_policy(), trajectory=trajectory
    )
    decisions = {}
    previous_decisions = {}
    recalls = {}
    next_minds = {}
    for round_result in trajectory.rounds:
        prior_state = round_result.cognitive_round.cognitive_round.prior_state
        prior_minds = {mind.agent_id: mind for mind in prior_state.minds}
        events = {
            event.action_id: event
            for event in round_result.next_story.rounds[-1].events
        }
        for decision in round_result.decisions:
            decisions[decision.content_hash] = (
                decision,
                prior_minds[decision.agent_id],
                prior_state.round_index,
                events[decision.intent.action_id],
                previous_decisions.get(decision.agent_id),
            )
            previous_decisions[decision.agent_id] = decision
        for recall in round_result.recalls:
            for admission in recall.admissions:
                if not admission.consolidated:
                    recalls[admission.content_hash] = (
                        recall,
                        round_result.next_cognitive_state.round_index,
                    )
        for mind in round_result.next_cognitive_state.minds:
            next_minds[(round_result.next_cognitive_state.round_index, mind.agent_id)] = mind

    for beat in projection.beats:
        support_triples = {
            (support.artifact_kind, support.artifact_id, support.artifact_hash)
            for support in beat.supporting_artifacts
        }
        if beat.phase in {NarrativeBeatPhase.BELIEF, NarrativeBeatPhase.DECISION}:
            current_hash = next(
                support.artifact_hash
                for support in beat.supporting_artifacts
                if support.artifact_kind == "cognitive_decision"
                and support.artifact_hash in decisions
                and decisions[support.artifact_hash][0].round_index == beat.round_index
            )
            decision, prior_mind, prior_round, event, prior_decision = decisions[
                current_hash
            ]
            assert beat.place_id == prior_mind.own_place_id
            assert (
                "cognitive_mind_state",
                f"{decision.agent_id}:{prior_round}:prior",
                prior_mind.content_hash,
            ) in support_triples
            assert (
                "world_event",
                event.event_id,
                event.content_hash,
            ) in support_triples
            if beat.phase is NarrativeBeatPhase.DECISION:
                assert prior_decision is not None
                assert (
                    "cognitive_decision",
                    f"{prior_decision.agent_id}:{prior_decision.round_index}",
                    prior_decision.content_hash,
                ) in support_triples
        elif beat.phase is NarrativeBeatPhase.MEMORY_RECALL:
            admission_hash = next(
                support.artifact_hash
                for support in beat.supporting_artifacts
                if support.artifact_kind == "memory_recall_admission"
            )
            recall, round_index = recalls[admission_hash]
            assert beat.place_id == recall.prior_mind.own_place_id
            assert (
                "cognitive_mind_state",
                f"{recall.prior_mind.agent_id}:{round_index}:recall-prior",
                recall.prior_mind.content_hash,
            ) in support_triples
        elif beat.phase is NarrativeBeatPhase.SOCIAL:
            next_mind = next_minds[(beat.round_index, beat.active_pov_agent_id)]
            assert beat.place_id == next_mind.own_place_id
            assert (
                "cognitive_mind_state",
                f"{next_mind.agent_id}:{beat.round_index}:next",
                next_mind.content_hash,
            ) in support_triples


def test_limited_decision_metadata_comes_only_from_exact_self_percepts(trajectory):
    projection = project_situated_narrative(
        trajectory.final_story,
        limited_policy("bob"),
        trajectory=trajectory,
    )
    bob_decisions = {
        decision.content_hash: decision
        for round_result in trajectory.rounds
        for decision in round_result.decisions
        if decision.agent_id == "bob"
    }
    exact_self_percepts = {
        (percept.round_index, percept.kind): percept
        for percept in perceptual_timeline(
            trajectory.final_story.perception_model,
            trajectory.final_story,
            "bob",
        )
        if percept.agent_id == "bob"
        and percept.actor_agent_id == "bob"
        and percept.fidelity.value == "exact"
    }
    decision_beats = [
        beat
        for beat in projection.beats
        if beat.phase in {NarrativeBeatPhase.BELIEF, NarrativeBeatPhase.DECISION}
    ]
    assert decision_beats
    for beat in decision_beats:
        support_by_hash = {
            support.artifact_hash: support for support in beat.supporting_artifacts
        }
        decision = next(
            decision
            for content_hash, decision in bob_decisions.items()
            if content_hash in support_by_hash
            and decision.round_index == beat.round_index
        )
        percept = exact_self_percepts[(decision.round_index, decision.intent.kind)]
        assert beat.source_event_id == percept.source_event_id
        assert beat.round_index == percept.round_index
        assert (
            "percept",
            percept.percept_id,
            percept.content_hash,
        ) in {
            (support.artifact_kind, support.artifact_id, support.artifact_hash)
            for support in beat.supporting_artifacts
        }
        assert all(
            support.artifact_kind != "world_event"
            for support in beat.supporting_artifacts
        )

    by_round = {}
    for beat in decision_beats:
        by_round.setdefault(beat.round_index, []).append(beat)
    assert all(
        [beat.sequence for beat in beats] == list(range(1, len(beats) + 1))
        for beats in by_round.values()
    )


def test_limited_decision_without_exact_self_percept_stays_event_unbound(trajectory):
    first_round = trajectory.rounds[0]
    memory_round = first_round.cognitive_round
    percept_round = memory_round.percept_cognitive_round
    cognitive_round = percept_round.cognitive_round
    bob_decision = next(
        decision for decision in cognitive_round.decisions if decision.agent_id == "bob"
    )
    forged_decision = replace(
        bob_decision,
        intent=replace(
            bob_decision.intent,
            kind=SituatedActionKind.WAIT,
            target_id=None,
        ),
    )
    forged_cognitive_round = replace(
        cognitive_round,
        decisions=tuple(
            forged_decision if decision.agent_id == "bob" else decision
            for decision in cognitive_round.decisions
        ),
    )
    forged_percept_round = replace(
        percept_round, cognitive_round=forged_cognitive_round
    )
    forged_memory_round = replace(
        memory_round, percept_cognitive_round=forged_percept_round
    )
    forged_first_round = replace(
        first_round, cognitive_round=forged_memory_round
    )
    forged_trajectory = replace(
        trajectory,
        rounds=(forged_first_round,) + trajectory.rounds[1:],
    )

    projection = project_situated_narrative(
        forged_trajectory.final_story,
        limited_policy("bob"),
        trajectory=forged_trajectory,
    )
    beat = next(
        beat
        for beat in projection.beats
        if any(
            support.artifact_kind == "cognitive_decision"
            and support.artifact_hash == forged_decision.content_hash
            for support in beat.supporting_artifacts
        )
    )
    assert beat.source_event_id is None
    assert all(
        support.artifact_kind not in {"world_event", "percept"}
        for support in beat.supporting_artifacts
    )


def test_action_reversal_requires_a_changed_prior_projected_decision(trajectory):
    projection = project_situated_narrative(
        trajectory.final_story, objective_policy(), trajectory=trajectory
    )
    previous = {}
    expected_hashes = set()
    first_round_hashes = set()
    for round_result in trajectory.rounds:
        for decision in round_result.decisions:
            prior = previous.get(decision.agent_id)
            if prior is None:
                first_round_hashes.add(decision.content_hash)
            elif prior != decision.selected_action_id:
                expected_hashes.add(decision.content_hash)
            previous[decision.agent_id] = decision.selected_action_id
    reversal_hashes = {
        next(
            support.artifact_hash
            for support in beat.supporting_artifacts
            if support.artifact_kind == "cognitive_decision"
            and support.artifact_id.endswith(f":{beat.round_index}")
        )
        for beat in projection.beats
        if beat.kind is NarrativeBeatKind.ACTION_REVERSAL
    }
    assert reversal_hashes == expected_hashes
    assert reversal_hashes.isdisjoint(first_round_hashes)


def test_trajectory_chronology_uses_explicit_phases_without_rewriting_event_sequence(
    trajectory,
):
    projection = project_situated_narrative(
        trajectory.final_story, objective_policy(), trajectory=trajectory
    )
    expected_phase = {
        NarrativeBeatKind.MEMORY_RECALL: NarrativeBeatPhase.MEMORY_RECALL,
        NarrativeBeatKind.BELIEF_SHIFT: NarrativeBeatPhase.BELIEF,
        NarrativeBeatKind.ACTION_REVERSAL: NarrativeBeatPhase.DECISION,
        NarrativeBeatKind.PHYSICAL: NarrativeBeatPhase.WORLD,
        NarrativeBeatKind.INFORMATION: NarrativeBeatPhase.WORLD,
        NarrativeBeatKind.CAUSAL_PAYOFF: NarrativeBeatPhase.WORLD,
        NarrativeBeatKind.CLAIM_REVISION: NarrativeBeatPhase.SOCIAL,
        NarrativeBeatKind.RELATIONSHIP_CHANGE: NarrativeBeatPhase.SOCIAL,
    }
    assert all(beat.phase is expected_phase[beat.kind] for beat in projection.beats)

    phase_order = {
        NarrativeBeatPhase.MEMORY_RECALL: 0,
        NarrativeBeatPhase.BELIEF: 1,
        NarrativeBeatPhase.DECISION: 2,
        NarrativeBeatPhase.WORLD: 3,
        NarrativeBeatPhase.SOCIAL: 4,
    }
    for round_index in {beat.round_index for beat in projection.beats}:
        round_beats = [
            beat for beat in projection.beats if beat.round_index == round_index
        ]
        assert round_beats == sorted(
            round_beats,
            key=lambda beat: (
                phase_order[beat.phase],
                beat.sequence,
                beat.active_pov_agent_id or "",
                beat.beat_id,
            ),
        )

    round_six = [beat for beat in projection.beats if beat.round_index == 6]
    round_seven = [beat for beat in projection.beats if beat.round_index == 7]
    assert [beat.phase for beat in round_six].index(NarrativeBeatPhase.MEMORY_RECALL) < [
        beat.phase for beat in round_six
    ].index(NarrativeBeatPhase.BELIEF)
    assert [beat.phase for beat in round_seven].index(NarrativeBeatPhase.BELIEF) < [
        beat.phase for beat in round_seven
    ].index(NarrativeBeatPhase.DECISION)
    assert [beat.phase for beat in round_seven].index(NarrativeBeatPhase.DECISION) < [
        beat.phase for beat in round_seven
    ].index(NarrativeBeatPhase.WORLD)
    assert [beat.phase for beat in round_six].index(NarrativeBeatPhase.WORLD) < [
        beat.phase for beat in round_six
    ].index(NarrativeBeatPhase.SOCIAL)

    accepted_events = {
        event.event_id: event
        for round_result in trajectory.rounds
        for event in round_result.next_story.rounds[-1].events
    }
    world_beats = [
        beat
        for beat in projection.beats
        if beat.phase is NarrativeBeatPhase.WORLD
        and beat.source_event_id in accepted_events
    ]
    assert all(
        beat.sequence == accepted_events[beat.source_event_id].sequence
        for beat in world_beats
    )
    decision_bound_beats = [
        beat
        for beat in projection.beats
        if beat.phase in {NarrativeBeatPhase.BELIEF, NarrativeBeatPhase.DECISION}
    ]
    assert all(
        beat.sequence == accepted_events[beat.source_event_id].sequence
        for beat in decision_bound_beats
    )


def test_social_beats_bind_only_exact_trigger_evidence_and_cite_complete_support(
    multi_source_trajectory,
):
    projection = project_situated_narrative(
        multi_source_trajectory.final_story,
        objective_policy(),
        trajectory=multi_source_trajectory,
    )
    social_beats = [
        beat
        for beat in projection.beats
        if beat.kind
        in {
            NarrativeBeatKind.CLAIM_REVISION,
            NarrativeBeatKind.RELATIONSHIP_CHANGE,
        }
    ]
    assert social_beats
    updates = {
        item.social_update.content_hash: item.social_update
        for item in multi_source_trajectory.rounds
    }
    evidence = {
        item.content_hash: item
        for round_result in multi_source_trajectory.rounds
        for item in round_result.social_update.admitted_evidence
    }
    changed_claims = {
        item.content_hash: item
        for round_result in multi_source_trajectory.rounds
        for item in round_result.social_update.next_state.claims
    }
    changed_relationships = {
        item.content_hash: item
        for round_result in multi_source_trajectory.rounds
        for item in round_result.social_update.next_state.relationships
    }
    assert {
        item.source_agent_id for item in changed_claims.values()
    } >= {"alice", "carol"}

    for beat in social_beats:
        supports = {
            item.artifact_kind: item for item in beat.supporting_artifacts
        }
        assert supports["social_memory_update"].artifact_hash in updates
        if "social_evidence" not in supports:
            assert beat.source_event_id is None
            assert beat.kind in {
                NarrativeBeatKind.CLAIM_REVISION,
                NarrativeBeatKind.RELATIONSHIP_CHANGE,
            }
            continue
        trigger = evidence[supports["social_evidence"].artifact_hash]
        assert beat.source_event_id == trigger.event_id
        if beat.kind is NarrativeBeatKind.CLAIM_REVISION:
            claim = changed_claims[supports["consolidated_claim"].artifact_hash]
            assert trigger.observer_agent_id == claim.observer_agent_id
            assert trigger.topic_id == claim.topic_id
            if claim.status is SituatedClaimStatus.ACTIVE:
                assert trigger.source_agent_id == claim.source_agent_id
                assert trigger.symbol_id == claim.symbol_id
            else:
                assert trigger.source_agent_id is None
        else:
            relationship = changed_relationships[
                supports["source_relationship"].artifact_hash
            ]
            assert trigger.source_agent_id is None
            assert trigger.observer_agent_id == relationship.observer_agent_id
            assert any(
                claim.observer_agent_id == relationship.observer_agent_id
                and claim.source_agent_id == relationship.source_agent_id
                and claim.topic_id == trigger.topic_id
                for claim in changed_claims.values()
            )
    assert any(beat.source_event_id is None for beat in social_beats)
    assert any(beat.source_event_id is not None for beat in social_beats)


def test_same_update_claim_creation_and_supersession_stay_event_unbound(
    newly_superseded_claim_trajectory,
):
    update = newly_superseded_claim_trajectory.rounds[0].social_update
    claims = [
        claim
        for claim in update.next_state.claims
        if claim.observer_agent_id == "bob"
        and claim.source_agent_id == "alice"
        and claim.topic_id == "restructuring"
    ]
    assert update.prior_state.claims == ()
    assert {(claim.symbol_id, claim.status) for claim in claims} == {
        ("denied", SituatedClaimStatus.SUPERSEDED),
        ("approved", SituatedClaimStatus.ACTIVE),
    }
    assert {
        (item.symbol_id, item.kind)
        for item in update.admitted_evidence
    } == {
        ("denied", SituatedSocialEvidenceKind.TESTIMONY),
        ("approved", SituatedSocialEvidenceKind.TESTIMONY),
    }

    projection = project_situated_narrative(
        newly_superseded_claim_trajectory.final_story,
        objective_policy(),
        trajectory=newly_superseded_claim_trajectory,
    )
    beats_by_claim_hash = {
        support.artifact_hash: beat
        for beat in projection.beats
        if beat.kind is NarrativeBeatKind.CLAIM_REVISION
        for support in beat.supporting_artifacts
        if support.artifact_kind == "consolidated_claim"
    }
    for claim in claims:
        beat = beats_by_claim_hash[claim.content_hash]
        supports = {
            support.artifact_kind: support
            for support in beat.supporting_artifacts
        }
        assert beat.source_event_id is None
        assert "social_evidence" not in supports
        assert supports["social_memory_update"].artifact_hash == update.content_hash
        assert supports["consolidated_claim"].artifact_hash == claim.content_hash


def test_same_update_support_and_supersession_stay_event_unbound(
    supported_then_superseded_claim_trajectory,
):
    update = supported_then_superseded_claim_trajectory.rounds[0].social_update
    prior = next(
        claim
        for claim in update.prior_state.claims
        if claim.observer_agent_id == "bob"
        and claim.source_agent_id == "alice"
        and claim.topic_id == "restructuring"
        and claim.symbol_id == "denied"
    )
    changed = next(
        claim for claim in update.next_state.claims if claim.claim_id == prior.claim_id
    )
    assert prior.status is SituatedClaimStatus.ACTIVE
    assert changed.status is SituatedClaimStatus.SUPERSEDED
    assert changed.support_count == prior.support_count + 1
    assert set(changed.event_ids) - set(prior.event_ids) == {
        next(
            item.event_id
            for item in update.admitted_evidence
            if item.symbol_id == "denied"
        )
    }
    assert any(
        item.symbol_id == "approved"
        and item.kind is SituatedSocialEvidenceKind.TESTIMONY
        for item in update.admitted_evidence
    )

    projection = project_situated_narrative(
        supported_then_superseded_claim_trajectory.final_story,
        objective_policy(),
        trajectory=supported_then_superseded_claim_trajectory,
    )
    beat = next(
        beat
        for beat in projection.beats
        if beat.kind is NarrativeBeatKind.CLAIM_REVISION
        and any(
            support.artifact_kind == "consolidated_claim"
            and support.artifact_hash == changed.content_hash
            for support in beat.supporting_artifacts
        )
    )
    supports = {
        support.artifact_kind: support for support in beat.supporting_artifacts
    }
    assert beat.source_event_id is None
    assert "social_evidence" not in supports
    assert supports["social_memory_update"].artifact_hash == update.content_hash
    assert supports["consolidated_claim"].artifact_hash == changed.content_hash


def test_authored_trajectory_keeps_unbound_aggregate_social_beats(
    multi_source_trajectory,
):
    authored = tuple(
        item.event_id for item in objective_timeline(multi_source_trajectory.final_story)
    )
    projection = project_situated_narrative(
        multi_source_trajectory.final_story,
        objective_policy(
            temporal_order=NarrativeTemporalOrder.AUTHORED,
            authored_event_order=authored,
        ),
        trajectory=multi_source_trajectory,
    )
    assert any(
        beat.phase is NarrativeBeatPhase.SOCIAL and beat.source_event_id is None
        for beat in projection.beats
    )
    bound_order = []
    for beat in projection.beats:
        if beat.source_event_id is None:
            continue
        if not bound_order or beat.source_event_id != bound_order[-1]:
            bound_order.append(beat.source_event_id)
    assert tuple(bound_order) == authored


def test_existing_claim_and_directed_relationship_bind_the_exact_verification(
    prior_claim_trajectory,
):
    projection = project_situated_narrative(
        prior_claim_trajectory.final_story,
        objective_policy(),
        trajectory=prior_claim_trajectory,
    )
    social_beats = [
        beat
        for beat in projection.beats
        if beat.kind
        in {
            NarrativeBeatKind.CLAIM_REVISION,
            NarrativeBeatKind.RELATIONSHIP_CHANGE,
        }
    ]
    assert social_beats
    evidence_by_hash = {
        item.content_hash: item
        for round_result in prior_claim_trajectory.rounds
        for item in round_result.social_update.admitted_evidence
    }
    for beat in social_beats:
        supports = {
            item.artifact_kind: item for item in beat.supporting_artifacts
        }
        trigger = evidence_by_hash[supports["social_evidence"].artifact_hash]
        assert trigger.kind is SituatedSocialEvidenceKind.VERIFICATION
        assert trigger.observer_agent_id == "bob"
        assert trigger.source_agent_id is None
        assert beat.source_event_id == trigger.event_id


def test_social_changes_use_the_post_world_destination_place(moving_social_trajectory):
    round_result = moving_social_trajectory.rounds[0]
    prior_bob = next(
        item
        for item in round_result.cognitive_round.cognitive_round.prior_state.minds
        if item.agent_id == "bob"
    )
    next_bob = next(
        item
        for item in round_result.next_cognitive_state.minds
        if item.agent_id == "bob"
    )
    assert prior_bob.own_place_id != next_bob.own_place_id

    projection = project_situated_narrative(
        moving_social_trajectory.final_story,
        objective_policy(),
        trajectory=moving_social_trajectory,
    )
    first_round_social = [
        beat
        for beat in projection.beats
        if beat.round_index == round_result.next_cognitive_state.round_index
        and beat.phase is NarrativeBeatPhase.SOCIAL
        and beat.active_pov_agent_id == "bob"
    ]
    assert first_round_social
    assert {beat.place_id for beat in first_round_social} == {
        next_bob.own_place_id
    }


def test_foreign_social_evidence_event_is_rejected(trajectory):
    round_result = trajectory.rounds[0]
    evidence = round_result.social_update.admitted_evidence
    forged_evidence = (replace(evidence[0], event_id="foreign-event"),) + evidence[1:]
    forged_update = replace(
        round_result.social_update,
        admitted_evidence=forged_evidence,
    )
    forged_round = replace(round_result, social_update=forged_update)
    forged_trajectory = replace(
        trajectory,
        rounds=(forged_round,) + trajectory.rounds[1:],
    )

    with pytest.raises(ValueError, match="social evidence event"):
        project_situated_narrative(
            forged_trajectory.final_story,
            objective_policy(),
            trajectory=forged_trajectory,
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
    assert projection.beats[0].place_id is None
    assert projection.scenes[0].place_id is None


def test_unknown_location_beats_each_start_their_own_scene():
    story = detected_story()
    story, _ = _append_single_alice_tell(story, "alice-tells-again")
    projection = project_situated_narrative(story, limited_policy("bob"))
    assert len(projection.beats) == 2
    assert all(beat.place_id is None for beat in projection.beats)
    assert len(projection.scenes) == 2
    assert all(scene.place_id is None for scene in projection.scenes)
    assert all(len(scene.beat_ids) == 1 for scene in projection.scenes)
    assert '"place_id": null' in json.dumps(projection.to_dict(), sort_keys=True)
    assert projection.content_hash == project_situated_narrative(
        story, limited_policy("bob")
    ).content_hash


def test_known_world_place_named_undisclosed_preserves_normal_grouping():
    projection = project_situated_narrative(
        one_place_story(round_count=2, place_id="undisclosed"),
        objective_policy(),
    )
    assert {beat.place_id for beat in projection.beats} == {"undisclosed"}
    assert len(projection.scenes) == 1
    assert projection.scenes[0].place_id == "undisclosed"
    assert len(projection.scenes[0].beat_ids) == 2


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


def test_internal_candidates_are_not_boundary_or_event_gap_anchors(trajectory):
    projection = project_situated_narrative(
        trajectory.final_story,
        limited_policy(
            "bob",
            salience_weights={kind: 0.0 for kind in NarrativeBeatKind},
            minimum_salience=1.0,
            maximum_event_omission_gap=2,
            required_causal_coverage=0.0,
        ),
        trajectory=trajectory,
    )
    assert all(beat.phase is NarrativeBeatPhase.WORLD for beat in projection.beats)

    percepts = perceptual_timeline(
        trajectory.final_story.perception_model,
        trajectory.final_story,
        "bob",
    )
    position_by_event = {
        percept.source_event_id: index for index, percept in enumerate(percepts)
    }
    selected_positions = [
        position_by_event[beat.source_event_id] for beat in projection.beats
    ]
    assert selected_positions[0] == 0
    assert selected_positions[-1] == len(percepts) - 1
    assert all(
        right - left <= 2
        for left, right in zip(selected_positions, selected_positions[1:])
    )


def test_selection_does_not_conflate_support_versions_with_same_artifact_id():
    from narrative_dynamics.abm.situated_projection import (
        _Candidate,
        _candidate_beat_id,
        _select_candidates,
    )
    from narrative_dynamics.abm.situated_projection_contracts import (
        NarrativeEntitlement,
        NarrativeSupportRef,
    )

    def candidate(artifact_hash: str, sequence: int, magnitude: float):
        support = NarrativeSupportRef(
            "cognitive_decision", "alice:1", artifact_hash
        )
        entitlement = NarrativeEntitlement(
            f"entitlement-{sequence}",
            NarrativeEntitlementScope.PRIVATE,
            "alice",
            1,
            (),
            (support,),
        )
        return _Candidate(
            None,
            1,
            sequence,
            "office",
            "alice",
            ("alice",),
            NarrativeBeatKind.BELIEF_SHIFT,
            support,
            entitlement,
            (),
            magnitude=magnitude,
            phase=NarrativeBeatPhase.BELIEF,
        )

    selected_version = candidate("sha256:" + "a" * 64, 1, 1.0)
    omitted_version = candidate("sha256:" + "b" * 64, 2, 0.0)
    selected = _select_candidates(
        (selected_version, omitted_version),
        limited_policy(
            "alice",
            minimum_salience=0.5,
            required_causal_coverage=0.0,
        ),
    )
    assert selected == (selected_version,)
    same_support_at_another_sequence = replace(selected_version, sequence=2)
    assert _candidate_beat_id(selected_version) != _candidate_beat_id(
        same_support_at_another_sequence
    )


def test_causal_payoff_weight_controls_selection_kind_and_salience():
    story, (_, cause_id, payoff_id) = detected_causal_story()
    story, last_id = _append_single_alice_tell(story, "alice-last")
    projection = project_situated_narrative(
        story,
        objective_policy(
            salience_weights={
                NarrativeBeatKind.INFORMATION: 0.0,
                NarrativeBeatKind.CAUSAL_PAYOFF: 1.0,
            },
            minimum_salience=1.0,
            required_causal_coverage=0.0,
        ),
    )
    by_event = {beat.source_event_id: beat for beat in projection.beats}
    assert set(by_event) == {
        objective_timeline(story)[0].event_id,
        cause_id,
        payoff_id,
        last_id,
    }
    assert by_event[payoff_id].kind is NarrativeBeatKind.CAUSAL_PAYOFF
    assert by_event[payoff_id].salience == 1.0
    assert by_event[payoff_id].cause_beat_ids == (by_event[cause_id].beat_id,)


def test_cause_free_selected_event_keeps_its_underlying_kind_and_weight():
    world = office_model()
    story = SituatedStory(
        world.model_id,
        world.content_hash,
        initialize_situated_world(world),
    )
    story = advance_situated_story(
        world,
        story,
        (SituatedActionIntent("first", "alice", SituatedActionKind.WAIT),),
    )
    story = advance_situated_story(
        world,
        story,
        (SituatedActionIntent(
            "cause", "alice", SituatedActionKind.TAKE, target_id="memo"
        ),),
    )
    cause_id = next(
        event.event_id
        for event in story.rounds[-1].events
        if event.actor_agent_id == "alice"
    )
    story = advance_situated_story(
        world,
        story,
        (SituatedActionIntent(
            "payoff",
            "alice",
            SituatedActionKind.TELL,
            message="status update",
            source_event_ids=(cause_id,),
        ),),
    )
    payoff_id = next(
        event.event_id
        for event in story.rounds[-1].events
        if event.actor_agent_id == "alice"
    )
    story = advance_situated_story(
        world,
        story,
        (SituatedActionIntent("last", "alice", SituatedActionKind.WAIT),),
    )

    projection = project_situated_narrative(
        story,
        objective_policy(
            salience_weights={
                NarrativeBeatKind.PHYSICAL: 0.0,
                NarrativeBeatKind.INFORMATION: 1.0,
                NarrativeBeatKind.CAUSAL_PAYOFF: 0.0,
            },
            minimum_salience=1.0,
            required_causal_coverage=0.0,
        ),
    )

    by_event = {beat.source_event_id: beat for beat in projection.beats}
    assert cause_id not in by_event
    assert by_event[payoff_id].kind is NarrativeBeatKind.INFORMATION
    assert by_event[payoff_id].salience == 1.0
    assert by_event[payoff_id].cause_beat_ids == ()


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


def test_multi_pov_perceived_endpoints_do_not_disclose_objective_causal_edges():
    story, _ = detected_causal_story()
    projection = project_situated_narrative(
        story,
        NarrativeProjectionPolicy(
            "alice-and-bob",
            "1.0",
            NarrativeAuthority.MULTI_POV,
            pov_agent_ids=("alice", "bob"),
            required_causal_coverage=1.0,
        ),
    )
    assert {beat.active_pov_agent_id for beat in projection.beats} == {
        "alice",
        "bob",
    }
    assert all(not beat.cause_beat_ids for beat in projection.beats)
    assert all(
        beat.kind is not NarrativeBeatKind.CAUSAL_PAYOFF
        for beat in projection.beats
    )


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
