"""R2.M5e.1: legitimate V19 trust history before a new native TELL.

Compare confirmed and contradicted *actual source histories* in one
immutable V19 world/model, then let the same native SQLite-backed
cognitive/social engine produce a new TELL and Bob's subsequent action.
This qualifies a source mechanism, NOT an isolated trust-only causal
estimate: prior private evidence and belief differ between branches.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import (
    ObservationChannel, SituatedActionIntent, SituatedActionKind,
)
from narrative_dynamics.abm.situated_story import replay_situated_story
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryRecallCue,
)
from narrative_dynamics.abm.situated_percept_cognition import (
    admit_situated_percepts, perceptual_timeline,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptFidelity,
)

_SCHEMA = "r2-m5e1-v19-confirmation-contradiction-new-tell-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_HISTORY = (
    ("confirmed_history", "The restructuring is approved.", 0.6, "confirmed"),
    ("contradicted_history", "The restructuring is denied.", 0.3, "contradicted"),
)
_ROOT = frozenset({
    "schema", "source_owner", "source_world_hash", "source_perception_hash",
    "source_cognitive_model_hash", "source_memory_model_hash",
    "source_social_model_hash", "initial_world_hash",
    "native_direct_trust_enabled", "authored_common_action_ids",
    "single_history_intervention", "same_world_positions_before_native_learning",
    "source_story_lineage", "historical_context_match",
    "trust_only_causal_effect", "post_trust_action_provenance",
    "cross_mechanism_ranking", "sample_size_per_arm",
    "confidence_interval", "external_validation", "arms",
})
_ARM = frozenset({
    "historical_story_hash", "historical_world_state_hash",
    "historical_tell_event_hash", "historical_tell_event_id",
    "historical_tell_fidelity_bob", "initial_cognitive_checkpoint_hash",
    "initial_social_checkpoint_hash", "first_native_round_hash",
    "first_native_story_hash", "source_social_trust_before_native",
    "source_social_trust_after_learning", "history_admitted_testimony",
    "history_admitted_private_verification", "history_claim_status",
    "bob_belief_after_source_history", "native_new_tell_round",
    "native_new_tell_event_id",
    "native_new_tell_event_hash", "native_new_tell_fidelity_bob",
    "native_new_tell_causal_source_ids",
    "native_bob_direct_tell_admission_id",
    "native_bob_direct_tell_source_trust",
    "native_bob_direct_tell_prior_approved",
    "native_bob_direct_tell_posterior_approved",
    "native_bob_decision_hash", "native_bob_selected_action",
    "native_bob_feasible_actions", "native_bob_actual_event_hash",
    "native_bob_actual_outcome", "native_bob_actual_success",
    "native_bob_source_admitted_new_tell", "native_bob_prior_action_belief",
    "native_bob_posterior_action_belief", "native_world_after_round_hash",
    "native_social_after_round_hash",
})


def _digest(x: object) -> bool:
    return type(x) is str and _HASH.fullmatch(x) is not None


@dataclass(frozen=True)
class SourceHistoryNewTellReport:
    """Immutable redacted source-derived dossier. Never a trust-only estimator."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("trust history dossier requires source JSON")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict or set(d) != _ROOT
            or d["schema"] != _SCHEMA
            or d["source_owner"] != "V19_ONE_WORLD_AND_ORIGINAL_SOCIAL_COGNITION"
            or d["native_direct_trust_enabled"] is not True
            or d["single_history_intervention"] != "PREVIOUS_SOURCE_TELL_CONTENT_ONLY"
            or d["same_world_positions_before_native_learning"] is not True
            or d["source_story_lineage"] != "DIVERGENT_HISTORICAL_STORIES"
            or d["historical_context_match"] != "NOT_MATCHED_PRIVATE_EVIDENCE_OR_BELIEF"
            or d["trust_only_causal_effect"] != "NOT_IDENTIFIED"
            or d["post_trust_action_provenance"] != "NATIVE_V19_WORLD_EVENT"
            or d["cross_mechanism_ranking"] != "FORBIDDEN"
            or d["sample_size_per_arm"] != 1
            or d["confidence_interval"] != "NOT_ESTIMABLE"
            or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or any(not _digest(d.get(k)) for k in (
                "source_world_hash", "source_perception_hash",
                "source_cognitive_model_hash", "source_memory_model_hash",
                "source_social_model_hash", "initial_world_hash",
            ))
            or type(d["authored_common_action_ids"]) is not list
            or len(d["authored_common_action_ids"]) != 7
            or len(set(d["authored_common_action_ids"])) != 7
            or type(d["arms"]) is not dict
            or set(d["arms"]) != {x[0] for x in _HISTORY}
        ):
            raise ValueError("M5e.1 source cannot imply unverified power or trust causality")
        ids = []
        for name, _, trust, status in _HISTORY:
            arm = d["arms"][name]
            if (
                type(arm) is not dict or set(arm) != _ARM
                or any(not _digest(arm.get(k)) for k in (
                    "historical_story_hash", "historical_world_state_hash",
                    "historical_tell_event_hash", "initial_cognitive_checkpoint_hash",
                    "initial_social_checkpoint_hash", "first_native_round_hash",
                    "first_native_story_hash", "native_new_tell_event_hash",
                    "native_bob_decision_hash", "native_bob_actual_event_hash",
                    "native_world_after_round_hash", "native_social_after_round_hash",
                ))
                or arm["historical_tell_fidelity_bob"] != "exact"
                or arm["native_new_tell_fidelity_bob"] != "exact"
                or arm["native_new_tell_round"] != 9
                or arm["history_claim_status"] != status
                or arm["source_social_trust_before_native"] != 0.5
                or arm["source_social_trust_after_learning"] != trust
                or arm["native_bob_direct_tell_source_trust"] != trust
                or arm["history_admitted_testimony"] != 1
                or arm["history_admitted_private_verification"] != 1
                or arm["native_bob_source_admitted_new_tell"] is not True
                or type(arm["native_new_tell_causal_source_ids"]) is not list
                or len(arm["native_new_tell_causal_source_ids"]) != 1
                or type(arm["native_bob_feasible_actions"]) is not list
                or arm["native_bob_feasible_actions"] !=
                   sorted(set(arm["native_bob_feasible_actions"]))
                or arm["native_bob_selected_action"] not in
                   arm["native_bob_feasible_actions"]
                or type(arm["native_bob_actual_success"]) is not bool
                or type(arm["native_bob_actual_outcome"]) is not str
                or not arm["native_bob_actual_outcome"]
                or any(type(arm[k]) not in (int, float) or not 0 <= arm[k] <= 1
                       for k in (
                           "bob_belief_after_source_history",
                           "native_bob_direct_tell_prior_approved",
                           "native_bob_direct_tell_posterior_approved",
                           "native_bob_prior_action_belief",
                           "native_bob_posterior_action_belief",
                       ))
            ):
                raise ValueError("M5e.1 source history or native trust admission forged")
            ids.append((arm["historical_tell_event_id"],
                        arm["historical_tell_event_hash"],
                        arm["historical_story_hash"]))
        if (ids[0][0] != ids[1][0] or ids[0][1] == ids[1][1]
                or ids[0][2] == ids[1][2]):
            raise ValueError("one historical event ID requires two distinct Story revisions")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_verified_trust_then_new_tell() -> SourceHistoryNewTellReport:
    from tests.test_network_epistemic_influence import source_epistemic_case

    original, root, _, _ = source_epistemic_case()
    old_memory = original.percept_memory_model
    world = old_memory.cognitive_model.world_model
    perception = old_memory.perception_model
    # Source V19 social policy enables the exact M5e.0 newly accepted
    # direct-TELL weight. One policy/model is shared by BOTH histories.
    # One source cognitive model allows the ORIGINAL planner to choose
    # repeatable TELL after trust is learned. Bob remains at open because
    # his original MOVE is eligible only when his own place is records.
    cognition = replace(
        old_memory.cognitive_model,
        agents=tuple(
            replace(agent, actions=tuple(
                replace(action, repeatable=True)
                if action.action_id == "tell" else action
                for action in agent.actions
            )) if agent.agent_id == "alice" else agent
            for agent in old_memory.cognitive_model.agents
        ),
    )
    social = replace(
        original.social_memory_model,
        memory_cognitive_model=replace(
            original.social_memory_model.memory_cognitive_model,
            cognitive_model=cognition,
        ),
        policy=replace(
            original.social_memory_model.policy,
            trust_weight_direct_testimony=True,
        ),
    )
    alice_cue = SituatedMemoryRecallCue(
        "m5e1-alice-inspect", "restructuring",
        required_place_ids=("open",),
        event_kinds=(SituatedActionKind.INSPECT,),
        channels=(ObservationChannel.INSPECTION,),
    )
    memory = replace(
        old_memory,
        cognitive_model=cognition,
        agents=tuple(
            replace(agent, cues=(alice_cue,))
            if agent.agent_id == "alice" else agent
            for agent in old_memory.agents
        ),
    )
    # One historical TELL message is the only changed input. All actual
    # V19 authored action IDs and other intents match exactly.
    inspected = SituatedActionIntent(
        "m5e1-alice-inspect", "alice", SituatedActionKind.INSPECT,
        target_id="memo",
    )
    opened = SituatedActionIntent(
        "m5e1-alice-authorized-open", "alice",
        SituatedActionKind.OPEN_PASSAGE, target_id="records-open",
    )
    moved = SituatedActionIntent(
        "m5e1-alice-move", "alice", SituatedActionKind.MOVE,
        target_id="records-open",
    )
    # Original source event ID follows the source-owned canonical
    # synchronous round, not a hand-written social claim.
    first_three = replay_situated_story(
        world, root.initial_state,
        ((inspected,), (opened,), (moved,)),
        perception_model=perception,
    )
    observed_inspection = next(
        e for e in first_three.rounds[0].events
        if e.action_id == inspected.action_id and e.success
    )
    bob_move = SituatedActionIntent(
        "m5e1-bob-enter-records", "bob", SituatedActionKind.MOVE,
        target_id="open-records",
    )
    bob_inspect = SituatedActionIntent(
        "m5e1-bob-own-inspect", "bob", SituatedActionKind.INSPECT,
        target_id="memo",
    )
    bob_return = SituatedActionIntent(
        "m5e1-bob-return-open", "bob", SituatedActionKind.MOVE,
        target_id="records-open",
    )
    snapshots = {}
    data = {}
    for label, previous_message, expected_trust, _ in _HISTORY:
        historical_tell = SituatedActionIntent(
            "m5e1-alice-prior-tell", "alice", SituatedActionKind.TELL,
            message=previous_message,
            source_event_ids=(observed_inspection.event_id,),
        )
        schedule = (
            (inspected,), (opened,), (moved,),
            (historical_tell,), (bob_move,), (bob_inspect,), (bob_return,),
        )
        story = replay_situated_story(
            world, root.initial_state, schedule, perception_model=perception,
        )
        if not all(
            next(e for e in row.events
                 if e.action_id == act[0].action_id).success
            for row, act in zip(story.rounds, schedule, strict=True)
        ):
            raise ValueError("M5e.1 expected all native authored history actions")
        body = {
            a.agent_id: a.place_id for a in story.current_state.agents
        }
        gate = {
            p.passage_id: p.open for p in story.current_state.passages
        }
        snapshots[label] = (body, gate)
        prior = initialize_situated_percept_memory_cognition(memory, story)
        social_prior = initialize_situated_social_memory(social, prior)
        older = next(
            e for e in story.rounds[3].events
            if e.action_id == historical_tell.action_id
        )
        bob_historical = [
            p for p in perceptual_timeline(perception, story, "bob")
            if p.source_event_id == older.event_id
            and p.source_event_hash == older.content_hash
            and p.fidelity is SituatedPerceptFidelity.EXACT
            and p.kind is SituatedActionKind.TELL
            and p.actor_agent_id == "alice"
        ]
        if len(bob_historical) != 1:
            raise ValueError("M5e.1 prior testimony must be original Bob EXACT")
        with TemporaryDirectory(prefix=f"v19-m5e1-{label}-") as tmp:
            db = str(Path(tmp) / "memory.sqlite3")
            first = simulate_situated_percept_social_cognitive_round(
                db, memory, social, story, prior, social_prior,
            )
            bob_before = next(x.trust for x in social_prior.relationships
                              if x.observer_agent_id == "bob"
                              and x.source_agent_id == "alice")
            bob_after = next(x.trust for x in first.next_social_state.relationships
                             if x.observer_agent_id == "bob"
                             and x.source_agent_id == "alice")
            histories = [
                e for e in first.social_update.admitted_evidence
                if e.observer_agent_id == "bob"
            ]
            past_tell = sum(
                e.kind.value == "testimony" and e.source_agent_id == "alice"
                for e in histories
            )
            own_verify = sum(
                e.kind.value == "verification" and e.source_agent_id is None
                for e in histories
            )
            statuses = sorted(
                c.status.value for c in first.next_social_state.claims
                if c.observer_agent_id == "bob"
            )
            if (bob_before != 0.5 or bob_after != expected_trust
                    or past_tell != 1 or own_verify != 1):
                raise ValueError("M5e.1 source evidence did not produce learned trust")
            # Round 8: original social owner processes source history,
            # learns source trust 0.6 (confirmed) or 0.3 (contradicted).
            # A contemporaneous TELL before the end-of-round trust
            # update is NOT accepted as the post-revision target.
            prior_native_tells = [
                e for e in first.next_story.rounds[-1].events
                if e.actor_agent_id == "alice"
                and e.kind is SituatedActionKind.TELL and e.success
            ]
            if len(prior_native_tells) != 1:
                raise ValueError("M5e.1 first source round must emit Alice native TELL")
            # Round 9: AFTER trust has been learned, the SAME source
            # action planner produces a truly NEW, repeatable TELL.
            second = simulate_situated_percept_social_cognitive_round(
                db, memory, social,
                first.next_story, first.next_cognitive_state,
                first.next_social_state,
            )
            new_tells = [
                e for e in second.next_story.rounds[-1].events
                if e.actor_agent_id == "alice"
                and e.kind is SituatedActionKind.TELL and e.success
            ]
            if len(new_tells) != 1:
                raise ValueError("M5e.1 no new Alice TELL after native trust learning")
            new_tell = new_tells[0]
            # The original planner must name the most recently actually
            # observed eligible source: its own native round-8 TELL. This
            # is stricter than accepting an arbitrary earlier event hash.
            if (new_tell.round_index != 9 or
                    new_tell.cause_event_ids != (prior_native_tells[0].event_id,)):
                raise ValueError("M5e.1 new native TELL lacks preceding source event")
            current_trust = next(
                x.trust for x in second.next_social_state.relationships
                if x.observer_agent_id == "bob"
                and x.source_agent_id == "alice"
            )
            if current_trust != bob_after:
                raise ValueError("M5e.1 trust drifted before later TELL admission")
            bob_percepts = perceptual_timeline(
                perception, second.next_story, "bob",
            )
            new_received = [
                p for p in bob_percepts
                if p.source_event_id == new_tell.event_id
                and p.source_event_hash == new_tell.content_hash
                and p.actor_agent_id == "alice"
                and p.kind is SituatedActionKind.TELL
                and p.fidelity is SituatedPerceptFidelity.EXACT
            ]
            if len(new_received) != 1:
                raise ValueError("M5e.1 Bob did not receive the later source EXACT TELL")
            bob_mind = next(
                x for x in second.next_cognitive_state.minds
                if x.agent_id == "bob"
            )
            bob_model = next(
                x for x in memory.cognitive_model.agents if x.agent_id == "bob"
            )
            direct = admit_situated_percepts(
                bob_model, bob_mind, bob_percepts,
                _source_trust_by_source={"alice": current_trust},
            )
            new_admit = [
                x for x in direct.admissions
                if x.event_id == new_tell.event_id
            ]
            if len(new_admit) != 1 or new_admit[0].source_trust != current_trust:
                raise ValueError("M5e.1 learned trust was not used for new TELL")
            # Round 10: original native Agent decision must actually
            # consume the new TELL, select and execute one world action.
            third = simulate_situated_percept_social_cognitive_round(
                db, memory, social,
                second.next_story, second.next_cognitive_state,
                second.next_social_state,
            )
        decision = next(x for x in third.decisions if x.agent_id == "bob")
        if new_admit[0].observation_id not in decision.admitted_observation_ids:
            raise ValueError("M5e.1 native decision did not consume new TELL")
        event = next(
            x for x in third.next_story.rounds[-1].events
            if x.action_id == decision.intent.action_id
            and x.actor_agent_id == "bob"
        )
        result = {
            "historical_story_hash": story.content_hash,
            "historical_world_state_hash": story.current_state.content_hash,
            "historical_tell_event_hash": older.content_hash,
            "historical_tell_event_id": older.event_id,
            "historical_tell_fidelity_bob": bob_historical[0].fidelity.value,
            "initial_cognitive_checkpoint_hash": prior.content_hash,
            "initial_social_checkpoint_hash": social_prior.content_hash,
            "first_native_round_hash": first.content_hash,
            "first_native_story_hash": first.next_story.content_hash,
            "source_social_trust_before_native": bob_before,
            "source_social_trust_after_learning": bob_after,
            "history_admitted_testimony": past_tell,
            "history_admitted_private_verification": own_verify,
            "history_claim_status": statuses[0] if len(statuses) == 1 else "INVALID",
            "bob_belief_after_source_history": next(
                a.posterior_belief.probabilities["approved"]
                for a in first.decisions if a.agent_id == "bob"
            ),
            "native_new_tell_round": new_tell.round_index,
            "native_new_tell_event_id": new_tell.event_id,
            "native_new_tell_event_hash": new_tell.content_hash,
            "native_new_tell_fidelity_bob": new_received[0].fidelity.value,
            "native_new_tell_causal_source_ids": list(new_tell.cause_event_ids),
            "native_bob_direct_tell_admission_id": new_admit[0].observation_id,
            "native_bob_direct_tell_source_trust": new_admit[0].source_trust,
            "native_bob_direct_tell_prior_approved":
                new_admit[0].prior_belief.probabilities["approved"],
            "native_bob_direct_tell_posterior_approved":
                new_admit[0].posterior_belief.probabilities["approved"],
            "native_bob_decision_hash": decision.content_hash,
            "native_bob_selected_action": decision.selected_action_id,
            "native_bob_feasible_actions": list(decision.feasible_action_ids),
            "native_bob_actual_event_hash": event.content_hash,
            "native_bob_actual_outcome": event.outcome,
            "native_bob_actual_success": event.success,
            "native_bob_source_admitted_new_tell": True,
            "native_bob_prior_action_belief":
                decision.prior_belief.probabilities["approved"],
            "native_bob_posterior_action_belief":
                decision.posterior_belief.probabilities["approved"],
            "native_world_after_round_hash": third.next_story.content_hash,
            "native_social_after_round_hash": third.next_social_state.content_hash,
        }
        data[label] = result
    a, b = (snapshots[k] for k, *_ in _HISTORY)
    if a != b:
        raise ValueError("M5e.1 historical intervention altered the physical world")
    root_data = {
        "schema": _SCHEMA,
        "source_owner": "V19_ONE_WORLD_AND_ORIGINAL_SOCIAL_COGNITION",
        "source_world_hash": world.content_hash,
        "source_perception_hash": perception.content_hash,
        "source_cognitive_model_hash": memory.cognitive_model.content_hash,
        "source_memory_model_hash": memory.content_hash,
        "source_social_model_hash": social.content_hash,
        "initial_world_hash": root.initial_state.content_hash,
        "native_direct_trust_enabled": social.policy.trust_weight_direct_testimony,
        "authored_common_action_ids": [
            inspected.action_id, opened.action_id, moved.action_id,
            "m5e1-alice-prior-tell", bob_move.action_id,
            bob_inspect.action_id, bob_return.action_id,
        ],
        "single_history_intervention": "PREVIOUS_SOURCE_TELL_CONTENT_ONLY",
        "same_world_positions_before_native_learning": True,
        "source_story_lineage": "DIVERGENT_HISTORICAL_STORIES",
        "historical_context_match": "NOT_MATCHED_PRIVATE_EVIDENCE_OR_BELIEF",
        "trust_only_causal_effect": "NOT_IDENTIFIED",
        "post_trust_action_provenance": "NATIVE_V19_WORLD_EVENT",
        "cross_mechanism_ranking": "FORBIDDEN",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "arms": data,
    }
    return SourceHistoryNewTellReport(
        json.dumps(root_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
