"""R2.M5g.1: neutral audit learning → fresh native target TELL → native choice.

Two original V19 source histories share one registered target/audit model.
The original social owner learns directional trust; the next ORIGINAL
native Agent planner emits new target testimony, and Bob's subsequent
SQLite-backed cognitive decision admits it using the learned trust.
No historical event, private memory, trust value or action is fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from grounded_goal_softmax import finite_softmax
from grounded_goal_softmax import finite_softmax
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_neutral_audit_trust import (
    _AUDIT_ROWS,
    _neutral_source_models,
    observe_neutral_training_trust,
)
from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
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
from narrative_dynamics.abm.situated_story import replay_situated_story

_SCHEMA = "r2-m5g1-native-neutral-audit-post-trust-target-choice-v1"
_HASH = re.compile(r"sha256:[a-f0-9]{64}")
_CASES = (("confirmed_audit", 0.6), ("contradicted_audit", 0.3))
_ROOT = frozenset({
    "schema", "source_owner", "parent_m5g0_report_hash",
    "source_world_model_hash", "source_perception_model_hash",
    "source_cognitive_model_hash", "source_social_model_hash",
    "source_social_policy_hash", "source_initial_world_hash",
    "source_bob_beta", "registered_model_change", "training_source_symbol_neutral",
    "original_reward_policy_preserved", "native_social_learning_round",
    "new_target_tell_round", "native_bob_decision_round",
    "same_world_and_action_roster", "same_declared_target_message",
    "training_target_belief_match", "pre_new_target_belief_match",
    "source_private_history_match", "actual_action_comparison",
    "sample_size_per_arm", "confidence_interval", "external_validation",
    "trust_only_causal_effect", "cross_mechanism_ranking", "arms",
})
_ARM = frozenset({
    "training_story_hash", "training_audit_event_hash",
    "training_source_claim_status", "training_bob_belief_approved",
    "training_bob_trust_alice_before", "training_bob_trust_alice_after",
    "training_social_state_hash", "training_native_round_hash",
    "new_target_tell_event_id", "new_target_tell_event_hash",
    "new_target_tell_cause_event_ids", "new_target_tell_fidelity_bob",
    "new_target_tell_source_observed_by_alice", "post_learning_story_hash",
    "native_bob_pre_target_approved", "native_bob_direct_prior_approved",
    "native_bob_direct_posterior_approved",
    "native_bob_direct_tell_trust", "native_bob_direct_admission_id",
    "native_bob_admission_consumed_by_decision",
    "native_bob_observation_floor", "native_bob_recalled_memory_count",
    "native_bob_active_target_claim_count",
    "native_bob_active_audit_claim_count",
    "native_bob_recent_observation_ids_hash",
    "native_bob_previous_selected_actions_hash",
    "native_bob_action_values", "native_bob_action_policy",
    "native_bob_feasible_actions", "native_bob_selected_action",
    "native_bob_decision_hash", "native_bob_world_event_hash",
    "native_bob_world_event_kind", "native_bob_world_event_outcome",
    "native_bob_world_event_success", "next_native_story_hash",
    "next_native_social_state_hash",
})


def _sha(s: object) -> bool:
    return type(s) is str and _HASH.fullmatch(s) is not None


def _number(n: object) -> bool:
    return type(n) in (int, float) and math.isfinite(n)


@dataclass(frozen=True)
class NeutralAuditTargetDecisionReport:
    """Strict source-provenance dossier; never a population causal result."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("M5g1 native source report needs canonical JSON")
        x = json.loads(self._canonical_json)
        if (
            type(x) is not dict or set(x) != _ROOT or x["schema"] != _SCHEMA
            or x["source_owner"] != "V19_ONE_SQLITE_SOCIAL_COGNITIVE_WORLD_OWNER"
            or any(not _sha(x.get(k)) for k in (
                "parent_m5g0_report_hash", "source_world_model_hash",
                "source_perception_model_hash", "source_cognitive_model_hash",
                "source_social_model_hash", "source_social_policy_hash",
                "source_initial_world_hash",
            ))
            or not _number(x.get("source_bob_beta"))
            or x["source_bob_beta"] <= 0
            or x["registered_model_change"] !=
               "ALICE_AND_BOB_EXISTING_TELL_REPEATABLE_TRUE_ONLY"
            or x["training_source_symbol_neutral"] is not True
            or x["original_reward_policy_preserved"] is not True
            or x["native_social_learning_round"] != 8
            or x["new_target_tell_round"] != 9
            or x["native_bob_decision_round"] != 10
            or x["same_world_and_action_roster"] is not True
            or x["same_declared_target_message"] is not True
            or x["training_target_belief_match"] != "EXACT_AT_CHECKPOINT_0_5"
            or x["pre_new_target_belief_match"] not in (
                "EXACT", "NOT_MATCHED_AFTER_NATIVE_INTERMEDIATE_EVENTS",
            )
            or x["source_private_history_match"] != "NOT_MATCHED"
            or x["actual_action_comparison"] not in (
                "DIFFERENT_MAP", "SAME_MAP",
            )
            or x["sample_size_per_arm"] != 1
            or x["confidence_interval"] != "NOT_ESTIMABLE"
            or x["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or x["trust_only_causal_effect"] != "NOT_IDENTIFIED"
            or x["cross_mechanism_ranking"] != "FORBIDDEN"
            or type(x["arms"]) is not dict
            or set(x["arms"]) != {"confirmed_audit", "contradicted_audit"}
        ):
            raise ValueError("M5g1 cannot claim unsupported target behavior causality")
        sources = []
        for name, trust in _CASES:
            a = x["arms"][name]
            if (
                type(a) is not dict or set(a) != _ARM
                or any(not _sha(a.get(k)) for k in (
                    "training_story_hash", "training_audit_event_hash",
                    "training_social_state_hash", "training_native_round_hash",
                    "new_target_tell_event_hash", "post_learning_story_hash",
                    "native_bob_recent_observation_ids_hash",
                    "native_bob_previous_selected_actions_hash",
                    "native_bob_decision_hash", "native_bob_world_event_hash",
                    "next_native_story_hash", "next_native_social_state_hash",
                ))
                or a["training_source_claim_status"] != (
                    "confirmed" if name == "confirmed_audit" else "contradicted"
                )
                or a["training_bob_belief_approved"] != 0.5
                or a["training_bob_trust_alice_before"] != 0.5
                or a["training_bob_trust_alice_after"] != trust
                or a["new_target_tell_fidelity_bob"] != "exact"
                or a["new_target_tell_source_observed_by_alice"] is not True
                or a["native_bob_direct_tell_trust"] != trust
                or a["native_bob_admission_consumed_by_decision"] is not True
                or type(a["new_target_tell_event_id"]) is not str
                or not a["new_target_tell_event_id"]
                or type(a["new_target_tell_cause_event_ids"]) is not list
                or len(a["new_target_tell_cause_event_ids"]) != 1
                or type(a["native_bob_direct_admission_id"]) is not str
                or not a["native_bob_direct_admission_id"]
                or any(not _number(a[k]) or not 0 <= a[k] <= 1 for k in (
                    "native_bob_pre_target_approved",
                    "native_bob_direct_prior_approved",
                    "native_bob_direct_posterior_approved",
                ))
                or any(type(a[k]) is not int or a[k] < 0 for k in (
                    "native_bob_observation_floor", "native_bob_recalled_memory_count",
                    "native_bob_active_target_claim_count",
                    "native_bob_active_audit_claim_count",
                ))
                or type(a["native_bob_feasible_actions"]) is not list
                or len(a["native_bob_feasible_actions"]) < 2
                or a["native_bob_feasible_actions"] !=
                   sorted(set(a["native_bob_feasible_actions"]))
                or type(a["native_bob_action_values"]) is not dict
                or type(a["native_bob_action_policy"]) is not dict
                or set(a["native_bob_feasible_actions"]) !=
                   set(a["native_bob_action_values"])
                or set(a["native_bob_action_values"]) !=
                   set(a["native_bob_action_policy"])
                or any(not _number(v) for v in a["native_bob_action_values"].values())
                or any(not _number(v) or not 0 <= v <= 1 for v in
                       a["native_bob_action_policy"].values())
                or not math.isclose(math.fsum(a["native_bob_action_policy"].values()),
                                    1.0, abs_tol=1e-12, rel_tol=0)
                or any(
                    not math.isclose(
                        float(a["native_bob_action_policy"][action]),float(value),
                        abs_tol=1e-12,rel_tol=0,
                    )
                    for action,value in finite_softmax(
                        a["native_bob_action_values"],beta=x["source_bob_beta"],
                    ).items()
                )
                or a["native_bob_selected_action"] not in
                   a["native_bob_feasible_actions"]
                or a["native_bob_selected_action"] != min(
                    key for key in a["native_bob_action_policy"]
                    if a["native_bob_action_policy"][key] ==
                       max(a["native_bob_action_policy"].values())
                )
                or a["native_bob_world_event_kind"] !=
                   a["native_bob_selected_action"]
                or not a["native_bob_world_event_outcome"]
                or type(a["native_bob_world_event_success"]) is not bool
            ):
                raise ValueError("M5g1 source decision/trust evidence forged")
            sources.append(a)
        c, d = sources
        matches = c["native_bob_pre_target_approved"] == d["native_bob_pre_target_approved"]
        if x["pre_new_target_belief_match"] != (
            "EXACT" if matches else "NOT_MATCHED_AFTER_NATIVE_INTERMEDIATE_EVENTS"
        ):
            raise ValueError("M5g1 cannot hide intermediate belief drift")
        if x["actual_action_comparison"] != (
            "DIFFERENT_MAP" if c["native_bob_selected_action"] !=
            d["native_bob_selected_action"] else "SAME_MAP"
        ):
            raise ValueError("M5g1 selected action comparison not source-derived")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_neutral_audit_then_new_target_choice() -> NeutralAuditTargetDecisionReport:
    """Original V19 source/SQLite engine with neutral training and target decision."""
    parent = observe_neutral_training_trust()
    parent_data = parent.to_dict()
    memory, social, root = _neutral_source_models(
        allow_post_audit_target_tell=True,
    )
    # New source model is one declared version; same in BOTH histories.
    reference, _, _ = _neutral_source_models()
    if memory.cognitive_model.content_hash == reference.cognitive_model.content_hash:
        raise ValueError("M5g1 repeatable target action needs an explicit model revision")
    for agent in memory.cognitive_model.agents:
        older = next(a for a in reference.cognitive_model.agents if a.agent_id == agent.agent_id)
        if (agent.rewards != older.rewards
                or agent.likelihoods != older.likelihoods
                or agent.goals != older.goals
                or agent.beta != older.beta
                or agent.action_schedule != older.action_schedule):
            raise ValueError("M5g1 must preserve the original neutral source reward policy")
    world = memory.cognitive_model.world_model
    perception = memory.perception_model
    source_model = memory.cognitive_model
    if (world.content_hash != parent_data["source_world_model_hash"]
            or perception.content_hash != parent_data["source_perception_model_hash"]
            or root.initial_state.content_hash !=
               parent_data["source_initial_world_hash"]):
        raise ValueError("M5g1 must preserve exact parent source world and perception")
    bob_cognitive_model = next(a for a in source_model.agents if a.agent_id=="bob")
    alice = next(a for a in source_model.agents if a.agent_id == "alice")
    alice_target = next(a for a in alice.actions if a.action_id == "tell")
    if alice_target.message is None or not alice_target.repeatable:
        raise ValueError("M5g1 native target source TELL must be repeatable")
    names = (
        ("m5g-alice-inspect", "m5g-alice-authorized-open", "m5g-alice-move"),
        ("m5g-bob-enter-records", "m5g-bob-own-inspect", "m5g-bob-return-open"),
    )
    first_three = (
        (SituatedActionIntent(names[0][0], "alice", SituatedActionKind.INSPECT,
                              target_id="memo"),),
        (SituatedActionIntent(names[0][1], "alice", SituatedActionKind.OPEN_PASSAGE,
                              target_id="records-open"),),
        (SituatedActionIntent(names[0][2], "alice", SituatedActionKind.MOVE,
                              target_id="records-open"),),
    )
    preliminary = replay_situated_story(
        world, root.initial_state, first_three, perception_model=perception,
    )
    observed_inspect = next(e for e in preliminary.rounds[0].events if e.success)
    trailing = (
        (SituatedActionIntent(names[1][0], "bob", SituatedActionKind.MOVE,
                              target_id="open-records"),),
        (SituatedActionIntent(names[1][1], "bob", SituatedActionKind.INSPECT,
                              target_id="memo"),),
        (SituatedActionIntent(names[1][2], "bob", SituatedActionKind.MOVE,
                              target_id="records-open"),),
    )
    results: dict[str, dict] = {}
    for label, original_message, _, original_status, expected_trust in _AUDIT_ROWS:
        original_audit = SituatedActionIntent(
            "m5g-alice-audit-training", "alice", SituatedActionKind.TELL,
            message=original_message, source_event_ids=(observed_inspect.event_id,),
        )
        schedule = (
            first_three[0],first_three[1],first_three[2],
            (original_audit,),trailing[0],trailing[1],trailing[2],
        )
        story = replay_situated_story(
            world, root.initial_state, schedule, perception_model=perception,
        )
        if (story.content_hash !=
                parent_data["arms"][label]["source_historical_story_hash"]):
            raise ValueError("M5g1 authored source history diverged from reviewed M5g0")
        prior = initialize_situated_percept_memory_cognition(memory, story)
        prior_social = initialize_situated_social_memory(social, prior)
        with TemporaryDirectory(prefix="v19-neutral-target-"+label+"-") as temp:
            db = str(Path(temp)/"memory.sqlite3")
            first = simulate_situated_percept_social_cognitive_round(
                db,memory,social,story,prior,prior_social,
            )
            trust = next(t.trust for t in first.next_social_state.relationships
                         if t.observer_agent_id=="bob" and t.source_agent_id=="alice")
            prior_trust = next(t.trust for t in prior_social.relationships
                               if t.observer_agent_id=="bob" and t.source_agent_id=="alice")
            bob_first = next(d for d in first.decisions if d.agent_id=="bob")
            audit_claims = [
                c for c in first.next_social_state.claims
                if c.observer_agent_id=="bob" and c.source_agent_id=="alice"
                and c.topic_id=="source-audit"
            ]
            if (trust != expected_trust or prior_trust!=0.5
                    or bob_first.prior_belief.probabilities["approved"] != 0.5
                    or bob_first.posterior_belief.probabilities["approved"] != 0.5
                    or len(audit_claims)!=1
                    or audit_claims[0].status.value!=original_status):
                raise ValueError("M5g1 authentic original neutral training changed target belief")
            # Round 9 AFTER native trust learning, using original planner
            # and exactly this branch's source-owned SQLite state.
            second = simulate_situated_percept_social_cognitive_round(
                db,memory,social,
                first.next_story,first.next_cognitive_state,
                first.next_social_state,
            )
            current_alice = next(d for d in second.decisions if d.agent_id=="alice")
            new_events = [
                e for e in second.next_story.rounds[-1].events
                if e.actor_agent_id=="alice"
                and e.kind is SituatedActionKind.TELL and e.success
            ]
            if len(new_events)!=1 or new_events[0].round_index!=9:
                raise ValueError("M5g1 native Alice produced no NEW target TELL at round 9")
            newer = new_events[0]
            if current_alice.intent.message != alice_target.message:
                raise ValueError("M5g1 new native TELL is not the declared target message")
            alice_mind = next(m for m in first.next_cognitive_state.minds
                              if m.agent_id=="alice")
            available_events = {
                e.event_id:e for round_result in first.next_story.rounds
                for e in round_result.events
            }
            if (len(newer.cause_event_ids)!=1
                    or newer.cause_event_ids!=current_alice.intent.source_event_ids
                    or newer.cause_event_ids[0] not in available_events
                    or available_events[newer.cause_event_ids[0]].kind not in (
                        SituatedActionKind.INSPECT,SituatedActionKind.TELL)
                    or newer.cause_event_ids[0] not in alice_mind.observed_event_ids):
                raise ValueError("M5g1 target TELL source is not native and observed")
            bob_percepts = [
                p for p in perceptual_timeline(perception, second.next_story, "bob")
                if p.source_event_id==newer.event_id
                and p.source_event_hash==newer.content_hash
                and p.kind is SituatedActionKind.TELL
                and p.actor_agent_id=="alice"
                and p.fidelity is SituatedPerceptFidelity.EXACT
            ]
            if len(bob_percepts)!=1:
                raise ValueError("M5g1 target TELL did not reach Bob as EXACT")
            bob_mind = next(m for m in second.next_cognitive_state.minds
                            if m.agent_id=="bob")
            bob_model = next(m for m in source_model.agents if m.agent_id=="bob")
            trust_current = next(t.trust for t in second.next_social_state.relationships
                                 if t.observer_agent_id=="bob" and t.source_agent_id=="alice")
            if trust_current!=expected_trust:
                raise ValueError("M5g1 directional trust drifted before target admission")
            direct = admit_situated_percepts(
                bob_model,bob_mind,perceptual_timeline(perception,second.next_story,"bob"),
                _source_trust_by_source={"alice":trust_current},
            )
            admissions = [a for a in direct.admissions if a.event_id==newer.event_id]
            if len(admissions)!=1 or admissions[0].source_trust!=expected_trust:
                raise ValueError("M5g1 original TELL admission lost current trust source")
            third = simulate_situated_percept_social_cognitive_round(
                db,memory,social,
                second.next_story,second.next_cognitive_state,second.next_social_state,
            )
        dec = next(item for item in third.decisions if item.agent_id=="bob")
        if admissions[0].observation_id not in dec.admitted_observation_ids:
            raise ValueError("M5g1 real Bob decision did not consume the new target TELL")
        event = next(e for e in third.next_story.rounds[-1].events
                     if e.actor_agent_id=="bob"
                     and e.action_id==dec.intent.action_id)
        mind = next(m for m in second.next_cognitive_state.minds
                    if m.agent_id=="bob")
        bob_claims = [
            c for c in second.next_social_state.claims
            if c.observer_agent_id=="bob"
        ]
        if not math.isclose(
            math.fsum(dec.action_policy.values()),1.0,abs_tol=1e-12,rel_tol=0
        ) or len(dec.feasible_action_ids)<2:
            raise ValueError("M5g1 original Bob lacks a native two-action choice")
        results[label]={
            "training_story_hash":story.content_hash,
            "training_audit_event_hash":next(
                e.content_hash for e in story.rounds[3].events
                if e.action_id==original_audit.action_id
            ),
            "training_source_claim_status":audit_claims[0].status.value,
            "training_bob_belief_approved":bob_first.posterior_belief.probabilities["approved"],
            "training_bob_trust_alice_before":prior_trust,
            "training_bob_trust_alice_after":trust,
            "training_social_state_hash":first.next_social_state.content_hash,
            "training_native_round_hash":first.content_hash,
            "new_target_tell_event_id":newer.event_id,
            "new_target_tell_event_hash":newer.content_hash,
            "new_target_tell_cause_event_ids":list(newer.cause_event_ids),
            "new_target_tell_fidelity_bob":bob_percepts[0].fidelity.value,
            "new_target_tell_source_observed_by_alice":True,
            "post_learning_story_hash":second.next_story.content_hash,
            "native_bob_pre_target_approved":mind.belief.probabilities["approved"],
            "native_bob_direct_prior_approved":admissions[0].prior_belief.probabilities["approved"],
            "native_bob_direct_posterior_approved":admissions[0].posterior_belief.probabilities["approved"],
            "native_bob_direct_tell_trust":admissions[0].source_trust,
            "native_bob_direct_admission_id":admissions[0].observation_id,
            "native_bob_admission_consumed_by_decision":True,
            "native_bob_observation_floor":mind.observation_floor_round,
            "native_bob_recalled_memory_count":len(mind.recalled_memory_ids),
            "native_bob_active_target_claim_count":sum(
                c.status.value=="active" and c.topic_id=="restructuring" for c in bob_claims
            ),
            "native_bob_active_audit_claim_count":sum(
                c.status.value=="active" and c.topic_id=="source-audit" for c in bob_claims
            ),
            "native_bob_recent_observation_ids_hash":stable_content_hash(
                list(mind.processed_observation_ids)
            ),
            "native_bob_previous_selected_actions_hash":stable_content_hash(
                list(mind.selected_action_ids)
            ),
            "native_bob_action_values":dict(dec.action_values),
            "native_bob_action_policy":dict(dec.action_policy),
            "native_bob_feasible_actions":list(dec.feasible_action_ids),
            "native_bob_selected_action":dec.selected_action_id,
            "native_bob_decision_hash":dec.content_hash,
            "native_bob_world_event_hash":event.content_hash,
            "native_bob_world_event_kind":event.kind.value,
            "native_bob_world_event_outcome":event.outcome,
            "native_bob_world_event_success":event.success,
            "next_native_story_hash":third.next_story.content_hash,
            "next_native_social_state_hash":third.next_social_state.content_hash,
        }
    yes,no=results["confirmed_audit"],results["contradicted_audit"]
    model_scope={
        "schema":_SCHEMA,
        "source_owner":"V19_ONE_SQLITE_SOCIAL_COGNITIVE_WORLD_OWNER",
        "parent_m5g0_report_hash":parent.content_hash,
        "source_world_model_hash":world.content_hash,
        "source_perception_model_hash":perception.content_hash,
        "source_cognitive_model_hash":source_model.content_hash,
        "source_social_model_hash":social.content_hash,
        "source_social_policy_hash":social.policy.content_hash,
        "source_initial_world_hash":root.initial_state.content_hash,
        "source_bob_beta":bob_cognitive_model.beta,
        "registered_model_change":"ALICE_AND_BOB_EXISTING_TELL_REPEATABLE_TRUE_ONLY",
        "training_source_symbol_neutral":True,
        "original_reward_policy_preserved":True,
        "native_social_learning_round":8,
        "new_target_tell_round":9,
        "native_bob_decision_round":10,
        "same_world_and_action_roster":True,
        "same_declared_target_message":True,
        "training_target_belief_match":"EXACT_AT_CHECKPOINT_0_5",
        "pre_new_target_belief_match":(
            "EXACT" if yes["native_bob_pre_target_approved"] ==
                       no["native_bob_pre_target_approved"]
            else "NOT_MATCHED_AFTER_NATIVE_INTERMEDIATE_EVENTS"
        ),
        "source_private_history_match":"NOT_MATCHED",
        "actual_action_comparison":(
            "DIFFERENT_MAP" if yes["native_bob_selected_action"] !=
                               no["native_bob_selected_action"] else "SAME_MAP"
        ),
        "sample_size_per_arm":1,
        "confidence_interval":"NOT_ESTIMABLE",
        "external_validation":"NO_EXTERNAL_VALIDATION",
        "trust_only_causal_effect":"NOT_IDENTIFIED",
        "cross_mechanism_ranking":"FORBIDDEN",
        "arms":results,
    }
    return NeutralAuditTargetDecisionReport(json.dumps(
        model_scope,sort_keys=True,separators=(",",":"),ensure_ascii=False,
    ))
