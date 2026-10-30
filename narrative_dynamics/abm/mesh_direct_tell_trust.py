"""R2.M5e.0: direct EXACT testimony uses *source-owned* trust, opt-in.

Compare one identical original V19 Story, perception and cognitive model
under explicitly registered source-social policy variants. This is a
deterministic protocol qualification, not a real-world or trust-only
causal effect. Native actions and social evidence retain their owners.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionKind, SituatedActionIntent, ObservationChannel
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
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
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryRecallCue,
)
from narrative_dynamics.abm.situated_story import replay_situated_story

_SCHEMA = "r2-m5e0-v19-source-direct-tell-trust-v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SPECS = (
    ("legacy_unweighted", False, 0.5, 1.0),
    ("weighted_zero", True, 0.0, 0.0),
    ("weighted_half", True, 0.5, 0.5),
    ("weighted_one", True, 1.0, 1.0),
)
_ARM_KEYS = frozenset({
    "social_model_hash", "policy_hash", "configured_social_source_trust",
    "weight_direct_tell", "actual_social_source_trust",
    "actual_effective_direct_tell_trust",
    "native_source_tell_event_hash", "native_bob_exact_percept_hash",
    "direct_tell_admission_id", "direct_tell_prior_approved",
    "direct_tell_posterior_approved", "native_non_tell_direct_weights",
    "native_new_tell_round", "native_tell_admitted_to_bob_decision",
    "native_cognitive_round_hash", "native_source_story_after_hash",
    "native_social_state_after_hash", "native_bob_decision_hash",
    "native_bob_selected_action", "native_bob_feasible_actions",
    "native_bob_prior_approved", "native_bob_posterior_approved",
    "native_bob_event_hash", "native_bob_event_outcome",
    "native_bob_event_success", "native_social_claim_statuses",
    "native_social_trust_after", "native_bob_admitted_testimony",
})
_ROOT_KEYS = frozenset({
    "schema", "source_owner", "base_world_hash", "base_story_hash",
    "base_cognitive_model_hash", "base_perception_model_hash",
    "base_initial_cognitive_state_hash", "source_policy_variants",
    "source_tell_event_id", "source_tell_event_hash",
    "source_subject_id", "source_observer_id", "source_trust_direction",
    "source_direct_percept_fidelity", "native_decision_rule",
    "source_world_branch_comparison", "trust_only_causal_claim",
    "cross_mechanism_ranking", "sample_size_per_arm",
    "confidence_interval", "external_validation", "arms",
})


def _hash(x: object) -> bool:
    return type(x) is str and _DIGEST.fullmatch(x) is not None


@dataclass(frozen=True)
class DirectTellSourceTrustReport:
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("M5e source evidence requires canonical JSON")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict or set(d) != _ROOT_KEYS
            or d["schema"] != _SCHEMA
            or d["source_owner"] != "V19_SINGLE_DIRECT_COGNITION_AND_SOCIAL_OWNER"
            or d["source_subject_id"] != "alice"
            or d["source_observer_id"] != "bob"
            or d["source_trust_direction"] != "BOB_OBSERVES_ALICE"
            or d["source_direct_percept_fidelity"] != "exact"
            or d["native_decision_rule"] != "ORIGINAL_V19_SOFTMAX_MAP"
            or d["source_world_branch_comparison"] != "SAME_WORLD_STORY_DIFFERENT_SOCIAL_POLICY"
            or d["trust_only_causal_claim"] != "NOT_ESTABLISHED"
            or d["cross_mechanism_ranking"] != "FORBIDDEN"
            or d["sample_size_per_arm"] != 1
            or d["confidence_interval"] != "NOT_ESTIMABLE"
            or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or any(not _hash(d.get(k)) for k in (
                "base_world_hash", "base_story_hash", "base_cognitive_model_hash",
                "base_perception_model_hash", "base_initial_cognitive_state_hash",
                "source_tell_event_hash",
            ))
            or type(d["source_tell_event_id"]) is not str
            or not d["source_tell_event_id"]
            or type(d["source_policy_variants"]) is not list
            or d["source_policy_variants"] != [x[0] for x in _SPECS]
            or type(d["arms"]) is not dict
            or set(d["arms"]) != set(d["source_policy_variants"])
        ):
            raise ValueError("M5e report cannot fabricate social trust or causal conclusions")
        for name, flag, configured, expected_trust in _SPECS:
            a = d["arms"][name]
            if (
                type(a) is not dict or set(a) != _ARM_KEYS
                or any(not _hash(a.get(k)) for k in (
                    "social_model_hash", "policy_hash",
                    "native_source_tell_event_hash", "native_bob_exact_percept_hash",
                    "native_cognitive_round_hash", "native_source_story_after_hash",
                    "native_social_state_after_hash", "native_bob_decision_hash",
                    "native_bob_event_hash",
                ))
                or a["weight_direct_tell"] is not flag
                or a["configured_social_source_trust"] != configured
                or a["actual_social_source_trust"] != configured
                or a["actual_effective_direct_tell_trust"] != expected_trust
                or a["native_source_tell_event_hash"] != d["source_tell_event_hash"]
                or type(a["direct_tell_admission_id"]) is not str
                or not a["direct_tell_admission_id"]
                or a["native_new_tell_round"] != 4
                or a["native_tell_admitted_to_bob_decision"] is not True
                or type(a["native_non_tell_direct_weights"]) is not list
                or any(weight != 1.0 for weight in a["native_non_tell_direct_weights"])
                or any(type(a[k]) not in (int, float) or not 0 <= a[k] <= 1
                       for k in (
                           "direct_tell_prior_approved",
                           "direct_tell_posterior_approved",
                           "native_bob_prior_approved",
                           "native_bob_posterior_approved",
                           "native_social_trust_after",
                       ))
                or type(a["native_bob_selected_action"]) is not str
                or not a["native_bob_selected_action"]
                or type(a["native_bob_feasible_actions"]) is not list
                or a["native_bob_feasible_actions"] != sorted(set(a["native_bob_feasible_actions"]))
                or a["native_bob_selected_action"] not in a["native_bob_feasible_actions"]
                or type(a["native_bob_event_outcome"]) is not str
                or not a["native_bob_event_outcome"]
                or type(a["native_bob_event_success"]) is not bool
                or type(a["native_bob_admitted_testimony"]) is not int
                or a["native_bob_admitted_testimony"] != 1
                or type(a["native_social_claim_statuses"]) is not list
                or any(item not in {"active", "contradicted", "confirmed", "forgotten", "superseded"}
                       for item in a["native_social_claim_statuses"])
            ):
                raise ValueError("M5e source report contradicts its protocol variant")
        full = d["arms"]["legacy_unweighted"]
        zero = d["arms"]["weighted_zero"]
        half = d["arms"]["weighted_half"]
        one = d["arms"]["weighted_one"]
        if (
            full["direct_tell_posterior_approved"] != one["direct_tell_posterior_approved"]
            or zero["direct_tell_posterior_approved"] != zero["direct_tell_prior_approved"]
            or half["direct_tell_posterior_approved"] in (
                full["direct_tell_posterior_approved"],
                zero["direct_tell_posterior_approved"],
            )
            or len({d["arms"][k]["native_bob_decision_hash"] for k in d["arms"]}) < 2
        ):
            raise ValueError("M5e direct source trust must have a real model-local response")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_direct_tell_source_trust() -> DirectTellSourceTrustReport:
    """Make a *new* native V19 TELL after a legitimate source checkpoint.

    All social-policy arms share the exact same prior Story and cognition.
    The original action planner emits the new message in native round 4;
    the original percept/social/cognitive owner admits it in round 5.
    """
    from tests.test_network_epistemic_influence import source_epistemic_case

    net, root, _, _ = source_epistemic_case()
    old_memory = net.percept_memory_model
    world = old_memory.cognitive_model.world_model
    perception = old_memory.perception_model
    social = net.social_memory_model
    # An authored Alice INSPECT and authorized OPEN/MOVE are genuine world
    # actions. The cognitive checkpoint is created only AFTER these rounds,
    # so the next TELL is a new direct percept, not stale checkpoint history.
    alice_inspection_cue = SituatedMemoryRecallCue(
        "m5e-alice-own-inspect", "restructuring",
        required_place_ids=("open",),
        event_kinds=(SituatedActionKind.INSPECT,),
        channels=(ObservationChannel.INSPECTION,),
    )
    memory = replace(
        old_memory,
        agents=tuple(
            replace(agent, cues=(alice_inspection_cue,))
            if agent.agent_id == "alice" else agent
            for agent in old_memory.agents
        ),
    )
    early_actions = (
        (SituatedActionIntent("m5e-alice-inspect", "alice",
                              SituatedActionKind.INSPECT, target_id="memo"),),
        (SituatedActionIntent("m5e-alice-authorized-open", "alice",
                              SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),),
        (SituatedActionIntent("m5e-alice-move", "alice",
                              SituatedActionKind.MOVE, target_id="records-open"),),
    )
    story = replay_situated_story(
        world, root.initial_state, early_actions, perception_model=perception,
    )
    source_actions = [next(e for e in row.events if e.action_id == actions[0].action_id)
                      for row, actions in zip(story.rounds, early_actions, strict=True)]
    if (len(source_actions) != 3 or not all(item.success for item in source_actions)
            or story.current_state.round_index != 3):
        raise ValueError("M5e prerequisite must use actual V19 INSPECT OPEN MOVE")
    cognitive = initialize_situated_percept_memory_cognition(memory, story)
    agents = {a.agent_id: a for a in memory.cognitive_model.agents}
    bob_model = agents["bob"]
    arms = {}
    accepted_tells = {}
    for name, flag, trust, expected_effective_trust in _SPECS:
        policy = replace(
            social.policy, initial_source_trust=trust,
            trust_weight_direct_testimony=flag,
        )
        bounded_social = replace(social, policy=policy)
        prior_social = initialize_situated_social_memory(bounded_social, cognitive)
        with TemporaryDirectory(prefix=f"v19-m5e-direct-{name}-") as tmp:
            db = str(Path(tmp) / "memory.sqlite3")
            first = simulate_situated_percept_social_cognitive_round(
                db, memory, bounded_social, story, cognitive, prior_social,
            )
            # Alice's action MUST come from the existing V19 cognitive owner,
            # be a successful TELL and cite her actually observed INSPECT.
            emitted = [
                x for x in first.next_story.rounds[-1].events
                if x.kind is SituatedActionKind.TELL
                and x.actor_agent_id == "alice"
                and x.success
            ]
            if len(emitted) != 1:
                raise ValueError("M5e native planner did not produce a fresh Alice TELL")
            told = emitted[0]
            if told.round_index != 4 or told.cause_event_ids != (source_actions[0].event_id,):
                raise ValueError("M5e fresh TELL must cite the actual native INSPECT")
            accepted_tells[name] = told
            bob_percepts = perceptual_timeline(
                perception, first.next_story, "bob"
            )
            exact_tell = [
                x for x in bob_percepts
                if x.source_event_id == told.event_id
                and x.source_event_hash == told.content_hash
                and x.fidelity is SituatedPerceptFidelity.EXACT
                and x.kind is SituatedActionKind.TELL
                and x.actor_agent_id == "alice"
            ]
            if len(exact_tell) != 1:
                raise ValueError("M5e Bob must receive a new source-owned EXACT TELL")
            actual_trust = next(
                x.trust for x in first.next_social_state.relationships
                if x.observer_agent_id == "bob" and x.source_agent_id == "alice"
            )
            bob_mind = next(x for x in first.next_cognitive_state.minds
                            if x.agent_id == "bob")
            direct = admit_situated_percepts(
                bob_model, bob_mind, bob_percepts,
                _source_trust_by_source=(
                    {"alice": actual_trust} if flag else None
                ),
            )
            told_admissions = [
                x for x in direct.admissions if x.event_id == told.event_id
            ]
            if len(told_admissions) != 1 or told_admissions[0].source_trust != expected_effective_trust:
                raise ValueError("M5e source-owned new direct TELL trust admission missing")
            not_tell_weights = [
                x.source_trust for x in direct.admissions
                if x.event_id != told.event_id
            ]
            # The second original native round is the actual decision which
            # consumes the new TELL and current source social trust.
            second = simulate_situated_percept_social_cognitive_round(
                db, memory, bounded_social,
                first.next_story, first.next_cognitive_state,
                first.next_social_state,
            )
        bob_decision = next(x for x in second.decisions if x.agent_id == "bob")
        if told_admissions[0].observation_id not in bob_decision.admitted_observation_ids:
            raise ValueError("M5e native Bob decision did not consume the new TELL")
        bob_event = next(x for x in second.next_story.rounds[-1].events
                         if x.action_id == bob_decision.intent.action_id
                         and x.actor_agent_id == "bob")
        direct_social_trust_after = next(
            x.trust for x in second.next_social_state.relationships
            if x.observer_agent_id == "bob" and x.source_agent_id == "alice"
        )
        admitted_testimony = sum(
            x.kind.value == "testimony" and x.observer_agent_id == "bob"
            and x.source_agent_id == "alice"
            for x in second.social_update.admitted_evidence
        )
        arms[name] = {
            "social_model_hash": bounded_social.content_hash,
            "policy_hash": policy.content_hash,
            "configured_social_source_trust": trust,
            "weight_direct_tell": flag,
            "actual_social_source_trust": actual_trust,
            "actual_effective_direct_tell_trust": told_admissions[0].source_trust,
            "native_source_tell_event_hash": told.content_hash,
            "native_bob_exact_percept_hash": exact_tell[0].content_hash,
            "direct_tell_admission_id": told_admissions[0].observation_id,
            "direct_tell_prior_approved":
                told_admissions[0].prior_belief.probabilities["approved"],
            "direct_tell_posterior_approved":
                told_admissions[0].posterior_belief.probabilities["approved"],
            "native_non_tell_direct_weights": not_tell_weights,
            "native_new_tell_round": told.round_index,
            "native_tell_admitted_to_bob_decision": True,
            "native_cognitive_round_hash": second.content_hash,
            "native_source_story_after_hash": second.next_story.content_hash,
            "native_social_state_after_hash": second.next_social_state.content_hash,
            "native_bob_decision_hash": bob_decision.content_hash,
            "native_bob_selected_action": bob_decision.selected_action_id,
            "native_bob_feasible_actions": list(bob_decision.feasible_action_ids),
            "native_bob_prior_approved":
                bob_decision.prior_belief.probabilities["approved"],
            "native_bob_posterior_approved":
                bob_decision.posterior_belief.probabilities["approved"],
            "native_bob_event_hash": bob_event.content_hash,
            "native_bob_event_outcome": bob_event.outcome,
            "native_bob_event_success": bob_event.success,
            "native_social_claim_statuses": sorted(
                x.status.value for x in second.next_social_state.claims
                if x.observer_agent_id == "bob"
            ),
            "native_social_trust_after": direct_social_trust_after,
            "native_bob_admitted_testimony": admitted_testimony,
        }
    source_events = {(t.event_id, t.content_hash) for t in accepted_tells.values()}
    if len(source_events) != 1:
        raise ValueError("M5e four social policy arms changed the authored TELL source event")
    tell_id, tell_hash = next(iter(source_events))
    result = {
        "schema": _SCHEMA,
        "source_owner": "V19_SINGLE_DIRECT_COGNITION_AND_SOCIAL_OWNER",
        "base_world_hash": world.content_hash,
        "base_story_hash": story.content_hash,
        "base_cognitive_model_hash": memory.cognitive_model.content_hash,
        "base_perception_model_hash": perception.content_hash,
        "base_initial_cognitive_state_hash": cognitive.content_hash,
        "source_policy_variants": [x[0] for x in _SPECS],
        "source_tell_event_id": tell_id,
        "source_tell_event_hash": tell_hash,
        "source_subject_id": "alice",
        "source_observer_id": "bob",
        "source_trust_direction": "BOB_OBSERVES_ALICE",
        "source_direct_percept_fidelity": "exact",
        "native_decision_rule": "ORIGINAL_V19_SOFTMAX_MAP",
        "source_world_branch_comparison": "SAME_WORLD_STORY_DIFFERENT_SOCIAL_POLICY",
        "trust_only_causal_claim": "NOT_ESTABLISHED",
        "cross_mechanism_ranking": "FORBIDDEN",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "arms": arms,
    }
    return DirectTellSourceTrustReport(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
