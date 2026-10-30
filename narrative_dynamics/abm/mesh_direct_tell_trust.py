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
from narrative_dynamics.abm.situated import SituatedActionKind
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
    "direct_tell_posterior_approved", "direct_inspect_effective_trust",
    "direct_inspect_posterior_approved",
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
                or a["direct_inspect_effective_trust"] != 1.0
                or any(type(a[k]) not in (int, float) or not 0 <= a[k] <= 1
                       for k in (
                           "direct_tell_prior_approved",
                           "direct_tell_posterior_approved",
                           "direct_inspect_posterior_approved",
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
    """Execute one real V19 Story with four exact social source-policy variants."""
    from tests.test_network_epistemic_influence import source_epistemic_case

    net, origin, _, open_schedule = source_epistemic_case()
    memory = net.percept_memory_model
    world = memory.cognitive_model.world_model
    perception = memory.perception_model
    social = net.social_memory_model
    story = replay_situated_story(
        world, origin.initial_state, open_schedule, perception_model=perception,
    )
    cognitive = initialize_situated_percept_memory_cognition(memory, story)
    told = next(event for round_result in story.rounds for event in round_result.events
                if event.action_id == open_schedule[3][0].action_id)
    if not told.success or told.kind is not SituatedActionKind.TELL:
        raise ValueError("M5e source must have real successful V19 TELL")
    bob_percepts = perceptual_timeline(perception, story, "bob")
    exact_tell = [
        p for p in bob_percepts
        if p.source_event_id == told.event_id
        and p.fidelity is SituatedPerceptFidelity.EXACT
        and p.actor_agent_id == "alice"
        and p.kind is SituatedActionKind.TELL
    ]
    if len(exact_tell) != 1 or exact_tell[0].source_event_hash != told.content_hash:
        raise ValueError("M5e needs one native source-qualified Bob EXACT TELL")
    agents = {a.agent_id: a for a in memory.cognitive_model.agents}
    bob = agents["bob"]
    mind = next(m for m in cognitive.minds if m.agent_id == "bob")
    arms = {}
    for name, flag, trust, expected_effective_trust in _SPECS:
        policy = replace(
            social.policy, initial_source_trust=trust,
            trust_weight_direct_testimony=flag,
        )
        bounded_social = replace(social, policy=policy)
        prior_social = initialize_situated_social_memory(bounded_social, cognitive)
        actual_trust = next(
            x.trust for x in prior_social.relationships
            if x.observer_agent_id == "bob" and x.source_agent_id == "alice"
        )
        # This is *the same original V19 percept admission function* that
        # the integrated social-cognitive round calls, not a second updater.
        direct = admit_situated_percepts(
            bob, mind, bob_percepts,
            _source_trust_by_source=(
                {"alice": actual_trust} if flag else None
            ),
        )
        tell_admissions = [
            x for x in direct.admissions
            if x.event_id == told.event_id
        ]
        inspect_admissions = [
            x for x in direct.admissions if x.event_id != told.event_id
            and x.likelihood_action_id == "inspect"
        ]
        if (
            len(tell_admissions) != 1 or len(inspect_admissions) != 1
            or tell_admissions[0].source_trust != expected_effective_trust
            or inspect_admissions[0].source_trust != 1.0
        ):
            raise ValueError("M5e must separate external testimony from own inspection")
        with TemporaryDirectory(prefix=f"v19-m5e-direct-{name}-") as tmp:
            result = simulate_situated_percept_social_cognitive_round(
                str(Path(tmp) / "memory.sqlite3"),
                memory, bounded_social, story, cognitive, prior_social,
            )
        decision = next(x for x in result.decisions if x.agent_id == "bob")
        event = next(
            e for e in result.next_story.rounds[-1].events
            if e.action_id == decision.intent.action_id
            and e.actor_agent_id == "bob"
        )
        bob_social_after = next(
            x.trust for x in result.next_social_state.relationships
            if x.observer_agent_id == "bob" and x.source_agent_id == "alice"
        )
        admitted_testimony = sum(
            e.kind.value == "testimony" and e.observer_agent_id == "bob"
            and e.source_agent_id == "alice"
            for e in result.social_update.admitted_evidence
        )
        arms[name] = {
            "social_model_hash": bounded_social.content_hash,
            "policy_hash": policy.content_hash,
            "configured_social_source_trust": trust,
            "weight_direct_tell": flag,
            "actual_social_source_trust": actual_trust,
            "actual_effective_direct_tell_trust": tell_admissions[0].source_trust,
            "native_source_tell_event_hash": told.content_hash,
            "native_bob_exact_percept_hash": exact_tell[0].content_hash,
            "direct_tell_admission_id": tell_admissions[0].observation_id,
            "direct_tell_prior_approved":
                tell_admissions[0].prior_belief.probabilities["approved"],
            "direct_tell_posterior_approved":
                tell_admissions[0].posterior_belief.probabilities["approved"],
            "direct_inspect_effective_trust": inspect_admissions[0].source_trust,
            "direct_inspect_posterior_approved":
                inspect_admissions[0].posterior_belief.probabilities["approved"],
            "native_cognitive_round_hash": result.content_hash,
            "native_source_story_after_hash": result.next_story.content_hash,
            "native_social_state_after_hash": result.next_social_state.content_hash,
            "native_bob_decision_hash": decision.content_hash,
            "native_bob_selected_action": decision.selected_action_id,
            "native_bob_feasible_actions": list(decision.feasible_action_ids),
            "native_bob_prior_approved": decision.prior_belief.probabilities["approved"],
            "native_bob_posterior_approved":
                decision.posterior_belief.probabilities["approved"],
            "native_bob_event_hash": event.content_hash,
            "native_bob_event_outcome": event.outcome,
            "native_bob_event_success": event.success,
            "native_social_claim_statuses": sorted(
                c.status.value for c in result.next_social_state.claims
                if c.observer_agent_id == "bob"
            ),
            "native_social_trust_after": bob_social_after,
            "native_bob_admitted_testimony": admitted_testimony,
        }
    payload = {
        "schema": _SCHEMA,
        "source_owner": "V19_SINGLE_DIRECT_COGNITION_AND_SOCIAL_OWNER",
        "base_world_hash": world.content_hash,
        "base_story_hash": story.content_hash,
        "base_cognitive_model_hash": memory.cognitive_model.content_hash,
        "base_perception_model_hash": perception.content_hash,
        "base_initial_cognitive_state_hash": cognitive.content_hash,
        "source_policy_variants": [k for k, *_ in _SPECS],
        "source_tell_event_id": told.event_id,
        "source_tell_event_hash": told.content_hash,
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
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
