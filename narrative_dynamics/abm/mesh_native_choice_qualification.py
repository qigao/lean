"""R2.M5e.2: two genuine native choices after source-owned trust learning.

This is a paired *model-version qualification* of the pre-registered
repeatability flag on Bob's already-declared TELL. All rewards, likelihoods,
goals and both source histories remain unchanged. V19's own planner and
world resolver select and execute the actual actions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_source_history_trust import (
    _run_verified_trust_then_new_tell,
    observe_verified_trust_then_new_tell,
)
from tests.test_network_epistemic_influence import source_epistemic_case

_SCHEMA = "r2-m5e2-v19-two-native-choices-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_ROOT = frozenset({
    "schema", "source_owner", "pre_registered_model_change",
    "parent_m5e1_exact_report_hash", "two_choice_source_report_hash",
    "prior_cognitive_model_hash", "choice_cognitive_model_hash",
    "prior_social_model_hash", "choice_social_model_hash",
    "registered_social_policy_hash",
    "source_bob_rewards_hash", "source_bob_beta",
    "native_direct_trust_enabled", "native_action_clock",
    "trust_only_causal_effect", "historical_context_match",
    "sample_size_per_arm", "confidence_interval", "external_validation",
    "cross_mechanism_ranking", "native_selected_actions_differ",
    "native_action_probabilities_differ", "native_world_outcomes_differ",
    "arms",
})
_ARM = frozenset({
    "source_history_story_hash", "source_trust_bob_alice",
    "new_exact_tell_source_trust", "bob_new_tell_prior_approved",
    "bob_new_tell_posterior_approved", "parent_feasible_actions",
    "native_feasible_actions", "native_action_values",
    "native_action_policy", "native_selected_action_id",
    "native_selected_action_value", "native_selected_action_probability",
    "native_decision_hash", "native_world_event_hash",
    "native_world_event_kind", "native_world_event_outcome",
    "native_world_event_success", "source_post_action_story_hash",
    "native_new_tell_accepted",
})


def _sha(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _num(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class TwoChoiceSourceActionReport:
    """A strict, private-content-free source report; not causal identification."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("M5e.2 native choice dossier requires JSON text")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict or set(d) != _ROOT
            or d["schema"] != _SCHEMA
            or d["source_owner"] != "V19_NATIVE_COGNITIVE_SOCIAL_WORLD_AND_PLANNER"
            or d["pre_registered_model_change"] !=
               "BOB_EXISTING_TELL_REPEATABLE_FALSE_TO_TRUE_ONLY"
            or d["native_direct_trust_enabled"] is not True
            or d["native_action_clock"] != 10
            or d["trust_only_causal_effect"] != "NOT_IDENTIFIED"
            or d["historical_context_match"] !=
               "NOT_MATCHED_PRIVATE_EVIDENCE_OR_BELIEF"
            or d["sample_size_per_arm"] != 1
            or d["confidence_interval"] != "NOT_ESTIMABLE"
            or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or d["cross_mechanism_ranking"] != "FORBIDDEN"
            or not _num(d["source_bob_beta"])
            or d["source_bob_beta"] <= 0
            or any(not _sha(d.get(k)) for k in (
                "parent_m5e1_exact_report_hash",
                "two_choice_source_report_hash",
                "prior_cognitive_model_hash",
                "choice_cognitive_model_hash",
                "prior_social_model_hash",
                "choice_social_model_hash",
                "registered_social_policy_hash",
                "source_bob_rewards_hash",
            ))
            or d["prior_cognitive_model_hash"] ==
               d["choice_cognitive_model_hash"]
            or d["prior_social_model_hash"] ==
               d["choice_social_model_hash"]
            or type(d["arms"]) is not dict
            or set(d["arms"]) != {"confirmed_history", "contradicted_history"}
            or any(type(d[k]) is not bool for k in (
                "native_selected_actions_differ",
                "native_action_probabilities_differ",
                "native_world_outcomes_differ",
            ))
        ):
            raise ValueError("M5e.2 requires one bounded original native choice experiment")
        case_trust = {"confirmed_history": 0.6, "contradicted_history": 0.3}
        for name, expected_trust in case_trust.items():
            x = d["arms"][name]
            if (
                type(x) is not dict or set(x) != _ARM
                or any(not _sha(x.get(k)) for k in (
                    "source_history_story_hash", "native_decision_hash",
                    "native_world_event_hash", "source_post_action_story_hash",
                ))
                or x["source_trust_bob_alice"] != expected_trust
                or x["new_exact_tell_source_trust"] != expected_trust
                or x["native_new_tell_accepted"] is not True
                or x["parent_feasible_actions"] != ["wait"]
                or x["native_feasible_actions"] != ["tell", "wait"]
                or type(x["native_action_values"]) is not dict
                or type(x["native_action_policy"]) is not dict
                or set(x["native_action_values"]) != {"tell", "wait"}
                or set(x["native_action_policy"]) != {"tell", "wait"}
                or any(not _num(v) for v in x["native_action_values"].values())
                or any(not _num(v) or not 0 <= v <= 1
                       for v in x["native_action_policy"].values())
                or not math.isclose(
                    math.fsum(x["native_action_policy"].values()), 1.0,
                    abs_tol=1e-12, rel_tol=0,
                )
                or any(
                    not math.isclose(
                        float(x["native_action_policy"][action]),
                        float(value), abs_tol=1e-12, rel_tol=0,
                    )
                    for action, value in finite_softmax(
                        x["native_action_values"], beta=d["source_bob_beta"]
                    ).items()
                )
                or x["native_selected_action_id"] not in {"tell", "wait"}
                or x["native_selected_action_id"] != min(
                    action for action in x["native_action_policy"]
                    if x["native_action_policy"][action] ==
                       max(x["native_action_policy"].values())
                )
                or x["native_selected_action_probability"] !=
                   x["native_action_policy"][x["native_selected_action_id"]]
                or x["native_selected_action_value"] !=
                   x["native_action_values"][x["native_selected_action_id"]]
                or x["native_world_event_kind"] not in {"tell", "wait"}
                or x["native_world_event_kind"] != x["native_selected_action_id"]
                or type(x["native_world_event_success"]) is not bool
                or type(x["native_world_event_outcome"]) is not str
                or not x["native_world_event_outcome"]
                or any(not _num(x[k]) or not 0 <= x[k] <= 1 for k in (
                    "bob_new_tell_prior_approved",
                    "bob_new_tell_posterior_approved",
                ))
            ):
                raise ValueError("M5e.2 choice probabilities must be source-derived")
        c, t = d["arms"]["confirmed_history"], d["arms"]["contradicted_history"]
        if (
            d["native_selected_actions_differ"] is not
               (c["native_selected_action_id"] != t["native_selected_action_id"])
            or d["native_action_probabilities_differ"] is not
               (c["native_action_policy"] != t["native_action_policy"])
            or d["native_world_outcomes_differ"] is not (
                (c["native_world_event_kind"], c["native_world_event_outcome"]) !=
                (t["native_world_event_kind"], t["native_world_event_outcome"])
            )
        ):
            raise ValueError("M5e.2 native action comparison must not be fabricated")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_native_two_choice_source() -> TwoChoiceSourceActionReport:
    """Two genuine MAP alternatives with all original decision results intact."""
    old = observe_verified_trust_then_new_tell()
    new, receipts = _run_verified_trust_then_new_tell(bob_repeatable_tell=True)
    prior, source = old.to_dict(), new.to_dict()
    if (
        prior["source_world_hash"] != source["source_world_hash"]
        or prior["source_perception_hash"] != source["source_perception_hash"]
        or prior["initial_world_hash"] != source["initial_world_hash"]
        or prior["source_social_model_hash"] == source["source_social_model_hash"]
        or prior["source_cognitive_model_hash"] ==
           source["source_cognitive_model_hash"]
    ):
        raise ValueError("new choice model may change ONLY Bob's repeatability")
    baseline_net, _, _, _ = source_epistemic_case()
    bob = next(agent for agent in baseline_net.percept_memory_model.cognitive_model.agents
               if agent.agent_id == "bob")
    bob_reward_digest = stable_content_hash([item.to_dict() for item in bob.rewards])
    arms = {}
    for name in ("confirmed_history", "contradicted_history"):
        parent, variant = prior["arms"][name], source["arms"][name]
        decision, executed = receipts[name]
        if (
            parent["historical_story_hash"] != variant["historical_story_hash"]
            or parent["historical_tell_event_hash"] !=
               variant["historical_tell_event_hash"]
            or parent["native_bob_feasible_actions"] != ["wait"]
            or tuple(decision.feasible_action_ids) != ("tell", "wait")
            or executed.action_id != decision.intent.action_id
            or executed.actor_agent_id != "bob"
            or decision.round_index != 10
        ):
            raise ValueError("two-choice case changed the source event or original admissibility")
        arms[name] = {
            "source_history_story_hash": variant["historical_story_hash"],
            "source_trust_bob_alice": variant["source_social_trust_after_learning"],
            "new_exact_tell_source_trust":
                variant["native_bob_direct_tell_source_trust"],
            "bob_new_tell_prior_approved":
                variant["native_bob_direct_tell_prior_approved"],
            "bob_new_tell_posterior_approved":
                variant["native_bob_direct_tell_posterior_approved"],
            "parent_feasible_actions": parent["native_bob_feasible_actions"],
            "native_feasible_actions": list(decision.feasible_action_ids),
            "native_action_values": dict(decision.action_values),
            "native_action_policy": dict(decision.action_policy),
            "native_selected_action_id": decision.selected_action_id,
            "native_selected_action_value":
                decision.action_values[decision.selected_action_id],
            "native_selected_action_probability":
                decision.action_policy[decision.selected_action_id],
            "native_decision_hash": decision.content_hash,
            "native_world_event_hash": executed.content_hash,
            "native_world_event_kind": executed.kind.value,
            "native_world_event_outcome": executed.outcome,
            "native_world_event_success": executed.success,
            "source_post_action_story_hash": variant["native_world_after_round_hash"],
            "native_new_tell_accepted": variant["native_bob_source_admitted_new_tell"],
        }
    conf, contra = arms["confirmed_history"], arms["contradicted_history"]
    payload = {
        "schema": _SCHEMA,
        "source_owner": "V19_NATIVE_COGNITIVE_SOCIAL_WORLD_AND_PLANNER",
        "pre_registered_model_change":
            "BOB_EXISTING_TELL_REPEATABLE_FALSE_TO_TRUE_ONLY",
        "parent_m5e1_exact_report_hash": old.content_hash,
        "two_choice_source_report_hash": new.content_hash,
        "prior_cognitive_model_hash": prior["source_cognitive_model_hash"],
        "choice_cognitive_model_hash": source["source_cognitive_model_hash"],
        "source_bob_rewards_hash": bob_reward_digest,
        "source_bob_beta": bob.beta,
        "prior_social_model_hash": prior["source_social_model_hash"],
        "choice_social_model_hash": source["source_social_model_hash"],
        "registered_social_policy_hash": replace(
            baseline_net.social_memory_model.policy,
            trust_weight_direct_testimony=True,
        ).content_hash,
        "native_direct_trust_enabled": True,
        "native_action_clock": 10,
        "trust_only_causal_effect": "NOT_IDENTIFIED",
        "historical_context_match": "NOT_MATCHED_PRIVATE_EVIDENCE_OR_BELIEF",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "cross_mechanism_ranking": "FORBIDDEN",
        "native_selected_actions_differ":
            conf["native_selected_action_id"] != contra["native_selected_action_id"],
        "native_action_probabilities_differ":
            conf["native_action_policy"] != contra["native_action_policy"],
        "native_world_outcomes_differ":
            (conf["native_world_event_kind"], conf["native_world_event_outcome"]) !=
            (contra["native_world_event_kind"], contra["native_world_event_outcome"]),
        "arms": arms,
    }
    return TwoChoiceSourceActionReport(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
