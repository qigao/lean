"""R2.M5g.2 — complete 2x2 ORIGINAL V19 direct-TELL trust-channel ablation.

Within each genuine audit-history Story, run the SAME original V19
SQLite social/cognitive/world owner under either explicitly versioned
native direct-TELL trust policy: ON=source trust, OFF=full likelihood.
Historical social trust must still be learned, never edited by hand.
This is a deterministic software-protocol counterfactual, not a real
population estimate or a trust-only social causal claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_neutral_target_decision import (
    _run_neutral_audit_then_new_target_choice,
    observe_neutral_audit_then_new_target_choice,
)
from narrative_dynamics.abm.mesh_neutral_audit_trust import (
    _neutral_source_models,
)

_SCHEMA = "r2-m5g2-original-v19-native-direct-tell-trust-ablation-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_POLICIES = (("weighted", True), ("legacy", False))
_HISTORIES = (("confirmed_audit", 0.6), ("contradicted_audit", 0.3))
_ROOT = frozenset({
    "schema", "source_owner", "grid_design", "complete_factorial",
    "parent_m5g1_report_hash", "common_source_world_hash",
    "common_source_perception_hash", "common_source_cognitive_hash",
    "common_original_rewards_hash", "source_bob_beta",
    "model_social_policy_versions_distinct", "same_v19_world_and_model_cognition",
    "history_content_differs", "source_private_history_match",
    "within_history_event_identity_required", "trust_only_causal_effect",
    "native_protocol_mechanism_scope", "sample_size_per_arm",
    "confidence_interval", "external_validation", "cross_mechanism_ranking",
    "cells", "within_history_posterior_diff", "within_history_map_diff",
})
_CELL = frozenset({
    "protocol", "trust_weight_direct_testimony", "source_model_hash",
    "source_social_model_hash", "source_social_policy_hash",
    "source_world_hash", "source_perception_hash",
    "source_initial_world_hash", "history_arms",
})
_ARM = frozenset({
    "training_story_hash", "training_audit_event_hash",
    "learned_observer_source_trust", "source_target_tell_event_id",
    "source_target_tell_event_hash", "native_target_tell_received_fidelity",
    "source_target_tell_cause_event_ids", "post_learning_story_hash",
    "native_bob_prior_approved", "native_bob_direct_prior_approved",
    "native_bob_direct_posterior_approved", "native_effective_tell_weight",
    "native_admission_consumed", "native_feasible_actions",
    "native_action_values", "native_action_policy", "native_map_action",
    "native_world_event_hash", "native_world_event_kind",
    "native_world_outcome", "native_world_success",
    "native_direct_decision_hash",
    "native_private_memory_count", "native_active_audit_claims",
    "native_active_target_claims", "prior_selected_action_hash",
    "processed_observation_ids_hash",
})


def _sha(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class NativeTrustChannelAblationReport:
    """Four real native cells with immutable provenance and causal firewall."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("native ablation requires canonical JSON")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict or set(d) != _ROOT
            or d["schema"] != _SCHEMA
            or d["source_owner"] != "ONE_ORIGINAL_V19_SQLITE_COGNITION_SOCIAL_WORLD"
            or d["grid_design"] != "COMPLETE_TWO_FACTOR_NATIVE_PROTOCOL_ABLATION"
            or d["complete_factorial"] is not True
            or d["model_social_policy_versions_distinct"] is not True
            or d["same_v19_world_and_model_cognition"] is not True
            or d["history_content_differs"] is not True
            or d["source_private_history_match"] != "NOT_MATCHED"
            or d["within_history_event_identity_required"] is not True
            or d["trust_only_causal_effect"] != "NOT_IDENTIFIED"
            or d["native_protocol_mechanism_scope"] != "FINITE_SOFTWARE_POLICY_ABLATION"
            or d["sample_size_per_arm"] != 1
            or d["confidence_interval"] != "NOT_ESTIMABLE"
            or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or d["cross_mechanism_ranking"] != "FORBIDDEN"
            or not _finite(d.get("source_bob_beta"))
            or d["source_bob_beta"] <= 0
            or any(not _sha(d.get(k)) for k in (
                "parent_m5g1_report_hash", "common_source_world_hash",
                "common_source_perception_hash", "common_source_cognitive_hash",
                "common_original_rewards_hash",
            ))
            or type(d["cells"]) is not dict
            or set(d["cells"]) != {n for n, _ in _POLICIES}
            or type(d["within_history_posterior_diff"]) is not dict
            or type(d["within_history_map_diff"]) is not dict
            or set(d["within_history_posterior_diff"]) !=
                {n for n, _ in _HISTORIES}
            or set(d["within_history_map_diff"]) !=
                {n for n, _ in _HISTORIES}
        ):
            raise ValueError("native direct-TELL ablation requires all four sourced cells")
        model_hashes = []
        source_histories = {}
        for name, enabled in _POLICIES:
            cell = d["cells"][name]
            if (
                type(cell) is not dict or set(cell) != _CELL
                or cell["protocol"] != name
                or cell["trust_weight_direct_testimony"] is not enabled
                or cell["source_model_hash"] != d["common_source_cognitive_hash"]
                or cell["source_world_hash"] != d["common_source_world_hash"]
                or cell["source_perception_hash"] !=
                   d["common_source_perception_hash"]
                or any(not _sha(cell.get(k)) for k in (
                    "source_social_model_hash", "source_social_policy_hash",
                    "source_initial_world_hash",
                ))
                or type(cell["history_arms"]) is not dict
                or set(cell["history_arms"]) != {h for h, _ in _HISTORIES}
            ):
                raise ValueError("each native policy must use exact one V19 model")
            model_hashes.append((
                cell["source_social_model_hash"],
                cell["source_social_policy_hash"],
            ))
            for history, trust in _HISTORIES:
                a = cell["history_arms"][history]
                if (
                    type(a) is not dict or set(a) != _ARM
                    or any(not _sha(a.get(k)) for k in (
                        "training_story_hash", "training_audit_event_hash",
                        "source_target_tell_event_hash",
                        "post_learning_story_hash",
                        "native_world_event_hash", "native_direct_decision_hash",
                        "prior_selected_action_hash",
                        "processed_observation_ids_hash",
                    ))
                    or a["learned_observer_source_trust"] != trust
                    or a["native_effective_tell_weight"] != (
                        trust if enabled else 1.0
                    )
                    or a["native_target_tell_received_fidelity"] != "exact"
                    or a["native_admission_consumed"] is not True
                    or type(a["source_target_tell_event_id"]) is not str
                    or not a["source_target_tell_event_id"]
                    or type(a["source_target_tell_cause_event_ids"]) is not list
                    or len(a["source_target_tell_cause_event_ids"]) != 1
                    or any(not _finite(a[k]) or not 0 <= a[k] <= 1 for k in (
                        "native_bob_prior_approved",
                        "native_bob_direct_prior_approved",
                        "native_bob_direct_posterior_approved",
                    ))
                    or type(a["native_private_memory_count"]) is not int
                    or a["native_private_memory_count"] < 0
                    or any(type(a[k]) is not int or a[k] < 0
                           for k in (
                               "native_active_audit_claims",
                               "native_active_target_claims",
                           ))
                    or a["native_feasible_actions"] != ["tell", "wait"]
                    or type(a["native_action_values"]) is not dict
                    or type(a["native_action_policy"]) is not dict
                    or set(a["native_action_values"]) != {"tell", "wait"}
                    or set(a["native_action_policy"]) != {"tell", "wait"}
                    or any(not _finite(v) for v in a["native_action_values"].values())
                    or any(not _finite(v) or not 0 <= v <= 1
                           for v in a["native_action_policy"].values())
                    or not math.isclose(
                        math.fsum(a["native_action_policy"].values()), 1.0,
                        abs_tol=1e-12, rel_tol=0,
                    )
                    or any(
                        not math.isclose(
                            float(a["native_action_policy"][act]),float(value),
                            abs_tol=1e-12, rel_tol=0,
                        )
                        for act, value in finite_softmax(
                            a["native_action_values"], beta=d["source_bob_beta"]
                        ).items()
                    )
                    or a["native_map_action"] != min(
                        action for action in a["native_action_policy"]
                        if a["native_action_policy"][action] ==
                           max(a["native_action_policy"].values())
                    )
                    or a["native_world_event_kind"] != a["native_map_action"]
                    or not a["native_world_outcome"]
                    or type(a["native_world_success"]) is not bool
                ):
                    raise ValueError("native policy/decision evidence not source qualified")
                identity = (
                    a["training_story_hash"],
                    a["training_audit_event_hash"],
                    a["source_target_tell_event_id"],
                    a["source_target_tell_event_hash"],
                    a["source_target_tell_cause_event_ids"],
                    a["post_learning_story_hash"],
                    a["native_bob_prior_approved"],
                    a["native_bob_direct_prior_approved"],
                    a["native_feasible_actions"],
                    a["prior_selected_action_hash"],
                    a["processed_observation_ids_hash"],
                )
                source_histories.setdefault(history, []).append(identity)
        if (model_hashes[0][0] == model_hashes[1][0]
                or model_hashes[0][1] == model_hashes[1][1]
                or any(x[0] != x[1] for x in source_histories.values())
                or source_histories[_HISTORIES[0][0]][0][0] ==
                   source_histories[_HISTORIES[1][0]][0][0]):
            raise ValueError("policy switch must preserve each native source history")
        for hist, _ in _HISTORIES:
            on = d["cells"]["weighted"]["history_arms"][hist]
            off = d["cells"]["legacy"]["history_arms"][hist]
            diff = (on["native_bob_direct_posterior_approved"] !=
                    off["native_bob_direct_posterior_approved"])
            map_diff = on["native_map_action"] != off["native_map_action"]
            if (d["within_history_posterior_diff"][hist] is not diff
                    or d["within_history_map_diff"][hist] is not map_diff):
                raise ValueError("within-history source differences must be computed")
            # The same original target TELL is informative with the
            # full-likelihood OFF policy. No posterior update would
            # indicate stale, DETECTED-only or consolidated evidence.
            if not (off["native_bob_direct_posterior_approved"] >
                    off["native_bob_direct_prior_approved"]):
                raise ValueError("M5g2 legacy source TELL was not informatively admitted")
            if not (off["native_bob_direct_posterior_approved"] >
                    on["native_bob_direct_posterior_approved"]):
                raise ValueError("M5g2 original full-likelihood source not stronger than ON")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_native_trust_channel_ablation() -> NativeTrustChannelAblationReport:
    """Four original V19 source runs. Parent ON source bytes stay unchanged."""
    parent = observe_neutral_audit_then_new_target_choice()
    on = parent.to_dict()
    off = _run_neutral_audit_then_new_target_choice(
        trust_weight_direct_testimony=False,
    )
    if any(on[k] != off[k] for k in (
        "source_world_model_hash", "source_perception_model_hash",
        "source_cognitive_model_hash", "source_initial_world_hash",
        "source_bob_beta",
    )):
        raise ValueError("policy ablation changed the common V19 world or cognition")
    if (on["source_social_model_hash"] == off["source_social_model_hash"]
            or on["source_social_policy_hash"] == off["source_social_policy_hash"]):
        raise ValueError("ON/OFF must have different native social policy model hashes")
    memory, _, _ = _neutral_source_models(allow_post_audit_target_tell=True)
    bob = next(x for x in memory.cognitive_model.agents if x.agent_id=="bob")
    original_reward_hash = stable_content_hash([v.to_dict() for v in bob.rewards])
    cells = {}
    for name, enabled, data in (
        ("weighted", True, on),
        ("legacy", False, off),
    ):
        arms = {}
        for history, trust in _HISTORIES:
            a = data["arms"][history]
            if (
                a["training_bob_belief_approved"] != 0.5
                or a["native_bob_pre_target_approved"] != 0.5
                or a["native_bob_direct_prior_approved"] != 0.5
                or a["training_bob_trust_alice_after"] != trust
                or a["native_bob_direct_tell_trust"] != (trust if enabled else 1.0)
                or a["new_target_tell_fidelity_bob"] != "exact"
                or a["native_bob_admission_consumed_by_decision"] is not True
            ):
                raise ValueError("source-native ablation must retain learned trust and EXACT TELL")
            arms[history] = {
                "training_story_hash": a["training_story_hash"],
                "training_audit_event_hash": a["training_audit_event_hash"],
                "learned_observer_source_trust": a["training_bob_trust_alice_after"],
                "source_target_tell_event_id": a["new_target_tell_event_id"],
                "source_target_tell_event_hash": a["new_target_tell_event_hash"],
                "native_target_tell_received_fidelity": a["new_target_tell_fidelity_bob"],
                "source_target_tell_cause_event_ids": a["new_target_tell_cause_event_ids"],
                "post_learning_story_hash": a["post_learning_story_hash"],
                "native_bob_prior_approved": a["native_bob_pre_target_approved"],
                "native_bob_direct_prior_approved": a["native_bob_direct_prior_approved"],
                "native_bob_direct_posterior_approved":
                    a["native_bob_direct_posterior_approved"],
                "native_effective_tell_weight": a["native_bob_direct_tell_trust"],
                "native_admission_consumed": a["native_bob_admission_consumed_by_decision"],
                "native_feasible_actions": a["native_bob_feasible_actions"],
                "native_action_values": a["native_bob_action_values"],
                "native_action_policy": a["native_bob_action_policy"],
                "native_map_action": a["native_bob_selected_action"],
                "native_world_event_hash": a["native_bob_world_event_hash"],
                "native_world_event_kind": a["native_bob_world_event_kind"],
                "native_world_outcome": a["native_bob_world_event_outcome"],
                "native_world_success": a["native_bob_world_event_success"],
                "native_direct_decision_hash": a["native_bob_decision_hash"],
                "native_private_memory_count": a["native_bob_recalled_memory_count"],
                "native_active_audit_claims": a["native_bob_active_audit_claim_count"],
                "native_active_target_claims": a["native_bob_active_target_claim_count"],
                "prior_selected_action_hash":
                    a["native_bob_previous_selected_actions_hash"],
                "processed_observation_ids_hash":
                    a["native_bob_recent_observation_ids_hash"],
            }
        cells[name] = {
            "protocol": name,
            "trust_weight_direct_testimony": enabled,
            "source_model_hash": data["source_cognitive_model_hash"],
            "source_social_model_hash": data["source_social_model_hash"],
            "source_social_policy_hash": data["source_social_policy_hash"],
            "source_world_hash": data["source_world_model_hash"],
            "source_perception_hash": data["source_perception_model_hash"],
            "source_initial_world_hash": data["source_initial_world_hash"],
            "history_arms": arms,
        }
    result = {
        "schema": _SCHEMA,
        "source_owner": "ONE_ORIGINAL_V19_SQLITE_COGNITION_SOCIAL_WORLD",
        "grid_design": "COMPLETE_TWO_FACTOR_NATIVE_PROTOCOL_ABLATION",
        "complete_factorial": True,
        "parent_m5g1_report_hash": parent.content_hash,
        "common_source_world_hash": on["source_world_model_hash"],
        "common_source_perception_hash": on["source_perception_model_hash"],
        "common_source_cognitive_hash": on["source_cognitive_model_hash"],
        "common_original_rewards_hash": original_reward_hash,
        "source_bob_beta": on["source_bob_beta"],
        "model_social_policy_versions_distinct": True,
        "same_v19_world_and_model_cognition": True,
        "history_content_differs": True,
        "source_private_history_match": "NOT_MATCHED",
        "within_history_event_identity_required": True,
        "trust_only_causal_effect": "NOT_IDENTIFIED",
        "native_protocol_mechanism_scope": "FINITE_SOFTWARE_POLICY_ABLATION",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "cross_mechanism_ranking": "FORBIDDEN",
        "cells": cells,
        "within_history_posterior_diff": {
            hist: (cells["weighted"]["history_arms"][hist]
                   ["native_bob_direct_posterior_approved"] !=
                   cells["legacy"]["history_arms"][hist]
                   ["native_bob_direct_posterior_approved"])
            for hist, _ in _HISTORIES
        },
        "within_history_map_diff": {
            hist: (cells["weighted"]["history_arms"][hist]["native_map_action"] !=
                   cells["legacy"]["history_arms"][hist]["native_map_action"])
            for hist, _ in _HISTORIES
        },
    }
    return NativeTrustChannelAblationReport(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
