"""R2.M5f: complete exploratory native V19 response-cost regime grid.

The entire bounded grid is selected AFTER inspecting M5e.2 baseline
outcomes; it is not blind preregistration or isolated trust causality.
Every grid cell runs the *same original* V19 planner, SQLite-backed
social/cognition and world resolver. No result selection and no hidden
reward tuning after this full grid is fixed in source.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_native_choice_qualification import (
    observe_native_two_choice_source,
)
from narrative_dynamics.abm.mesh_source_history_trust import (
    _run_verified_trust_then_new_tell,
)
from tests.test_network_epistemic_influence import source_epistemic_case

_SCHEMA = "r2-m5f-exploratory-native-speaking-cost-grid-v1"
# Fixed before the first M5f source execution; never silently extend/shrink
# these cases in response to desired simulated behavior.
_COSTS = (0.0, 2.0, 3.0, 4.0, 5.0)
_HISTORY = ("confirmed_history", "contradicted_history")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_ROOT_KEYS = frozenset({
    "schema", "source_owner", "grid_costs", "grid_design",
    "registered_model_change", "parent_m5e2_report_hash",
    "baseline_m5e2_source_report_hash", "source_world_hash",
    "source_perception_hash", "registered_social_policy_hash",
    "original_bob_reward_table_hash", "source_bob_beta",
    "grid_completeness", "discovered_switch_costs",
    "cells", "source_history_comparison",
    "scientific_scope", "trust_only_causal_effect",
    "sample_size_per_arm", "confidence_interval",
    "external_validation", "cross_mechanism_ranking",
})
_CELL_KEYS = frozenset({
    "speaking_cost", "source_cognitive_model_hash",
    "source_social_model_hash", "source_memory_model_hash",
    "bob_reward_table_hash", "native_history_report_hash",
    "same_original_world_hash", "same_original_perception_hash",
    "history_arms", "native_selected_actions_differ",
    "native_world_outcomes_differ", "native_policy_probabilities_differ",
})
_ARM_KEYS = frozenset({
    "prior_source_history_hash", "native_bob_learned_trust",
    "native_new_tell_posterior_approved",
    "native_action_feasible_ids", "native_action_values",
    "native_action_policy", "native_selected_action_id",
    "native_selected_action_probability",
    "native_decision_hash", "native_world_event_hash",
    "native_world_event_kind", "native_world_event_outcome",
    "native_world_event_success", "native_clock_round",
    "new_source_tell_admitted",
})


def _digest(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class NativeResponseGridReport:
    """Validated read-only source-conformance report, NOT empirical effect."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("M5f report requires immutable canonical JSON")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict or set(d) != _ROOT_KEYS
            or d["schema"] != _SCHEMA
            or d["source_owner"] != "V19_NATIVE_PLANNER_AND_SQLITE_SOCIAL_COGNITION"
            or d["grid_costs"] != list(_COSTS)
            or d["grid_design"] != "EXPLORATORY_SELECTED_AFTER_M5E2_BASELINE"
            or d["registered_model_change"] !=
                "SYMMETRIC_BOB_TELL_REWARD_MINUS_COST_EACH_HYPOTHESIS"
            or d["grid_completeness"] != "ALL_REGISTERED_CELLS_INCLUDED"
            or d["source_history_comparison"] != "CONFIRMED_VS_CONTRADICTED"
            or d["scientific_scope"] != "FINITE_MODEL_SENSITIVITY_NOT_CAUSAL_INFERENCE"
            or d["trust_only_causal_effect"] != "NOT_IDENTIFIED"
            or d["sample_size_per_arm"] != 1
            or d["confidence_interval"] != "NOT_ESTIMABLE"
            or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or d["cross_mechanism_ranking"] != "FORBIDDEN"
            or any(not _digest(d.get(k)) for k in (
                "parent_m5e2_report_hash",
                "baseline_m5e2_source_report_hash",
                "source_world_hash", "source_perception_hash",
                "registered_social_policy_hash",
                "original_bob_reward_table_hash",
            ))
            or not _number(d["source_bob_beta"])
            or d["source_bob_beta"] <= 0
            or type(d["cells"]) is not list
            or len(d["cells"]) != len(_COSTS)
            or type(d["discovered_switch_costs"]) is not list
        ):
            raise ValueError("M5f grid must disclose every source cell and limitations")
        source_ids = set()
        switch_costs = []
        reference_histories = None
        for entry, cost in zip(d["cells"], _COSTS, strict=True):
            if (
                type(entry) is not dict or set(entry) != _CELL_KEYS
                or type(entry["speaking_cost"]) not in (int, float)
                or entry["speaking_cost"] != cost
                or entry["same_original_world_hash"] != d["source_world_hash"]
                or entry["same_original_perception_hash"] !=
                    d["source_perception_hash"]
                or any(not _digest(entry.get(k)) for k in (
                    "source_cognitive_model_hash",
                    "source_social_model_hash",
                    "source_memory_model_hash",
                    "bob_reward_table_hash", "native_history_report_hash",
                ))
                or type(entry["history_arms"]) is not dict
                or set(entry["history_arms"]) != set(_HISTORY)
                or any(type(entry[k]) is not bool for k in (
                    "native_selected_actions_differ",
                    "native_world_outcomes_differ",
                    "native_policy_probabilities_differ",
                ))
            ):
                raise ValueError("M5f has invalid or unreviewed source cell")
            if entry["source_cognitive_model_hash"] in source_ids:
                raise ValueError("different speaking costs cannot reuse a model identity")
            source_ids.add(entry["source_cognitive_model_hash"])
            for name, trust in (("confirmed_history", 0.6),
                                ("contradicted_history", 0.3)):
                arm = entry["history_arms"][name]
                if (
                    type(arm) is not dict or set(arm) != _ARM_KEYS
                    or any(not _digest(arm.get(k)) for k in (
                        "prior_source_history_hash",
                        "native_decision_hash", "native_world_event_hash",
                    ))
                    or arm["native_bob_learned_trust"] != trust
                    or not _number(arm["native_new_tell_posterior_approved"])
                    or not 0 <= arm["native_new_tell_posterior_approved"] <= 1
                    or arm["native_action_feasible_ids"] != ["tell", "wait"]
                    or type(arm["native_action_values"]) is not dict
                    or type(arm["native_action_policy"]) is not dict
                    or set(arm["native_action_values"]) != {"tell", "wait"}
                    or set(arm["native_action_policy"]) != {"tell", "wait"}
                    or any(not _number(v) for v in
                           arm["native_action_values"].values())
                    or any(not _number(v) or not 0 <= v <= 1 for v in
                           arm["native_action_policy"].values())
                    or not math.isclose(
                        math.fsum(arm["native_action_policy"].values()), 1.0,
                        rel_tol=0, abs_tol=1e-12,
                    )
                    or any(not math.isclose(float(arm["native_action_policy"][key]),
                                            float(value), abs_tol=1e-12, rel_tol=0)
                           for key, value in finite_softmax(
                               arm["native_action_values"], beta=d["source_bob_beta"]
                           ).items())
                    or arm["native_selected_action_id"] not in {"tell", "wait"}
                    or arm["native_selected_action_id"] != min(
                        key for key in arm["native_action_policy"]
                        if arm["native_action_policy"][key] ==
                           max(arm["native_action_policy"].values())
                    )
                    or arm["native_selected_action_probability"] !=
                       arm["native_action_policy"][arm["native_selected_action_id"]]
                    or arm["native_world_event_kind"] !=
                       arm["native_selected_action_id"]
                    or type(arm["native_world_event_outcome"]) is not str
                    or not arm["native_world_event_outcome"]
                    or type(arm["native_world_event_success"]) is not bool
                    or arm["native_clock_round"] != 10
                    or arm["new_source_tell_admitted"] is not True
                ):
                    raise ValueError("M5f actual native action/reward policy was forged")
            # Costs alter only the registered native cognitive model;
            # the original seven AUTHORED V19 world/history action
            # records must remain identical across cost versions.
            history_hashes = tuple(
                entry["history_arms"][name]["prior_source_history_hash"]
                for name in _HISTORY
            )
            if reference_histories is None:
                reference_histories = history_hashes
            elif reference_histories != history_hashes:
                raise ValueError("M5f cost versions changed source historical Stories")
            yes = entry["history_arms"]["confirmed_history"]
            no = entry["history_arms"]["contradicted_history"]
            switched = (yes["native_selected_action_id"] !=
                        no["native_selected_action_id"])
            outcome = (
                (yes["native_world_event_kind"], yes["native_world_event_outcome"])
                != (no["native_world_event_kind"], no["native_world_event_outcome"])
            )
            policy = yes["native_action_policy"] != no["native_action_policy"]
            if (entry["native_selected_actions_differ"] is not switched
                    or entry["native_world_outcomes_differ"] is not outcome
                    or entry["native_policy_probabilities_differ"] is not policy):
                raise ValueError("M5f selected native outcomes do not match actual records")
            if switched:
                switch_costs.append(cost)
        if d["discovered_switch_costs"] != switch_costs:
            raise ValueError("M5f must retain every discovered source-grid outcome")
        if (
            d["cells"][0]["native_history_report_hash"] !=
                d["baseline_m5e2_source_report_hash"]
        ):
            raise ValueError("M5f zero-cost cell must exactly match reviewed M5e2 source")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_native_speaking_cost_grid() -> NativeResponseGridReport:
    """Run EVERY predeclared grid cell through the original V19 source owner."""
    parent = observe_native_two_choice_source()
    baseline = parent.to_dict()
    original_net, _, _, _ = source_epistemic_case()
    bob = next(a for a in original_net.percept_memory_model.cognitive_model.agents
               if a.agent_id == "bob")
    original_rewards = [r.to_dict() for r in bob.rewards]
    policy_hash = replace(
        original_net.social_memory_model.policy,
        trust_weight_direct_testimony=True,
    ).content_hash
    cells = []
    switches = []
    for cost in _COSTS:
        report, decisions = _run_verified_trust_then_new_tell(
            bob_repeatable_tell=True, bob_tell_cost=cost,
        )
        raw = report.to_dict()
        expected_rewards = [
            {**r, "value": r["value"] - cost}
            if r["action_id"] == "tell" else r
            for r in original_rewards
        ]
        arms = {}
        for name in _HISTORY:
            source = raw["arms"][name]
            decision, world_event = decisions[name]
            if (
                source["native_bob_direct_tell_source_trust"] !=
                    source["source_social_trust_after_learning"]
                or decision.round_index != 10
                or tuple(decision.feasible_action_ids) != ("tell", "wait")
                or decision.content_hash != source["native_bob_decision_hash"]
                or world_event.content_hash != source["native_bob_actual_event_hash"]
                or world_event.actor_agent_id != "bob"
                or world_event.action_id != decision.intent.action_id
            ):
                raise ValueError("M5f source decision or event does not bind original V19")
            arms[name] = {
                "prior_source_history_hash": source["historical_story_hash"],
                "native_bob_learned_trust":
                    source["source_social_trust_after_learning"],
                "native_new_tell_posterior_approved":
                    source["native_bob_direct_tell_posterior_approved"],
                "native_action_feasible_ids": list(decision.feasible_action_ids),
                "native_action_values": dict(decision.action_values),
                "native_action_policy": dict(decision.action_policy),
                "native_selected_action_id": decision.selected_action_id,
                "native_selected_action_probability":
                    decision.action_policy[decision.selected_action_id],
                "native_decision_hash": decision.content_hash,
                "native_world_event_hash": world_event.content_hash,
                "native_world_event_kind": world_event.kind.value,
                "native_world_event_outcome": world_event.outcome,
                "native_world_event_success": world_event.success,
                "native_clock_round": decision.round_index,
                "new_source_tell_admitted": source["native_bob_source_admitted_new_tell"],
            }
        confirmed, contradicted = arms[_HISTORY[0]], arms[_HISTORY[1]]
        selected_diff = (confirmed["native_selected_action_id"] !=
                         contradicted["native_selected_action_id"])
        if selected_diff:
            switches.append(cost)
        cells.append({
            "speaking_cost": cost,
            "source_cognitive_model_hash": raw["source_cognitive_model_hash"],
            "source_social_model_hash": raw["source_social_model_hash"],
            "source_memory_model_hash": raw["source_memory_model_hash"],
            "bob_reward_table_hash": stable_content_hash(expected_rewards),
            "native_history_report_hash": report.content_hash,
            "same_original_world_hash": raw["source_world_hash"],
            "same_original_perception_hash": raw["source_perception_hash"],
            "history_arms": arms,
            "native_selected_actions_differ": selected_diff,
            "native_world_outcomes_differ": (
                (confirmed["native_world_event_kind"],
                 confirmed["native_world_event_outcome"]) !=
                (contradicted["native_world_event_kind"],
                 contradicted["native_world_event_outcome"])
            ),
            "native_policy_probabilities_differ":
                confirmed["native_action_policy"] !=
                contradicted["native_action_policy"],
        })
    if cells[0]["native_history_report_hash"] != baseline["two_choice_source_report_hash"]:
        raise ValueError("M5f zero speaking cost changed source-authoritative M5e2 history")
    payload = {
        "schema": _SCHEMA,
        "source_owner": "V19_NATIVE_PLANNER_AND_SQLITE_SOCIAL_COGNITION",
        "grid_costs": list(_COSTS),
        "grid_design": "EXPLORATORY_SELECTED_AFTER_M5E2_BASELINE",
        "registered_model_change":
            "SYMMETRIC_BOB_TELL_REWARD_MINUS_COST_EACH_HYPOTHESIS",
        "parent_m5e2_report_hash": parent.content_hash,
        "baseline_m5e2_source_report_hash": baseline["two_choice_source_report_hash"],
        "source_world_hash": cells[0]["same_original_world_hash"],
        "source_perception_hash": cells[0]["same_original_perception_hash"],
        "registered_social_policy_hash": policy_hash,
        "original_bob_reward_table_hash": stable_content_hash(original_rewards),
        "source_bob_beta": bob.beta,
        "grid_completeness": "ALL_REGISTERED_CELLS_INCLUDED",
        "discovered_switch_costs": switches,
        "cells": cells,
        "source_history_comparison": "CONFIRMED_VS_CONTRADICTED",
        "scientific_scope": "FINITE_MODEL_SENSITIVITY_NOT_CAUSAL_INFERENCE",
        "trust_only_causal_effect": "NOT_IDENTIFIED",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "cross_mechanism_ranking": "FORBIDDEN",
    }
    return NativeResponseGridReport(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
