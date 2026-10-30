"""R2.M5d: actual post-evidence native Agent action; no fake trust-only causality.

This observer advances the original V19/SQLite cognitive+social engines twice
after each already qualified M5c Story. It does not write a new behavior
policy, alter a belief/trust state, create an Agent action or use a synthetic
source edge. Branch differences remain observational and source-qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.mesh_epistemic_influence import (
    observe_epistemic_key_player_pair, _trust, _validate_schedules,
)
from narrative_dynamics.abm.situated import SituatedActionKind, SituatedActionIntent
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import (
    SituatedStory, replay_situated_story,
)

_SCHEMA = "r2-m5d-v19-post-evidence-native-actions-v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_ARM_KEYS = frozenset({
    "source_story_before_cognition_hash", "source_after_first_cognition_hash",
    "source_after_post_evidence_action_hash",
    "first_native_round_hash", "post_evidence_native_round_hash",
    "first_social_state_hash", "post_evidence_social_state_hash",
    "source_trust_bob_alice_after_first_round",
    "post_evidence_action_round", "native_bob_action",
    "native_bob_action_kind", "native_bob_action_event_id",
    "native_bob_action_event_hash", "native_bob_action_success",
    "native_bob_action_outcome", "native_bob_feasible_actions",
    "native_bob_decision_hash", "native_bob_policy_probability",
    "native_bob_prior_belief_approved", "native_bob_posterior_belief_approved",
    "native_bob_recalled_alice_tell", "new_trust_weighted_tell_recall_count",
    "second_round_actual_tell_event_count",
})
_ROOT_KEYS = frozenset({
    "schema", "source_owner", "parent_epistemic_source_report_hash",
    "source_network_model_hash", "source_initial_story_hash",
    "source_world_model_hash", "same_original_action_ids",
    "branches", "actual_native_actions_observed",
    "source_branches_divergent", "selected_action_ids_differ",
    "action_kind_or_outcome_differs", "prior_trust_values",
    "trust_isolation", "causal_attribution", "sample_size_per_arm",
    "confidence_interval", "external_validation",
    "post_trust_action_effect", "cross_mechanism_ranking",
})


def _digest(x: object) -> bool:
    return type(x) is str and _HASH.fullmatch(x) is not None


@dataclass(frozen=True)
class PostEvidenceActualActionReport:
    """Finite source-action dossier with an explicit causal firewall."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("native action dossier requires canonical JSON text")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict or set(d) != _ROOT_KEYS
            or d["schema"] != _SCHEMA
            or d["source_owner"] != "V19_NATIVE_SQLITE_COGNITION_SOCIAL_AND_WORLD"
            or any(not _digest(d.get(k)) for k in (
                "parent_epistemic_source_report_hash",
                "source_network_model_hash", "source_initial_story_hash",
                "source_world_model_hash",
            ))
            or d["source_branches_divergent"] is not True
            or d["actual_native_actions_observed"] is not True
            or d["trust_isolation"] != "NOT_ISOLATED_BETWEEN_SOURCE_BRANCHES"
            or d["causal_attribution"] != "UNATTRIBUTED"
            or d["sample_size_per_arm"] != 1
            or d["confidence_interval"] != "NOT_ESTIMABLE"
            or d["external_validation"] != "NO_EXTERNAL_VALIDATION"
            or d["post_trust_action_effect"] != "TRUST_ONLY_CAUSAL_EFFECT_NOT_ESTABLISHED"
            or d["cross_mechanism_ranking"] != "FORBIDDEN"
            or d["prior_trust_values"] != {
                "control_wait": 0.5, "authorized_open": 0.3,
            }
            or type(d["same_original_action_ids"]) is not list
            or len(d["same_original_action_ids"]) != 5
            or len(set(d["same_original_action_ids"])) != 5
            or type(d["branches"]) is not dict
            or set(d["branches"]) != {"control_wait", "authorized_open"}
        ):
            raise ValueError("native response cannot assert unobserved causal influence")
        for name, arm in d["branches"].items():
            if (
                type(arm) is not dict or set(arm) != _ARM_KEYS
                or any(not _digest(arm.get(k)) for k in (
                    "source_story_before_cognition_hash",
                    "source_after_first_cognition_hash",
                    "source_after_post_evidence_action_hash",
                    "first_native_round_hash",
                    "post_evidence_native_round_hash",
                    "first_social_state_hash",
                    "post_evidence_social_state_hash",
                    "native_bob_action_event_hash",
                    "native_bob_decision_hash",
                ))
                or arm["source_after_first_cognition_hash"] ==
                   arm["source_after_post_evidence_action_hash"]
                or type(arm["post_evidence_action_round"]) is not int
                or arm["post_evidence_action_round"] != 8
                or type(arm["native_bob_action"]) is not str
                or not arm["native_bob_action"]
                or arm["native_bob_action_kind"] not in
                   {item.value for item in SituatedActionKind}
                or type(arm["native_bob_action_event_id"]) is not str
                or not arm["native_bob_action_event_id"]
                or type(arm["native_bob_action_outcome"]) is not str
                or not arm["native_bob_action_outcome"]
                or type(arm["native_bob_action_success"]) is not bool
                or type(arm["native_bob_feasible_actions"]) is not list
                or arm["native_bob_feasible_actions"] !=
                   sorted(set(arm["native_bob_feasible_actions"]))
                or arm["native_bob_action"] not in arm["native_bob_feasible_actions"]
                or type(arm["native_bob_policy_probability"]) not in (int, float)
                or not 0 < arm["native_bob_policy_probability"] <= 1
                or any(type(arm[k]) not in (float, int) or
                       not 0 <= arm[k] <= 1
                       for k in (
                           "native_bob_prior_belief_approved",
                           "native_bob_posterior_belief_approved",
                           "source_trust_bob_alice_after_first_round",
                       ))
                or arm["source_trust_bob_alice_after_first_round"] !=
                   d["prior_trust_values"][name]
                or type(arm["native_bob_recalled_alice_tell"]) is not list
                or any(type(x) is not dict or set(x) !=
                       {"source_event_id", "source_trust", "evidence_weight"}
                       or type(x["source_event_id"]) is not str
                       or not x["source_event_id"]
                       or type(x["source_trust"]) not in (int, float)
                       or x["source_trust"] != d["prior_trust_values"][name]
                       or type(x["evidence_weight"]) not in (int, float)
                       or not 0 <= x["evidence_weight"] <= 1
                       for x in arm["native_bob_recalled_alice_tell"])
                or type(arm["new_trust_weighted_tell_recall_count"]) is not int
                or arm["new_trust_weighted_tell_recall_count"] !=
                   len(arm["native_bob_recalled_alice_tell"])
                or type(arm["second_round_actual_tell_event_count"]) is not int
                or arm["second_round_actual_tell_event_count"] < 0
            ):
                raise ValueError("post-evidence action must bind actual source decisions")
        control = d["branches"]["control_wait"]
        opened = d["branches"]["authorized_open"]
        if (
            control["source_story_before_cognition_hash"] ==
                opened["source_story_before_cognition_hash"]
            or d["selected_action_ids_differ"] is not
                (control["native_bob_action"] != opened["native_bob_action"])
            or d["action_kind_or_outcome_differs"] is not (
                (control["native_bob_action_kind"],
                 control["native_bob_action_outcome"]) !=
                (opened["native_bob_action_kind"],
                 opened["native_bob_action_outcome"])
            )
        ):
            raise ValueError("cross-branch action differences must be source-computed")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def observe_post_evidence_actual_actions(
    model: SituatedNetworkRuntimeModel,
    initial_story: SituatedStory,
    *,
    control_schedule: tuple[tuple[SituatedActionIntent, ...], ...],
    intervention_schedule: tuple[tuple[SituatedActionIntent, ...], ...],
) -> PostEvidenceActualActionReport:
    """Execute two real V19 cognitive/social rounds per native Story branch."""
    if not isinstance(model, SituatedNetworkRuntimeModel):
        raise TypeError("post-evidence action requires exact V19 network model")
    if not isinstance(initial_story, SituatedStory) or initial_story.rounds:
        raise ValueError("post-evidence action requires a pristine V19 Story")
    _validate_schedules(control_schedule, intervention_schedule)
    parent = observe_epistemic_key_player_pair(
        model, initial_story,
        control_schedule=control_schedule,
        intervention_schedule=intervention_schedule,
    )
    p = parent.to_dict()
    if p["post_trust_action_effect"] != "NOT_ESTABLISHED":
        raise ValueError("parent source cannot assert imagined trust-caused action")
    memory_model, social_model = model.percept_memory_model, model.social_memory_model
    world = memory_model.cognitive_model.world_model
    arms = {}
    for name, schedule in (
        ("control_wait", control_schedule),
        ("authorized_open", intervention_schedule),
    ):
        story = replay_situated_story(
            world, initial_story.initial_state, schedule,
            perception_model=memory_model.perception_model,
        )
        if story.content_hash != p["branches"][name]["story_hash"]:
            raise ValueError("M5d source Story must reproduce prior M5c revision")
        cognitive = initialize_situated_percept_memory_cognition(memory_model, story)
        social = initialize_situated_social_memory(social_model, cognitive)
        with TemporaryDirectory(prefix=f"v19-post-trust-{name}-") as tmp:
            database = str(Path(tmp) / "memory.sqlite3")
            first = simulate_situated_percept_social_cognitive_round(
                database, memory_model, social_model, story, cognitive, social,
            )
            if first.next_social_state.content_hash != (
                    p["branches"][name]["source_social_after_hash"]):
                raise ValueError("first V19 native evidence round diverged from M5c")
            current_trust = _trust(first.next_social_state, "bob", "alice")
            second = simulate_situated_percept_social_cognitive_round(
                database, memory_model, social_model,
                first.next_story,
                first.next_cognitive_state,
                first.next_social_state,
            )
        first_bob = next(item for item in first.decisions if item.agent_id == "bob")
        action = next(item for item in second.decisions if item.agent_id == "bob")
        executed = next(
            event for event in second.next_story.rounds[-1].events
            if event.action_id == action.intent.action_id
            and event.actor_agent_id == "bob"
        )
        if (
            action.round_index != story.current_state.round_index + 2
            or action.round_index != 8
            or executed.kind is not action.intent.kind
            or second.next_story.current_state.round_index != 8
            or first_bob.round_index != 7
            or second.next_cognitive_state.story_hash !=
               second.next_story.content_hash
        ):
            raise ValueError("M5d requires an actual later native world action")
        tell_event = next(
            event for res in story.rounds for event in res.events
            if event.action_id == control_schedule[3][0].action_id
        )
        bob_recall = next(
            item for item in second.recalls if item.prior_mind.agent_id == "bob"
        )
        related = [
            {
                "source_event_id": adm.event_id,
                "source_trust": adm.source_trust,
                "evidence_weight": adm.evidence_weight,
            }
            for adm in bob_recall.admissions
            if adm.event_id == tell_event.event_id
            and tell_event.kind is SituatedActionKind.TELL
            and tell_event.actor_agent_id == "alice"
            and adm.rule_id is not None
            and not adm.consolidated
        ]
        if any(item["source_trust"] != current_trust for item in related):
            raise ValueError("M5d trust-weighted recall must use real current social trust")
        arms[name] = {
            "source_story_before_cognition_hash": story.content_hash,
            "source_after_first_cognition_hash": first.next_story.content_hash,
            "source_after_post_evidence_action_hash": second.next_story.content_hash,
            "first_native_round_hash": first.content_hash,
            "post_evidence_native_round_hash": second.content_hash,
            "first_social_state_hash": first.next_social_state.content_hash,
            "post_evidence_social_state_hash": second.next_social_state.content_hash,
            "source_trust_bob_alice_after_first_round": current_trust,
            "post_evidence_action_round": second.next_story.current_state.round_index,
            "native_bob_action": action.selected_action_id,
            "native_bob_action_kind": executed.kind.value,
            "native_bob_action_event_id": executed.event_id,
            "native_bob_action_event_hash": executed.content_hash,
            "native_bob_action_success": executed.success,
            "native_bob_action_outcome": executed.outcome,
            "native_bob_feasible_actions": list(action.feasible_action_ids),
            "native_bob_decision_hash": action.content_hash,
            "native_bob_policy_probability": action.action_policy[action.selected_action_id],
            "native_bob_prior_belief_approved":
                action.prior_belief.probabilities["approved"],
            "native_bob_posterior_belief_approved":
                action.posterior_belief.probabilities["approved"],
            "native_bob_recalled_alice_tell": related,
            "new_trust_weighted_tell_recall_count": len(related),
            "second_round_actual_tell_event_count": sum(
                e.kind is SituatedActionKind.TELL and e.success
                for e in second.next_story.rounds[-1].events
            ),
        }
    ctrl, alt = arms["control_wait"], arms["authorized_open"]
    document = {
        "schema": _SCHEMA,
        "source_owner": "V19_NATIVE_SQLITE_COGNITION_SOCIAL_AND_WORLD",
        "parent_epistemic_source_report_hash": parent.content_hash,
        "source_network_model_hash": model.content_hash,
        "source_initial_story_hash": initial_story.content_hash,
        "source_world_model_hash": world.content_hash,
        "same_original_action_ids": [
            control_schedule[i][0].action_id for i in (0, 2, 3, 4, 5)
        ],
        "branches": arms,
        "actual_native_actions_observed": True,
        "source_branches_divergent": True,
        "selected_action_ids_differ":
            ctrl["native_bob_action"] != alt["native_bob_action"],
        "action_kind_or_outcome_differs": (
            (ctrl["native_bob_action_kind"], ctrl["native_bob_action_outcome"]) !=
            (alt["native_bob_action_kind"], alt["native_bob_action_outcome"])
        ),
        "prior_trust_values": {
            "control_wait": 0.5, "authorized_open": 0.3,
        },
        "trust_isolation": "NOT_ISOLATED_BETWEEN_SOURCE_BRANCHES",
        "causal_attribution": "UNATTRIBUTED",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "post_trust_action_effect": "TRUST_ONLY_CAUSAL_EFFECT_NOT_ESTABLISHED",
        "cross_mechanism_ranking": "FORBIDDEN",
    }
    return PostEvidenceActualActionReport(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
