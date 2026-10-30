"""R2.M4c.1 source-owned within-family paired finite experiments.

This module observes existing BB, rewiring and evolving owners. It does not
create an alternative simulator, common social graph or causal inference engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.bb_runtime_contracts import BBRuntimeSeed, BBRuntimeTick
from narrative_dynamics.abm.rewiring_contracts import (
    EndogenousRewiringModel, initialize_rewiring_population,
)
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel, EvolvingPopulationState,
)
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind, PopulationLifecycleEvent,
)
from narrative_dynamics.abm.model_family_comparison import (
    observe_bb_growth, observe_rewiring, observe_evolving,
)
from narrative_dynamics.abm.native_metric_protocol import (
    NativeMetric, _bb_frames, _rewiring_frames, _evolving_frames,
)

_SCHEMA = "r2-m4c1-native-within-family-paired-v1"
_FAMILIES = {
    "bb_fitness_growth": ("BIRTH_TARGET_ONLY", "SAME_NATIVE_SEED_FRAME"),
    "directed_rewiring": ("FORMATION_THRESHOLD_ONLY", "MATCHED_INITIAL_COMPONENTS_DIFFERENT_MODEL_HASH"),
    "lifecycle_feedback": ("ENTER_EVENT_ONLY", "SAME_NATIVE_MODEL_AND_STATE"),
}


def _native_metrics(records: list) -> list[dict[str, object]]:
    return [metric.to_dict() for metric in records]


def _delta(control: list, intervention: list) -> list[str]:
    if len(control) != len(intervention):
        raise ValueError("paired metric horizons must match")
    result = []
    for left, right in zip(control, intervention, strict=True):
        if (left.family != right.family or left.metric_id != right.metric_id
                or left.native_time != right.native_time
                or left.unit != right.unit or left.phase != right.phase
                or left.denominator_kind != right.denominator_kind
                or left.denominator != right.denominator
                or left.population_scope != right.population_scope):
            raise ValueError("within-family pair requires identical units, phases, rosters and denominators")
        result.append(str(Fraction(right.exact_fraction) - Fraction(left.exact_fraction)))
    return result


def run_bb_target_pair(
    seed: BBRuntimeSeed,
    control_program: tuple[BBRuntimeTick, ...],
    intervention_program: tuple[BBRuntimeTick, ...],
) -> dict[str, object]:
    """Author exactly one birth-target change, with identical later idle tick."""
    if not isinstance(seed, BBRuntimeSeed) or any(
        type(program) is not tuple or len(program) != 2
        or any(not isinstance(x, BBRuntimeTick) for x in program)
        for program in (control_program, intervention_program)
    ):
        raise TypeError("BB paired birth requires typed source and two native ticks per arm")
    c0, c1 = control_program
    i0, i1 = intervention_program
    if (c0.kind != "birth" or i0.kind != "birth"
            or c1.kind != "idle" or i1.kind != "idle"
            or c1.birth is not None or i1.birth is not None
            or c0.birth is None or i0.birth is None
            or c0.birth.fitness != i0.birth.fitness
            or c0.birth.agent != i0.birth.agent
            or c0.birth.targets == i0.birth.targets):
        raise ValueError("BB intervention must change only ordered birth targets")
    control = observe_bb_growth(seed, 1, control_program)
    altered = observe_bb_growth(seed, 1, intervention_program)
    if control["initial_frame_hash"] != altered["initial_frame_hash"]:
        raise ValueError("BB paired runs must start at identical native genesis")
    for arm in (control, altered):
        if [x["tick_kind"] for x in arm["steps"]] != ["birth", "idle"]:
            raise ValueError("BB pair requires birth-then-idle native clock")
    c_metrics, i_metrics = _bb_frames(control), _bb_frames(altered)
    c_last, i_last = control["steps"][-1], altered["steps"][-1]
    if c_last["native_undirected_topology"] == i_last["native_undirected_topology"]:
        raise ValueError("BB target intervention did not change actual undirected topology")
    return {
        "family": "bb_fitness_growth",
        "intervention_kind": "BIRTH_TARGET_ONLY",
        "origin_identity": "SAME_NATIVE_SEED_FRAME",
        "initial_source_hash": control["initial_frame_hash"],
        "horizon": 2,
        "clock": "GLOBAL_BB_TICK",
        "control": {
            "source_program_tick_hashes": [x["tick_hash"] for x in control["steps"]],
            "final_source_hash": control["final_frame_hash"],
            "metrics": _native_metrics(c_metrics),
            "last_native_topology": c_last["native_undirected_topology"],
            "native_transmissions_per_tick": [
                x["native_transmissions"] for x in control["steps"]
            ],
        },
        "intervention": {
            "source_program_tick_hashes": [x["tick_hash"] for x in altered["steps"]],
            "final_source_hash": altered["final_frame_hash"],
            "metrics": _native_metrics(i_metrics),
            "last_native_topology": i_last["native_undirected_topology"],
            "native_transmissions_per_tick": [
                x["native_transmissions"] for x in altered["steps"]
            ],
        },
        "native_metric_delta": _delta(c_metrics, i_metrics),
        "topology_changed": True,
    }


def run_rewiring_threshold_pair(
    control_model: EndogenousRewiringModel,
    intervention_model: EndogenousRewiringModel,
    *,
    initial_beliefs: dict[str, float],
) -> dict[str, object]:
    """Vary one model-native formation threshold; retain all initial components."""
    if (not isinstance(control_model, EndogenousRewiringModel)
            or not isinstance(intervention_model, EndogenousRewiringModel)
            or type(initial_beliefs) is not dict):
        raise TypeError("rewiring pair requires exact native models and initial beliefs")
    if (control_model.base_model.content_hash != intervention_model.base_model.content_hash
            or control_model.model_id != intervention_model.model_id
            or control_model.version != intervention_model.version
            or control_model.dissolution_similarity != intervention_model.dissolution_similarity
            or control_model.formation_similarity == intervention_model.formation_similarity):
        raise ValueError("rewiring pair may change only formation similarity")
    ctrl_initial = initialize_rewiring_population(
        control_model, beliefs=initial_beliefs
    )
    alt_initial = initialize_rewiring_population(
        intervention_model, beliefs=initial_beliefs
    )
    if (ctrl_initial.agents != alt_initial.agents
            or ctrl_initial.edge_topology != alt_initial.edge_topology
            or ctrl_initial.model_hash == alt_initial.model_hash):
        raise ValueError("rewiring initial Agent/topology components must match")
    control = observe_rewiring(control_model, ctrl_initial, 2)
    altered = observe_rewiring(intervention_model, alt_initial, 2)
    c_metrics, i_metrics = _rewiring_frames(control), _rewiring_frames(altered)
    c0, i0 = control["steps"][0], altered["steps"][0]
    if c0["prior_active_candidate_edges"] != i0["prior_active_candidate_edges"]:
        raise ValueError("rewiring pair must share pre-intervention candidate activity")
    if c0["native_transmissions"] != i0["native_transmissions"]:
        raise ValueError("rewiring first round must propagate using same prior edges")
    return {
        "family": "directed_rewiring",
        "intervention_kind": "FORMATION_THRESHOLD_ONLY",
        "origin_identity": "MATCHED_INITIAL_COMPONENTS_DIFFERENT_MODEL_HASH",
        "shared_base_model_hash": control_model.base_model.content_hash,
        "shared_native_initial_components_hash": stable_content_hash({
            "agents": [x.to_dict() for x in ctrl_initial.agents],
            "topology": [x.to_dict() for x in ctrl_initial.edge_topology],
        }),
        "horizon": 2,
        "clock": "NATIVE_REWIRING_ROUND",
        "control": {
            "native_formation_threshold": control_model.formation_similarity,
            "model_hash": control_model.content_hash,
            "initial_source_hash": ctrl_initial.content_hash,
            "final_source_hash": control["final_state_hash"],
            "metrics": _native_metrics(c_metrics),
            "prior_active_edges_per_round": [
                x["prior_active_candidate_edges"] for x in control["steps"]
            ],
            "next_active_edges_per_round": [
                x["next_active_candidate_edges"] for x in control["steps"]
            ],
            "native_transmissions_per_round": [
                x["native_transmissions"] for x in control["steps"]
            ],
        },
        "intervention": {
            "native_formation_threshold": intervention_model.formation_similarity,
            "model_hash": intervention_model.content_hash,
            "initial_source_hash": alt_initial.content_hash,
            "final_source_hash": altered["final_state_hash"],
            "metrics": _native_metrics(i_metrics),
            "prior_active_edges_per_round": [
                x["prior_active_candidate_edges"] for x in altered["steps"]
            ],
            "next_active_edges_per_round": [
                x["next_active_candidate_edges"] for x in altered["steps"]
            ],
            "native_transmissions_per_round": [
                x["native_transmissions"] for x in altered["steps"]
            ],
        },
        "native_metric_delta": _delta(c_metrics, i_metrics),
        "first_round_transmission_equal": True,
    }


def run_evolving_entry_pair(
    model: EvolvingNetworkModel,
    initial: EvolvingPopulationState,
    *,
    entrant: PopulationLifecycleEvent,
) -> dict[str, object]:
    """Only change one ENTER event; hold feedback absent in both native arms."""
    if (not isinstance(model, EvolvingNetworkModel)
            or not isinstance(initial, EvolvingPopulationState)
            or not isinstance(entrant, PopulationLifecycleEvent)):
        raise TypeError("evolving pair requires exact typed lifecycle source")
    if entrant.kind is not LifecycleEventKind.ENTER or entrant.belief is not None:
        raise ValueError("evolving intervention must be one default-belief ENTER")
    no_events = ((), ())
    enter_events = ((entrant,), ())
    no_feedback = ((), ())
    control = observe_evolving(model, initial, no_events, no_feedback)
    altered = observe_evolving(model, initial, enter_events, no_feedback)
    if (control["initial_state_hash"] != altered["initial_state_hash"]
            or control["model_hash"] != altered["model_hash"]):
        raise ValueError("evolving pair must share exact initial source and model")
    if (any(step["feedback_targets"] for step in control["steps"])
            or any(step["feedback_targets"] for step in altered["steps"])):
        raise ValueError("entry-only experiment must not add external feedback")
    c_metrics, i_metrics = _evolving_frames(control), _evolving_frames(altered)
    return {
        "family": "lifecycle_feedback",
        "intervention_kind": "ENTER_EVENT_ONLY",
        "origin_identity": "SAME_NATIVE_MODEL_AND_STATE",
        "initial_source_hash": initial.content_hash,
        "model_hash": model.content_hash,
        "shared_feedback_schedule": "EMPTY_BOTH_ARMS",
        "horizon": 2,
        "clock": "NATIVE_EVOLVING_ROUND",
        "control": {
            "event_schedule_hash": stable_content_hash([[], []]),
            "final_source_hash": control["final_state_hash"],
            "metrics": _native_metrics(c_metrics),
            "native_members_after": [x["native_members_after"] for x in control["steps"]],
            "native_transmissions_per_round": [
                x["native_transmissions"] for x in control["steps"]
            ],
            "external_feedback_targets": [x["feedback_targets"] for x in control["steps"]],
        },
        "intervention": {
            "event_schedule_hash": stable_content_hash([[entrant.to_dict()], []]),
            "final_source_hash": altered["final_state_hash"],
            "metrics": _native_metrics(i_metrics),
            "native_members_after": [x["native_members_after"] for x in altered["steps"]],
            "native_transmissions_per_round": [
                x["native_transmissions"] for x in altered["steps"]
            ],
            "external_feedback_targets": [x["feedback_targets"] for x in altered["steps"]],
        },
        "native_metric_delta": _delta(c_metrics, i_metrics),
    }



# Validate the structure of immutable observations independently of the
# original source replay. This cannot authenticate an actor or replace source
# reexecution; the checked-in vector remains the exact engine-owned evidence.
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_NATIVE_CLOCKS = {
    "bb_fitness_growth": "GLOBAL_BB_TICK",
    "directed_rewiring": "NATIVE_REWIRING_ROUND",
    "lifecycle_feedback": "NATIVE_EVOLVING_ROUND",
}


def _valid_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _edges(items: object, *, directed: bool, label: str) -> set:
    if type(items) is not list:
        raise ValueError(label + ": expected native edge list")
    result = set()
    for edge in items:
        if (type(edge) is not list or len(edge) != (3 if directed else 2)
                or any(type(x) is not str or not x for x in edge)
                or edge[0] == edge[1]):
            raise ValueError(label + ": malformed native edge")
        identity = tuple(edge) if directed else frozenset(edge)
        result.add(identity)
    if len(result) != len(items):
        raise ValueError(label + ": repeated or reversed native edge")
    return result


def _round_edges(items: object, *, label: str) -> list[set]:
    if type(items) is not list or len(items) != 2:
        raise ValueError(label + ": expected two native rounds")
    return [_edges(x, directed=True, label=label) for x in items]


def _audit_native_paired_case(family: str, case: dict) -> None:
    """Require locally consistent source identities, native units and deltas."""
    if case.get("clock") != _NATIVE_CLOCKS[family]:
        raise ValueError("paired case changed native clock")
    parsed = {}
    for label in ("control", "intervention"):
        arm = case[label]
        if not _valid_digest(arm.get("final_source_hash")):
            raise ValueError("paired arm requires native final source hash")
        try:
            rows = [NativeMetric(**item) for item in arm["metrics"]]
        except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
            raise ValueError("paired arm has invalid typed native metric") from exc
        if (len(rows) != 2 or any(m.family != family for m in rows)
                or [m.native_time for m in rows] != [1, 2]
                or rows[-1].frame_source_hash != arm["final_source_hash"]
                or rows[0].frame_source_hash == rows[-1].frame_source_hash
                or any(not _valid_digest(m.run_source_hash)
                       or not _valid_digest(m.frame_source_hash) for m in rows)):
            raise ValueError("paired arm lost original model/clock/frame identity")
        parsed[label] = rows
    if case["native_metric_delta"] != _delta(
        parsed["control"], parsed["intervention"]
    ):
        raise ValueError("paired native metric delta differs from exact fractions")
    if family == "bb_fitness_growth":
        source = case.get("initial_source_hash")
        if (not _valid_digest(source) or case.get("topology_changed") is not True
                or any(m.run_source_hash != source
                       for rows in parsed.values() for m in rows)):
            raise ValueError("BB paired birth must share exact genesis source")
        topologies = []
        ticks = []
        for label in ("control", "intervention"):
            arm = case[label]
            topology = _edges(
                arm.get("last_native_topology"), directed=False,
                label="BB undirected topology",
            )
            count = len({node for edge in topology for node in edge})
            denominator = parsed[label][-1].denominator
            if (len(topology) != parsed[label][-1].numerator
                    or count * (count - 1) // 2 != denominator):
                raise ValueError("BB native undirected endpoints/denominator disagree")
            program = arm.get("source_program_tick_hashes")
            if (type(program) is not list or len(program) != 2
                    or not all(_valid_digest(x) for x in program)):
                raise ValueError("BB source tick hashes must identify both steps")
            transmissions = _round_edges(
                arm.get("native_transmissions_per_tick"),
                label="BB native transmissions",
            )
            if any((edge[2] != "bb" or frozenset(edge[:2]) not in topology)
                   for round_edges in transmissions for edge in round_edges):
                raise ValueError("BB transmissions must use actual undirected birth edges")
            topologies.append(topology)
            ticks.append(program)
        if (topologies[0] == topologies[1] or ticks[0][0] == ticks[1][0]
                or ticks[0][1] != ticks[1][1]):
            raise ValueError("BB paired intervention must change only its birth tick")
    elif family == "directed_rewiring":
        if (not _valid_digest(case.get("shared_base_model_hash"))
                or not _valid_digest(case.get("shared_native_initial_components_hash"))
                or case.get("first_round_transmission_equal") is not True):
            raise ValueError("rewiring pair lost shared initial components")
        model_hashes, initial_hashes, thresholds = [], [], []
        all_propagations = []
        all_priors = []
        for label in ("control", "intervention"):
            arm = case[label]
            model_hash, initial_hash = arm.get("model_hash"), arm.get("initial_source_hash")
            threshold = arm.get("native_formation_threshold")
            if (not _valid_digest(model_hash) or not _valid_digest(initial_hash)
                    or type(threshold) not in (float, int)
                    or not math.isfinite(threshold) or not 0 <= threshold <= 1
                    or any(m.run_source_hash != initial_hash for m in parsed[label])):
                raise ValueError("rewiring arm changed model/initial source identity")
            priors = _round_edges(arm.get("prior_active_edges_per_round"),
                                  label="prior directed candidate topology")
            posts = _round_edges(arm.get("next_active_edges_per_round"),
                                 label="post directed candidate topology")
            tx = _round_edges(arm.get("native_transmissions_per_round"),
                              label="rewiring propagation")
            if (priors[1] != posts[0]
                    or any(not transmitted.issubset(prior)
                           for transmitted, prior in zip(tx, priors, strict=True))
                    or any(len(post) != metric.numerator
                           for post, metric in zip(posts, parsed[label], strict=True))):
                raise ValueError("rewiring must propagate old directed edges before rewire")
            model_hashes.append(model_hash)
            initial_hashes.append(initial_hash)
            thresholds.append(threshold)
            all_priors.append(priors)
            all_propagations.append(tx)
        if (len(set(model_hashes)) != 2 or len(set(initial_hashes)) != 2
                or thresholds[0] == thresholds[1]
                or all_priors[0][0] != all_priors[1][0]
                or all_propagations[0][0] != all_propagations[1][0]):
            raise ValueError("rewiring pair must change only its native threshold")
    else:
        shared_source, model_hash = case.get("initial_source_hash"), case.get("model_hash")
        if (not _valid_digest(shared_source) or not _valid_digest(model_hash)
                or case.get("shared_feedback_schedule") != "EMPTY_BOTH_ARMS"
                or any(m.run_source_hash != shared_source
                       for rows in parsed.values() for m in rows)):
            raise ValueError("evolving pair must share exact model, initial state and no-feedback")
        memberships = []
        schedules = []
        for label in ("control", "intervention"):
            arm = case[label]
            if not _valid_digest(arm.get("event_schedule_hash")):
                raise ValueError("evolving event schedule hash required")
            frames = arm.get("native_members_after")
            if type(frames) is not list or len(frames) != 2:
                raise ValueError("evolving pair requires two native membership frames")
            if arm.get("external_feedback_targets") != [[], []]:
                raise ValueError("evolving entry-only comparison forbids external feedback")
            transmissions = _round_edges(
                arm.get("native_transmissions_per_round"),
                label="evolving native transmissions",
            )
            member_ids = []
            for idx, frame in enumerate(frames):
                if type(frame) is not list or not frame:
                    raise ValueError("evolving membership frame must have real catalog")
                catalog = {}
                for member in frame:
                    if (type(member) is not dict or set(member) !=
                            {"agent_id", "status", "belief"}
                            or type(member["agent_id"]) is not str
                            or not member["agent_id"]
                            or member["status"] not in {"active", "inactive", "dead"}
                            or type(member["belief"]) not in (int, float)
                            or not math.isfinite(member["belief"])
                            or not 0 <= member["belief"] <= 1
                            or member["agent_id"] in catalog):
                        raise ValueError("evolving member catalog malformed")
                    catalog[member["agent_id"]] = member["status"]
                if (len(catalog) != parsed[label][idx].denominator
                        or sum(status == "active" for status in catalog.values())
                        != parsed[label][idx].numerator
                        or any(edge[0] not in catalog or edge[1] not in catalog
                               or catalog[edge[0]] != "active"
                               or catalog[edge[1]] != "active"
                               for edge in transmissions[idx])):
                    raise ValueError("evolving metric does not match active catalog/transmissions")
                member_ids.append(set(catalog))
            if member_ids[0] != member_ids[1]:
                raise ValueError("evolving catalog identity cannot silently change")
            memberships.append(frames)
            schedules.append(arm["event_schedule_hash"])
        if (schedules[0] != stable_content_hash([[], []])
                or schedules[0] == schedules[1]
                or [m["agent_id"] for m in memberships[0][0]] !=
                   [m["agent_id"] for m in memberships[1][0]]):
            raise ValueError("evolving pair must differ by one native ENTER event")
        for idx in (0, 1):
            before = {m["agent_id"]: m["status"] for m in memberships[0][idx]}
            after = {m["agent_id"]: m["status"] for m in memberships[1][idx]}
            changed = [k for k in before if before[k] != after[k]]
            if (len(changed) != 1 or before[changed[0]] != "inactive"
                    or after[changed[0]] != "active"):
                raise ValueError("evolving entry must change one actual member")


@dataclass(frozen=True)
class NativePairedExperiment:
    """Audited immutable source dossier; not a graph model or statistical result."""
    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("paired dossier must store canonical JSON")
        d = json.loads(self._canonical_json)
        if (type(d) is not dict or d.get("schema") != _SCHEMA
                or d.get("source_owner") != "INDEPENDENT_NATIVE_PAIRED_REPLAYS"
                or d.get("normalization") != "NONE"
                or d.get("cross_family_numeric_ranking") != "FORBIDDEN"
                or d.get("causal_attribution") != "UNATTRIBUTED"
                or d.get("sample_size_per_arm") != 1
                or d.get("confidence_interval") != "NOT_ESTIMABLE"
                or d.get("external_validation") != "NO_EXTERNAL_VALIDATION"
                or d.get("paired_estimand_scope") != "MODEL_LOCAL_DETERMINISTIC_DESCRIPTION"
                or type(d.get("families")) is not dict
                or set(d["families"]) != set(_FAMILIES)
                or type(d.get("v19_reference")) is not dict
                or d["v19_reference"].get("story_lineage") != "DIVERGENT_STORY_BRANCH"
                or d["v19_reference"].get("comparison_status") != "SEPARATE_SOURCE_REFERENCE_ONLY"):
            raise ValueError("paired dossier cannot assert universal graph or population causality")
        for family, (kind, origin) in _FAMILIES.items():
            case = d["families"][family]
            if (type(case) is not dict or case.get("family") != family
                    or case.get("intervention_kind") != kind
                    or case.get("origin_identity") != origin
                    or case.get("horizon") != 2
                    or type(case.get("native_metric_delta")) is not list
                    or len(case["native_metric_delta"]) != 2):
                raise ValueError("paired source case violates native family contract")
            for arm in ("control", "intervention"):
                data = case.get(arm)
                if type(data) is not dict or type(data.get("metrics")) is not list:
                    raise ValueError("paired native source arms must be complete")
                if len(data["metrics"]) != 2:
                    raise ValueError("paired native metric horizon cannot be altered")
            _audit_native_paired_case(family, case)
        reference = d["v19_reference"]
        if (reference.get("source_owner") != "SITUATED_V19_STORY_COGNITION_SOCIAL_MEMORY"
                or not all(_valid_digest(reference.get(key)) for key in (
                    "initial_world_hash", "control_story_hash", "intervention_story_hash",
                ))
                or reference["control_story_hash"] == reference["intervention_story_hash"]):
            raise ValueError("V19 source-owned story branches must not be merged")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def compare_native_paired_experiments(
    bb: dict[str, object],
    rewiring: dict[str, object],
    evolving: dict[str, object],
    v19_reference: dict[str, object],
) -> NativePairedExperiment:
    if (not all(type(x) is dict for x in (bb, rewiring, evolving, v19_reference))
            or v19_reference.get("schema") != "v19-controlled-coupled-evolution-v1"
            or v19_reference.get("cross_branch", {}).get("story_lineage") != "DIVERGENT_STORY_BRANCH"
            or v19_reference.get("cross_branch", {}).get("causal_attribution") != "UNATTRIBUTED"):
        raise ValueError("paired dossier requires independent native families and source-owned V19")
    root = {
        "schema": _SCHEMA,
        "source_owner": "INDEPENDENT_NATIVE_PAIRED_REPLAYS",
        "normalization": "NONE",
        "cross_family_numeric_ranking": "FORBIDDEN",
        "causal_attribution": "UNATTRIBUTED",
        "sample_size_per_arm": 1,
        "confidence_interval": "NOT_ESTIMABLE",
        "external_validation": "NO_EXTERNAL_VALIDATION",
        "paired_estimand_scope": "MODEL_LOCAL_DETERMINISTIC_DESCRIPTION",
        "families": {
            "bb_fitness_growth": bb,
            "directed_rewiring": rewiring,
            "lifecycle_feedback": evolving,
        },
        "v19_reference": {
            "source_owner": "SITUATED_V19_STORY_COGNITION_SOCIAL_MEMORY",
            "story_lineage": v19_reference["cross_branch"]["story_lineage"],
            "initial_world_hash": v19_reference["initial_world_hash"],
            "control_story_hash": v19_reference["arms"]["control_no_gate_change"]["story_hash"],
            "intervention_story_hash": v19_reference["arms"]["intervention_open_gate"]["story_hash"],
            "comparison_status": "SEPARATE_SOURCE_REFERENCE_ONLY",
        },
    }
    return NativePairedExperiment(
        json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
