"""R2.M4b source-bound *model-native* BB/rewiring/evolving comparison.

These are three DIFFERENT existing ABM owners. A BB undirected fitness
birth edge, a rewiring candidate directed edge, an evolving lifecycle
edge, and a V19 private-message relation are not equivalent measurements.
No second graph engine, normalized cross-family score or causal theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.bb_runtime import replay_bb_population
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeError, BBRuntimeReplay, BBRuntimeSeed, BBRuntimeTick,
)
from narrative_dynamics.abm.rewiring import simulate_rewiring_population
from narrative_dynamics.abm.rewiring_contracts import (
    EndogenousRewiringModel, RewiringPopulationState,
)
from narrative_dynamics.abm.evolving import simulate_evolving_population
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel, EvolvingPopulationState,
)
from narrative_dynamics.abm.adaptive_contracts import TruthFeedback
from narrative_dynamics.abm.lifecycle_contracts import PopulationLifecycleEvent

_SCHEMA = "v19-abm-model-family-qualification-v1"
_MAX_STEPS = 8


@dataclass(frozen=True)
class NativeModelFamilyComparison:
    """Immutable, audit-safe cross-family dossier; not an ABM state owner."""

    _canonical_json: str

    def __post_init__(self) -> None:
        if type(self._canonical_json) is not str:
            raise TypeError("comparison must store canonical immutable JSON")
        d = json.loads(self._canonical_json)
        if (
            type(d) is not dict
            or d.get("schema") != _SCHEMA
            or d.get("causal_attribution") != "UNATTRIBUTED"
            or d.get("cross_family_numeric_ranking") != "FORBIDDEN"
            or d.get("source_owner") != "INDEPENDENT_NATIVE_MODELS"
            or not isinstance(d.get("families"), dict)
            or set(d["families"]) != {"bb_fitness_growth", "directed_rewiring", "lifecycle_feedback"}
            or d.get("v19_reference", {}).get("story_lineage") != "DIVERGENT_STORY_BRANCH"
        ):
            raise ValueError("comparison must not claim equivalent graphs or causal rankings")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _transmission_ids(items: tuple) -> list[list[str]]:
    return [
        [t.source_agent_id, t.target_agent_id, t.relation_type]
        for t in items
    ]


def _active_directed(state) -> list[list[str]]:
    """Describe native candidate activity, NOT its V19 trust meaning."""
    return [
        [e.edge.source_agent_id, e.edge.target_agent_id, e.edge.relation_type]
        for e in state.edge_topology if e.active
    ]


def observe_bb_growth(
    seed: BBRuntimeSeed,
    m: int,
    ticks: tuple[BBRuntimeTick, ...],
) -> dict[str, object]:
    """Check BB's exact finite birth/idle tick sequence with its native owner."""
    if not isinstance(seed, BBRuntimeSeed) or type(ticks) is not tuple:
        raise TypeError("BB observation requires typed seed and tuple of ticks")
    if not 1 <= len(ticks) <= _MAX_STEPS:
        raise ValueError("BB comparison requires 1 to 8 authored ticks")
    replay = replay_bb_population(seed, m, ticks)
    if isinstance(replay, BBRuntimeError):
        raise ValueError("BB source replay rejected authored tick: "
                         + replay.stage + ":" + replay.code)
    if not isinstance(replay, BBRuntimeReplay):
        raise TypeError("BB comparison requires native BBRuntimeReplay")
    if replay_bb_population(seed, m, ticks) != replay:
        raise ValueError("BB source replay is not deterministic")
    records = []
    for step in replay.transitions:
        if step.next_frame.parent_frame_hash != step.prior.content_hash:
            raise ValueError("BB frame chain lost parent identity")
        records.append({
            "native_tick": step.next_frame.tick_count,
            "native_birth_index": step.next_frame.birth_count,
            "native_population_round": step.next_frame.population.round_index,
            "tick_kind": step.tick.kind,
            "tick_hash": stable_content_hash(step.tick.to_dict()),
            "parent_frame_hash": step.prior.content_hash,
            "next_frame_hash": step.next_frame.content_hash,
            "native_undirected_topology": [
                [step.next_frame.agent_ids[u], step.next_frame.agent_ids[v]]
                for u, v in step.next_frame.topology.edges
            ],
            "native_agent_ids": list(step.next_frame.agent_ids),
            "native_transmissions": _transmission_ids(step.round_result.transmissions),
            "exact_tick_mass": str(step.tick_mass),
            "exact_cumulative_trace_mass": str(step.next_frame.trace_mass),
        })
    return {
        "native_owner": "BBRuntimeReplay",
        "edge_semantics": "UNDIRECTED_FITNESS_WEIGHTED_BIRTH",
        "time_semantics": "GLOBAL_TICK_AND_BIRTH_INDEX_NOT_V19_STORY_ROUND",
        "trust_semantics": "NOT_REPRESENTED_AS_V19_SOCIAL_TRUST",
        "initial_frame_hash": replay.initial.content_hash,
        "final_frame_hash": replay.final.content_hash,
        "seed_hash": stable_content_hash({
            "network":seed.network.to_dict(), "agents":[a.to_dict() for a in seed.agents]
        }),
        "m": m,
        "initial_undirected_topology": [
            [replay.initial.agent_ids[u], replay.initial.agent_ids[v]]
            for u, v in replay.initial.topology.edges
        ],
        "final_node_count": replay.final.topology.node_count,
        "final_birth_count": replay.final.birth_count,
        "final_native_beliefs": {
            a.agent_id:a.belief for a in replay.final.population.agents
        },
        "steps": records,
    }


def observe_rewiring(
    model: EndogenousRewiringModel,
    initial: RewiringPopulationState,
    rounds: int,
) -> dict[str, object]:
    """Native directed similarity/hysteresis edge changes AFTER propagation."""
    if not isinstance(model, EndogenousRewiringModel) or not isinstance(initial, RewiringPopulationState):
        raise TypeError("rewiring observation requires its exact model/state")
    if type(rounds) is not int or not 1 <= rounds <= _MAX_STEPS:
        raise ValueError("rewiring comparison requires 1 to 8 rounds")
    trajectory = simulate_rewiring_population(model, initial, rounds=rounds)
    if simulate_rewiring_population(model, initial, rounds=rounds) != trajectory:
        raise ValueError("rewiring model results are not source-deterministic")
    records = []
    for result in trajectory.rounds:
        if result.next_state.parent_state_hash != result.prior_state.content_hash:
            raise ValueError("rewiring result has a stale source parent")
        records.append({
            "round": result.round_index,
            "source_state_hash": result.prior_state.content_hash,
            "next_state_hash": result.next_state.content_hash,
            "prior_active_candidate_edges": _active_directed(result.prior_state),
            "next_active_candidate_edges": _active_directed(result.next_state),
            "native_transmissions": _transmission_ids(result.transmissions),
            "changed_candidate_edges": [
                [e.edge.source_agent_id, e.edge.target_agent_id, e.edge.relation_type]
                for e in result.edge_updates if e.changed
            ],
            "native_beliefs": {
                a.agent_id:a.belief for a in result.next_state.agents
            },
        })
    return {
        "native_owner": "RewiringPopulationState",
        "edge_semantics": "DIRECTED_FIXED_CANDIDATES_SIMILARITY_HYSTERESIS",
        "time_semantics": "PROPAGATE_OLD_TOPOLOGY_THEN_REWIRE_FOR_NEXT_ROUND",
        "trust_semantics": "NO_PRIVATE_V19_TESTIMONY_OR_SOCIAL_CLAIM_STATE",
        "model_hash": model.content_hash,
        "initial_state_hash": initial.content_hash,
        "final_state_hash": trajectory.final_state.content_hash,
        "formation_similarity": model.formation_similarity,
        "dissolution_similarity": model.dissolution_similarity,
        "candidate_count": len(initial.edge_topology),
        "steps": records,
    }


def observe_evolving(
    model: EvolvingNetworkModel,
    initial: EvolvingPopulationState,
    event_schedule: tuple[tuple[PopulationLifecycleEvent, ...], ...],
    feedback_schedule: tuple[tuple[TruthFeedback, ...], ...],
) -> dict[str, object]:
    """Native lifecycle/external truth-feedback + propagation + rewiring."""
    if not isinstance(model, EvolvingNetworkModel) or not isinstance(initial, EvolvingPopulationState):
        raise TypeError("evolving comparison requires its exact model/state")
    if type(event_schedule) is not tuple or type(feedback_schedule) is not tuple:
        raise TypeError("evolving source schedules must be immutable tuples")
    if not 1 <= len(event_schedule) <= _MAX_STEPS or len(event_schedule) != len(feedback_schedule):
        raise ValueError("evolving comparison requires 1 to 8 paired rounds")
    trajectory = simulate_evolving_population(
        model, initial, event_schedule=event_schedule,
        feedback_schedule=feedback_schedule,
    )
    if simulate_evolving_population(
        model, initial, event_schedule=event_schedule,
        feedback_schedule=feedback_schedule,
    ) != trajectory:
        raise ValueError("evolving source replay is not deterministic")
    records = []
    for result in trajectory.rounds:
        if result.next_state.parent_state_hash != result.prior_state.content_hash:
            raise ValueError("evolving result lost its exact prior state")
        records.append({
            "round": result.round_index,
            "source_state_hash": result.prior_state.content_hash,
            "next_state_hash": result.next_state.content_hash,
            "native_members_after": [
                {"agent_id":x.agent_id, "status":x.status.value,
                 "belief":x.belief} for x in result.next_state.members
            ],
            "native_active_candidate_edges": _active_directed(result.next_state),
            "native_transmissions": _transmission_ids(result.transmissions),
            "lifecycle_events": [
                {"agent_id":x.agent_id, "kind":x.kind.value}
                for x in result.events
            ],
            "feedback_targets": [
                [x.edge.source_agent_id, x.edge.target_agent_id,
                 x.edge.relation_type] for x in result.feedback
            ],
            "native_trust_values": [
                {"source":x.edge.source_agent_id, "target":x.edge.target_agent_id,
                 "trust":x.trust, "feedback_count":x.feedback_count}
                for x in result.next_state.edge_trust
            ],
            "native_rewiring_changes": [
                [x.edge.source_agent_id, x.edge.target_agent_id]
                for x in result.rewiring_updates if x.changed
            ],
        })
    return {
        "native_owner": "EvolvingPopulationState",
        "edge_semantics": "DIRECTED_LIFECYCLE_TRUST_AND_SIMILARITY_HYSTERESIS",
        "time_semantics": "MEMBERSHIP_THEN_PROPAGATION_THEN_EXTERNAL_FEEDBACK_THEN_REWIRING",
        "trust_semantics": "EXTERNAL_TRUTH_FEEDBACK_NOT_V19_PRIVATE_INSPECT_VERIFICATION",
        "model_hash": model.content_hash,
        "initial_state_hash": initial.content_hash,
        "final_state_hash": trajectory.final_state.content_hash,
        "initial_active_members": [
            x.agent_id for x in initial.members if x.status.value == "active"
        ],
        "learning_rate": model.learning_rate,
        "initial_trust": model.initial_trust,
        "candidate_count": len(initial.edge_topology),
        "steps": records,
    }


def compare_native_model_families(
    bb: dict[str, object],
    rewiring: dict[str, object],
    evolving: dict[str, object],
    v19_reference: dict[str, object],
) -> NativeModelFamilyComparison:
    """Prepare a no-ranking dossier; never normalize incompatible edge kinds."""
    if (
        type(bb) is not dict or type(rewiring) is not dict or type(evolving) is not dict
        or type(v19_reference) is not dict
        or bb.get("native_owner") != "BBRuntimeReplay"
        or rewiring.get("native_owner") != "RewiringPopulationState"
        or evolving.get("native_owner") != "EvolvingPopulationState"
        or v19_reference.get("schema") != "v19-controlled-coupled-evolution-v1"
        or v19_reference.get("cross_branch", {}).get("story_lineage") != "DIVERGENT_STORY_BRANCH"
        or v19_reference.get("cross_branch", {}).get("causal_attribution") != "UNATTRIBUTED"
    ):
        raise ValueError("model family dossier requires three native sources and a V19 reference")
    root = {
        "schema": _SCHEMA,
        "source_owner": "INDEPENDENT_NATIVE_MODELS",
        "scope": "frozen finite model-native qualification, NOT equivalent V19 histories",
        "causal_attribution": "UNATTRIBUTED",
        "cross_family_numeric_ranking": "FORBIDDEN",
        "normalization": "NONE",
        "families": {
            "bb_fitness_growth":bb,
            "directed_rewiring":rewiring,
            "lifecycle_feedback":evolving,
        },
        "v19_reference": {
            "source_owner":"SITUATED_V19_STORY_COGNITION_SOCIAL_MEMORY",
            "story_lineage":v19_reference["cross_branch"]["story_lineage"],
            "initial_world_hash":v19_reference["initial_world_hash"],
            "model_hash":v19_reference["model_hash"],
            "control_story_hash":v19_reference["arms"]["control_no_gate_change"]["story_hash"],
            "intervention_story_hash":v19_reference["arms"]["intervention_open_gate"]["story_hash"],
            "controlled_intervention": "OPEN_PASSAGE_VS_WAIT_THEN_IDENTICAL_MOVE_TELL",
            "trust_semantics": "V19_PRIVATE_EXACT_PERCEPT_TESTIMONY_VERIFICATION",
        },
        "comparability": {
            "bb_vs_directed_others": "INCOMPARABLE_UNDIRECTED_BIRTH_VS_DIRECTED_CANDIDATES",
            "evolving_vs_v19_trust": "INCOMPARABLE_EXTERNAL_TRUTH_FEEDBACK_VS_PRIVATE_EVIDENCE",
            "rewiring_vs_v19_edges": "INCOMPARABLE_SIMILARITY_CANDIDATE_VS_SOCIAL_TRUST",
            "raw_edge_counts": "NOT_RANKABLE",
            "propagation_events": "MODEL_NATIVE_UNITS_ONLY",
        },
    }
    return NativeModelFamilyComparison(
        json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )
