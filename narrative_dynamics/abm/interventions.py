from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentState,
    PopulationState,
    SocialNetwork,
    _hash,
    _probability,
    _text,
)
from narrative_dynamics.abm.metrics import EmergenceMetrics, measure_emergence
from narrative_dynamics.abm.simulation import (
    PopulationTrajectory,
    _validate_model_state,
    simulate_population,
)
from narrative_dynamics.contracts import stable_content_hash


@dataclass(frozen=True)
class EdgeSelector:
    """Stable identity of one directed typed social edge."""

    source_agent_id: str
    target_agent_id: str
    relation_type: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_agent_id",
            _text(self.source_agent_id, label="edge selector source agent id"),
        )
        object.__setattr__(
            self,
            "target_agent_id",
            _text(self.target_agent_id, label="edge selector target agent id"),
        )
        object.__setattr__(
            self,
            "relation_type",
            _text(self.relation_type, label="edge selector relation type"),
        )
        if self.source_agent_id == self.target_agent_id:
            raise ValueError("edge selector cannot identify a self-edge")

    @property
    def identity(self) -> tuple[str, str, str]:
        return (
            self.source_agent_id,
            self.target_agent_id,
            self.relation_type,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "relation_type": self.relation_type,
        }


@dataclass(frozen=True)
class EdgeInfluenceChange:
    edge: EdgeSelector
    influence: float

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeSelector):
            raise TypeError("edge influence change requires EdgeSelector")
        object.__setattr__(
            self,
            "influence",
            _probability(self.influence, label="intervention edge influence"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"edge": self.edge.to_dict(), "influence": self.influence}


@dataclass(frozen=True)
class BeliefSeed:
    agent_id: str
    belief: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="belief seed agent id"),
        )
        object.__setattr__(
            self,
            "belief",
            _probability(self.belief, label="intervention belief seed"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"agent_id": self.agent_id, "belief": self.belief}


@dataclass(frozen=True)
class NetworkIntervention:
    """Declarative topology and initial-information treatment."""

    intervention_id: str
    disabled_edges: tuple[EdgeSelector, ...] = ()
    influence_changes: tuple[EdgeInfluenceChange, ...] = ()
    belief_seeds: tuple[BeliefSeed, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intervention_id",
            _text(self.intervention_id, label="network intervention id"),
        )
        for field_name, value, value_type in (
            ("disabled_edges", self.disabled_edges, EdgeSelector),
            ("influence_changes", self.influence_changes, EdgeInfluenceChange),
            ("belief_seeds", self.belief_seeds, BeliefSeed),
        ):
            if not isinstance(value, tuple):
                raise TypeError(f"intervention {field_name} must be a tuple")
            if any(not isinstance(item, value_type) for item in value):
                raise TypeError(
                    f"intervention {field_name} must contain {value_type.__name__} values"
                )

        disabled_ids = tuple(item.identity for item in self.disabled_edges)
        change_ids = tuple(item.edge.identity for item in self.influence_changes)
        seed_ids = tuple(item.agent_id for item in self.belief_seeds)
        if len(set(disabled_ids)) != len(disabled_ids):
            raise ValueError("intervention disabled edge identities must be unique")
        if len(set(change_ids)) != len(change_ids):
            raise ValueError("intervention influence edge identities must be unique")
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError("intervention belief seed agent ids must be unique")
        if set(disabled_ids) & set(change_ids):
            raise ValueError("an intervention edge cannot be both disabled and overridden")
        object.__setattr__(
            self,
            "disabled_edges",
            tuple(sorted(self.disabled_edges, key=lambda item: item.identity)),
        )
        object.__setattr__(
            self,
            "influence_changes",
            tuple(sorted(self.influence_changes, key=lambda item: item.edge.identity)),
        )
        object.__setattr__(
            self,
            "belief_seeds",
            tuple(sorted(self.belief_seeds, key=lambda item: item.agent_id)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention_id": self.intervention_id,
            "disabled_edges": [item.to_dict() for item in self.disabled_edges],
            "influence_changes": [
                item.to_dict() for item in self.influence_changes
            ],
            "belief_seeds": [item.to_dict() for item in self.belief_seeds],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AppliedIntervention:
    """Treatment model and rebound round-zero state with baseline lineage."""

    intervention: NetworkIntervention
    baseline_model_hash: str
    baseline_state_hash: str
    model: NetworkABMModel
    initial_state: PopulationState

    def __post_init__(self) -> None:
        if not isinstance(self.intervention, NetworkIntervention):
            raise TypeError("applied intervention requires NetworkIntervention")
        object.__setattr__(
            self,
            "baseline_model_hash",
            _hash(self.baseline_model_hash, label="baseline model hash"),
        )
        object.__setattr__(
            self,
            "baseline_state_hash",
            _hash(self.baseline_state_hash, label="baseline state hash"),
        )
        _validate_model_state(self.model, self.initial_state)
        if self.initial_state.round_index != 0:
            raise ValueError("applied intervention initial state must be round zero")

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention": self.intervention.to_dict(),
            "baseline_model_hash": self.baseline_model_hash,
            "baseline_state_hash": self.baseline_state_hash,
            "model": self.model.to_dict(),
            "initial_state": self.initial_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class InterventionComparison:
    """Paired deterministic baseline/treatment outcomes and macro deltas."""

    intervention: NetworkIntervention
    baseline: PopulationTrajectory
    treatment: PopulationTrajectory
    baseline_metrics: EmergenceMetrics
    treatment_metrics: EmergenceMetrics
    metric_deltas: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.intervention, NetworkIntervention):
            raise TypeError("intervention comparison requires NetworkIntervention")
        if not isinstance(self.baseline, PopulationTrajectory):
            raise TypeError("intervention comparison baseline must be PopulationTrajectory")
        if not isinstance(self.treatment, PopulationTrajectory):
            raise TypeError("intervention comparison treatment must be PopulationTrajectory")
        if not isinstance(self.baseline_metrics, EmergenceMetrics):
            raise TypeError("intervention comparison requires baseline EmergenceMetrics")
        if not isinstance(self.treatment_metrics, EmergenceMetrics):
            raise TypeError("intervention comparison requires treatment EmergenceMetrics")
        if len(self.baseline.rounds) != len(self.treatment.rounds):
            raise ValueError("intervention comparison trajectories must have equal rounds")
        if not isinstance(self.metric_deltas, Mapping):
            raise TypeError("intervention metric deltas must be a mapping")
        baseline_values = self.baseline_metrics.to_dict()
        treatment_values = self.treatment_metrics.to_dict()
        if set(self.metric_deltas) != set(baseline_values):
            raise ValueError("intervention metric deltas must cover every emergence metric")
        deltas: dict[str, float] = {}
        for name in sorted(baseline_values):
            raw = self.metric_deltas[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("intervention metric deltas must be numeric")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("intervention metric deltas must be finite")
            expected = treatment_values[name] - baseline_values[name]
            if value != expected:
                raise ValueError("intervention metric delta does not match paired metrics")
            deltas[name] = value
        object.__setattr__(self, "metric_deltas", MappingProxyType(deltas))

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention": self.intervention.to_dict(),
            "baseline_trajectory_hash": self.baseline.content_hash,
            "treatment_trajectory_hash": self.treatment.content_hash,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "treatment_metrics": self.treatment_metrics.to_dict(),
            "metric_deltas": dict(self.metric_deltas),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def apply_intervention(
    model: NetworkABMModel,
    initial_state: PopulationState,
    intervention: NetworkIntervention,
) -> AppliedIntervention:
    """Apply a treatment without mutating baseline model or population state."""

    _validate_model_state(model, initial_state)
    if initial_state.round_index != 0:
        raise ValueError("network interventions require a round-zero initial state")
    if not isinstance(intervention, NetworkIntervention):
        raise TypeError("network intervention application requires NetworkIntervention")

    edges = {item.identity: item for item in model.network.edges}
    requested = {
        item.identity for item in intervention.disabled_edges
    } | {item.edge.identity for item in intervention.influence_changes}
    if not requested.issubset(edges):
        raise ValueError("network intervention references an unknown edge")
    seeds = {item.agent_id: item.belief for item in intervention.belief_seeds}
    if not set(seeds).issubset(model.network.agent_ids):
        raise ValueError("network intervention references an unknown agent")

    disabled = {item.identity for item in intervention.disabled_edges}
    influences = {
        item.edge.identity: item.influence for item in intervention.influence_changes
    }
    treated_edges = tuple(
        replace(
            edge,
            active=False if edge.identity in disabled else edge.active,
            influence=influences.get(edge.identity, edge.influence),
        )
        for edge in model.network.edges
    )
    treated_model = NetworkABMModel(
        f"{model.model_id}::intervention::{intervention.content_hash}",
        model.version,
        model.agents,
        SocialNetwork(model.network.agent_ids, treated_edges),
    )

    profiles = {item.agent_id: item for item in model.agents}
    treated_states: list[NetworkAgentState] = []
    for prior in initial_state.agents:
        seeded = prior.agent_id in seeds
        belief = seeds[prior.agent_id] if seeded else prior.belief
        treated_states.append(
            NetworkAgentState(
                prior.agent_id,
                belief,
                max(1, prior.exposure_count) if seeded else prior.exposure_count,
                belief >= profiles[prior.agent_id].broadcast_threshold,
            )
        )
    treated_initial = PopulationState(
        treated_model.model_id,
        treated_model.content_hash,
        0,
        None,
        tuple(treated_states),
    )
    return AppliedIntervention(
        intervention,
        model.content_hash,
        initial_state.content_hash,
        treated_model,
        treated_initial,
    )


def compare_intervention(
    model: NetworkABMModel,
    initial_state: PopulationState,
    intervention: NetworkIntervention,
    *,
    rounds: int,
) -> InterventionComparison:
    """Run a paired baseline/treatment contrast for the same round horizon."""

    applied = apply_intervention(model, initial_state, intervention)
    baseline = simulate_population(model, initial_state, rounds=rounds)
    treatment = simulate_population(
        applied.model,
        applied.initial_state,
        rounds=rounds,
    )
    baseline_metrics = measure_emergence(model, baseline.final_state)
    treatment_metrics = measure_emergence(applied.model, treatment.final_state)
    baseline_values = baseline_metrics.to_dict()
    treatment_values = treatment_metrics.to_dict()
    deltas = {
        name: treatment_values[name] - baseline_values[name]
        for name in sorted(baseline_values)
    }
    return InterventionComparison(
        intervention,
        baseline,
        treatment,
        baseline_metrics,
        treatment_metrics,
        deltas,
    )


def no_propagation_intervention(
    model: NetworkABMModel,
    *,
    intervention_id: str = "no-propagation",
) -> NetworkIntervention:
    """Construct the V1 null model by disabling every active social edge."""

    if not isinstance(model, NetworkABMModel):
        raise TypeError("no-propagation intervention requires NetworkABMModel")
    return NetworkIntervention(
        intervention_id,
        disabled_edges=tuple(
            EdgeSelector(
                edge.source_agent_id,
                edge.target_agent_id,
                edge.relation_type,
            )
            for edge in model.network.edges
            if edge.active
        ),
    )


__all__ = (
    "EdgeSelector",
    "EdgeInfluenceChange",
    "BeliefSeed",
    "NetworkIntervention",
    "AppliedIntervention",
    "InterventionComparison",
    "apply_intervention",
    "compare_intervention",
    "no_propagation_intervention",
)
