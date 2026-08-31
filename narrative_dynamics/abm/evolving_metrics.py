from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.abm.contracts import _nonnegative_integer, _probability
from narrative_dynamics.abm.evolving import _validate_evolving_model_state
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel,
    EvolvingPopulationState,
)
from narrative_dynamics.abm.lifecycle_contracts import LifecycleStatus
from narrative_dynamics.contracts import stable_content_hash


@dataclass(frozen=True)
class EvolvingSystemMetrics:
    """One cross-layer macro snapshot of the unified evolving system."""

    catalog_population: int
    active_population: int
    inactive_population: int
    dead_population: int
    active_share: float
    dead_share: float
    mean_active_belief: float | None
    active_adoption_rate: float | None
    mean_trust: float
    learned_edge_rate: float
    catalog_edge_count: int
    effective_active_edge_count: int
    effective_active_edge_rate: float
    cumulative_entries: int
    cumulative_exits: int
    cumulative_rewirings: int

    def __post_init__(self) -> None:
        for field_name in (
            "catalog_population",
            "active_population",
            "inactive_population",
            "dead_population",
            "catalog_edge_count",
            "effective_active_edge_count",
            "cumulative_entries",
            "cumulative_exits",
            "cumulative_rewirings",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"evolving system metric {field_name}",
                ),
            )
        if self.catalog_population == 0:
            raise ValueError("evolving system catalog population must be positive")
        if self.catalog_edge_count == 0:
            raise ValueError("evolving system catalog edge count must be positive")
        if (
            self.active_population
            + self.inactive_population
            + self.dead_population
            != self.catalog_population
        ):
            raise ValueError("evolving status counts must sum to catalog population")
        if self.effective_active_edge_count > self.catalog_edge_count:
            raise ValueError("evolving effective edges cannot exceed catalog edges")
        for field_name in (
            "active_share",
            "dead_share",
            "mean_trust",
            "learned_edge_rate",
            "effective_active_edge_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"evolving system metric {field_name}",
                ),
            )
        expected_active_share = self.active_population / self.catalog_population
        expected_dead_share = self.dead_population / self.catalog_population
        expected_edge_rate = (
            self.effective_active_edge_count / self.catalog_edge_count
        )
        if not math.isclose(
            self.active_share,
            expected_active_share,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("evolving active share must match population counts")
        if not math.isclose(
            self.dead_share,
            expected_dead_share,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("evolving dead share must match population counts")
        if not math.isclose(
            self.effective_active_edge_rate,
            expected_edge_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("evolving effective edge rate must match edge counts")
        active_metrics = (self.mean_active_belief, self.active_adoption_rate)
        if self.active_population == 0:
            if any(item is not None for item in active_metrics):
                raise ValueError(
                    "evolving active belief metrics must be None without active agents"
                )
        else:
            if any(item is None for item in active_metrics):
                raise ValueError(
                    "evolving active belief metrics cannot be None with active agents"
                )
            object.__setattr__(
                self,
                "mean_active_belief",
                _probability(
                    self.mean_active_belief,
                    label="evolving mean active belief",
                ),
            )
            object.__setattr__(
                self,
                "active_adoption_rate",
                _probability(
                    self.active_adoption_rate,
                    label="evolving active adoption rate",
                ),
            )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "catalog_population": self.catalog_population,
            "active_population": self.active_population,
            "inactive_population": self.inactive_population,
            "dead_population": self.dead_population,
            "active_share": self.active_share,
            "dead_share": self.dead_share,
            "mean_active_belief": self.mean_active_belief,
            "active_adoption_rate": self.active_adoption_rate,
            "mean_trust": self.mean_trust,
            "learned_edge_rate": self.learned_edge_rate,
            "catalog_edge_count": self.catalog_edge_count,
            "effective_active_edge_count": self.effective_active_edge_count,
            "effective_active_edge_rate": self.effective_active_edge_rate,
            "cumulative_entries": self.cumulative_entries,
            "cumulative_exits": self.cumulative_exits,
            "cumulative_rewirings": self.cumulative_rewirings,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def measure_evolving_system(
    model: EvolvingNetworkModel,
    state: EvolvingPopulationState,
) -> EvolvingSystemMetrics:
    """Measure membership, cognition, trust, and effective structure together."""

    _validate_evolving_model_state(model, state)
    catalog_population = len(state.members)
    active = tuple(
        item for item in state.members if item.status is LifecycleStatus.ACTIVE
    )
    active_ids = {item.agent_id for item in active}
    inactive_population = sum(
        item.status is LifecycleStatus.INACTIVE for item in state.members
    )
    dead_population = sum(
        item.status is LifecycleStatus.DEAD for item in state.members
    )
    profiles = {item.agent_id: item for item in model.base_model.agents}
    active_population = len(active)
    mean_active_belief = (
        sum(item.belief for item in active) / active_population
        if active
        else None
    )
    active_adoption_rate = (
        sum(
            item.belief >= profiles[item.agent_id].adoption_threshold
            for item in active
        )
        / active_population
        if active
        else None
    )
    catalog_edge_count = len(state.edge_topology)
    effective_edges = tuple(
        item
        for item in state.edge_topology
        if item.active
        and item.edge.source_agent_id in active_ids
        and item.edge.target_agent_id in active_ids
    )
    return EvolvingSystemMetrics(
        catalog_population,
        active_population,
        inactive_population,
        dead_population,
        active_population / catalog_population,
        dead_population / catalog_population,
        mean_active_belief,
        active_adoption_rate,
        sum(item.trust for item in state.edge_trust) / len(state.edge_trust),
        sum(item.feedback_count > 0 for item in state.edge_trust)
        / len(state.edge_trust),
        catalog_edge_count,
        len(effective_edges),
        len(effective_edges) / catalog_edge_count,
        sum(item.entry_count for item in state.members),
        sum(item.exit_count for item in state.members),
        sum(item.rewiring_count for item in state.edge_topology),
    )


__all__ = (
    "EvolvingSystemMetrics",
    "measure_evolving_system",
)
