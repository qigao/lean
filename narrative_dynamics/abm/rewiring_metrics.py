from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.abm.contracts import _nonnegative_integer, _probability
from narrative_dynamics.abm.rewiring import _validate_rewiring_model_state
from narrative_dynamics.abm.rewiring_contracts import (
    EndogenousRewiringModel,
    RewiringPopulationState,
)
from narrative_dynamics.contracts import stable_content_hash


@dataclass(frozen=True)
class NetworkStructureMetrics:
    """Macro density, similarity, churn, and fragmentation observables."""

    catalog_edge_count: int
    active_edge_count: int
    active_edge_rate: float
    mean_active_similarity: float
    rewired_edge_rate: float
    cumulative_rewirings: int
    weak_component_count: int
    largest_component_share: float

    def __post_init__(self) -> None:
        for field_name in (
            "catalog_edge_count",
            "active_edge_count",
            "cumulative_rewirings",
            "weak_component_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"network structure metric {field_name}",
                ),
            )
        if self.catalog_edge_count == 0:
            raise ValueError("network structure catalog edge count must be positive")
        if self.active_edge_count > self.catalog_edge_count:
            raise ValueError("network structure active edges cannot exceed catalog edges")
        if self.weak_component_count == 0:
            raise ValueError("network structure weak component count must be positive")
        for field_name in (
            "active_edge_rate",
            "mean_active_similarity",
            "rewired_edge_rate",
            "largest_component_share",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"network structure metric {field_name}",
                ),
            )
        expected_active_rate = self.active_edge_count / self.catalog_edge_count
        if not math.isclose(
            self.active_edge_rate,
            expected_active_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("network structure active edge rate must match edge counts")
        if self.largest_component_share == 0.0:
            raise ValueError("network structure largest component share must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "catalog_edge_count": self.catalog_edge_count,
            "active_edge_count": self.active_edge_count,
            "active_edge_rate": self.active_edge_rate,
            "mean_active_similarity": self.mean_active_similarity,
            "rewired_edge_rate": self.rewired_edge_rate,
            "cumulative_rewirings": self.cumulative_rewirings,
            "weak_component_count": self.weak_component_count,
            "largest_component_share": self.largest_component_share,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _component_sizes(
    agent_ids: tuple[str, ...],
    active_edges: tuple[tuple[str, str], ...],
) -> tuple[int, ...]:
    adjacency = {agent_id: set() for agent_id in agent_ids}
    for source, target in active_edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    unseen = set(agent_ids)
    sizes: list[int] = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            agent_id = stack.pop()
            size += 1
            neighbors = adjacency[agent_id] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def measure_network_structure(
    model: EndogenousRewiringModel,
    state: RewiringPopulationState,
) -> NetworkStructureMetrics:
    """Measure current topology structure and persistent rewiring churn."""

    _validate_rewiring_model_state(model, state)
    catalog_edge_count = len(state.edge_topology)
    active = tuple(item for item in state.edge_topology if item.active)
    active_edge_count = len(active)
    component_sizes = _component_sizes(
        tuple(item.agent_id for item in state.agents),
        tuple(
            (item.edge.source_agent_id, item.edge.target_agent_id)
            for item in active
        ),
    )
    return NetworkStructureMetrics(
        catalog_edge_count,
        active_edge_count,
        active_edge_count / catalog_edge_count,
        (
            sum(item.last_similarity for item in active) / active_edge_count
            if active
            else 0.0
        ),
        sum(item.rewiring_count > 0 for item in state.edge_topology)
        / catalog_edge_count,
        sum(item.rewiring_count for item in state.edge_topology),
        len(component_sizes),
        component_sizes[0] / len(state.agents),
    )


__all__ = (
    "NetworkStructureMetrics",
    "measure_network_structure",
)
