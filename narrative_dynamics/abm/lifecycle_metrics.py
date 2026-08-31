from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.abm.contracts import _nonnegative_integer, _probability
from narrative_dynamics.abm.lifecycle import _validate_model_state
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleStatus,
    PopulationLifecycleModel,
    PopulationLifecycleState,
)
from narrative_dynamics.contracts import stable_content_hash


@dataclass(frozen=True)
class PopulationLifecycleMetrics:
    """Macro membership summary over the complete identity catalog."""

    catalog_population: int
    active_population: int
    inactive_population: int
    dead_population: int
    active_share: float
    dead_share: float
    cumulative_entries: int
    cumulative_exits: int

    def __post_init__(self) -> None:
        for field_name in (
            "catalog_population",
            "active_population",
            "inactive_population",
            "dead_population",
            "cumulative_entries",
            "cumulative_exits",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"population lifecycle metric {field_name}",
                ),
            )
        if self.catalog_population == 0:
            raise ValueError("population lifecycle catalog population must be positive")
        if (
            self.active_population
            + self.inactive_population
            + self.dead_population
            != self.catalog_population
        ):
            raise ValueError("lifecycle status counts must sum to catalog population")
        object.__setattr__(
            self,
            "active_share",
            _probability(self.active_share, label="population lifecycle active share"),
        )
        object.__setattr__(
            self,
            "dead_share",
            _probability(self.dead_share, label="population lifecycle dead share"),
        )
        expected_active = self.active_population / self.catalog_population
        expected_dead = self.dead_population / self.catalog_population
        if not math.isclose(self.active_share, expected_active, abs_tol=1e-12):
            raise ValueError("population lifecycle active share must match active count")
        if not math.isclose(self.dead_share, expected_dead, abs_tol=1e-12):
            raise ValueError("population lifecycle dead share must match dead count")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "catalog_population": self.catalog_population,
            "active_population": self.active_population,
            "inactive_population": self.inactive_population,
            "dead_population": self.dead_population,
            "active_share": self.active_share,
            "dead_share": self.dead_share,
            "cumulative_entries": self.cumulative_entries,
            "cumulative_exits": self.cumulative_exits,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def measure_population_lifecycle(
    model: PopulationLifecycleModel,
    state: PopulationLifecycleState,
) -> PopulationLifecycleMetrics:
    """Measure current membership and cumulative population flow."""

    _validate_model_state(model, state)
    catalog_population = len(state.members)
    active_population = sum(
        item.status is LifecycleStatus.ACTIVE for item in state.members
    )
    inactive_population = sum(
        item.status is LifecycleStatus.INACTIVE for item in state.members
    )
    dead_population = sum(
        item.status is LifecycleStatus.DEAD for item in state.members
    )
    return PopulationLifecycleMetrics(
        catalog_population,
        active_population,
        inactive_population,
        dead_population,
        active_population / catalog_population,
        dead_population / catalog_population,
        sum(item.entry_count for item in state.members),
        sum(item.exit_count for item in state.members),
    )


__all__ = (
    "PopulationLifecycleMetrics",
    "measure_population_lifecycle",
)
