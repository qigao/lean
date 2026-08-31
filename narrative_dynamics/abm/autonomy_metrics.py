from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.abm.autonomy import _validate_autonomous_model_state
from narrative_dynamics.abm.autonomy_contracts import (
    AutonomousNetworkModel,
    AutonomousPopulationState,
)
from narrative_dynamics.abm.contracts import _nonnegative_integer, _probability
from narrative_dynamics.abm.lifecycle_contracts import LifecycleStatus
from narrative_dynamics.contracts import stable_content_hash


@dataclass(frozen=True)
class AgentAutonomyMetrics:
    """Cumulative autonomous behavior, resources, and current sharing."""

    decision_count: int
    verification_count: int
    verification_rate: float
    silent_decision_count: int
    silence_rate: float
    autonomous_exit_count: int
    exit_rate: float
    initial_verification_budget: int
    remaining_verification_budget: int
    budget_utilization: float
    active_population: int
    active_sharing_count: int
    active_sharing_rate: float

    def __post_init__(self) -> None:
        for field_name in (
            "decision_count",
            "verification_count",
            "silent_decision_count",
            "autonomous_exit_count",
            "initial_verification_budget",
            "remaining_verification_budget",
            "active_population",
            "active_sharing_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"agent autonomy metric {field_name}",
                ),
            )
        if self.verification_count > self.decision_count:
            raise ValueError("autonomy verification count cannot exceed decisions")
        if self.silent_decision_count > self.decision_count:
            raise ValueError("autonomy silent count cannot exceed decisions")
        if self.autonomous_exit_count > self.decision_count:
            raise ValueError("autonomy exit count cannot exceed decisions")
        if self.remaining_verification_budget > self.initial_verification_budget:
            raise ValueError("remaining autonomy budget cannot exceed initial budget")
        if self.active_sharing_count > self.active_population:
            raise ValueError("autonomy sharing count cannot exceed active population")
        for field_name in (
            "verification_rate",
            "silence_rate",
            "exit_rate",
            "budget_utilization",
            "active_sharing_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"agent autonomy metric {field_name}",
                ),
            )
        expected_verification_rate = (
            self.verification_count / self.decision_count
            if self.decision_count
            else 0.0
        )
        expected_silence_rate = (
            self.silent_decision_count / self.decision_count
            if self.decision_count
            else 0.0
        )
        expected_exit_rate = (
            self.autonomous_exit_count / self.decision_count
            if self.decision_count
            else 0.0
        )
        expected_budget_utilization = (
            (
                self.initial_verification_budget
                - self.remaining_verification_budget
            )
            / self.initial_verification_budget
            if self.initial_verification_budget
            else 0.0
        )
        expected_sharing_rate = (
            self.active_sharing_count / self.active_population
            if self.active_population
            else 0.0
        )
        for field_name, expected in (
            ("verification_rate", expected_verification_rate),
            ("silence_rate", expected_silence_rate),
            ("exit_rate", expected_exit_rate),
            ("budget_utilization", expected_budget_utilization),
            ("active_sharing_rate", expected_sharing_rate),
        ):
            if not math.isclose(
                getattr(self, field_name),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                label = field_name.replace("_", " ")
                raise ValueError(f"autonomy {label} must match counts")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "decision_count": self.decision_count,
            "verification_count": self.verification_count,
            "verification_rate": self.verification_rate,
            "silent_decision_count": self.silent_decision_count,
            "silence_rate": self.silence_rate,
            "autonomous_exit_count": self.autonomous_exit_count,
            "exit_rate": self.exit_rate,
            "initial_verification_budget": self.initial_verification_budget,
            "remaining_verification_budget": self.remaining_verification_budget,
            "budget_utilization": self.budget_utilization,
            "active_population": self.active_population,
            "active_sharing_count": self.active_sharing_count,
            "active_sharing_rate": self.active_sharing_rate,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def measure_agent_autonomy(
    model: AutonomousNetworkModel,
    state: AutonomousPopulationState,
) -> AgentAutonomyMetrics:
    """Measure cumulative decisions, resources, exits, and current sharing."""

    _validate_autonomous_model_state(model, state)
    profiles = {
        item.agent_id: item for item in model.evolving_model.base_model.agents
    }
    policies = {item.role: item for item in model.role_policies}
    initial_budget = sum(
        policies[profiles[item.agent_id].role].initial_verification_budget
        for item in state.agents
    )
    decision_count = sum(item.decision_count for item in state.agents)
    verification_count = sum(item.verification_count for item in state.agents)
    silent_count = sum(item.silent_decision_count for item in state.agents)
    exit_count = sum(
        item.exit_count for item in state.evolving_state.members
    )
    active_ids = {
        item.agent_id
        for item in state.evolving_state.members
        if item.status is LifecycleStatus.ACTIVE
    }
    active_population = len(active_ids)
    active_sharing_count = sum(
        item.agent_id in active_ids and item.sharing for item in state.agents
    )
    remaining_budget = sum(
        item.remaining_verification_budget for item in state.agents
    )
    return AgentAutonomyMetrics(
        decision_count,
        verification_count,
        verification_count / decision_count if decision_count else 0.0,
        silent_count,
        silent_count / decision_count if decision_count else 0.0,
        exit_count,
        exit_count / decision_count if decision_count else 0.0,
        initial_budget,
        remaining_budget,
        (
            (initial_budget - remaining_budget) / initial_budget
            if initial_budget
            else 0.0
        ),
        active_population,
        active_sharing_count,
        active_sharing_count / active_population if active_population else 0.0,
    )


__all__ = (
    "AgentAutonomyMetrics",
    "measure_agent_autonomy",
)
