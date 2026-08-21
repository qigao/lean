from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Goal:
    """A candidate instrumental state, not a primitive desire."""

    instrumentality: Mapping[str, float]
    cost: float = 0.0
    risk: float = 0.0


@dataclass
class AgentMotivation:
    """Minimal numerical reference model for endogenous goal scoring.

    `pressures` are primitive drive pressures. Candidate goals are scored only
    by how instrumentally they reduce those pressures, minus cost and risk.
    """

    pressures: Mapping[str, float]
    goals: Mapping[str, Goal]

    def score(self, goal_name: str) -> float:
        goal = self.goals[goal_name]
        instrumental_value = sum(
            max(0.0, pressure) * goal.instrumentality.get(drive, 0.0)
            for drive, pressure in self.pressures.items()
        )
        return instrumental_value - goal.cost - goal.risk
