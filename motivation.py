from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


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


def effective_pressure(boundary: Sequence[float], pressures: Sequence[float]) -> float:
    """Weighted perceived pressure inside an agent's motivational boundary."""
    return sum(weight * pressure for weight, pressure in zip(boundary, pressures))


def softmax_probability(beta: float, own_score: float, rival_score: float) -> float:
    """Binary softmax probability written in numerically stable logistic form."""
    return 1.0 / (1.0 + math.exp(beta * (rival_score - own_score)))


def learn_instrumentality(old: float, observed: float, rate: float) -> float:
    """Prediction-error learning update for goal instrumentality."""
    return old + rate * (observed - old)
