from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import random

from grounded_goal_softmax import grounded_goal_choice_probabilities
from grounded_hypothesis_space import GroundedHypothesisSpace
from narrative_dynamics.contracts import ModelRun, SimulationTrace, TraceEvent


@dataclass(frozen=True)
class GroundedGoalScenario:
    """One finite grounded epistemic decision problem.

    The adapter intentionally stores the existing grounded hypothesis space and
    goal-instrumentality maps instead of reproducing their mathematics.
    """

    id: str
    space: GroundedHypothesisSpace
    instrumentality: Mapping[str, Mapping[str, float]]
    pressure: float
    costs: Mapping[str, float] = field(default_factory=dict)
    risks: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("grounded goal scenario id must be non-empty")
        if not self.instrumentality:
            raise ValueError("grounded goal scenario must contain at least one goal")
        if not math.isfinite(float(self.pressure)):
            raise ValueError("grounded goal pressure must be finite")
        goals = set(self.instrumentality)
        if not set(self.costs).issubset(goals):
            raise ValueError("grounded goal costs contain an unknown goal")
        if not set(self.risks).issubset(goals):
            raise ValueError("grounded goal risks contain an unknown goal")


class GroundedGoalDecisionModel:
    """Sample a goal from the repository's existing grounded softmax model."""

    name = "grounded-goal-decision"

    def simulate(
        self,
        scenario: GroundedGoalScenario,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        if set(parameters) != {"beta"}:
            raise ValueError("grounded goal model requires exactly one beta parameter")
        beta = float(parameters["beta"])
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("grounded goal beta must be positive and finite")

        raw_policy = grounded_goal_choice_probabilities(
            scenario.space,
            scenario.instrumentality,
            pressure=float(scenario.pressure),
            beta=beta,
            costs=scenario.costs,
            risks=scenario.risks,
        )
        policy = {goal: float(raw_policy[goal]) for goal in sorted(raw_policy)}
        if not policy:
            raise ValueError("grounded goal policy must be non-empty")
        if any(not math.isfinite(value) or value <= 0.0 for value in policy.values()):
            raise ValueError("grounded goal policy must contain positive finite mass")
        total = sum(policy.values())
        if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("grounded goal policy must be normalized")

        draw = rng.random()
        cumulative = 0.0
        selected = next(reversed(policy))
        for goal, probability in policy.items():
            cumulative += probability
            if draw < cumulative:
                selected = goal
                break

        events = (
            TraceEvent(
                tick=0,
                kind="policy_computed",
                data={"probabilities": dict(policy)},
            ),
            TraceEvent(
                tick=1,
                kind="goal_selected",
                data={"goal": selected},
            ),
        )
        return ModelRun(
            events=events,
            outcome={
                "selected_goal": selected,
                "policy": dict(policy),
            },
        )


def selected_goal_metrics(trace: SimulationTrace) -> dict[str, float]:
    """Represent one sampled decision as stable one-hot goal coordinates."""

    policy = trace.outcome.get("policy")
    selected = trace.outcome.get("selected_goal")
    if not isinstance(policy, Mapping) or not policy:
        raise ValueError("trace outcome must contain a non-empty goal policy")
    if not isinstance(selected, str) or selected not in policy:
        raise ValueError("trace outcome selected goal must belong to its policy")

    goals: list[str] = []
    for goal, raw_probability in policy.items():
        if not isinstance(goal, str) or not goal:
            raise ValueError("goal labels must be non-empty strings")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as error:
            raise ValueError("goal policy probabilities must be numeric") from error
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("goal policy probabilities must be finite and non-negative")
        goals.append(goal)

    return {
        f"choice.{goal}": 1.0 if goal == selected else 0.0
        for goal in sorted(goals)
    }
