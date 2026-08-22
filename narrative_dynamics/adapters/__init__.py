"""Adapters from existing models into the canonical simulation protocol."""

from narrative_dynamics.adapters.grounded_goal import (
    GroundedGoalDecisionModel,
    GroundedGoalScenario,
    selected_goal_metrics,
)

__all__ = [
    "GroundedGoalDecisionModel",
    "GroundedGoalScenario",
    "selected_goal_metrics",
]
