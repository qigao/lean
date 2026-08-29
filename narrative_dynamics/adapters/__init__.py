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

# Install the Study V1 Spaceship presentation-index correction on the canonical
# two-stage adapter.  Upstream symbol0/symbol1 are independent binary indices,
# so equal values are valid.  Study-specific APIs remain outside this package
# root surface.
from . import narrative_two_stage as _narrative_two_stage
from . import two_stage_spaceship_configuration as _two_stage_spaceship_configuration

_two_stage_spaceship_configuration.install()
