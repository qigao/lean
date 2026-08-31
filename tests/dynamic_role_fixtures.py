from narrative_dynamics.abm.role_contracts import (
    DynamicRoleModel,
    RoleTransitionRule,
)
from tests.autonomy_fixtures import autonomous_model


def role_transition_rules(*, reverse: bool = False):
    rules = (
        RoleTransitionRule(
            "recipient",
            "relay",
            priority=0,
            minimum_belief=0.6,
            minimum_rounds_in_role=1,
        ),
        RoleTransitionRule(
            "relay",
            "source",
            priority=0,
            minimum_belief=0.5,
            minimum_rounds_in_role=1,
            minimum_verification_count=1,
        ),
        RoleTransitionRule(
            "source",
            "relay",
            priority=0,
            maximum_belief=0.4,
            minimum_rounds_in_role=1,
        ),
    )
    return tuple(reversed(rules)) if reverse else rules


def dynamic_role_model(*, reverse: bool = False) -> DynamicRoleModel:
    return DynamicRoleModel(
        "dynamic-roles",
        "1",
        autonomous_model(reverse=reverse),
        role_transition_rules(reverse=reverse),
    )


def role_state(state, agent_id: str):
    return next(item for item in state.agents if item.agent_id == agent_id)
