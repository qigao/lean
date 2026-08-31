from narrative_dynamics.abm.autonomy_contracts import (
    AutonomousNetworkModel,
    RoleDecisionPolicy,
)
from narrative_dynamics.abm.evolving_contracts import EvolvingNetworkModel
from tests.evolving_fixtures import evolving_base_model


def role_policies(*, reverse: bool = False):
    policies = (
        RoleDecisionPolicy("source", True, 0.5, 0.4, None, 1),
        RoleDecisionPolicy("relay", True, 0.5, 0.6, 0.2, 2),
        RoleDecisionPolicy("recipient", False, 0.5, 0.6, 0.2, 1),
    )
    return tuple(reversed(policies)) if reverse else policies


def autonomous_model(*, reverse: bool = False) -> AutonomousNetworkModel:
    evolving = EvolvingNetworkModel(
        "unified-evolution",
        "1",
        evolving_base_model(reverse=reverse),
        ("c", "a") if reverse else ("a", "c"),
        0.5,
        0.5,
        0.2,
        0.8,
    )
    return AutonomousNetworkModel(
        "agent-autonomy",
        "1",
        evolving,
        role_policies(reverse=reverse),
    )
