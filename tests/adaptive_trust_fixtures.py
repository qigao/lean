from narrative_dynamics.abm import (
    NetworkABMModel,
    NetworkAgentSpec,
    SocialEdge,
    SocialNetwork,
)
from narrative_dynamics.abm.adaptive_contracts import (
    AdaptivePopulationState,
    AdaptiveTrustModel,
    TruthFeedback,
    initialize_adaptive_population,
)
from narrative_dynamics.abm.interventions import EdgeSelector


def trust_base_model(*, reverse: bool = False) -> NetworkABMModel:
    agents = (
        NetworkAgentSpec("accurate", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("inaccurate", "source", 1.0, 0.5, 0.0),
        NetworkAgentSpec("target", "recipient", 1.0, 0.5, 0.5),
    )
    edges = (
        SocialEdge("accurate", "target", "report", 1.0),
        SocialEdge("inaccurate", "target", "report", 1.0),
    )
    roster = ("accurate", "inaccurate", "target")
    if reverse:
        agents = tuple(reversed(agents))
        edges = tuple(reversed(edges))
        roster = tuple(reversed(roster))
    return NetworkABMModel(
        "two-sources",
        "1",
        agents,
        SocialNetwork(roster, edges),
    )


def adaptive_case(
    *,
    learning_rate: float = 0.5,
    initial_trust: float = 0.5,
    reverse: bool = False,
) -> tuple[AdaptiveTrustModel, AdaptivePopulationState]:
    model = AdaptiveTrustModel(
        "adaptive-sources",
        "1",
        trust_base_model(reverse=reverse),
        learning_rate,
        initial_trust,
    )
    state = initialize_adaptive_population(
        model,
        beliefs={"accurate": 1.0, "inaccurate": 0.0},
    )
    return model, state


def truth_feedback(*, reverse: bool = False) -> tuple[TruthFeedback, ...]:
    feedback = (
        TruthFeedback(EdgeSelector("accurate", "target", "report"), 1.0),
        TruthFeedback(EdgeSelector("inaccurate", "target", "report"), 1.0),
    )
    return tuple(reversed(feedback)) if reverse else feedback


def adaptive_agent(state: AdaptivePopulationState, agent_id: str):
    return next(item for item in state.agents if item.agent_id == agent_id)


def edge_trust(
    state: AdaptivePopulationState,
    source_agent_id: str,
    target_agent_id: str,
):
    return next(
        item
        for item in state.edge_trust
        if item.edge.source_agent_id == source_agent_id
        and item.edge.target_agent_id == target_agent_id
    )
