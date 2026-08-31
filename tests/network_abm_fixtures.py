from narrative_dynamics.abm import (
    NetworkABMModel,
    NetworkAgentSpec,
    PopulationState,
    SocialEdge,
    SocialNetwork,
    initialize_population,
)


def line_case(*, reverse: bool = False) -> tuple[NetworkABMModel, PopulationState]:
    agents = (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 1.0, 0.5, 0.5),
        NetworkAgentSpec("c", "recipient", 1.0, 0.5, 0.5),
        NetworkAgentSpec("d", "isolated", 1.0, 0.5, 0.5),
    )
    edges = (
        SocialEdge("a", "b", "peer", 1.0),
        SocialEdge("b", "c", "peer", 1.0),
    )
    roster = ("a", "b", "c", "d")
    if reverse:
        agents = tuple(reversed(agents))
        edges = tuple(reversed(edges))
        roster = tuple(reversed(roster))
    model = NetworkABMModel(
        "line-network",
        "1",
        agents,
        SocialNetwork(roster, edges),
    )
    return model, initialize_population(model, beliefs={"a": 1.0})


def agent(state: PopulationState, agent_id: str):
    return next(item for item in state.agents if item.agent_id == agent_id)
