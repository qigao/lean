from narrative_dynamics.abm import (
    NetworkABMModel,
    NetworkAgentSpec,
    SocialEdge,
    SocialNetwork,
)


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
