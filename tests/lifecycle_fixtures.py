from narrative_dynamics.abm import (
    NetworkABMModel,
    NetworkAgentSpec,
    SocialEdge,
    SocialNetwork,
)


def lifecycle_base_model(*, reverse: bool = False) -> NetworkABMModel:
    agents = (
        NetworkAgentSpec("a", "source", 1.0, 0.5, 0.5),
        NetworkAgentSpec("b", "relay", 1.0, 0.5, 0.5),
        NetworkAgentSpec("c", "observer", 1.0, 0.5, 0.5),
    )
    edges = (
        SocialEdge("a", "b", "peer", 1.0),
        SocialEdge("b", "c", "peer", 1.0),
    )
    if reverse:
        agents = tuple(reversed(agents))
        edges = tuple(reversed(edges))
    return NetworkABMModel(
        "lifecycle-line",
        "1",
        agents,
        SocialNetwork(
            tuple(reversed(("a", "b", "c"))) if reverse else ("a", "b", "c"),
            edges,
        ),
    )
