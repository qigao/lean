"""Deterministic network interaction and population emergence runtime."""

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentSpec,
    NetworkAgentState,
    PopulationState,
    SocialEdge,
    SocialNetwork,
    initialize_population,
)
from narrative_dynamics.abm.simulation import (
    InformationTransmission,
    NetworkRoundResult,
    PopulationTrajectory,
    simulate_population,
    simulate_round,
)
from narrative_dynamics.abm.metrics import EmergenceMetrics, measure_emergence


__all__ = (
    "NetworkAgentSpec",
    "SocialEdge",
    "SocialNetwork",
    "NetworkABMModel",
    "NetworkAgentState",
    "PopulationState",
    "initialize_population",
    "InformationTransmission",
    "NetworkRoundResult",
    "PopulationTrajectory",
    "simulate_round",
    "simulate_population",
    "EmergenceMetrics",
    "measure_emergence",
)
