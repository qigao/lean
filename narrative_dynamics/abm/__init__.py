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
from narrative_dynamics.abm.interventions import (
    AppliedIntervention,
    BeliefSeed,
    EdgeInfluenceChange,
    EdgeSelector,
    InterventionComparison,
    NetworkIntervention,
    apply_intervention,
    compare_intervention,
    no_propagation_intervention,
)
from narrative_dynamics.abm.adaptive_contracts import (
    AdaptivePopulationState,
    AdaptiveTrustModel,
    EdgeTrustState,
    TruthFeedback,
    initialize_adaptive_population,
)
from narrative_dynamics.abm.learning import (
    AdaptiveRoundResult,
    AdaptiveTrajectory,
    EdgeTrustUpdate,
    population_view,
    simulate_adaptive_population,
    simulate_adaptive_round,
)
from narrative_dynamics.abm.adaptive_metrics import (
    AdaptiveTrustMetrics,
    measure_adaptive_trust,
)
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleMemberState,
    LifecycleStatus,
    PopulationLifecycleEvent,
    PopulationLifecycleModel,
    PopulationLifecycleState,
    initialize_lifecycle_population,
)
from narrative_dynamics.abm.lifecycle import (
    LifecycleRoundResult,
    LifecycleTrajectory,
    active_population_view,
    simulate_lifecycle_population,
    simulate_lifecycle_round,
)
from narrative_dynamics.abm.lifecycle_metrics import (
    PopulationLifecycleMetrics,
    measure_population_lifecycle,
)
from narrative_dynamics.abm.rewiring_contracts import (
    EdgeTopologyState,
    EndogenousRewiringModel,
    RewiringPopulationState,
    initialize_rewiring_population,
)
from narrative_dynamics.abm.rewiring import (
    EdgeRewiringUpdate,
    RewiringRoundResult,
    RewiringTrajectory,
    rewiring_population_view,
    simulate_rewiring_population,
    simulate_rewiring_round,
)
from narrative_dynamics.abm.rewiring_metrics import (
    NetworkStructureMetrics,
    measure_network_structure,
)
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel,
    EvolvingPopulationState,
    initialize_evolving_population,
)
from narrative_dynamics.abm.evolving import (
    EvolvingRoundResult,
    EvolvingTrajectory,
    evolving_active_population_view,
    simulate_evolving_population,
    simulate_evolving_round,
)
from narrative_dynamics.abm.evolving_metrics import (
    EvolvingSystemMetrics,
    measure_evolving_system,
)
from narrative_dynamics.abm.autonomy_contracts import (
    AgentActionIntent,
    AutonomousAgentState,
    AutonomousNetworkModel,
    AutonomousPopulationState,
    RoleDecisionPolicy,
    SharingDecision,
    TruthObservation,
    initialize_autonomous_population,
)
from narrative_dynamics.abm.autonomy import (
    AutonomousRoundResult,
    AutonomousTrajectory,
    simulate_autonomous_population,
    simulate_autonomous_round,
)
from narrative_dynamics.abm.autonomy_metrics import (
    AgentAutonomyMetrics,
    measure_agent_autonomy,
)
from narrative_dynamics.abm.role_contracts import (
    DynamicRoleAgentState,
    DynamicRoleModel,
    DynamicRolePopulationState,
    RoleTransitionRecord,
    RoleTransitionRule,
    initialize_dynamic_role_population,
)
from narrative_dynamics.abm.roles import (
    DynamicRoleRoundResult,
    DynamicRoleTrajectory,
    simulate_dynamic_role_population,
    simulate_dynamic_role_round,
)
from narrative_dynamics.abm.role_metrics import (
    RoleDynamicsMetrics,
    RolePopulationCount,
    measure_role_dynamics,
)
from narrative_dynamics.abm.calibration_contracts import (
    ABMCalibrationCandidate,
    ABMCalibrationWeights,
    CalibrationSplit,
    EmpiricalABMCase,
    EmpiricalABMDataset,
    ObservedABMSnapshot,
)


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
    "EdgeSelector",
    "EdgeInfluenceChange",
    "BeliefSeed",
    "NetworkIntervention",
    "AppliedIntervention",
    "InterventionComparison",
    "apply_intervention",
    "compare_intervention",
    "no_propagation_intervention",
    "AdaptiveTrustModel",
    "EdgeTrustState",
    "AdaptivePopulationState",
    "TruthFeedback",
    "initialize_adaptive_population",
    "EdgeTrustUpdate",
    "AdaptiveRoundResult",
    "AdaptiveTrajectory",
    "population_view",
    "simulate_adaptive_round",
    "simulate_adaptive_population",
    "AdaptiveTrustMetrics",
    "measure_adaptive_trust",
    "LifecycleStatus",
    "LifecycleEventKind",
    "PopulationLifecycleEvent",
    "PopulationLifecycleModel",
    "LifecycleMemberState",
    "PopulationLifecycleState",
    "initialize_lifecycle_population",
    "LifecycleRoundResult",
    "LifecycleTrajectory",
    "active_population_view",
    "simulate_lifecycle_round",
    "simulate_lifecycle_population",
    "PopulationLifecycleMetrics",
    "measure_population_lifecycle",
    "EndogenousRewiringModel",
    "EdgeTopologyState",
    "RewiringPopulationState",
    "initialize_rewiring_population",
    "EdgeRewiringUpdate",
    "RewiringRoundResult",
    "RewiringTrajectory",
    "rewiring_population_view",
    "simulate_rewiring_round",
    "simulate_rewiring_population",
    "NetworkStructureMetrics",
    "measure_network_structure",
    "EvolvingNetworkModel",
    "EvolvingPopulationState",
    "initialize_evolving_population",
    "EvolvingRoundResult",
    "EvolvingTrajectory",
    "evolving_active_population_view",
    "simulate_evolving_round",
    "simulate_evolving_population",
    "EvolvingSystemMetrics",
    "measure_evolving_system",
    "SharingDecision",
    "RoleDecisionPolicy",
    "TruthObservation",
    "AgentActionIntent",
    "AutonomousAgentState",
    "AutonomousNetworkModel",
    "AutonomousPopulationState",
    "initialize_autonomous_population",
    "AutonomousRoundResult",
    "AutonomousTrajectory",
    "simulate_autonomous_round",
    "simulate_autonomous_population",
    "AgentAutonomyMetrics",
    "measure_agent_autonomy",
    "RoleTransitionRule",
    "RoleTransitionRecord",
    "DynamicRoleAgentState",
    "DynamicRoleModel",
    "DynamicRolePopulationState",
    "initialize_dynamic_role_population",
    "DynamicRoleRoundResult",
    "DynamicRoleTrajectory",
    "simulate_dynamic_role_round",
    "simulate_dynamic_role_population",
    "RolePopulationCount",
    "RoleDynamicsMetrics",
    "measure_role_dynamics",
    "CalibrationSplit",
    "ObservedABMSnapshot",
    "EmpiricalABMCase",
    "EmpiricalABMDataset",
    "ABMCalibrationCandidate",
    "ABMCalibrationWeights",
)
