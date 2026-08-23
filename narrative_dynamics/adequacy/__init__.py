"""Model-adequacy diagnostics for finite research models."""

from narrative_dynamics.adequacy.coverage import (
    AxisCoverage,
    CoverageAxis,
    CoverageBin,
    FinalTestCoverageReport,
    ScenarioCoverageAxis,
    diagnose_final_test_coverage,
)
from narrative_dynamics.adequacy.interaction import (
    FactorInteractionReport,
    FactorSource,
    InteractionCorner,
    LocalFactor,
    MetricInteraction,
    local_factor_interaction_report,
)
from narrative_dynamics.adequacy.misspecification import (
    MisspecificationCandidate,
    MisspecificationReport,
    assess_misspecification,
)

__all__ = [
    "AxisCoverage",
    "CoverageAxis",
    "CoverageBin",
    "FactorInteractionReport",
    "FactorSource",
    "FinalTestCoverageReport",
    "InteractionCorner",
    "LocalFactor",
    "MetricInteraction",
    "MisspecificationCandidate",
    "MisspecificationReport",
    "ScenarioCoverageAxis",
    "assess_misspecification",
    "diagnose_final_test_coverage",
    "local_factor_interaction_report",
]
