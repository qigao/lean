from __future__ import annotations

from enum import Enum

from narrative_dynamics.observations.external import (
    EXTERNAL_CLAIM_SCOPE,
    EXTERNAL_EVIDENCE_ORIGIN,
    ExternalValidationError,
)


class ExternalScoreRole(str, Enum):
    BRIER = "brier"
    LOG = "log"


class PredictiveAdequacyStatus(str, Enum):
    MET = "predictive_adequacy_met"
    NOT_MET = "predictive_adequacy_not_met"


class PredictiveSeparationStatus(str, Enum):
    SEPARATED = "predictively_separated_under_protocol"
    NOT_SEPARATED = "not_predictively_separated_under_protocol"


class ExternalConstraintStatus(str, Enum):
    CONSTRAINED = "constrained_under_external_protocol"
    NOT_CONSTRAINED = "not_constrained_under_external_protocol"


__all__ = [
    "EXTERNAL_CLAIM_SCOPE",
    "EXTERNAL_EVIDENCE_ORIGIN",
    "ExternalConstraintStatus",
    "ExternalScoreRole",
    "ExternalValidationError",
    "PredictiveAdequacyStatus",
    "PredictiveSeparationStatus",
]
