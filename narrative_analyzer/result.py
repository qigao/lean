"""Immutable analyzer verdicts and provenance records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class ClaimStatus(Enum):
    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    status: ClaimStatus
    theorem: str | None
    assumptions: tuple[str, ...]
    exact_values: Mapping[str, str]
    note: str | None

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id must not be empty")
        if self.status in (ClaimStatus.PROVED, ClaimStatus.DISPROVED):
            if self.theorem is None or not self.theorem.strip():
                raise ValueError(
                    f"{self.status.value} claim {self.claim_id} requires theorem provenance"
                )
        elif self.status is ClaimStatus.UNKNOWN:
            if self.note is None or not self.note.strip():
                raise ValueError(
                    f"UNKNOWN claim {self.claim_id} requires an explanatory note"
                )
        else:
            raise ValueError(f"unsupported claim status: {self.status!r}")

        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(
            self,
            "exact_values",
            MappingProxyType(dict(self.exact_values)),
        )


@dataclass(frozen=True)
class AnalysisResult:
    model_summary: Mapping[str, object]
    claims: tuple[ClaimResult, ...]

    def __post_init__(self) -> None:
        claims = tuple(self.claims)
        claim_ids = [claim.claim_id for claim in claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim ids must be unique")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(
            self,
            "model_summary",
            MappingProxyType(dict(self.model_summary)),
        )

    @property
    def consensus_claim(self) -> ClaimResult | None:
        by_id = {claim.claim_id: claim for claim in self.claims}
        for claim_id in ("path2_consensus", "pathn_consensus_exists"):
            candidate = by_id.get(claim_id)
            if candidate is not None and candidate.status is not ClaimStatus.UNKNOWN:
                return candidate
        return None

    @property
    def consensus_status(self) -> ClaimStatus:
        claim = self.consensus_claim
        if claim is None:
            return ClaimStatus.UNKNOWN
        return claim.status


class AnalyzerError(RuntimeError):
    """Base class for analyzer execution failures, not theorem verdicts."""


class CertificateGenerationError(AnalyzerError):
    pass


class CertificateCompileError(AnalyzerError):
    pass


class CertificateTimeoutError(AnalyzerError):
    pass


class ProvenanceMismatchError(AnalyzerError):
    pass
