from __future__ import annotations

from dataclasses import dataclass

from narrative_dynamics.abm.adaptive_contracts import (
    AdaptivePopulationState,
    AdaptiveTrustModel,
)
from narrative_dynamics.abm.contracts import _probability
from narrative_dynamics.abm.learning import _validate_adaptive_model_state
from narrative_dynamics.contracts import stable_content_hash


@dataclass(frozen=True)
class AdaptiveTrustMetrics:
    """Macro summary of learned reliability across the social network."""

    mean_trust: float
    min_trust: float
    max_trust: float
    learned_edge_rate: float

    def __post_init__(self) -> None:
        for field_name in (
            "mean_trust",
            "min_trust",
            "max_trust",
            "learned_edge_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"adaptive trust metric {field_name}",
                ),
            )
        if not self.min_trust <= self.mean_trust <= self.max_trust:
            raise ValueError("adaptive trust metrics must satisfy min <= mean <= max")

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_trust": self.mean_trust,
            "min_trust": self.min_trust,
            "max_trust": self.max_trust,
            "learned_edge_rate": self.learned_edge_rate,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def measure_adaptive_trust(
    model: AdaptiveTrustModel,
    state: AdaptivePopulationState,
) -> AdaptiveTrustMetrics:
    """Measure the distribution and feedback coverage of edge-local trust."""

    _validate_adaptive_model_state(model, state)
    trusts = tuple(item.trust for item in state.edge_trust)
    count = len(trusts)
    return AdaptiveTrustMetrics(
        sum(trusts) / count,
        min(trusts),
        max(trusts),
        sum(item.feedback_count > 0 for item in state.edge_trust) / count,
    )


__all__ = (
    "AdaptiveTrustMetrics",
    "measure_adaptive_trust",
)
