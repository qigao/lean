from __future__ import annotations

from dataclasses import dataclass

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    PopulationState,
    _probability,
)
from narrative_dynamics.abm.simulation import _validate_model_state
from narrative_dynamics.contracts import stable_content_hash


def _unit_interval(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class EmergenceMetrics:
    """Canonical macro projection of one population snapshot."""

    mean_belief: float
    adoption_rate: float
    broadcasting_rate: float
    informed_rate: float
    consensus: float
    polarization: float

    def __post_init__(self) -> None:
        for field_name in (
            "mean_belief",
            "adoption_rate",
            "broadcasting_rate",
            "informed_rate",
            "consensus",
            "polarization",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"emergence metric {field_name}",
                ),
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_belief": self.mean_belief,
            "adoption_rate": self.adoption_rate,
            "broadcasting_rate": self.broadcasting_rate,
            "informed_rate": self.informed_rate,
            "consensus": self.consensus,
            "polarization": self.polarization,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def measure_emergence(
    model: NetworkABMModel,
    state: PopulationState,
) -> EmergenceMetrics:
    """Project persistent agent states into bounded group-level observables."""

    _validate_model_state(model, state)
    profiles = {item.agent_id: item for item in model.agents}
    beliefs = tuple(item.belief for item in state.agents)
    count = len(beliefs)
    mean = sum(beliefs) / count
    adoption_rate = sum(
        item.belief >= profiles[item.agent_id].adoption_threshold
        for item in state.agents
    ) / count
    broadcasting_rate = sum(item.broadcasting for item in state.agents) / count
    informed_rate = sum(item.exposure_count > 0 for item in state.agents) / count
    consensus = 1.0 - (max(beliefs) - min(beliefs))
    polarization = 4.0 * sum((value - mean) ** 2 for value in beliefs) / count
    return EmergenceMetrics(
        _unit_interval(mean),
        _unit_interval(adoption_rate),
        _unit_interval(broadcasting_rate),
        _unit_interval(informed_rate),
        _unit_interval(consensus),
        _unit_interval(polarization),
    )


__all__ = (
    "EmergenceMetrics",
    "measure_emergence",
)
