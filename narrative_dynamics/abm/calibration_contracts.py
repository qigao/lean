from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from narrative_dynamics.abm.autonomy_contracts import TruthObservation
from narrative_dynamics.abm.contracts import (
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.abm.lifecycle_contracts import PopulationLifecycleEvent
from narrative_dynamics.contracts import stable_content_hash


class CalibrationSplit(str, Enum):
    TRAIN = "train"
    HOLDOUT = "holdout"


_SNAPSHOT_FIELDS = (
    "active_share",
    "mean_active_belief",
    "mean_trust",
    "learned_edge_rate",
    "effective_active_edge_rate",
    "rewired_edge_rate",
    "verification_rate",
    "active_sharing_rate",
    "transition_rate",
    "role_entropy",
)


@dataclass(frozen=True)
class ObservedABMSnapshot:
    """One empirical or predicted macro observation at a positive round."""

    round_index: int
    active_share: float
    mean_active_belief: float
    mean_trust: float
    learned_edge_rate: float
    effective_active_edge_rate: float
    rewired_edge_rate: float
    verification_rate: float
    active_sharing_rate: float
    transition_rate: float
    role_entropy: float

    def __post_init__(self) -> None:
        round_index = _nonnegative_integer(
            self.round_index,
            label="observed ABM snapshot round index",
        )
        if round_index == 0:
            raise ValueError("observed ABM snapshot round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        for field_name in _SNAPSHOT_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"observed ABM snapshot {field_name}",
                ),
            )

    @property
    def metric_items(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in _SNAPSHOT_FIELDS)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "round_index": self.round_index,
            **dict(self.metric_items),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _event_key(value: PopulationLifecycleEvent) -> str:
    return value.agent_id


def _truth_key(value: TruthObservation) -> tuple[str, str, str]:
    return value.edge.identity


@dataclass(frozen=True)
class EmpiricalABMCase:
    """Frozen initial conditions, schedules, and observed trajectory."""

    case_id: str
    split: CalibrationSplit
    initial_beliefs: tuple[tuple[str, float], ...]
    environment_event_schedule: tuple[tuple[PopulationLifecycleEvent, ...], ...]
    truth_observation_schedule: tuple[tuple[TruthObservation, ...], ...]
    observations: tuple[ObservedABMSnapshot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _text(self.case_id, label="empirical ABM case id"),
        )
        if not isinstance(self.split, CalibrationSplit):
            raise TypeError("empirical ABM case split must be CalibrationSplit")
        if not isinstance(self.initial_beliefs, tuple):
            raise TypeError("empirical ABM initial beliefs must be a tuple")
        beliefs: list[tuple[str, float]] = []
        for item in self.initial_beliefs:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("empirical ABM beliefs must contain agent/value pairs")
            agent_id, belief = item
            beliefs.append(
                (
                    _text(agent_id, label="empirical ABM belief agent id"),
                    _probability(belief, label="empirical ABM initial belief"),
                )
            )
        belief_ids = tuple(item[0] for item in beliefs)
        if len(set(belief_ids)) != len(belief_ids):
            raise ValueError("empirical ABM belief agent ids must be unique")
        object.__setattr__(self, "initial_beliefs", tuple(sorted(beliefs)))
        schedules = (
            (
                "environment_event_schedule",
                self.environment_event_schedule,
                PopulationLifecycleEvent,
                _event_key,
            ),
            (
                "truth_observation_schedule",
                self.truth_observation_schedule,
                TruthObservation,
                _truth_key,
            ),
        )
        for field_name, schedule, value_type, key in schedules:
            if not isinstance(schedule, tuple):
                raise TypeError(f"empirical ABM {field_name} must be a tuple")
            canonical_rounds = []
            for values in schedule:
                if not isinstance(values, tuple):
                    raise TypeError(
                        f"empirical ABM {field_name} rounds must be tuples"
                    )
                if any(not isinstance(item, value_type) for item in values):
                    raise TypeError(
                        f"empirical ABM {field_name} has invalid values"
                    )
                identities = tuple(key(item) for item in values)
                if len(set(identities)) != len(identities):
                    raise ValueError(
                        f"empirical ABM {field_name} round identities must be unique"
                    )
                canonical_rounds.append(tuple(sorted(values, key=key)))
            object.__setattr__(self, field_name, tuple(canonical_rounds))
        if (
            len(self.environment_event_schedule)
            != len(self.truth_observation_schedule)
        ):
            raise ValueError("empirical ABM case schedules must have equal lengths")
        if not self.environment_event_schedule:
            raise ValueError("empirical ABM case schedules must be nonempty")
        if not isinstance(self.observations, tuple):
            raise TypeError("empirical ABM observations must be a tuple")
        if any(not isinstance(item, ObservedABMSnapshot) for item in self.observations):
            raise TypeError(
                "empirical ABM observations must contain ObservedABMSnapshot values"
            )
        if len(self.observations) != len(self.environment_event_schedule):
            raise ValueError("empirical ABM observations must cover every schedule round")
        expected_rounds = tuple(range(1, len(self.observations) + 1))
        if tuple(item.round_index for item in self.observations) != expected_rounds:
            raise ValueError("empirical ABM observations must cover consecutive rounds")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "split": self.split.value,
            "initial_beliefs": self.initial_beliefs,
            "environment_event_schedule": [
                [item.to_dict() for item in values]
                for values in self.environment_event_schedule
            ],
            "truth_observation_schedule": [
                [item.to_dict() for item in values]
                for values in self.truth_observation_schedule
            ],
            "observations": [item.to_dict() for item in self.observations],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class EmpiricalABMDataset:
    dataset_id: str
    version: str
    cases: tuple[EmpiricalABMCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_id",
            _text(self.dataset_id, label="empirical ABM dataset id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="empirical ABM dataset version"),
        )
        if not isinstance(self.cases, tuple):
            raise TypeError("empirical ABM dataset cases must be a tuple")
        cases = tuple(self.cases)
        if any(not isinstance(item, EmpiricalABMCase) for item in cases):
            raise TypeError("empirical ABM dataset has invalid cases")
        case_ids = tuple(item.case_id for item in cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("empirical ABM dataset case ids must be unique")
        splits = {item.split for item in cases}
        if splits != {CalibrationSplit.TRAIN, CalibrationSplit.HOLDOUT}:
            raise ValueError("empirical ABM dataset requires TRAIN and HOLDOUT cases")
        object.__setattr__(
            self,
            "cases",
            tuple(sorted(cases, key=lambda item: item.case_id)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "cases": [item.to_dict() for item in self.cases],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ABMCalibrationCandidate:
    learning_rate: float
    dissolution_similarity: float
    formation_similarity: float

    def __post_init__(self) -> None:
        for field_name in (
            "learning_rate",
            "dissolution_similarity",
            "formation_similarity",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"ABM calibration candidate {field_name}",
                ),
            )
        if self.dissolution_similarity >= self.formation_similarity:
            raise ValueError(
                "candidate dissolution similarity must be strictly below formation"
            )

    @property
    def canonical_tuple(self) -> tuple[float, float, float]:
        return (
            self.learning_rate,
            self.dissolution_similarity,
            self.formation_similarity,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "dissolution_similarity": self.dissolution_similarity,
            "formation_similarity": self.formation_similarity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _nonnegative_weight(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ABMCalibrationWeights:
    active_share: float
    mean_active_belief: float
    mean_trust: float
    learned_edge_rate: float
    effective_active_edge_rate: float
    rewired_edge_rate: float
    verification_rate: float
    active_sharing_rate: float
    transition_rate: float
    role_entropy: float

    def __post_init__(self) -> None:
        for field_name in _SNAPSHOT_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _nonnegative_weight(
                    getattr(self, field_name),
                    label=f"ABM calibration weight {field_name}",
                ),
            )
        if not any(value > 0.0 for _, value in self.metric_items):
            raise ValueError("ABM calibration requires at least one positive weight")

    @classmethod
    def uniform(cls) -> ABMCalibrationWeights:
        return cls(*([1.0] * len(_SNAPSHOT_FIELDS)))

    @property
    def metric_items(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in _SNAPSHOT_FIELDS)

    def to_dict(self) -> dict[str, float]:
        return dict(self.metric_items)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = (
    "CalibrationSplit",
    "ObservedABMSnapshot",
    "EmpiricalABMCase",
    "EmpiricalABMDataset",
    "ABMCalibrationCandidate",
    "ABMCalibrationWeights",
)
