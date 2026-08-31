from __future__ import annotations

from dataclasses import dataclass
import math

from narrative_dynamics.abm.contracts import (
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.abm.lifecycle_contracts import LifecycleStatus
from narrative_dynamics.abm.role_contracts import (
    DynamicRoleModel,
    DynamicRolePopulationState,
)
from narrative_dynamics.abm.roles import _validate_dynamic_role_model_state
from narrative_dynamics.contracts import stable_content_hash


def _nonnegative_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class RolePopulationCount:
    """Current active count and population share for one configured role."""

    role: str
    count: int
    share: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _text(self.role, label="role metric role"))
        object.__setattr__(
            self,
            "count",
            _nonnegative_integer(self.count, label="role metric count"),
        )
        object.__setattr__(
            self,
            "share",
            _probability(self.share, label="role metric share"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "count": self.count, "share": self.share}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RoleDynamicsMetrics:
    """Role composition, diversity, transitions, and active tenure."""

    active_population: int
    active_role_counts: tuple[RolePopulationCount, ...]
    role_entropy: float
    decision_count: int
    transition_count: int
    transition_rate: float
    active_rounds_in_role: int
    mean_active_rounds_in_role: float

    def __post_init__(self) -> None:
        for field_name in (
            "active_population",
            "decision_count",
            "transition_count",
            "active_rounds_in_role",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"role dynamics metric {field_name}",
                ),
            )
        if self.transition_count > self.decision_count:
            raise ValueError("role transition count cannot exceed decisions")
        if not isinstance(self.active_role_counts, tuple):
            raise TypeError("active role counts must be a tuple")
        counts = tuple(self.active_role_counts)
        if not counts:
            raise ValueError("active role counts must include configured roles")
        if any(not isinstance(item, RolePopulationCount) for item in counts):
            raise TypeError(
                "active role counts must contain RolePopulationCount values"
            )
        roles = tuple(item.role for item in counts)
        if len(set(roles)) != len(roles):
            raise ValueError("active role count roles must be unique")
        object.__setattr__(
            self,
            "active_role_counts",
            tuple(sorted(counts, key=lambda item: item.role)),
        )
        if sum(item.count for item in counts) != self.active_population:
            raise ValueError("active role counts must sum to active population")
        for item in counts:
            expected_share = (
                item.count / self.active_population
                if self.active_population
                else 0.0
            )
            if not math.isclose(
                item.share,
                expected_share,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("active role shares must match counts")
        object.__setattr__(
            self,
            "role_entropy",
            _probability(self.role_entropy, label="role dynamics entropy"),
        )
        positive_shares = tuple(item.share for item in counts if item.share > 0.0)
        expected_entropy = (
            -sum(item * math.log(item) for item in positive_shares)
            / math.log(len(counts))
            if self.active_population and len(counts) > 1
            else 0.0
        )
        if not math.isclose(
            self.role_entropy,
            expected_entropy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("role entropy must match active role shares")
        object.__setattr__(
            self,
            "transition_rate",
            _probability(
                self.transition_rate,
                label="role dynamics transition rate",
            ),
        )
        expected_transition_rate = (
            self.transition_count / self.decision_count
            if self.decision_count
            else 0.0
        )
        if not math.isclose(
            self.transition_rate,
            expected_transition_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("role transition rate must match counts")
        object.__setattr__(
            self,
            "mean_active_rounds_in_role",
            _nonnegative_number(
                self.mean_active_rounds_in_role,
                label="mean active rounds in role",
            ),
        )
        expected_mean = (
            self.active_rounds_in_role / self.active_population
            if self.active_population
            else 0.0
        )
        if not math.isclose(
            self.mean_active_rounds_in_role,
            expected_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("mean active role tenure must match counts")

    def to_dict(self) -> dict[str, object]:
        return {
            "active_population": self.active_population,
            "active_role_counts": [
                item.to_dict() for item in self.active_role_counts
            ],
            "role_entropy": self.role_entropy,
            "decision_count": self.decision_count,
            "transition_count": self.transition_count,
            "transition_rate": self.transition_rate,
            "active_rounds_in_role": self.active_rounds_in_role,
            "mean_active_rounds_in_role": self.mean_active_rounds_in_role,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def measure_role_dynamics(
    model: DynamicRoleModel,
    state: DynamicRolePopulationState,
) -> RoleDynamicsMetrics:
    """Measure current role emergence and cumulative role changes."""

    _validate_dynamic_role_model_state(model, state)
    active_ids = {
        item.agent_id
        for item in state.autonomy_state.evolving_state.members
        if item.status is LifecycleStatus.ACTIVE
    }
    active_population = len(active_ids)
    role_by_id = {item.agent_id: item for item in state.agents}
    configured_roles = tuple(
        sorted(item.role for item in model.autonomy_model.role_policies)
    )
    role_counts = tuple(
        RolePopulationCount(
            role,
            sum(
                agent_id in active_ids and item.current_role == role
                for agent_id, item in role_by_id.items()
            ),
            (
                sum(
                    agent_id in active_ids and item.current_role == role
                    for agent_id, item in role_by_id.items()
                )
                / active_population
                if active_population
                else 0.0
            ),
        )
        for role in configured_roles
    )
    positive_shares = tuple(item.share for item in role_counts if item.share > 0.0)
    entropy = (
        -sum(item * math.log(item) for item in positive_shares)
        / math.log(len(configured_roles))
        if active_population and len(configured_roles) > 1
        else 0.0
    )
    decision_count = sum(
        item.decision_count for item in state.autonomy_state.agents
    )
    transition_count = sum(item.transition_count for item in state.agents)
    active_tenure = sum(
        item.rounds_in_role
        for item in state.agents
        if item.agent_id in active_ids
    )
    return RoleDynamicsMetrics(
        active_population,
        role_counts,
        entropy,
        decision_count,
        transition_count,
        transition_count / decision_count if decision_count else 0.0,
        active_tenure,
        active_tenure / active_population if active_population else 0.0,
    )


__all__ = (
    "RolePopulationCount",
    "RoleDynamicsMetrics",
    "measure_role_dynamics",
)
