from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.abm.autonomy_contracts import (
    AutonomousNetworkModel,
    AutonomousPopulationState,
    initialize_autonomous_population,
)
from narrative_dynamics.abm.contracts import (
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.contracts import stable_content_hash


@dataclass(frozen=True)
class RoleTransitionRule:
    """Deterministic local evidence required for one directed role change."""

    from_role: str
    to_role: str
    priority: int
    minimum_belief: float | None = None
    maximum_belief: float | None = None
    minimum_rounds_in_role: int = 0
    minimum_decision_count: int = 0
    minimum_verification_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_role",
            _text(self.from_role, label="role transition source role"),
        )
        object.__setattr__(
            self,
            "to_role",
            _text(self.to_role, label="role transition destination role"),
        )
        if self.from_role == self.to_role:
            raise ValueError("role transition cannot be a self-transition")
        object.__setattr__(
            self,
            "priority",
            _nonnegative_integer(self.priority, label="role transition priority"),
        )
        if self.minimum_belief is not None:
            object.__setattr__(
                self,
                "minimum_belief",
                _probability(
                    self.minimum_belief,
                    label="role transition minimum belief",
                ),
            )
        if self.maximum_belief is not None:
            object.__setattr__(
                self,
                "maximum_belief",
                _probability(
                    self.maximum_belief,
                    label="role transition maximum belief",
                ),
            )
        if (
            self.minimum_belief is not None
            and self.maximum_belief is not None
            and self.minimum_belief >= self.maximum_belief
        ):
            raise ValueError("role transition belief interval must be nonempty")
        for field_name in (
            "minimum_rounds_in_role",
            "minimum_decision_count",
            "minimum_verification_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"role transition {field_name}",
                ),
            )
        if (
            self.minimum_belief is None
            and self.maximum_belief is None
            and self.minimum_rounds_in_role == 0
            and self.minimum_decision_count == 0
            and self.minimum_verification_count == 0
        ):
            raise ValueError("role transition requires a substantive condition")

    def matches(
        self,
        *,
        belief: float,
        completed_rounds_in_role: int,
        decision_count: int,
        verification_count: int,
    ) -> bool:
        """Return whether already-validated local evidence satisfies this rule."""

        return (
            (self.minimum_belief is None or belief >= self.minimum_belief)
            and (self.maximum_belief is None or belief < self.maximum_belief)
            and completed_rounds_in_role >= self.minimum_rounds_in_role
            and decision_count >= self.minimum_decision_count
            and verification_count >= self.minimum_verification_count
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "from_role": self.from_role,
            "to_role": self.to_role,
            "priority": self.priority,
            "minimum_belief": self.minimum_belief,
            "maximum_belief": self.maximum_belief,
            "minimum_rounds_in_role": self.minimum_rounds_in_role,
            "minimum_decision_count": self.minimum_decision_count,
            "minimum_verification_count": self.minimum_verification_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RoleTransitionRecord:
    """Auditable local evidence for one role change at a round boundary."""

    round_index: int
    agent_id: str
    from_role: str
    to_role: str
    rule_priority: int
    observed_belief: float
    completed_rounds_in_from_role: int
    decision_count: int
    verification_count: int

    def __post_init__(self) -> None:
        round_index = _nonnegative_integer(
            self.round_index,
            label="role transition record round index",
        )
        if round_index == 0:
            raise ValueError("role transition record round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="role transition record agent id"),
        )
        object.__setattr__(
            self,
            "from_role",
            _text(self.from_role, label="role transition record source role"),
        )
        object.__setattr__(
            self,
            "to_role",
            _text(self.to_role, label="role transition record destination role"),
        )
        if self.from_role == self.to_role:
            raise ValueError("role transition record requires distinct roles")
        for field_name in (
            "rule_priority",
            "completed_rounds_in_from_role",
            "decision_count",
            "verification_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"role transition record {field_name}",
                ),
            )
        if self.completed_rounds_in_from_role == 0:
            raise ValueError("role transition requires a completed source-role round")
        if self.decision_count == 0:
            raise ValueError("role transition requires an autonomous decision")
        if self.verification_count > self.decision_count:
            raise ValueError("role transition verifications cannot exceed decisions")
        object.__setattr__(
            self,
            "observed_belief",
            _probability(self.observed_belief, label="role transition observed belief"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "from_role": self.from_role,
            "to_role": self.to_role,
            "rule_priority": self.rule_priority,
            "observed_belief": self.observed_belief,
            "completed_rounds_in_from_role": self.completed_rounds_in_from_role,
            "decision_count": self.decision_count,
            "verification_count": self.verification_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class DynamicRoleAgentState:
    """Current role plus persistent active tenure and transition history."""

    agent_id: str
    current_role: str
    rounds_in_role: int
    transition_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="dynamic role agent id"),
        )
        object.__setattr__(
            self,
            "current_role",
            _text(self.current_role, label="dynamic current role"),
        )
        for field_name in ("rounds_in_role", "transition_count"):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"dynamic role {field_name}",
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "current_role": self.current_role,
            "rounds_in_role": self.rounds_in_role,
            "transition_count": self.transition_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _rule_key(rule: RoleTransitionRule) -> tuple[str, int, str]:
    return (rule.from_role, rule.priority, rule.to_role)


@dataclass(frozen=True)
class DynamicRoleModel:
    """V6 autonomous network plus deterministic role-transition rules."""

    model_id: str
    version: str
    autonomy_model: AutonomousNetworkModel
    transition_rules: tuple[RoleTransitionRule, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="dynamic role model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="dynamic role model version"),
        )
        if not isinstance(self.autonomy_model, AutonomousNetworkModel):
            raise TypeError("dynamic role model requires AutonomousNetworkModel")
        if not isinstance(self.transition_rules, tuple):
            raise TypeError("dynamic role transition rules must be a tuple")
        rules = tuple(self.transition_rules)
        if not rules:
            raise ValueError("dynamic role model requires transition rules")
        if any(not isinstance(item, RoleTransitionRule) for item in rules):
            raise TypeError(
                "dynamic role rules must contain RoleTransitionRule values"
            )
        roles = {item.role for item in self.autonomy_model.role_policies}
        if any(
            item.from_role not in roles or item.to_role not in roles for item in rules
        ):
            raise ValueError("dynamic role rules must reference configured role catalog")
        priorities = tuple((item.from_role, item.priority) for item in rules)
        if len(set(priorities)) != len(priorities):
            raise ValueError("outgoing role transition rules require unique priority")
        object.__setattr__(
            self,
            "transition_rules",
            tuple(sorted(rules, key=_rule_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "autonomy_model": self.autonomy_model.to_dict(),
            "transition_rules": [item.to_dict() for item in self.transition_rules],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class DynamicRolePopulationState:
    """Dynamic assignments bound to one exact embedded V6 state snapshot."""

    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    autonomy_state: AutonomousPopulationState
    agents: tuple[DynamicRoleAgentState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="dynamic role population model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="dynamic role population model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="dynamic role population round index",
        )
        object.__setattr__(self, "round_index", round_index)
        if round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("dynamic role round zero cannot have a parent")
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(
                    self.parent_state_hash,
                    label="dynamic role parent state hash",
                ),
            )
        if not isinstance(self.autonomy_state, AutonomousPopulationState):
            raise TypeError("dynamic role state requires AutonomousPopulationState")
        if self.autonomy_state.round_index != round_index:
            raise ValueError("dynamic role and autonomy round index must match")
        if not isinstance(self.agents, tuple):
            raise TypeError("dynamic role agents must be a tuple")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("dynamic role state requires agent roles")
        if any(not isinstance(item, DynamicRoleAgentState) for item in agents):
            raise TypeError(
                "dynamic role agents must contain DynamicRoleAgentState values"
            )
        agent_ids = tuple(item.agent_id for item in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("dynamic role agent ids must be unique")
        object.__setattr__(
            self,
            "agents",
            tuple(sorted(agents, key=lambda item: item.agent_id)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "autonomy_state": self.autonomy_state.to_dict(),
            "agents": [item.to_dict() for item in self.agents],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_dynamic_role_population(
    model: DynamicRoleModel,
    *,
    beliefs: Mapping[str, float] | None = None,
) -> DynamicRolePopulationState:
    """Initialize V6 resources and catalogued roles as a V7 state."""

    if not isinstance(model, DynamicRoleModel):
        raise TypeError("dynamic role initialization requires DynamicRoleModel")
    autonomy_state = initialize_autonomous_population(
        model.autonomy_model,
        beliefs=beliefs,
    )
    profiles = {
        item.agent_id: item
        for item in model.autonomy_model.evolving_model.base_model.agents
    }
    roles = tuple(
        DynamicRoleAgentState(agent_id, profiles[agent_id].role, 0, 0)
        for agent_id in model.autonomy_model.evolving_model.base_model.network.agent_ids
    )
    return DynamicRolePopulationState(
        model.model_id,
        model.content_hash,
        0,
        None,
        autonomy_state,
        roles,
    )


__all__ = (
    "RoleTransitionRule",
    "RoleTransitionRecord",
    "DynamicRoleAgentState",
    "DynamicRoleModel",
    "DynamicRolePopulationState",
    "initialize_dynamic_role_population",
)
