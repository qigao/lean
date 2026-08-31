from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from narrative_dynamics.abm.contracts import (
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel,
    EvolvingPopulationState,
    initialize_evolving_population,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.lifecycle_contracts import LifecycleStatus
from narrative_dynamics.contracts import stable_content_hash


class SharingDecision(str, Enum):
    """Autonomous outgoing-information choice for the following round."""

    SHARE = "share"
    SILENT = "silent"


@dataclass(frozen=True)
class RoleDecisionPolicy:
    """Deterministic autonomy policy shared by agents with one role."""

    role: str
    sharing_enabled: bool
    share_belief_threshold: float
    verify_trust_threshold: float
    exit_belief_threshold: float | None
    initial_verification_budget: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _text(self.role, label="autonomy policy role"))
        if not isinstance(self.sharing_enabled, bool):
            raise TypeError("autonomy policy sharing enabled must be boolean")
        object.__setattr__(
            self,
            "share_belief_threshold",
            _probability(
                self.share_belief_threshold,
                label="autonomy share belief threshold",
            ),
        )
        object.__setattr__(
            self,
            "verify_trust_threshold",
            _probability(
                self.verify_trust_threshold,
                label="autonomy verify trust threshold",
            ),
        )
        if self.exit_belief_threshold is not None:
            object.__setattr__(
                self,
                "exit_belief_threshold",
                _probability(
                    self.exit_belief_threshold,
                    label="autonomy exit belief threshold",
                ),
            )
        object.__setattr__(
            self,
            "initial_verification_budget",
            _nonnegative_integer(
                self.initial_verification_budget,
                label="autonomy initial verification budget",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "sharing_enabled": self.sharing_enabled,
            "share_belief_threshold": self.share_belief_threshold,
            "verify_trust_threshold": self.verify_trust_threshold,
            "exit_belief_threshold": self.exit_belief_threshold,
            "initial_verification_budget": self.initial_verification_budget,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class TruthObservation:
    """Environment truth supplied for an agent-selected verification edge."""

    edge: EdgeSelector
    observed_truth: float

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeSelector):
            raise TypeError("autonomy truth observation requires EdgeSelector")
        object.__setattr__(
            self,
            "observed_truth",
            _probability(
                self.observed_truth,
                label="autonomy truth observation",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "edge": self.edge.to_dict(),
            "observed_truth": self.observed_truth,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AgentActionIntent:
    """Auditable local decision emitted after one propagation round."""

    round_index: int
    agent_id: str
    role: str
    observed_belief: float
    incoming_transmission_count: int
    mean_incoming_trust: float | None
    sharing_decision: SharingDecision
    verification_edge: EdgeSelector | None
    exit_requested: bool
    verification_budget_before: int
    verification_budget_after: int

    def __post_init__(self) -> None:
        round_index = _nonnegative_integer(
            self.round_index,
            label="agent action intent round index",
        )
        if round_index == 0:
            raise ValueError("agent action intent round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="agent action intent agent id"),
        )
        object.__setattr__(self, "role", _text(self.role, label="agent action role"))
        object.__setattr__(
            self,
            "observed_belief",
            _probability(self.observed_belief, label="agent action observed belief"),
        )
        incoming_count = _nonnegative_integer(
            self.incoming_transmission_count,
            label="agent action incoming transmission count",
        )
        object.__setattr__(self, "incoming_transmission_count", incoming_count)
        if incoming_count == 0:
            if self.mean_incoming_trust is not None:
                raise ValueError(
                    "mean incoming trust must be None without incoming transmissions"
                )
        else:
            if self.mean_incoming_trust is None:
                raise ValueError(
                    "mean incoming trust is required with incoming transmissions"
                )
            object.__setattr__(
                self,
                "mean_incoming_trust",
                _probability(
                    self.mean_incoming_trust,
                    label="agent action mean incoming trust",
                ),
            )
        if not isinstance(self.sharing_decision, SharingDecision):
            raise TypeError("agent action sharing decision must be SharingDecision")
        if self.verification_edge is not None and not isinstance(
            self.verification_edge,
            EdgeSelector,
        ):
            raise TypeError("agent action verification edge must be EdgeSelector or None")
        if self.verification_edge is not None and incoming_count == 0:
            raise ValueError("agent cannot verify without an incoming transmission")
        if not isinstance(self.exit_requested, bool):
            raise TypeError("agent action exit requested must be boolean")
        before = _nonnegative_integer(
            self.verification_budget_before,
            label="agent action verification budget before",
        )
        after = _nonnegative_integer(
            self.verification_budget_after,
            label="agent action verification budget after",
        )
        object.__setattr__(self, "verification_budget_before", before)
        object.__setattr__(self, "verification_budget_after", after)
        expected_after = before - int(self.verification_edge is not None)
        if expected_after < 0 or after != expected_after:
            raise ValueError("agent verification budget delta must match selection")
        if self.exit_requested and self.sharing_decision is not SharingDecision.SILENT:
            raise ValueError("exiting agent must be silent")

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "role": self.role,
            "observed_belief": self.observed_belief,
            "incoming_transmission_count": self.incoming_transmission_count,
            "mean_incoming_trust": self.mean_incoming_trust,
            "sharing_decision": self.sharing_decision.value,
            "verification_edge": (
                None if self.verification_edge is None else self.verification_edge.to_dict()
            ),
            "exit_requested": self.exit_requested,
            "verification_budget_before": self.verification_budget_before,
            "verification_budget_after": self.verification_budget_after,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AutonomousAgentState:
    """Persistent autonomous choice, resources, and cumulative behavior."""

    agent_id: str
    sharing: bool
    remaining_verification_budget: int
    verification_count: int
    decision_count: int
    silent_decision_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="autonomous state agent id"),
        )
        if not isinstance(self.sharing, bool):
            raise TypeError("autonomous agent sharing must be boolean")
        for field_name in (
            "remaining_verification_budget",
            "verification_count",
            "decision_count",
            "silent_decision_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    label=f"autonomous agent {field_name}",
                ),
            )
        if self.verification_count > self.decision_count:
            raise ValueError("autonomous verification count cannot exceed decisions")
        if self.silent_decision_count > self.decision_count:
            raise ValueError("autonomous silent count cannot exceed decisions")

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "sharing": self.sharing,
            "remaining_verification_budget": self.remaining_verification_budget,
            "verification_count": self.verification_count,
            "decision_count": self.decision_count,
            "silent_decision_count": self.silent_decision_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AutonomousNetworkModel:
    """Unified evolving network plus a complete role decision-policy catalog."""

    model_id: str
    version: str
    evolving_model: EvolvingNetworkModel
    role_policies: tuple[RoleDecisionPolicy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="autonomous network model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="autonomous network model version"),
        )
        if not isinstance(self.evolving_model, EvolvingNetworkModel):
            raise TypeError("autonomous network requires EvolvingNetworkModel")
        if not isinstance(self.role_policies, tuple):
            raise TypeError("autonomous role policies must be a tuple")
        policies = tuple(self.role_policies)
        if not policies:
            raise ValueError("autonomous network requires role policies")
        if any(not isinstance(item, RoleDecisionPolicy) for item in policies):
            raise TypeError("autonomous policies must contain RoleDecisionPolicy values")
        roles = tuple(item.role for item in policies)
        if len(set(roles)) != len(roles):
            raise ValueError("autonomous policy roles must be unique")
        expected_roles = {item.role for item in self.evolving_model.base_model.agents}
        if set(roles) != expected_roles:
            raise ValueError("autonomous policies must cover the exact role catalog")
        object.__setattr__(
            self,
            "role_policies",
            tuple(sorted(policies, key=lambda item: item.role)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "evolving_model": self.evolving_model.to_dict(),
            "role_policies": [item.to_dict() for item in self.role_policies],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AutonomousPopulationState:
    """Autonomy resources bound to one exact embedded evolving snapshot."""

    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    evolving_state: EvolvingPopulationState
    agents: tuple[AutonomousAgentState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="autonomous population model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="autonomous population model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="autonomous population round index",
        )
        object.__setattr__(self, "round_index", round_index)
        if round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("autonomous population round zero cannot have a parent")
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(
                    self.parent_state_hash,
                    label="autonomous population parent state hash",
                ),
            )
        if not isinstance(self.evolving_state, EvolvingPopulationState):
            raise TypeError("autonomous population requires EvolvingPopulationState")
        if self.evolving_state.round_index != round_index:
            raise ValueError("autonomous and evolving round index must match")
        if not isinstance(self.agents, tuple):
            raise TypeError("autonomous population agents must be a tuple")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("autonomous population requires agent resources")
        if any(not isinstance(item, AutonomousAgentState) for item in agents):
            raise TypeError(
                "autonomous population agents must contain AutonomousAgentState values"
            )
        agent_ids = tuple(item.agent_id for item in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("autonomous population agent ids must be unique")
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
            "evolving_state": self.evolving_state.to_dict(),
            "agents": [item.to_dict() for item in self.agents],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_autonomous_population(
    model: AutonomousNetworkModel,
    *,
    beliefs: Mapping[str, float] | None = None,
) -> AutonomousPopulationState:
    """Initialize embedded evolution plus role-derived budgets and choices."""

    if not isinstance(model, AutonomousNetworkModel):
        raise TypeError("autonomy initialization requires AutonomousNetworkModel")
    evolving_state = initialize_evolving_population(
        model.evolving_model,
        beliefs=beliefs,
    )
    policies = {item.role: item for item in model.role_policies}
    profiles = {
        item.agent_id: item for item in model.evolving_model.base_model.agents
    }
    members = {item.agent_id: item for item in evolving_state.members}
    resources: list[AutonomousAgentState] = []
    for agent_id in model.evolving_model.base_model.network.agent_ids:
        member = members[agent_id]
        profile = profiles[agent_id]
        policy = policies[profile.role]
        sharing = (
            member.status is LifecycleStatus.ACTIVE
            and policy.sharing_enabled
            and member.belief >= policy.share_belief_threshold
            and member.belief >= profile.broadcast_threshold
        )
        resources.append(
            AutonomousAgentState(
                agent_id,
                sharing,
                policy.initial_verification_budget,
                0,
                0,
                0,
            )
        )
    return AutonomousPopulationState(
        model.model_id,
        model.content_hash,
        0,
        None,
        evolving_state,
        tuple(resources),
    )


__all__ = (
    "SharingDecision",
    "RoleDecisionPolicy",
    "TruthObservation",
    "AgentActionIntent",
    "AutonomousAgentState",
    "AutonomousNetworkModel",
    "AutonomousPopulationState",
    "initialize_autonomous_population",
)
