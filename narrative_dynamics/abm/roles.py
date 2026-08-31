from __future__ import annotations

from dataclasses import dataclass

from narrative_dynamics.abm.autonomy import (
    AutonomousRoundResult,
    _simulate_autonomous_round_with_roles,
    _validate_autonomous_model_state,
)
from narrative_dynamics.abm.autonomy_contracts import TruthObservation
from narrative_dynamics.abm.contracts import _hash, _nonnegative_integer, _text
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleStatus,
    PopulationLifecycleEvent,
)
from narrative_dynamics.abm.role_contracts import (
    DynamicRoleAgentState,
    DynamicRoleModel,
    DynamicRolePopulationState,
    RoleTransitionRecord,
)
from narrative_dynamics.contracts import stable_content_hash


def _validate_dynamic_role_model_state(
    model: DynamicRoleModel,
    state: DynamicRolePopulationState,
) -> None:
    if not isinstance(model, DynamicRoleModel):
        raise TypeError("dynamic role simulation requires DynamicRoleModel")
    if not isinstance(state, DynamicRolePopulationState):
        raise TypeError("dynamic role simulation requires DynamicRolePopulationState")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("dynamic role state does not bind exact model identity")
    if state.round_index != state.autonomy_state.round_index:
        raise ValueError("dynamic role and autonomy round index must match")
    _validate_autonomous_model_state(model.autonomy_model, state.autonomy_state)
    catalog_ids = set(
        model.autonomy_model.evolving_model.base_model.network.agent_ids
    )
    role_by_id = {item.agent_id: item for item in state.agents}
    if set(role_by_id) != catalog_ids:
        raise ValueError("dynamic role state must cover exact agent catalog")
    configured_roles = {item.role for item in model.autonomy_model.role_policies}
    if any(item.current_role not in configured_roles for item in state.agents):
        raise ValueError("dynamic role state must use configured roles")
    resources = {item.agent_id: item for item in state.autonomy_state.agents}
    for item in state.agents:
        decisions = resources[item.agent_id].decision_count
        if item.transition_count > decisions:
            raise ValueError("dynamic role transition history exceeds decisions")
        if item.rounds_in_role > decisions:
            raise ValueError("dynamic role tenure exceeds decisions")
    if state.round_index == 0:
        profiles = {
            item.agent_id: item
            for item in model.autonomy_model.evolving_model.base_model.agents
        }
        for item in state.agents:
            if item.current_role != profiles[item.agent_id].role:
                raise ValueError("round-zero dynamic roles must match agent profiles")
            if item.rounds_in_role != 0 or item.transition_count != 0:
                raise ValueError("round-zero dynamic role history must be empty")


def _transition_key(value: RoleTransitionRecord) -> str:
    return value.agent_id


@dataclass(frozen=True)
class DynamicRoleRoundResult:
    """One V6 autonomy round followed by synchronous V7 role changes."""

    model_id: str
    model_hash: str
    round_index: int
    prior_state: DynamicRolePopulationState
    autonomy_result: AutonomousRoundResult
    transitions: tuple[RoleTransitionRecord, ...]
    next_state: DynamicRolePopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="dynamic role round model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="dynamic role round model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="dynamic role round index",
        )
        if round_index == 0:
            raise ValueError("dynamic role round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        if not isinstance(self.prior_state, DynamicRolePopulationState):
            raise TypeError("dynamic role round requires prior dynamic role state")
        if not isinstance(self.autonomy_result, AutonomousRoundResult):
            raise TypeError("dynamic role round requires AutonomousRoundResult")
        if not isinstance(self.next_state, DynamicRolePopulationState):
            raise TypeError("dynamic role round requires next dynamic role state")
        if not isinstance(self.transitions, tuple):
            raise TypeError("dynamic role transitions must be a tuple")
        transitions = tuple(self.transitions)
        if any(not isinstance(item, RoleTransitionRecord) for item in transitions):
            raise TypeError(
                "dynamic role transitions must contain RoleTransitionRecord values"
            )
        agent_ids = tuple(item.agent_id for item in transitions)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("dynamic role transition agent ids must be unique")
        object.__setattr__(
            self,
            "transitions",
            tuple(sorted(transitions, key=_transition_key)),
        )
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("dynamic role round states must bind exact model identity")
        if round_index != self.prior_state.round_index + 1:
            raise ValueError("dynamic role round must immediately follow prior state")
        if self.next_state.round_index != round_index:
            raise ValueError("dynamic role next state must bind exact round index")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("dynamic role next state must bind exact prior state hash")
        if self.autonomy_result.round_index != round_index:
            raise ValueError("dynamic role and autonomy result rounds must match")
        if self.autonomy_result.prior_state != self.prior_state.autonomy_state:
            raise ValueError("dynamic role round must embed exact prior autonomy state")
        if self.autonomy_result.next_state != self.next_state.autonomy_state:
            raise ValueError("dynamic role round must embed exact next autonomy state")
        prior_roles = {item.agent_id: item for item in self.prior_state.agents}
        next_roles = {item.agent_id: item for item in self.next_state.agents}
        transition_by_id = {item.agent_id: item for item in self.transitions}
        members = {
            item.agent_id: item
            for item in self.next_state.autonomy_state.evolving_state.members
        }
        resources = {
            item.agent_id: item for item in self.next_state.autonomy_state.agents
        }
        if set(prior_roles) != set(next_roles):
            raise ValueError("dynamic role round agent catalogs must match")
        for agent_id, prior in prior_roles.items():
            next_item = next_roles[agent_id]
            transition = transition_by_id.get(agent_id)
            active = members[agent_id].status is LifecycleStatus.ACTIVE
            if transition is None:
                expected_tenure = prior.rounds_in_role + int(active)
                if (
                    next_item.current_role != prior.current_role
                    or next_item.transition_count != prior.transition_count
                    or next_item.rounds_in_role != expected_tenure
                ):
                    raise ValueError("dynamic role unchanged-state history is inconsistent")
                continue
            if not active:
                raise ValueError("inactive agent cannot have a role transition")
            resource = resources[agent_id]
            member = members[agent_id]
            if (
                transition.round_index != round_index
                or transition.from_role != prior.current_role
                or transition.to_role != next_item.current_role
                or transition.completed_rounds_in_from_role
                != prior.rounds_in_role + 1
                or transition.decision_count != resource.decision_count
                or transition.verification_count != resource.verification_count
                or transition.observed_belief != member.belief
                or next_item.rounds_in_role != 0
                or next_item.transition_count != prior.transition_count + 1
            ):
                raise ValueError("dynamic role transition state history is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "prior_state_hash": self.prior_state.content_hash,
            "autonomy_result": self.autonomy_result.to_dict(),
            "transitions": [item.to_dict() for item in self.transitions],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class DynamicRoleTrajectory:
    """Validated positive chain of role-aware autonomous rounds."""

    model_id: str
    model_hash: str
    initial_state: DynamicRolePopulationState
    rounds: tuple[DynamicRoleRoundResult, ...]
    final_state: DynamicRolePopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="dynamic role trajectory model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="dynamic role trajectory model hash"),
        )
        if not isinstance(self.initial_state, DynamicRolePopulationState):
            raise TypeError("dynamic role trajectory requires initial state")
        if not isinstance(self.final_state, DynamicRolePopulationState):
            raise TypeError("dynamic role trajectory requires final state")
        if not isinstance(self.rounds, tuple):
            raise TypeError("dynamic role trajectory rounds must be a tuple")
        if not self.rounds:
            raise ValueError("dynamic role trajectory requires at least one round")
        if any(not isinstance(item, DynamicRoleRoundResult) for item in self.rounds):
            raise TypeError(
                "dynamic role trajectory rounds must contain DynamicRoleRoundResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("dynamic role trajectory states must bind exact model identity")
        current = self.initial_state
        for result in self.rounds:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError("dynamic role trajectory rounds must bind exact model identity")
            if result.prior_state != current:
                raise ValueError("dynamic role trajectory state chain is discontinuous")
            current = result.next_state
        if current != self.final_state:
            raise ValueError("dynamic role trajectory final state must equal chain tail")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "initial_state": self.initial_state.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "final_state": self.final_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def simulate_dynamic_role_round(
    model: DynamicRoleModel,
    prior_state: DynamicRolePopulationState,
    *,
    environment_events: tuple[PopulationLifecycleEvent, ...] = (),
    truth_observations: tuple[TruthObservation, ...] = (),
) -> DynamicRoleRoundResult:
    """Run V6 with current roles, then change roles for the next round."""

    _validate_dynamic_role_model_state(model, prior_state)
    role_by_agent = {
        item.agent_id: item.current_role for item in prior_state.agents
    }
    autonomy_result = _simulate_autonomous_round_with_roles(
        model.autonomy_model,
        prior_state.autonomy_state,
        environment_events=environment_events,
        truth_observations=truth_observations,
        role_by_agent=role_by_agent,
    )
    members = {
        item.agent_id: item
        for item in autonomy_result.next_state.evolving_state.members
    }
    resources = {
        item.agent_id: item for item in autonomy_result.next_state.agents
    }
    rules_by_role = {
        role: tuple(
            item for item in model.transition_rules if item.from_role == role
        )
        for role in {item.current_role for item in prior_state.agents}
    }
    transitions: list[RoleTransitionRecord] = []
    next_roles: list[DynamicRoleAgentState] = []
    for prior in prior_state.agents:
        member = members[prior.agent_id]
        if member.status is not LifecycleStatus.ACTIVE:
            next_roles.append(prior)
            continue
        completed_tenure = prior.rounds_in_role + 1
        resource = resources[prior.agent_id]
        matching = tuple(
            rule
            for rule in rules_by_role.get(prior.current_role, ())
            if rule.matches(
                belief=member.belief,
                completed_rounds_in_role=completed_tenure,
                decision_count=resource.decision_count,
                verification_count=resource.verification_count,
            )
        )
        if not matching:
            next_roles.append(
                DynamicRoleAgentState(
                    prior.agent_id,
                    prior.current_role,
                    completed_tenure,
                    prior.transition_count,
                )
            )
            continue
        rule = matching[0]
        transitions.append(
            RoleTransitionRecord(
                autonomy_result.round_index,
                prior.agent_id,
                prior.current_role,
                rule.to_role,
                rule.priority,
                member.belief,
                completed_tenure,
                resource.decision_count,
                resource.verification_count,
            )
        )
        next_roles.append(
            DynamicRoleAgentState(
                prior.agent_id,
                rule.to_role,
                0,
                prior.transition_count + 1,
            )
        )
    next_state = DynamicRolePopulationState(
        model.model_id,
        model.content_hash,
        autonomy_result.round_index,
        prior_state.content_hash,
        autonomy_result.next_state,
        tuple(next_roles),
    )
    return DynamicRoleRoundResult(
        model.model_id,
        model.content_hash,
        autonomy_result.round_index,
        prior_state,
        autonomy_result,
        tuple(transitions),
        next_state,
    )


def simulate_dynamic_role_population(
    model: DynamicRoleModel,
    initial_state: DynamicRolePopulationState,
    *,
    environment_event_schedule: tuple[tuple[PopulationLifecycleEvent, ...], ...],
    truth_observation_schedule: tuple[tuple[TruthObservation, ...], ...],
) -> DynamicRoleTrajectory:
    """Run paired environment and selected-truth schedules through V7."""

    _validate_dynamic_role_model_state(model, initial_state)
    if not isinstance(environment_event_schedule, tuple) or not isinstance(
        truth_observation_schedule,
        tuple,
    ):
        raise TypeError("dynamic role schedules must be tuples")
    if not environment_event_schedule or not truth_observation_schedule:
        raise ValueError("dynamic role schedules must be nonempty")
    if len(environment_event_schedule) != len(truth_observation_schedule):
        raise ValueError("dynamic role schedules must have equal lengths")
    if any(not isinstance(item, tuple) for item in environment_event_schedule):
        raise TypeError("dynamic role environment schedule entries must be tuples")
    if any(not isinstance(item, tuple) for item in truth_observation_schedule):
        raise TypeError("dynamic role truth schedule entries must be tuples")
    current = initial_state
    rounds: list[DynamicRoleRoundResult] = []
    for events, observations in zip(
        environment_event_schedule,
        truth_observation_schedule,
        strict=True,
    ):
        result = simulate_dynamic_role_round(
            model,
            current,
            environment_events=events,
            truth_observations=observations,
        )
        rounds.append(result)
        current = result.next_state
    return DynamicRoleTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(rounds),
        current,
    )


__all__ = (
    "DynamicRoleRoundResult",
    "DynamicRoleTrajectory",
    "simulate_dynamic_role_round",
    "simulate_dynamic_role_population",
)
