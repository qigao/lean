from __future__ import annotations

from dataclasses import dataclass, replace

from narrative_dynamics.abm.adaptive_contracts import (
    AdaptivePopulationState,
    AdaptiveTrustModel,
    EdgeTrustState,
    TruthFeedback,
)
from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    PopulationState,
    SocialNetwork,
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.simulation import (
    InformationTransmission,
    simulate_round,
)
from narrative_dynamics.contracts import stable_content_hash


def _selector_key(value: EdgeSelector) -> tuple[str, str, str]:
    return value.identity


def _feedback_key(value: TruthFeedback) -> tuple[str, str, str]:
    return value.edge.identity


def _update_key(value: EdgeTrustUpdate) -> tuple[str, str, str]:
    return value.edge.identity


def _transmission_key(
    value: InformationTransmission,
) -> tuple[str, str, str]:
    return (
        value.source_agent_id,
        value.target_agent_id,
        value.relation_type,
    )


@dataclass(frozen=True)
class EdgeTrustUpdate:
    """Auditable prediction-error update for one transmitted edge."""

    edge: EdgeSelector
    prior_trust: float
    signal: float
    observed_truth: float
    accuracy: float
    next_trust: float
    feedback_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeSelector):
            raise TypeError("edge trust update requires EdgeSelector")
        for field_name in (
            "prior_trust",
            "signal",
            "observed_truth",
            "accuracy",
            "next_trust",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(
                    getattr(self, field_name),
                    label=f"edge trust update {field_name}",
                ),
            )
        count = _nonnegative_integer(
            self.feedback_count,
            label="edge trust update feedback count",
        )
        if count == 0:
            raise ValueError("edge trust update feedback count must be positive")
        object.__setattr__(self, "feedback_count", count)
        expected_accuracy = 1.0 - abs(self.signal - self.observed_truth)
        if self.accuracy != expected_accuracy:
            raise ValueError("edge trust update accuracy does not match signal and truth")

    def to_dict(self) -> dict[str, object]:
        return {
            "edge": self.edge.to_dict(),
            "prior_trust": self.prior_trust,
            "signal": self.signal,
            "observed_truth": self.observed_truth,
            "accuracy": self.accuracy,
            "next_trust": self.next_trust,
            "feedback_count": self.feedback_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AdaptiveRoundResult:
    """One synchronous belief round followed by edge-local trust learning."""

    model_id: str
    model_hash: str
    round_index: int
    prior_state: AdaptivePopulationState
    transmissions: tuple[InformationTransmission, ...]
    feedback: tuple[TruthFeedback, ...]
    trust_updates: tuple[EdgeTrustUpdate, ...]
    next_state: AdaptivePopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="adaptive round model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="adaptive round model hash"),
        )
        round_index = _nonnegative_integer(
            self.round_index,
            label="adaptive round index",
        )
        if round_index == 0:
            raise ValueError("adaptive round index must be positive")
        object.__setattr__(self, "round_index", round_index)
        if not isinstance(self.prior_state, AdaptivePopulationState):
            raise TypeError("adaptive round prior state must be AdaptivePopulationState")
        if not isinstance(self.next_state, AdaptivePopulationState):
            raise TypeError("adaptive round next state must be AdaptivePopulationState")
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("adaptive round states must bind exact model identity")
        if round_index != self.prior_state.round_index + 1:
            raise ValueError("adaptive round must immediately follow prior state")
        if self.next_state.round_index != round_index:
            raise ValueError("adaptive next state must bind exact round index")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("adaptive next state must bind exact prior state hash")

        for field_name, values, value_type, key in (
            ("transmissions", self.transmissions, InformationTransmission, _transmission_key),
            ("feedback", self.feedback, TruthFeedback, _feedback_key),
            ("trust_updates", self.trust_updates, EdgeTrustUpdate, _update_key),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"adaptive round {field_name} must be a tuple")
            if any(not isinstance(item, value_type) for item in values):
                raise TypeError(
                    f"adaptive round {field_name} must contain {value_type.__name__} values"
                )
            identities = tuple(key(item) for item in values)
            if len(set(identities)) != len(identities):
                raise ValueError(f"adaptive round {field_name} identities must be unique")
            object.__setattr__(self, field_name, tuple(sorted(values, key=key)))
        if any(item.round_index != round_index for item in self.transmissions):
            raise ValueError("adaptive transmissions must bind exact round index")
        if {_feedback_key(item) for item in self.feedback} != {
            _update_key(item) for item in self.trust_updates
        }:
            raise ValueError("adaptive trust updates must cover exact feedback identities")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "prior_state_hash": self.prior_state.content_hash,
            "transmissions": [item.to_dict() for item in self.transmissions],
            "feedback": [item.to_dict() for item in self.feedback],
            "trust_updates": [item.to_dict() for item in self.trust_updates],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AdaptiveTrajectory:
    """Validated multi-round adaptive belief and trust state chain."""

    model_id: str
    model_hash: str
    initial_state: AdaptivePopulationState
    rounds: tuple[AdaptiveRoundResult, ...]
    final_state: AdaptivePopulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="adaptive trajectory model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="adaptive trajectory model hash"),
        )
        if not isinstance(self.initial_state, AdaptivePopulationState):
            raise TypeError("adaptive trajectory initial state must be AdaptivePopulationState")
        if not isinstance(self.final_state, AdaptivePopulationState):
            raise TypeError("adaptive trajectory final state must be AdaptivePopulationState")
        if not isinstance(self.rounds, tuple):
            raise TypeError("adaptive trajectory rounds must be a tuple")
        rounds = tuple(self.rounds)
        if not rounds:
            raise ValueError("adaptive trajectory requires at least one round")
        if any(not isinstance(item, AdaptiveRoundResult) for item in rounds):
            raise TypeError(
                "adaptive trajectory rounds must contain AdaptiveRoundResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError("adaptive trajectory states must bind exact model identity")
        current = self.initial_state
        for result in rounds:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError("adaptive trajectory rounds must bind exact model identity")
            if result.prior_state != current:
                raise ValueError("adaptive trajectory state chain is discontinuous")
            current = result.next_state
        if current != self.final_state:
            raise ValueError("adaptive trajectory final state must equal chain tail")
        object.__setattr__(self, "rounds", rounds)

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


def _validate_adaptive_model_state(
    model: AdaptiveTrustModel,
    state: AdaptivePopulationState,
) -> None:
    if not isinstance(model, AdaptiveTrustModel):
        raise TypeError("adaptive simulation requires AdaptiveTrustModel")
    if not isinstance(state, AdaptivePopulationState):
        raise TypeError("adaptive simulation requires AdaptivePopulationState")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("adaptive state does not bind exact model identity")
    if {item.agent_id for item in state.agents} != set(
        model.base_model.network.agent_ids
    ):
        raise ValueError("adaptive state must contain the exact base model roster")
    profiles = {item.agent_id: item for item in model.base_model.agents}
    for item in state.agents:
        expected = item.belief >= profiles[item.agent_id].broadcast_threshold
        if item.broadcasting is not expected:
            raise ValueError("adaptive agent broadcasting is inconsistent with threshold")
    expected_edges = {item.identity for item in model.base_model.network.edges}
    actual_edges = {item.edge.identity for item in state.edge_trust}
    if actual_edges != expected_edges:
        raise ValueError("adaptive state must contain exact base edge trust coverage")


def population_view(
    model: AdaptiveTrustModel,
    state: AdaptivePopulationState,
) -> PopulationState:
    """Project adaptive agents to a V1-compatible population metric view."""

    _validate_adaptive_model_state(model, state)
    return PopulationState(
        model.base_model.model_id,
        model.base_model.content_hash,
        state.round_index,
        state.parent_state_hash,
        state.agents,
    )


def _effective_model_and_state(
    model: AdaptiveTrustModel,
    state: AdaptivePopulationState,
) -> tuple[NetworkABMModel, PopulationState]:
    trust = {item.edge.identity: item.trust for item in state.edge_trust}
    effective_edges = tuple(
        replace(edge, influence=edge.influence * trust[edge.identity])
        for edge in model.base_model.network.edges
    )
    effective_model = NetworkABMModel(
        f"{model.base_model.model_id}::adaptive::{state.content_hash}",
        model.base_model.version,
        model.base_model.agents,
        SocialNetwork(model.base_model.network.agent_ids, effective_edges),
    )
    effective_state = PopulationState(
        effective_model.model_id,
        effective_model.content_hash,
        state.round_index,
        state.parent_state_hash,
        state.agents,
    )
    return effective_model, effective_state


def simulate_adaptive_round(
    model: AdaptiveTrustModel,
    prior_state: AdaptivePopulationState,
    *,
    feedback: tuple[TruthFeedback, ...] = (),
) -> AdaptiveRoundResult:
    """Propagate with prior trust, then learn from same-round truth feedback."""

    _validate_adaptive_model_state(model, prior_state)
    if not isinstance(feedback, tuple):
        raise TypeError("adaptive round feedback must be a tuple")
    if any(not isinstance(item, TruthFeedback) for item in feedback):
        raise TypeError("adaptive round feedback must contain TruthFeedback values")
    feedback_ids = tuple(_feedback_key(item) for item in feedback)
    if len(set(feedback_ids)) != len(feedback_ids):
        raise ValueError("adaptive round feedback edge identities must be unique")
    canonical_feedback = tuple(sorted(feedback, key=_feedback_key))

    effective_model, effective_state = _effective_model_and_state(
        model,
        prior_state,
    )
    propagated = simulate_round(effective_model, effective_state)
    transmissions = propagated.transmissions
    by_edge = {_transmission_key(item): item for item in transmissions}
    if not set(feedback_ids).issubset(by_edge):
        raise ValueError("adaptive truth feedback targets an edge that did not transmit")

    feedback_by_edge = {_feedback_key(item): item for item in canonical_feedback}
    next_trust: list[EdgeTrustState] = []
    updates: list[EdgeTrustUpdate] = []
    for old in prior_state.edge_trust:
        item = feedback_by_edge.get(old.edge.identity)
        if item is None:
            next_trust.append(old)
            continue
        transmission = by_edge[old.edge.identity]
        accuracy = 1.0 - abs(transmission.signal - item.observed_truth)
        learned = old.trust + model.learning_rate * (accuracy - old.trust)
        learned = min(1.0, max(0.0, learned))
        count = old.feedback_count + 1
        next_trust.append(EdgeTrustState(old.edge, learned, count))
        updates.append(
            EdgeTrustUpdate(
                old.edge,
                old.trust,
                transmission.signal,
                item.observed_truth,
                accuracy,
                learned,
                count,
            )
        )

    next_state = AdaptivePopulationState(
        model.model_id,
        model.content_hash,
        prior_state.round_index + 1,
        prior_state.content_hash,
        propagated.next_state.agents,
        tuple(next_trust),
    )
    return AdaptiveRoundResult(
        model.model_id,
        model.content_hash,
        next_state.round_index,
        prior_state,
        transmissions,
        canonical_feedback,
        tuple(updates),
        next_state,
    )


def simulate_adaptive_population(
    model: AdaptiveTrustModel,
    initial_state: AdaptivePopulationState,
    *,
    feedback_schedule: tuple[tuple[TruthFeedback, ...], ...],
) -> AdaptiveTrajectory:
    """Run one adaptive round for every tuple in a nonempty feedback schedule."""

    _validate_adaptive_model_state(model, initial_state)
    if not isinstance(feedback_schedule, tuple):
        raise TypeError("adaptive feedback schedule must be a tuple")
    if not feedback_schedule:
        raise ValueError("adaptive feedback schedule must be nonempty")
    if any(not isinstance(item, tuple) for item in feedback_schedule):
        raise TypeError("adaptive feedback schedule entries must be tuples")
    current = initial_state
    results: list[AdaptiveRoundResult] = []
    for feedback in feedback_schedule:
        result = simulate_adaptive_round(model, current, feedback=feedback)
        results.append(result)
        current = result.next_state
    return AdaptiveTrajectory(
        model.model_id,
        model.content_hash,
        initial_state,
        tuple(results),
        current,
    )


__all__ = (
    "EdgeTrustUpdate",
    "AdaptiveRoundResult",
    "AdaptiveTrajectory",
    "population_view",
    "simulate_adaptive_round",
    "simulate_adaptive_population",
)
