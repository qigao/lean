from __future__ import annotations

from dataclasses import dataclass
import re

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import GenericNarrative
from narrative_dynamics.narrative.observation_projection import (
    ObservationProjectionModelSpec,
)
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
    RuntimeIntentionalDecisionResolutionError,
    RuntimeIntentionalDecisionResult,
    run_runtime_intentional_decision,
)
from narrative_dynamics.narrative.runtime_perception import (
    RuntimeEvidenceLedger,
    RuntimePerceptAdmissionError,
    RuntimePerceptAdmissionResult,
    admit_world_percepts,
    runtime_evidence_ledger_from_story,
)
from narrative_dynamics.narrative.world import (
    ActionIntent,
    WorldState,
    WorldStepResult,
    WorldTransitionError,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class SimulationError(ValueError):
    """A deterministic runtime simulation contract could not be satisfied."""


class SimulationStepError(SimulationError):
    """One atomic runtime simulation round failed."""


class SimulationTrajectoryError(SimulationError):
    """A finite runtime simulation trajectory failed."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _step(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _cutoff(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _step(value, label=label)


def _agent_key(agent: RuntimeAgentSpec) -> str:
    return agent.agent_id


def _agent_step_key(step: SimulationAgentStep) -> str:
    return step.agent_id


def _domain_identity(
    domain_id: str,
    domain_version: str,
    domain_spec_hash: str,
) -> tuple[str, str, str]:
    return (domain_id, domain_version, domain_spec_hash)


@dataclass(frozen=True)
class RuntimeAgentSpec:
    agent_id: str
    decision_template_id: str
    intentional_model: RuntimeIntentionalDecisionModelSpec

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="runtime agent id"),
        )
        object.__setattr__(
            self,
            "decision_template_id",
            _text(
                self.decision_template_id,
                label="runtime agent decision template id",
            ),
        )
        if not isinstance(
            self.intentional_model,
            RuntimeIntentionalDecisionModelSpec,
        ):
            raise TypeError(
                "runtime agent intentional model must be "
                "RuntimeIntentionalDecisionModelSpec"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "decision_template_id": self.decision_template_id,
            "intentional_model_hash": self.intentional_model.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationModelSpec:
    model_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    agents: tuple[RuntimeAgentSpec, ...]
    world_model: WorldTransitionModelSpec
    observation_model: ObservationProjectionModelSpec

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="simulation model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="simulation model version"),
        )
        object.__setattr__(
            self,
            "domain_id",
            _text(self.domain_id, label="simulation domain id"),
        )
        object.__setattr__(
            self,
            "domain_version",
            _text(self.domain_version, label="simulation domain version"),
        )
        object.__setattr__(
            self,
            "domain_spec_hash",
            _hash(self.domain_spec_hash, label="simulation domain spec hash"),
        )
        if not isinstance(self.agents, tuple):
            raise TypeError("simulation agents must be a tuple")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("simulation model requires at least one runtime agent")
        if any(not isinstance(item, RuntimeAgentSpec) for item in agents):
            raise TypeError("simulation agents must contain RuntimeAgentSpec values")
        if len({item.agent_id for item in agents}) != len(agents):
            raise ValueError("simulation agent ids must be unique")
        if len({item.decision_template_id for item in agents}) != len(agents):
            raise ValueError("simulation decision template ids must be unique")
        if not isinstance(self.world_model, WorldTransitionModelSpec):
            raise TypeError(
                "simulation world model must be WorldTransitionModelSpec"
            )
        if not isinstance(
            self.observation_model,
            ObservationProjectionModelSpec,
        ):
            raise TypeError(
                "simulation observation model must be "
                "ObservationProjectionModelSpec"
            )
        identity = _domain_identity(
            self.domain_id,
            self.domain_version,
            self.domain_spec_hash,
        )
        if _domain_identity(
            self.world_model.domain_id,
            self.world_model.domain_version,
            self.world_model.domain_spec_hash,
        ) != identity:
            raise ValueError(
                "simulation world model domain identity must match simulation domain"
            )
        if _domain_identity(
            self.observation_model.domain_id,
            self.observation_model.domain_version,
            self.observation_model.domain_spec_hash,
        ) != identity:
            raise ValueError(
                "simulation observation model domain identity must match simulation domain"
            )
        object.__setattr__(self, "agents", tuple(sorted(agents, key=_agent_key)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "agents": [item.to_dict() for item in self.agents],
            "world_model_hash": self.world_model.content_hash,
            "observation_model_hash": self.observation_model.content_hash,
            "scheduler_implementation_identity": measure_implementation(
                SimulationModelSpec
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationState:
    model_id: str
    model_hash: str
    step_index: int
    world_state: WorldState
    evidence_ledger: RuntimeEvidenceLedger

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="simulation state model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="simulation state model hash"),
        )
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="simulation state step index"),
        )
        if not isinstance(self.world_state, WorldState):
            raise TypeError("simulation state world_state must be WorldState")
        if not isinstance(self.evidence_ledger, RuntimeEvidenceLedger):
            raise TypeError(
                "simulation state evidence_ledger must be RuntimeEvidenceLedger"
            )
        if self.step_index != self.world_state.step_index:
            raise ValueError("simulation step must match world step")
        if self.step_index != self.evidence_ledger.current_step_index:
            raise ValueError("simulation step must match evidence ledger step")
        if (
            self.evidence_ledger.current_world_state_hash
            != self.world_state.content_hash
        ):
            raise ValueError(
                "simulation ledger must bind exact current world state"
            )
        world_identity = (
            self.world_state.domain_id,
            self.world_state.domain_version,
            self.world_state.domain_spec_hash,
            self.world_state.source_story_hash,
            self.world_state.source_at_time,
        )
        ledger_identity = (
            self.evidence_ledger.domain_id,
            self.evidence_ledger.domain_version,
            self.evidence_ledger.domain_spec_hash,
            self.evidence_ledger.source_story_hash,
            self.evidence_ledger.source_at_time,
        )
        if world_identity != ledger_identity:
            raise ValueError(
                "simulation world and ledger source identities must match exactly"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "step_index": self.step_index,
            "world_state": self.world_state.to_dict(),
            "evidence_ledger": self.evidence_ledger.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationAgentStep:
    agent_id: str
    decision_template_id: str
    decision_result: RuntimeIntentionalDecisionResult
    action_intent: ActionIntent

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="simulation agent step agent id"),
        )
        object.__setattr__(
            self,
            "decision_template_id",
            _text(
                self.decision_template_id,
                label="simulation agent step decision template id",
            ),
        )
        if not isinstance(
            self.decision_result,
            RuntimeIntentionalDecisionResult,
        ):
            raise TypeError(
                "simulation agent step decision_result must be "
                "RuntimeIntentionalDecisionResult"
            )
        if not isinstance(self.action_intent, ActionIntent):
            raise TypeError("simulation agent step action_intent must be ActionIntent")
        if self.agent_id != self.decision_result.belief_state.agent_id:
            raise ValueError(
                "simulation agent step agent must match decision belief agent"
            )
        if self.decision_template_id != self.decision_result.decision_id:
            raise ValueError(
                "simulation agent step template must match decision result"
            )
        if self.action_intent.decision_id != self.decision_result.decision_id:
            raise ValueError(
                "simulation action intent decision must match decision result"
            )
        if (
            self.action_intent.selected_action
            != self.decision_result.selected_action
        ):
            raise ValueError(
                "simulation action intent action must match decision result"
            )
        if (
            self.action_intent.selection_model_id
            != self.decision_result.model_id
        ):
            raise ValueError(
                "simulation action intent model must match decision result"
            )
        if (
            self.action_intent.selection_result_hash
            != self.decision_result.content_hash
        ):
            raise ValueError(
                "simulation action intent result hash must bind decision result"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "decision_template_id": self.decision_template_id,
            "decision_result": self.decision_result.to_dict(),
            "action_intent": self.action_intent.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationStepResult:
    model_id: str
    model_hash: str
    step_index: int
    prior_state: SimulationState
    agent_steps: tuple[SimulationAgentStep, ...]
    world_step: WorldStepResult
    admission_result: RuntimePerceptAdmissionResult
    next_state: SimulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="simulation step result model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="simulation step result model hash"),
        )
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="simulation step result step index"),
        )
        if not isinstance(self.prior_state, SimulationState):
            raise TypeError("simulation step prior_state must be SimulationState")
        if not isinstance(self.next_state, SimulationState):
            raise TypeError("simulation step next_state must be SimulationState")
        if not isinstance(self.agent_steps, tuple):
            raise TypeError("simulation agent steps must be a tuple")
        agent_steps = tuple(self.agent_steps)
        if not agent_steps:
            raise ValueError("simulation step requires at least one agent step")
        if any(not isinstance(item, SimulationAgentStep) for item in agent_steps):
            raise TypeError(
                "simulation agent steps must contain SimulationAgentStep values"
            )
        if len({item.agent_id for item in agent_steps}) != len(agent_steps):
            raise ValueError("simulation step agent ids must be unique")
        if len({item.decision_template_id for item in agent_steps}) != len(agent_steps):
            raise ValueError("simulation step template ids must be unique")
        agent_steps = tuple(sorted(agent_steps, key=_agent_step_key))
        if not isinstance(self.world_step, WorldStepResult):
            raise TypeError("simulation world_step must be WorldStepResult")
        if not isinstance(
            self.admission_result,
            RuntimePerceptAdmissionResult,
        ):
            raise TypeError(
                "simulation admission_result must be RuntimePerceptAdmissionResult"
            )
        for state in (self.prior_state, self.next_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError(
                    "simulation step states must bind exact model identity"
                )
        if self.step_index != self.next_state.step_index:
            raise ValueError("simulation result step must equal next state step")
        if self.step_index != self.prior_state.step_index + 1:
            raise ValueError("simulation result step must increment prior state")
        if self.world_step.prior_state != self.prior_state.world_state:
            raise ValueError(
                "simulation world step must bind exact prior world state"
            )
        if self.world_step.next_state != self.next_state.world_state:
            raise ValueError(
                "simulation world step must bind exact next world state"
            )
        if (
            self.admission_result.prior_ledger_hash
            != self.prior_state.evidence_ledger.content_hash
        ):
            raise ValueError(
                "simulation admission must bind exact prior evidence ledger"
            )
        if (
            self.admission_result.next_ledger
            != self.next_state.evidence_ledger
        ):
            raise ValueError(
                "simulation admission must bind exact next evidence ledger"
            )
        if (
            self.admission_result.projection_result.source_world_step_hash
            != self.world_step.content_hash
        ):
            raise ValueError(
                "simulation admission projection must bind exact world step"
            )
        scheduled = tuple(
            sorted(
                (item.action_intent.to_dict() for item in agent_steps),
                key=lambda value: (
                    value["decision_id"],
                    value["selected_action"],
                ),
            )
        )
        executed = tuple(
            sorted(
                (item.intent.to_dict() for item in self.world_step.transitions),
                key=lambda value: (
                    value["decision_id"],
                    value["selected_action"],
                ),
            )
        )
        if scheduled != executed:
            raise ValueError(
                "simulation step transitions must equal scheduled intents exactly"
            )
        object.__setattr__(self, "agent_steps", agent_steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "step_index": self.step_index,
            "prior_state": self.prior_state.to_dict(),
            "agent_steps": [item.to_dict() for item in self.agent_steps],
            "world_step": self.world_step.to_dict(),
            "admission_result": self.admission_result.to_dict(),
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationTrajectory:
    model_id: str
    model_hash: str
    initial_state: SimulationState
    steps: tuple[SimulationStepResult, ...]
    final_state: SimulationState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="simulation trajectory model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="simulation trajectory model hash"),
        )
        if not isinstance(self.initial_state, SimulationState):
            raise TypeError(
                "simulation trajectory initial_state must be SimulationState"
            )
        if not isinstance(self.final_state, SimulationState):
            raise TypeError(
                "simulation trajectory final_state must be SimulationState"
            )
        if not isinstance(self.steps, tuple):
            raise TypeError("simulation trajectory steps must be a tuple")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("simulation trajectory must contain at least one step")
        if any(not isinstance(item, SimulationStepResult) for item in steps):
            raise TypeError(
                "simulation trajectory steps must contain SimulationStepResult values"
            )
        for state in (self.initial_state, self.final_state):
            if state.model_id != self.model_id or state.model_hash != self.model_hash:
                raise ValueError(
                    "simulation trajectory states must bind exact model identity"
                )
        current = self.initial_state
        expected_step = self.initial_state.step_index + 1
        for result in steps:
            if result.model_id != self.model_id or result.model_hash != self.model_hash:
                raise ValueError(
                    "simulation trajectory steps must bind exact model identity"
                )
            if result.prior_state != current:
                raise ValueError(
                    "simulation trajectory prior state chain is discontinuous"
                )
            if result.step_index != expected_step:
                raise ValueError(
                    "simulation trajectory step indices must be consecutive"
                )
            current = result.next_state
            expected_step += 1
        if current != self.final_state:
            raise ValueError(
                "simulation trajectory final state must equal chain tail"
            )
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "initial_state": self.initial_state.to_dict(),
            "steps": [item.to_dict() for item in self.steps],
            "final_state": self.final_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validate_execution_bindings(
    story: GenericNarrative,
    domain: DomainSpec,
    model: SimulationModelSpec,
    source_at_time: int | None,
) -> None:
    validate_narrative(story, domain)
    if not isinstance(model, SimulationModelSpec):
        raise TypeError("simulation execution requires SimulationModelSpec")
    cutoff = _cutoff(source_at_time, label="simulation source cutoff")
    domain_identity = _domain_identity(
        domain.domain_id,
        domain.version,
        domain.content_hash,
    )
    if _domain_identity(
        model.domain_id,
        model.domain_version,
        model.domain_spec_hash,
    ) != domain_identity:
        raise ValueError(
            "simulation model domain identity does not match DomainSpec"
        )
    entities = {entity.id: entity for entity in story.entities}
    decisions = {decision.id: decision for decision in story.decisions}
    for agent in model.agents:
        if agent.agent_id not in entities:
            raise ValueError("simulation agent is not declared by the story")
        decision = decisions.get(agent.decision_template_id)
        if decision is None:
            raise ValueError(
                "simulation decision template is not declared by the story"
            )
        if decision.actor_id != agent.agent_id:
            raise ValueError(
                "simulation decision template actor does not match configured agent"
            )
        if decision.type_name not in agent.intentional_model.supported_decision_types:
            raise ValueError(
                "simulation runtime model does not support decision template type"
            )
        if cutoff is not None and decision.logical_time > cutoff:
            raise ValueError(
                "simulation decision template occurs after source cutoff"
            )


def _validate_prior_state(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: SimulationState,
    model: SimulationModelSpec,
) -> None:
    if not isinstance(prior_state, SimulationState):
        raise TypeError("simulation prior state must be SimulationState")
    if not isinstance(model, SimulationModelSpec):
        raise TypeError("simulation step requires SimulationModelSpec")
    if prior_state.model_id != model.model_id:
        raise ValueError("simulation prior state model id does not match model")
    if prior_state.model_hash != model.content_hash:
        raise ValueError("simulation prior state model hash does not match model")

    # Reconstruct the public state so constructor-bypassing forgeries are
    # rejected before any runtime cognition, world, or projection hook.
    SimulationState(
        model_id=prior_state.model_id,
        model_hash=prior_state.model_hash,
        step_index=prior_state.step_index,
        world_state=prior_state.world_state,
        evidence_ledger=prior_state.evidence_ledger,
    )

    cutoff = prior_state.world_state.source_at_time
    _validate_execution_bindings(story, domain, model, cutoff)
    domain_identity = _domain_identity(
        domain.domain_id,
        domain.version,
        domain.content_hash,
    )
    if _domain_identity(
        prior_state.world_state.domain_id,
        prior_state.world_state.domain_version,
        prior_state.world_state.domain_spec_hash,
    ) != domain_identity:
        raise ValueError("simulation prior world domain identity does not match domain")
    if _domain_identity(
        prior_state.evidence_ledger.domain_id,
        prior_state.evidence_ledger.domain_version,
        prior_state.evidence_ledger.domain_spec_hash,
    ) != domain_identity:
        raise ValueError("simulation prior ledger domain identity does not match domain")
    if prior_state.world_state.source_story_hash != story.content_hash:
        raise ValueError("simulation prior world does not bind exact story")
    if prior_state.evidence_ledger.source_story_hash != story.content_hash:
        raise ValueError("simulation prior ledger does not bind exact story")


def simulation_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    model: SimulationModelSpec,
    *,
    at_time: int | None = None,
) -> SimulationState:
    try:
        cutoff = _cutoff(at_time, label="simulation source cutoff")
        _validate_execution_bindings(story, domain, model, cutoff)
        world = world_state_from_story(story, domain, at_time=cutoff)
        ledger = runtime_evidence_ledger_from_story(
            story,
            domain,
            at_time=cutoff,
        )
        return SimulationState(
            model_id=model.model_id,
            model_hash=model.content_hash,
            step_index=0,
            world_state=world,
            evidence_ledger=ledger,
        )
    except SimulationError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise SimulationError(
            "simulation state could not be initialized"
        ) from error


def simulate_step(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: SimulationState,
    model: SimulationModelSpec,
) -> SimulationStepResult:
    try:
        _validate_prior_state(story, domain, prior_state, model)
    except SimulationStepError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise SimulationStepError(
            "simulation prior state validation failed"
        ) from error

    agent_steps: list[SimulationAgentStep] = []
    for agent in model.agents:
        try:
            result = run_runtime_intentional_decision(
                story,
                domain,
                agent.decision_template_id,
                prior_state.evidence_ledger,
                agent.intentional_model,
            )
        except RuntimeIntentionalDecisionResolutionError as error:
            raise SimulationStepError(
                f"simulation cognition failed for agent {agent.agent_id}"
            ) from error

        if result.step_index != prior_state.step_index:
            raise SimulationStepError(
                "runtime decision step does not match simulation prior"
            )
        if (
            result.belief_state.ledger_hash
            != prior_state.evidence_ledger.content_hash
        ):
            raise SimulationStepError(
                "runtime decision does not bind shared prior ledger"
            )
        try:
            intent = ActionIntent(
                decision_id=result.decision_id,
                selected_action=result.selected_action,
                selection_model_id=result.model_id,
                selection_result_hash=result.content_hash,
            )
            agent_steps.append(
                SimulationAgentStep(
                    agent_id=agent.agent_id,
                    decision_template_id=agent.decision_template_id,
                    decision_result=result,
                    action_intent=intent,
                )
            )
        except (TypeError, ValueError, KeyError) as error:
            raise SimulationStepError(
                f"simulation action intent failed for agent {agent.agent_id}"
            ) from error

    try:
        world_step = advance_world_step(
            story,
            domain,
            prior_state.world_state,
            model.world_model,
            tuple(item.action_intent for item in agent_steps),
        )
    except WorldTransitionError as error:
        raise SimulationStepError(
            "simulation world transition failed"
        ) from error

    try:
        admission = admit_world_percepts(
            story,
            domain,
            world_step,
            model.observation_model,
            prior_state.evidence_ledger,
        )
    except RuntimePerceptAdmissionError as error:
        raise SimulationStepError(
            "simulation percept admission failed"
        ) from error

    try:
        next_state = SimulationState(
            model_id=model.model_id,
            model_hash=model.content_hash,
            step_index=prior_state.step_index + 1,
            world_state=world_step.next_state,
            evidence_ledger=admission.next_ledger,
        )
        return SimulationStepResult(
            model_id=model.model_id,
            model_hash=model.content_hash,
            step_index=next_state.step_index,
            prior_state=prior_state,
            agent_steps=tuple(agent_steps),
            world_step=world_step,
            admission_result=admission,
            next_state=next_state,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SimulationStepError(
            "simulation next-state construction failed"
        ) from error


def simulate_trajectory(
    story: GenericNarrative,
    domain: DomainSpec,
    initial_state: SimulationState,
    model: SimulationModelSpec,
    *,
    rounds: int,
) -> SimulationTrajectory:
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise SimulationTrajectoryError(
            "simulation rounds must be a positive integer"
        )
    if not isinstance(initial_state, SimulationState):
        raise SimulationTrajectoryError(
            "simulation initial state must be SimulationState"
        )
    if not isinstance(model, SimulationModelSpec):
        raise SimulationTrajectoryError(
            "simulation trajectory requires SimulationModelSpec"
        )

    current = initial_state
    steps: list[SimulationStepResult] = []
    for _ in range(rounds):
        try:
            result = simulate_step(
                story,
                domain,
                current,
                model,
            )
        except SimulationStepError as error:
            raise SimulationTrajectoryError(
                f"simulation trajectory failed at step {current.step_index + 1}"
            ) from error
        steps.append(result)
        current = result.next_state

    try:
        return SimulationTrajectory(
            model_id=model.model_id,
            model_hash=model.content_hash,
            initial_state=initial_state,
            steps=tuple(steps),
            final_state=current,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SimulationTrajectoryError(
            "simulation trajectory construction failed"
        ) from error
