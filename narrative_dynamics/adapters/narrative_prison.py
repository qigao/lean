from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import ModelRun, Scenario, stable_content_hash
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    DomainSpec,
    EntityTypeSpec,
    EventTypeSpec,
    ParameterSpec,
    SemanticHookBinding,
    StateDelta,
    StateDeltaOp,
    StateEffectSpec,
    StateVariableSpec,
    ValueTypeSpec,
    validate_narrative,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    Observation,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionModelSpec,
    run_runtime_decision,
)
from narrative_dynamics.narrative.runtime_perception import (
    RuntimeEvidenceLedger,
    runtime_evidence_ledger_from_story,
)
from narrative_dynamics.narrative.runtime_reactive import (
    RuntimeReactiveDecisionModelSpec,
)


_BENCHMARK_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "narrative-prison-held-out-v1"
_FAMILIES = ("reactive", "intentional", "planning")
_NAMES = {
    "reactive": "generic-narrative-prison-reactive",
    "intentional": "generic-narrative-prison-intentional",
    "planning": "generic-narrative-prison-planning",
}
_ACTIONS = ("scout", "escape", "submit")
_TERMINAL_ACTIONS = ("escape", "submit")
_REQUIRED_SCENARIO_FIELDS = frozenset(
    {
        "prior_weak",
        "signal_accuracy",
        "guard_persistence",
        "escape_reward",
        "capture_cost",
        "submit_reward",
        "scout_cost",
        "discount",
        "horizon",
    }
)


@dataclass(frozen=True)
class _ScenarioValues:
    prior_weak: float
    signal_accuracy: float
    guard_persistence: float
    escape_reward: float
    capture_cost: float
    submit_reward: float
    scout_cost: float
    discount: float
    horizon: int


@dataclass(frozen=True)
class _BenchmarkCase:
    values: _ScenarioValues
    domain: DomainSpec
    story: GenericNarrative
    decision_id: str
    ledger: RuntimeEvidenceLedger
    guard_cell: StateCellRef
    phase_cell: StateCellRef
    signal_cell: StateCellRef


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _probability(value: object, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _scenario_values(scenario: object) -> _ScenarioValues:
    if not isinstance(scenario, Scenario):
        raise TypeError("narrative prison benchmark requires Scenario")
    payload = scenario.payload
    if set(payload) != _REQUIRED_SCENARIO_FIELDS:
        raise ValueError(
            "narrative prison scenario must contain exactly its declared fields"
        )
    prior_weak = _probability(payload["prior_weak"], label="prior weak probability")
    signal_accuracy = _probability(
        payload["signal_accuracy"], label="signal accuracy"
    )
    if signal_accuracy < 0.5:
        raise ValueError("signal accuracy must be at least 0.5")
    guard_persistence = _probability(
        payload["guard_persistence"], label="guard persistence"
    )
    escape_reward = _finite_number(payload["escape_reward"], label="escape reward")
    capture_cost = _finite_number(payload["capture_cost"], label="capture cost")
    submit_reward = _finite_number(payload["submit_reward"], label="submit reward")
    scout_cost = _finite_number(payload["scout_cost"], label="scout cost")
    if escape_reward < 0.0 or capture_cost < 0.0 or scout_cost < 0.0:
        raise ValueError("escape reward and prison costs must be non-negative")
    discount = _probability(payload["discount"], label="discount")
    horizon = payload["horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError("narrative prison horizon must be an integer")
    if horizon not in (1, 2):
        raise ValueError("narrative prison horizon must be 1 or 2")
    return _ScenarioValues(
        prior_weak=prior_weak,
        signal_accuracy=signal_accuracy,
        guard_persistence=guard_persistence,
        escape_reward=escape_reward,
        capture_cost=capture_cost,
        submit_reward=submit_reward,
        scout_cost=scout_cost,
        discount=discount,
        horizon=horizon,
    )


def _beta(parameters: Mapping[str, float]) -> float:
    if not isinstance(parameters, Mapping) or set(parameters) != {"beta"}:
        raise ValueError("narrative prison benchmark requires exactly one beta parameter")
    beta = _finite_number(parameters["beta"], label="inverse temperature beta")
    if beta <= 0.0:
        raise ValueError("inverse temperature beta must be positive")
    return beta


class _SeedPrisonStateHook:
    def __call__(self, prior_state, event):
        prisoner = event.arguments["prisoner"].value
        if not isinstance(prisoner, EntityRef):
            raise TypeError("seed prisoner argument must contain EntityRef")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    prisoner.entity_id,
                    "prisoner.episode_phase",
                    event.arguments["phase"],
                ),
                StateDeltaOp(
                    "set",
                    prisoner.entity_id,
                    "prisoner.signal",
                    event.arguments["signal"],
                ),
            )
        )


def _build_benchmark_case(
    scenario: Scenario,
    values: _ScenarioValues,
) -> _BenchmarkCase:
    domain = DomainSpec(
        domain_id="narrative-prison-benchmark",
        version=_BENCHMARK_VERSION,
        entity_types=(EntityTypeSpec("Guard"), EntityTypeSpec("Prisoner")),
        value_types=(
            ValueTypeSpec("PrisonerRef", "entity_ref", entity_type="Prisoner"),
            ValueTypeSpec("GuardStatus", "enum", ("weak", "strong")),
            ValueTypeSpec("EpisodePhase", "enum", ("active", "terminal")),
            ValueTypeSpec("Signal", "enum", ("none", "clear", "alarm")),
        ),
        state_variables=(
            StateVariableSpec("guard.status", "Guard", "GuardStatus"),
            StateVariableSpec(
                "prisoner.episode_phase",
                "Prisoner",
                "EpisodePhase",
            ),
            StateVariableSpec("prisoner.signal", "Prisoner", "Signal"),
        ),
        event_types=(
            EventTypeSpec(
                "seed-prison-state",
                None,
                (
                    ParameterSpec("prisoner", "PrisonerRef"),
                    ParameterSpec("phase", "EpisodePhase"),
                    ParameterSpec("signal", "Signal"),
                ),
                (
                    StateEffectSpec("prisoner.episode_phase", "prisoner"),
                    StateEffectSpec("prisoner.signal", "prisoner"),
                ),
                "seed-prison-state",
            ),
        ),
        action_types=(ActionTypeSpec("prison-action", ()),),
        decision_types=(
            DecisionTypeSpec("prison-choice", "Prisoner", "prison-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "seed-prison-state",
                "author deterministic prison benchmark seed state",
                _SeedPrisonStateHook(),
            ),
        ),
    )
    prisoner = Entity("prisoner", "Prisoner")
    guard = Entity("guard", "Guard")
    guard_cell = StateCellRef(EntityRef(guard.id, guard.type_name), "guard.status")
    phase_cell = StateCellRef(
        EntityRef(prisoner.id, prisoner.type_name),
        "prisoner.episode_phase",
    )
    signal_cell = StateCellRef(
        EntityRef(prisoner.id, prisoner.type_name),
        "prisoner.signal",
    )
    seed = NarrativeEvent(
        "seed-prison-state",
        0,
        "seed-prison-state",
        None,
        {
            "prisoner": TypedValue(
                "PrisonerRef",
                EntityRef(prisoner.id, prisoner.type_name),
            ),
            "phase": TypedValue("EpisodePhase", "active"),
            "signal": TypedValue("Signal", "none"),
        },
    )
    action_ids = _TERMINAL_ACTIONS if values.horizon == 1 else _ACTIONS
    decision = Decision(
        "prison-choice",
        1,
        prisoner.id,
        "prison-choice",
        (guard_cell, phase_cell, signal_cell),
        tuple(ActionOption(action, "prison-action", {}) for action in action_ids),
    )
    story = GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=(prisoner, guard),
        events=(seed,),
        observations=(Observation("observe-prison-seed", prisoner.id, seed.id),),
        claims=(),
        receptions=(),
        decisions=(decision,),
    )
    validate_narrative(story, domain)
    ledger = runtime_evidence_ledger_from_story(story, domain, at_time=decision.logical_time)
    return _BenchmarkCase(
        values=values,
        domain=domain,
        story=story,
        decision_id=decision.id,
        ledger=ledger,
        guard_cell=guard_cell,
        phase_cell=phase_cell,
        signal_cell=signal_cell,
    )


def _signal_probability(values: _ScenarioValues, signal: str) -> float:
    clear = (
        values.prior_weak * values.signal_accuracy
        + (1.0 - values.prior_weak) * (1.0 - values.signal_accuracy)
    )
    return clear if signal == "clear" else 1.0 - clear


class _ReactiveScoreHook:
    def __call__(self, context):
        parameters = context.parameters
        prior_weak = float(parameters["prior_weak"])
        signal_accuracy = float(parameters["signal_accuracy"])
        escape_reward = float(parameters["escape_reward"])
        capture_cost = float(parameters["capture_cost"])
        submit_reward = float(parameters["submit_reward"])
        scout_cost = float(parameters["scout_cost"])
        discount = float(parameters["discount"])
        beta = float(parameters["beta"])
        horizon = int(parameters["horizon"])

        direct_escape = (
            prior_weak * escape_reward
            - (1.0 - prior_weak) * capture_cost
        )
        scores = {"escape": direct_escape, "submit": submit_reward}
        if horizon == 2:
            cue_strength = 2.0 * signal_accuracy - 1.0
            cue_scale = (escape_reward + capture_cost) / 2.0
            cue_delta = cue_strength * cue_scale
            terminal_value = 0.0
            for signal, escape_value in (
                ("clear", direct_escape + cue_delta),
                ("alarm", direct_escape - cue_delta),
            ):
                route_values = {
                    "escape": escape_value,
                    "submit": submit_reward,
                }
                route_policy = finite_softmax(route_values, beta=beta)
                signal_mass = (
                    prior_weak * signal_accuracy
                    + (1.0 - prior_weak) * (1.0 - signal_accuracy)
                    if signal == "clear"
                    else prior_weak * (1.0 - signal_accuracy)
                    + (1.0 - prior_weak) * signal_accuracy
                )
                terminal_value += signal_mass * math.fsum(
                    route_policy[action] * route_values[action]
                    for action in _TERMINAL_ACTIONS
                )
            scores["scout"] = -scout_cost + discount * terminal_value
        return scores


def _build_reactive_model(
    case: _BenchmarkCase,
    beta: float,
) -> RuntimeReactiveDecisionModelSpec:
    values = case.values
    return RuntimeReactiveDecisionModelSpec(
        "generic-narrative-prison-reactive",
        _BENCHMARK_VERSION,
        ("prison-choice",),
        (case.guard_cell,),
        {
            "prior_weak": values.prior_weak,
            "signal_accuracy": values.signal_accuracy,
            "escape_reward": values.escape_reward,
            "capture_cost": values.capture_cost,
            "submit_reward": values.submit_reward,
            "scout_cost": values.scout_cost,
            "discount": values.discount,
            "horizon": values.horizon,
            "beta": beta,
        },
        beta,
        _ReactiveScoreHook(),
    )


def _build_intentional_model(case: _BenchmarkCase, beta: float):
    raise NotImplementedError("intentional benchmark family is not implemented yet")


def _build_planning_model(case: _BenchmarkCase, beta: float):
    raise NotImplementedError("planning benchmark family is not implemented yet")


def _build_runtime_decision_model(
    family: str,
    case: _BenchmarkCase,
    beta: float,
):
    builder = {
        "reactive": _build_reactive_model,
        "intentional": _build_intentional_model,
        "planning": _build_planning_model,
    }[family]
    return builder(case, beta)


@dataclass(frozen=True)
class _NarrativePrisonBenchmarkModel:
    family: str

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError("narrative prison benchmark family is unsupported")

    @property
    def name(self) -> str:
        return _NAMES[self.family]

    @property
    def version(self) -> str:
        return _BENCHMARK_VERSION

    @property
    def implementation_revision(self) -> str:
        return _IMPLEMENTATION_REVISION

    def simulate(self, scenario, parameters, rng) -> ModelRun:
        values = _scenario_values(scenario)
        beta = _beta(parameters)
        case = _build_benchmark_case(scenario, values)
        nested = _build_runtime_decision_model(self.family, case, beta)
        dispatch = run_runtime_decision(
            case.story,
            case.domain,
            case.decision_id,
            case.ledger,
            RuntimeDecisionModelSpec(self.family, nested),
        )
        initial_policy = {action: 0.0 for action in _ACTIONS}
        for action, probability in dispatch.action_policy.items():
            initial_policy[action] = probability
        return ModelRun(
            events=(),
            outcome={
                "model_kind": self.family,
                "initial_policy": initial_policy,
                "selected_action": dispatch.selected_action,
                "runtime_dispatch_hash": dispatch.content_hash,
                "runtime_dispatch": dispatch.to_dict(),
                "translated_story_hash": case.story.content_hash,
                "translated_domain_hash": case.domain.content_hash,
                "runtime_ledger_hash": case.ledger.content_hash,
            },
        )


@dataclass(frozen=True)
class NarrativePrisonModelSource:
    family: str

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError(
                "narrative prison family must be reactive, intentional, or planning"
            )

    @property
    def name(self) -> str:
        return _NAMES[self.family]

    @property
    def version(self) -> str:
        return _BENCHMARK_VERSION

    @property
    def implementation_revision(self) -> str:
        return _IMPLEMENTATION_REVISION

    @property
    def lifecycle(self) -> str:
        return "fresh_per_batch"

    def instantiate(self) -> _NarrativePrisonBenchmarkModel:
        return _NarrativePrisonBenchmarkModel(self.family)

    def manifest_identity(self) -> dict[str, object]:
        family_builder = {
            "reactive": _build_reactive_model,
            "intentional": _build_intentional_model,
            "planning": _build_planning_model,
        }[self.family]
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "implementation_revision": self.implementation_revision,
            "lifecycle": self.lifecycle,
            "source_implementation_identity": measure_implementation(
                NarrativePrisonModelSource
            ).manifest_identity(),
            "model_implementation_identity": measure_implementation(
                _NarrativePrisonBenchmarkModel
            ).manifest_identity(),
            "benchmark_builder_implementation_identity": measure_implementation(
                _build_benchmark_case
            ).manifest_identity(),
            "family_builder_implementation_identity": measure_implementation(
                family_builder
            ).manifest_identity(),
            "dispatch_implementation_identity": measure_implementation(
                run_runtime_decision
            ).manifest_identity(),
        }


def create_narrative_prison_reactive_source() -> NarrativePrisonModelSource:
    return NarrativePrisonModelSource("reactive")


def create_narrative_prison_intentional_source() -> NarrativePrisonModelSource:
    return NarrativePrisonModelSource("intentional")


def create_narrative_prison_planning_source() -> NarrativePrisonModelSource:
    return NarrativePrisonModelSource("planning")


__all__ = [
    "NarrativePrisonModelSource",
    "create_narrative_prison_intentional_source",
    "create_narrative_prison_planning_source",
    "create_narrative_prison_reactive_source",
]
