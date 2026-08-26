from __future__ import annotations

from dataclasses import fields, replace
import inspect
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    ParameterSpec,
    StateDelta,
    StateDeltaOp,
)
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    GoalModelSpec,
    GoalSpec,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionModelSpec,
    ObserverProjectionSpec,
)
from narrative_dynamics.narrative.runtime_cognition import RuntimeBeliefModelSpec
from narrative_dynamics.narrative.uncertain import UncertainBeliefModelSpec
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionTransitionSpec,
    WorldTransitionModelSpec,
)
from tests.test_narrative_observation_projection import (
    make_projection_domain,
    make_projection_story,
)

_SIMULATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_intention import (
        RuntimeIntentionalDecisionModelSpec,
        RuntimeIntentionalDecisionResolutionError,
    )
    from narrative_dynamics.narrative.simulation import (
        RuntimeAgentSpec,
        SimulationAgentStep,
        SimulationError,
        SimulationModelSpec,
        SimulationState,
        SimulationStepError,
        SimulationStepResult,
        SimulationTrajectory,
        SimulationTrajectoryError,
        simulate_step,
        simulate_trajectory,
        simulation_state_from_story,
    )
except ImportError as error:
    _SIMULATION_IMPORT_ERROR = error


def _value_hash(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def _hash(label: str) -> str:
    return stable_content_hash({"scheduler-test": label})


def alert_cell() -> StateCellRef:
    return StateCellRef(EntityRef("svc", "Service"), "service.alert")


def _service_ref() -> TypedValue:
    return TypedValue("ServiceRef", EntityRef("svc", "Service"))


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


def make_scheduler_domain():
    base = make_projection_domain()
    return replace(
        base,
        action_types=base.action_types
        + (
            ActionTypeSpec(
                "alert-control-action",
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("raise", "AlertState"),
                ),
            ),
            ActionTypeSpec(
                "response-action",
                (ParameterSpec("respond", "AlertState"),),
            ),
        ),
        decision_types=base.decision_types
        + (
            DecisionTypeSpec(
                "scheduler-alert-choice",
                "Agent",
                "alert-control-action",
            ),
            DecisionTypeSpec(
                "scheduler-response-choice",
                "Agent",
                "response-action",
            ),
        ),
    )


def make_scheduler_story(domain):
    base = make_projection_story(domain)
    events = tuple(
        replace(
            event,
            arguments={
                **dict(event.arguments),
                "alert": TypedValue("AlertState", False),
            },
        )
        if event.id == "e6"
        else event
        for event in base.events
    )
    decisions = (
        Decision(
            "d-a1-scheduler",
            8,
            "a1",
            "scheduler-alert-choice",
            (alert_cell(),),
            (
                ActionOption(
                    "a1-raise-alert",
                    "alert-control-action",
                    {
                        "service": _service_ref(),
                        "raise": TypedValue("AlertState", True),
                    },
                ),
                ActionOption(
                    "a1-wait",
                    "alert-control-action",
                    {
                        "service": _service_ref(),
                        "raise": TypedValue("AlertState", False),
                    },
                ),
            ),
        ),
        Decision(
            "d-a2-scheduler",
            9,
            "a2",
            "scheduler-response-choice",
            (alert_cell(),),
            (
                ActionOption(
                    "a2-respond",
                    "response-action",
                    {"respond": TypedValue("AlertState", True)},
                ),
                ActionOption(
                    "a2-wait",
                    "response-action",
                    {"respond": TypedValue("AlertState", False)},
                ),
            ),
        ),
    )
    return replace(
        base,
        entities=base.entities + (Entity("a3", "Agent"),),
        events=events,
        observations=(),
        claims=(),
        receptions=(),
        decisions=decisions,
    )


class SchedulerPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        result = {}
        for value in hypotheses:
            if agent_id == "a2" and value.value is False:
                result[_value_hash(value)] = 0.9
            elif agent_id == "a2":
                result[_value_hash(value)] = 0.1
            else:
                result[_value_hash(value)] = 0.5
        return result


class InvalidSchedulerPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        return {}


class SchedulerSeedLikelihood:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        return {_value_hash(value): 1.0 for value in hypotheses}


class RuntimeAlertLikelihood:
    def __init__(self):
        self.calls = []

    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        self.calls.append((agent_id, percept_view))
        if percept_view.relation != "equals":
            return {_value_hash(value): 1.0 for value in hypotheses}
        return {
            _value_hash(value): (
                parameters["match"]
                if value == percept_view.value
                else parameters["mismatch"]
            )
            for value in hypotheses
        }


def make_runtime_belief_model(*, prior_hook=None, runtime_hook=None):
    if prior_hook is None:
        prior_hook = SchedulerPriorHook()
    if runtime_hook is None:
        runtime_hook = RuntimeAlertLikelihood()
    seed = UncertainBeliefModelSpec(
        "scheduler-seed",
        "1",
        {},
        prior_hook,
        SchedulerSeedLikelihood(),
    )
    return RuntimeBeliefModelSpec(
        "scheduler-belief",
        "1",
        seed,
        {"match": 0.99, "mismatch": 0.01},
        runtime_hook,
    )


def _bool_hash(value: bool) -> str:
    return _value_hash(TypedValue("AlertState", value))


def make_a_goal_model():
    false_hash = _bool_hash(False)
    true_hash = _bool_hash(True)
    return GoalModelSpec(
        "a-goals",
        "1",
        8.0,
        (
            GoalSpec(
                "act",
                2.0,
                {alert_cell(): 1.0},
                {alert_cell(): {false_hash: 1.0, true_hash: 1.0}},
            ),
            GoalSpec(
                "idle",
                1.0,
                {alert_cell(): 1.0},
                {alert_cell(): {false_hash: 1.0, true_hash: 1.0}},
            ),
        ),
    )


def make_b_goal_model():
    false_hash = _bool_hash(False)
    true_hash = _bool_hash(True)
    return GoalModelSpec(
        "b-goals",
        "1",
        8.0,
        (
            GoalSpec(
                "idle",
                1.0,
                {alert_cell(): 1.0},
                {alert_cell(): {false_hash: 1.0, true_hash: 0.0}},
            ),
            GoalSpec(
                "respond",
                1.0,
                {alert_cell(): 1.0},
                {alert_cell(): {false_hash: 0.0, true_hash: 1.0}},
            ),
        ),
    )


def make_a_choice_model():
    return ChoiceModelSpec(
        "a-choice",
        "1",
        8.0,
        {
            "act": {"a1-raise-alert": 3.0, "a1-wait": 0.0},
            "idle": {"a1-raise-alert": 0.0, "a1-wait": 3.0},
        },
    )


def make_b_choice_model():
    return ChoiceModelSpec(
        "b-choice",
        "1",
        8.0,
        {
            "idle": {"a2-respond": 0.0, "a2-wait": 3.0},
            "respond": {"a2-respond": 3.0, "a2-wait": 0.0},
        },
    )


def make_conflict_b_choice_model():
    return ChoiceModelSpec(
        "b-conflict-choice",
        "1",
        8.0,
        {
            "act": {"a2-raise-alert": 3.0, "a2-wait-alert": 0.0},
            "idle": {"a2-raise-alert": 0.0, "a2-wait-alert": 3.0},
        },
    )


class AlertControlTransition:
    def __init__(self):
        self.calls = 0

    def __call__(self, snapshot, decision, action):
        self.calls += 1
        if action.arguments["raise"].value is False:
            return StateDelta(())
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    "svc",
                    "service.alert",
                    TypedValue("AlertState", True),
                ),
            )
        )


class ResponseTransition:
    def __init__(self):
        self.calls = 0

    def __call__(self, snapshot, decision, action):
        self.calls += 1
        if action.arguments["respond"].value is False:
            return StateDelta(())
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    TypedValue("PhaseState", "active"),
                ),
            )
        )


class ObserveAlertForBAndPassive:
    def __init__(self):
        self.calls = []

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls.append((observer.id, step_index))
        if observer.id not in {"a2", "a3"}:
            return ()
        value = next_visible.get(alert_cell())
        return () if value is None else (
            ObservationFact(alert_cell(), "equals", value),
        )


class RaisingProjection:
    def __init__(self):
        self.calls = []

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls.append((observer.id, step_index))
        raise RuntimeError("scheduler projection boom")


class RaiseAtSecondStepProjection(ObserveAlertForBAndPassive):
    def __call__(self, prior_visible, next_visible, observer, step_index):
        if step_index == 2:
            self.calls.append((observer.id, step_index))
            raise RuntimeError("scheduler projection step two boom")
        return super().__call__(prior_visible, next_visible, observer, step_index)


def make_world_model(domain, *, alert_hook=None, response_hook=None):
    if alert_hook is None:
        alert_hook = AlertControlTransition()
    if response_hook is None:
        response_hook = ResponseTransition()
    return WorldTransitionModelSpec(
        "scheduler-world",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            ActionTransitionSpec(
                "alert-control-action",
                (ActionEffectSpec("service.alert", "argument", "service"),),
                alert_hook,
            ),
            ActionTransitionSpec(
                "response-action",
                (ActionEffectSpec("agent.phase", "actor"),),
                response_hook,
            ),
        ),
    )


def make_projection_model(domain, *, hook=None):
    if hook is None:
        hook = ObserveAlertForBAndPassive()
    capability = ObservationCapabilitySpec("service.alert", "any")
    return ObservationProjectionModelSpec(
        "scheduler-projection",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            ObserverProjectionSpec(
                "Agent",
                "vision",
                (capability,),
                (capability,),
                hook,
            ),
        ),
    )


def make_intentional_models(*, a_belief=None, b_belief=None):
    if a_belief is None:
        a_belief = make_runtime_belief_model()
    if b_belief is None:
        b_belief = make_runtime_belief_model()
    return (
        RuntimeIntentionalDecisionModelSpec(
            "a-runtime-intentional",
            "1",
            ("scheduler-alert-choice",),
            a_belief,
            make_a_goal_model(),
            make_a_choice_model(),
        ),
        RuntimeIntentionalDecisionModelSpec(
            "b-runtime-intentional",
            "1",
            ("scheduler-response-choice",),
            b_belief,
            make_b_goal_model(),
            make_b_choice_model(),
        ),
    )


def make_simulation_model(
    domain,
    *,
    reverse_agents: bool = False,
    a_belief=None,
    b_belief=None,
    alert_hook=None,
    response_hook=None,
    projection_hook=None,
):
    a_model, b_model = make_intentional_models(
        a_belief=a_belief,
        b_belief=b_belief,
    )
    agents = (
        RuntimeAgentSpec("a1", "d-a1-scheduler", a_model),
        RuntimeAgentSpec("a2", "d-a2-scheduler", b_model),
    )
    if reverse_agents:
        agents = tuple(reversed(agents))
    return SimulationModelSpec(
        "scheduler-model",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        agents,
        make_world_model(
            domain,
            alert_hook=alert_hook,
            response_hook=response_hook,
        ),
        make_projection_model(domain, hook=projection_hook),
    )


def make_case(**model_kwargs):
    domain = make_scheduler_domain()
    story = make_scheduler_story(domain)
    model = make_simulation_model(domain, **model_kwargs)
    return domain, story, model


def make_conflict_case():
    domain = make_scheduler_domain()
    story = make_scheduler_story(domain)
    b_conflict_decision = Decision(
        "d-a2-scheduler",
        9,
        "a2",
        "scheduler-alert-choice",
        (alert_cell(),),
        (
            ActionOption(
                "a2-raise-alert",
                "alert-control-action",
                {
                    "service": _service_ref(),
                    "raise": TypedValue("AlertState", True),
                },
            ),
            ActionOption(
                "a2-wait-alert",
                "alert-control-action",
                {
                    "service": _service_ref(),
                    "raise": TypedValue("AlertState", False),
                },
            ),
        ),
    )
    story = replace(
        story,
        decisions=(story.decisions[0], b_conflict_decision),
    )
    a_belief = make_runtime_belief_model()
    b_belief = make_runtime_belief_model()
    a_model = RuntimeIntentionalDecisionModelSpec(
        "a-runtime-intentional",
        "1",
        ("scheduler-alert-choice",),
        a_belief,
        make_a_goal_model(),
        make_a_choice_model(),
    )
    b_model = RuntimeIntentionalDecisionModelSpec(
        "b-conflict-intentional",
        "1",
        ("scheduler-alert-choice",),
        b_belief,
        make_a_goal_model(),
        make_conflict_b_choice_model(),
    )
    alert_hook = AlertControlTransition()
    projection_hook = ObserveAlertForBAndPassive()
    world_model = WorldTransitionModelSpec(
        "scheduler-world-conflict",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            ActionTransitionSpec(
                "alert-control-action",
                (ActionEffectSpec("service.alert", "argument", "service"),),
                alert_hook,
            ),
        ),
    )
    model = SimulationModelSpec(
        "scheduler-conflict-model",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            RuntimeAgentSpec("a1", "d-a1-scheduler", a_model),
            RuntimeAgentSpec("a2", "d-a2-scheduler", b_model),
        ),
        world_model,
        make_projection_model(domain, hook=projection_hook),
    )
    return domain, story, model, alert_hook, projection_hook


class NarrativeSimulationTests(unittest.TestCase):
    def require_simulation(self) -> None:
        if _SIMULATION_IMPORT_ERROR is not None:
            self.fail(
                "narrative simulation boundary is missing: "
                f"{_SIMULATION_IMPORT_ERROR}"
            )

    def test_simulation_model_identity_canonicalizes_agents_and_binds_nested_models(self):
        self.require_simulation()
        domain, _, first = make_case()
        _, _, reversed_model = make_case(reverse_agents=True)
        self.assertEqual(first.agents, tuple(sorted(first.agents, key=lambda item: item.agent_id)))
        self.assertEqual(first.content_hash, reversed_model.content_hash)
        changed_world = replace(first.world_model, version="2")
        changed_world_model = replace(first, world_model=changed_world)
        self.assertNotEqual(first.content_hash, changed_world_model.content_hash)
        changed_projection = replace(first.observation_model, version="2")
        changed_projection_model = replace(first, observation_model=changed_projection)
        self.assertNotEqual(first.content_hash, changed_projection_model.content_hash)
        self.assertEqual(first.domain_spec_hash, domain.content_hash)

    def test_simulation_model_rejects_duplicate_agents_templates_and_domain_mismatch(self):
        self.require_simulation()
        domain, _, model = make_case()
        first, second = model.agents
        with self.assertRaises((TypeError, ValueError)):
            replace(model, agents=(first, replace(second, agent_id=first.agent_id)))
        with self.assertRaises((TypeError, ValueError)):
            replace(
                model,
                agents=(
                    first,
                    replace(second, decision_template_id=first.decision_template_id),
                ),
            )
        with self.assertRaises((TypeError, ValueError)):
            replace(
                model,
                world_model=replace(model.world_model, domain_spec_hash=_hash("bad-domain")),
            )
        with self.assertRaises((TypeError, ValueError)):
            replace(
                model,
                observation_model=replace(
                    model.observation_model,
                    domain_version="2",
                ),
            )
        self.assertEqual(model.domain_id, domain.domain_id)

    def test_simulation_state_from_story_binds_exact_world_ledger_model_and_cutoff(self):
        self.require_simulation()
        domain, story, model = make_case()
        state = simulation_state_from_story(
            story,
            domain,
            model,
            at_time=9,
        )
        self.assertEqual(state.model_id, model.model_id)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertEqual(state.step_index, 0)
        self.assertEqual(state.world_state.step_index, 0)
        self.assertEqual(state.evidence_ledger.current_step_index, 0)
        self.assertEqual(state.world_state.source_at_time, 9)
        self.assertEqual(state.evidence_ledger.source_at_time, 9)
        self.assertEqual(
            state.evidence_ledger.current_world_state_hash,
            state.world_state.content_hash,
        )

    def test_initialization_rejects_story_agent_template_type_and_cutoff_mismatch_before_hooks(self):
        self.require_simulation()
        domain, story, model = make_case()
        b_hook = model.agents[1].intentional_model.belief_model.runtime_likelihood_hook

        missing_template_model = replace(
            model,
            agents=(
                model.agents[0],
                replace(model.agents[1], decision_template_id="missing-template"),
            ),
        )
        with self.assertRaises(SimulationError):
            simulation_state_from_story(
                story,
                domain,
                missing_template_model,
                at_time=9,
            )
        self.assertEqual(b_hook.calls, [])

        bad_actor_story = replace(
            story,
            decisions=tuple(
                replace(item, actor_id="a1")
                if item.id == "d-a2-scheduler"
                else item
                for item in story.decisions
            ),
        )
        with self.assertRaises(SimulationError):
            simulation_state_from_story(
                bad_actor_story,
                domain,
                model,
                at_time=9,
            )
        self.assertEqual(b_hook.calls, [])

        unsupported_b = replace(
            model.agents[1].intentional_model,
            supported_decision_types=("scheduler-alert-choice",),
        )
        unsupported_model = replace(
            model,
            agents=(
                model.agents[0],
                replace(model.agents[1], intentional_model=unsupported_b),
            ),
        )
        with self.assertRaises(SimulationError):
            simulation_state_from_story(
                story,
                domain,
                unsupported_model,
                at_time=9,
            )
        self.assertEqual(b_hook.calls, [])

        with self.assertRaises(SimulationError):
            simulation_state_from_story(story, domain, model, at_time=8)
        self.assertEqual(b_hook.calls, [])

        missing_agent_story = replace(
            story,
            entities=tuple(item for item in story.entities if item.id != "a2"),
        )
        with self.assertRaises(SimulationError):
            simulation_state_from_story(
                missing_agent_story,
                domain,
                model,
                at_time=9,
            )
        self.assertEqual(b_hook.calls, [])

    def test_static_round_runs_every_agent_once_against_same_prior_ledger_and_builds_exact_intents(self):
        self.require_simulation()
        domain = make_scheduler_domain()
        story = make_scheduler_story(domain)
        alert_hook = AlertControlTransition()
        response_hook = ResponseTransition()
        projection_hook = ObserveAlertForBAndPassive()
        model = make_simulation_model(
            domain,
            alert_hook=alert_hook,
            response_hook=response_hook,
            projection_hook=projection_hook,
        )
        prior = simulation_state_from_story(story, domain, model, at_time=9)
        result = simulate_step(story, domain, prior, model)
        self.assertEqual(tuple(item.agent_id for item in result.agent_steps), ("a1", "a2"))
        self.assertEqual(len(result.world_step.transitions), 2)
        self.assertEqual(alert_hook.calls, 1)
        self.assertEqual(response_hook.calls, 1)
        self.assertEqual(len(projection_hook.calls), 3)
        for agent_step in result.agent_steps:
            self.assertEqual(
                agent_step.decision_result.belief_state.ledger_hash,
                prior.evidence_ledger.content_hash,
            )
            self.assertEqual(agent_step.decision_result.step_index, prior.step_index)
            self.assertEqual(
                agent_step.action_intent.decision_id,
                agent_step.decision_result.decision_id,
            )
            self.assertEqual(
                agent_step.action_intent.selected_action,
                agent_step.decision_result.selected_action,
            )
            self.assertEqual(
                agent_step.action_intent.selection_result_hash,
                agent_step.decision_result.content_hash,
            )
        selected = {item.agent_id: item.decision_result.selected_action for item in result.agent_steps}
        self.assertEqual(selected, {"a1": "a1-raise-alert", "a2": "a2-wait"})

    def test_cognition_failure_blocks_all_world_projection_and_next_state(self):
        self.require_simulation()
        domain = make_scheduler_domain()
        story = make_scheduler_story(domain)
        alert_hook = AlertControlTransition()
        response_hook = ResponseTransition()
        projection_hook = ObserveAlertForBAndPassive()
        bad_belief = make_runtime_belief_model(prior_hook=InvalidSchedulerPriorHook())
        model = make_simulation_model(
            domain,
            b_belief=bad_belief,
            alert_hook=alert_hook,
            response_hook=response_hook,
            projection_hook=projection_hook,
        )
        prior = simulation_state_from_story(story, domain, model, at_time=9)
        with self.assertRaises(SimulationStepError) as caught:
            simulate_step(story, domain, prior, model)
        self.assertIsInstance(
            caught.exception.__cause__,
            RuntimeIntentionalDecisionResolutionError,
        )
        self.assertEqual(alert_hook.calls, 0)
        self.assertEqual(response_hook.calls, 0)
        self.assertEqual(projection_hook.calls, [])
        self.assertEqual(prior.step_index, 0)

    def test_world_failure_blocks_projection_and_preserves_prior_state(self):
        self.require_simulation()
        domain, story, model, alert_hook, projection_hook = make_conflict_case()
        prior = simulation_state_from_story(story, domain, model, at_time=9)
        before = prior.to_dict()
        with self.assertRaises(SimulationStepError):
            simulate_step(story, domain, prior, model)
        self.assertEqual(alert_hook.calls, 2)
        self.assertEqual(projection_hook.calls, [])
        self.assertEqual(prior.to_dict(), before)
        self.assertEqual(prior.step_index, 0)

    def test_projection_or_admission_failure_returns_no_successful_next_state(self):
        self.require_simulation()
        domain = make_scheduler_domain()
        story = make_scheduler_story(domain)
        alert_hook = AlertControlTransition()
        response_hook = ResponseTransition()
        projection_hook = RaisingProjection()
        model = make_simulation_model(
            domain,
            alert_hook=alert_hook,
            response_hook=response_hook,
            projection_hook=projection_hook,
        )
        prior = simulation_state_from_story(story, domain, model, at_time=9)
        before = prior.to_dict()
        with self.assertRaises(SimulationStepError):
            simulate_step(story, domain, prior, model)
        self.assertEqual(alert_hook.calls, 1)
        self.assertEqual(response_hook.calls, 1)
        self.assertGreater(len(projection_hook.calls), 0)
        self.assertEqual(prior.to_dict(), before)
        self.assertEqual(prior.step_index, 0)

    def test_agent_input_order_does_not_change_step_hash_or_semantics(self):
        self.require_simulation()
        domain = make_scheduler_domain()
        story = make_scheduler_story(domain)
        first = make_simulation_model(domain)
        second = make_simulation_model(domain, reverse_agents=True)
        self.assertEqual(first.content_hash, second.content_hash)
        first_state = simulation_state_from_story(story, domain, first, at_time=9)
        second_state = simulation_state_from_story(story, domain, second, at_time=9)
        self.assertEqual(first_state.content_hash, second_state.content_hash)
        first_step = simulate_step(story, domain, first_state, first)
        second_step = simulate_step(story, domain, second_state, second)
        self.assertEqual(first_step.content_hash, second_step.content_hash)
        self.assertEqual(
            tuple(item.agent_id for item in first_step.agent_steps),
            ("a1", "a2"),
        )
        self.assertEqual(first_step.agent_steps, second_step.agent_steps)

    def test_same_authored_template_repeats_across_rounds_without_logical_time_mutation(self):
        self.require_simulation()
        domain, story, model = make_case()
        original = {item.id: item.logical_time for item in story.decisions}
        initial = simulation_state_from_story(story, domain, model, at_time=9)
        trajectory = simulate_trajectory(
            story,
            domain,
            initial,
            model,
            rounds=2,
        )
        for step in trajectory.steps:
            by_agent = {item.agent_id: item for item in step.agent_steps}
            self.assertEqual(by_agent["a1"].decision_template_id, "d-a1-scheduler")
            self.assertEqual(by_agent["a2"].decision_template_id, "d-a2-scheduler")
        self.assertEqual(
            {item.id: item.logical_time for item in story.decisions},
            original,
        )
        self.assertEqual(original, {"d-a1-scheduler": 8, "d-a2-scheduler": 9})

    def test_passive_observer_evidence_and_scheduled_agent_blackout_are_legal(self):
        self.require_simulation()
        domain, story, model = make_case()
        initial = simulation_state_from_story(story, domain, model, at_time=9)
        first = simulate_step(story, domain, initial, model)
        observers = {
            item.observer_id
            for item in first.next_state.evidence_ledger.batches[-1].evidence
        }
        self.assertEqual(observers, {"a2", "a3"})
        self.assertNotIn("a3", {item.agent_id for item in first.agent_steps})
        self.assertIn("a1", {item.agent_id for item in first.agent_steps})
        second = simulate_step(story, domain, first.next_state, model)
        self.assertIn("a1", {item.agent_id for item in second.agent_steps})
        self.assertEqual(second.prior_state, first.next_state)

    def test_two_round_action_world_percept_belief_action_causal_closure_without_story_mutation(self):
        self.require_simulation()
        domain, story, model = make_case()
        original_hash = story.content_hash
        original_decisions = story.decisions
        original_observations = story.observations
        original_claims = story.claims
        original_receptions = story.receptions
        initial = simulation_state_from_story(story, domain, model, at_time=9)
        trajectory = simulate_trajectory(
            story,
            domain,
            initial,
            model,
            rounds=2,
        )
        self.assertEqual(len(trajectory.steps), 2)
        step0 = {item.agent_id: item for item in trajectory.steps[0].agent_steps}
        step1 = {item.agent_id: item for item in trajectory.steps[1].agent_steps}
        self.assertEqual(step0["a1"].decision_result.selected_action, "a1-raise-alert")
        self.assertEqual(step0["a2"].decision_result.selected_action, "a2-wait")
        self.assertEqual(
            trajectory.steps[0].next_state.world_state.values[alert_cell()],
            TypedValue("AlertState", True),
        )
        batch = trajectory.steps[0].next_state.evidence_ledger.batches[-1]
        a2_evidence = tuple(item for item in batch.evidence if item.observer_id == "a2")
        passive = tuple(item for item in batch.evidence if item.observer_id == "a3")
        self.assertEqual(len(a2_evidence), 1)
        self.assertEqual(a2_evidence[0].step_index, 1)
        self.assertEqual(a2_evidence[0].cell, alert_cell())
        self.assertEqual(a2_evidence[0].relation, "equals")
        self.assertEqual(a2_evidence[0].value, TypedValue("AlertState", True))
        self.assertEqual(len(passive), 1)
        round1_belief = step1["a2"].decision_result.belief_state
        view = round1_belief.cells[alert_cell()]
        self.assertNotEqual(view.posterior, view.seed_posterior)
        self.assertGreater(
            view.posterior.probability_of(TypedValue("AlertState", True)),
            view.posterior.probability_of(TypedValue("AlertState", False)),
        )
        self.assertEqual(step1["a2"].decision_result.selected_action, "a2-respond")
        self.assertEqual(story.content_hash, original_hash)
        self.assertEqual(story.decisions, original_decisions)
        self.assertEqual(story.observations, original_observations)
        self.assertEqual(story.claims, original_claims)
        self.assertEqual(story.receptions, original_receptions)

    def test_trajectory_chain_and_final_state_are_exact(self):
        self.require_simulation()
        domain, story, model = make_case()
        initial = simulation_state_from_story(story, domain, model, at_time=9)
        trajectory = simulate_trajectory(story, domain, initial, model, rounds=3)
        self.assertIsInstance(trajectory, SimulationTrajectory)
        self.assertEqual(trajectory.initial_state, initial)
        self.assertEqual(len(trajectory.steps), 3)
        current = initial
        for expected_step, result in enumerate(trajectory.steps, start=1):
            self.assertIsInstance(result, SimulationStepResult)
            self.assertEqual(result.prior_state, current)
            self.assertEqual(result.step_index, expected_step)
            self.assertEqual(result.next_state.step_index, expected_step)
            current = result.next_state
        self.assertEqual(trajectory.final_state, current)
        self.assertEqual(trajectory.final_state.step_index, 3)

    def test_deterministic_replay_produces_exact_same_trajectory_hash(self):
        self.require_simulation()
        domain = make_scheduler_domain()
        story = make_scheduler_story(domain)
        first_model = make_simulation_model(domain)
        second_model = make_simulation_model(domain)
        self.assertEqual(first_model.content_hash, second_model.content_hash)
        first_initial = simulation_state_from_story(
            story, domain, first_model, at_time=9
        )
        second_initial = simulation_state_from_story(
            story, domain, second_model, at_time=9
        )
        first = simulate_trajectory(
            story, domain, first_initial, first_model, rounds=3
        )
        second = simulate_trajectory(
            story, domain, second_initial, second_model, rounds=3
        )
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_round_validation_and_trajectory_failure_report_exact_step_with_typed_cause(self):
        self.require_simulation()
        domain, story, model = make_case()
        initial = simulation_state_from_story(story, domain, model, at_time=9)
        for rounds in (0, -1, True, 1.5):
            with self.subTest(rounds=rounds):
                with self.assertRaises(SimulationTrajectoryError):
                    simulate_trajectory(
                        story,
                        domain,
                        initial,
                        model,
                        rounds=rounds,
                    )

        projection_hook = RaiseAtSecondStepProjection()
        failing_model = make_simulation_model(
            domain,
            projection_hook=projection_hook,
        )
        failing_initial = simulation_state_from_story(
            story,
            domain,
            failing_model,
            at_time=9,
        )
        with self.assertRaises(SimulationTrajectoryError) as caught:
            simulate_trajectory(
                story,
                domain,
                failing_initial,
                failing_model,
                rounds=2,
            )
        self.assertIn("step 2", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, SimulationStepError)

    def test_forged_prior_simulation_state_rejects_before_any_agent_hook(self):
        self.require_simulation()
        domain = make_scheduler_domain()
        story = make_scheduler_story(domain)
        alert_hook = AlertControlTransition()
        response_hook = ResponseTransition()
        projection_hook = ObserveAlertForBAndPassive()
        a_runtime = RuntimeAlertLikelihood()
        b_runtime = RuntimeAlertLikelihood()
        model = make_simulation_model(
            domain,
            a_belief=make_runtime_belief_model(runtime_hook=a_runtime),
            b_belief=make_runtime_belief_model(runtime_hook=b_runtime),
            alert_hook=alert_hook,
            response_hook=response_hook,
            projection_hook=projection_hook,
        )
        prior = simulation_state_from_story(story, domain, model, at_time=9)
        forged = _forge(prior, step_index=1)
        with self.assertRaises(SimulationStepError):
            simulate_step(story, domain, forged, model)
        self.assertEqual(a_runtime.calls, [])
        self.assertEqual(b_runtime.calls, [])
        self.assertEqual(alert_hook.calls, 0)
        self.assertEqual(response_hook.calls, 0)
        self.assertEqual(projection_hook.calls, [])


if __name__ == "__main__":
    unittest.main()
