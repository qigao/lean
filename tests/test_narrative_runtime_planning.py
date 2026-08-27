from __future__ import annotations

from dataclasses import fields, replace
import inspect
import math
import unittest

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.intention import ChoiceModelSpec, GoalModelSpec, GoalSpec
from narrative_dynamics.narrative.ir import ActionOption, TypedValue
from narrative_dynamics.narrative.runtime_cognition import RuntimeUncertainBeliefState
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
    run_runtime_intentional_decision,
)
from narrative_dynamics.narrative.runtime_perception import (
    RuntimeEvidenceLedger,
    runtime_evidence_ledger_from_story,
)
from narrative_dynamics.narrative.runtime_reactive import (
    RuntimeReactiveDecisionModelSpec,
    run_runtime_reactive_decision,
)
from tests.test_narrative_runtime_cognition import (
    alert_cell,
    empty_runtime_case,
    make_runtime_belief_model,
    phase_cell,
)


_PLANNING_IMPORT_ERROR: ImportError | None = None
try:
    import narrative_dynamics.narrative.runtime_planning as runtime_planning_module
    from narrative_dynamics.narrative.runtime_planning import (
        PlanningBeliefState,
        PlanningBeliefUpdate,
        PlanningHiddenState,
        PlanningObservation,
        PlanningObservationContext,
        PlanningRewardContext,
        PlanningTransitionContext,
        PlanningValueRecord,
        RuntimePlanningBeliefContext,
        RuntimePlanningDecisionModelSpec,
        RuntimePlanningDecisionResolutionError,
        RuntimePlanningDecisionResult,
        run_runtime_planning_decision,
    )
except ImportError as error:
    _PLANNING_IMPORT_ERROR = error


def _hash(label: str) -> str:
    return stable_content_hash({"runtime-planning-test": label})


def _value_hash(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


class ProductCouplingHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        result = {}
        for state in context.hidden_states:
            probability = 1.0
            for cell, distribution in context.posterior.items():
                probability *= distribution.probability_of(state.cells[cell])
            result[state.state_id] = probability
        return result


class ConcentratedCouplingHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        first = context.hidden_states[0].state_id
        return {
            state.state_id: 1.0 if state.state_id == first else 0.0
            for state in context.hidden_states
        }


class IdentityTransitionHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return {
            state.state_id: 1.0 if state.state_id == context.state.state_id else 0.0
            for state in context.candidate_next_states
        }


class NoInformationObservationHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        first = context.observations[0].observation_id
        return {
            observation.observation_id: (
                1.0 if observation.observation_id == first else 0.0
            )
            for observation in context.observations
        }


class UniformObservationHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        mass = 1.0 / len(context.observations)
        return {item.observation_id: mass for item in context.observations}


class InformativeObservationHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        if context.action.id != "inspect":
            mass = 1.0 / len(context.observations)
            return {item.observation_id: mass for item in context.observations}
        return {
            item.observation_id: (
                1.0 if item.observation_id == context.next_state.state_id else 0.0
            )
            for item in context.observations
        }


class ParameterRewardHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return float(context.parameters["action_rewards"][context.action.id])


class DepthRewardHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return float(
            context.parameters["depth_rewards"][str(context.depth)][context.action.id]
        )


class ValueOfInformationRewardHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        if context.depth == 0:
            return -0.1 if context.action.id == "inspect" else 0.0
        if context.action.id == "act-a":
            return 2.0 if context.state.state_id == "active" else 0.0
        if context.action.id == "act-b":
            return 2.0 if context.state.state_id == "ready" else 0.0
        raise AssertionError("unexpected future planning action")


class StaticReactiveScoreHook:
    def __init__(self, scores):
        self.scores = dict(scores)
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return dict(self.scores)


class InvalidDistributionHook:
    def __init__(self, mode: str):
        self.mode = mode
        self.calls = []

    def _bad(self, ids):
        ids = tuple(ids)
        if self.mode == "nonmapping":
            return []
        if self.mode == "missing":
            return {item: 1.0 / max(1, len(ids) - 1) for item in ids[:-1]}
        if self.mode == "extra":
            result = {item: 1.0 / len(ids) for item in ids}
            result["extra"] = 0.0
            return result
        if self.mode == "bool":
            return {item: (True if index == 0 else 0.0) for index, item in enumerate(ids)}
        if self.mode == "nan":
            return {item: (math.nan if index == 0 else 0.0) for index, item in enumerate(ids)}
        if self.mode == "inf":
            return {item: (math.inf if index == 0 else 0.0) for index, item in enumerate(ids)}
        if self.mode == "negative":
            return {item: (-0.1 if index == 0 else 1.1 if index == 1 else 0.0) for index, item in enumerate(ids)}
        if self.mode == "nonunit":
            return {item: 0.1 for item in ids}
        raise AssertionError(self.mode)

    def __call__(self, context):
        self.calls.append(context)
        if hasattr(context, "candidate_next_states"):
            return self._bad(state.state_id for state in context.candidate_next_states)
        return self._bad(item.observation_id for item in context.observations)


class InvalidRewardHook:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return self.value


class RaisingHook:
    def __init__(self, label: str):
        self.label = label
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        raise RuntimeError(f"planning {self.label} boom")


class NarrativeRuntimePlanningTests(unittest.TestCase):
    def require_planning(self) -> None:
        if _PLANNING_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime planning boundary is missing: "
                f"{_PLANNING_IMPORT_ERROR}"
            )

    def make_model(
        self,
        *,
        joint=None,
        transition=None,
        observation=None,
        reward=None,
        planning_cells=None,
        observation_cells=(),
        hidden_states=None,
        observations=None,
        schedule=None,
        discount: float = 1.0,
        beta: float = 2.0,
        parameters=None,
        belief_model=None,
    ):
        self.require_planning()
        if planning_cells is None:
            planning_cells = (phase_cell(),)
        if hidden_states is None:
            hidden_states = (
                PlanningHiddenState(
                    "active",
                    {phase_cell(): TypedValue("PhaseState", "active")},
                ),
                PlanningHiddenState(
                    "ready",
                    {phase_cell(): TypedValue("PhaseState", "ready")},
                ),
            )
        if observations is None:
            observations = (PlanningObservation("none", {}),)
        if schedule is None:
            schedule = (("a1-active", "a1-ready"),)
        if parameters is None:
            parameters = {
                "action_rewards": {"a1-active": 2.0, "a1-ready": 0.0},
                "nested": {"flags": [True, None]},
            }
        if belief_model is None:
            belief_model = make_runtime_belief_model()
        return RuntimePlanningDecisionModelSpec(
            "runtime-planning",
            "1",
            ("phase-choice",),
            tuple(planning_cells),
            tuple(observation_cells),
            belief_model,
            tuple(hidden_states),
            tuple(observations),
            tuple(tuple(row) for row in schedule),
            discount,
            beta,
            parameters,
            joint or ProductCouplingHook(),
            transition or IdentityTransitionHook(),
            observation or NoInformationObservationHook(),
            reward or ParameterRewardHook(),
        )

    def _single_goal_intentional_model(self, scores, *, beta: float):
        cell = phase_cell()
        ready = _value_hash(TypedValue("PhaseState", "ready"))
        active = _value_hash(TypedValue("PhaseState", "active"))
        goals = tuple(
            GoalSpec(
                goal_id,
                1.0,
                {cell: 1.0},
                {cell: {ready: 0.0, active: 0.0}},
            )
            for goal_id in ("left", "right")
        )
        return RuntimeIntentionalDecisionModelSpec(
            "science-intentional",
            "1",
            ("phase-choice",),
            make_runtime_belief_model(),
            GoalModelSpec("science-goal", "1", 1.0, goals),
            ChoiceModelSpec(
                "science-choice",
                "1",
                beta,
                {goal.goal_id: dict(scores) for goal in goals},
            ),
        )

    def _reactive_model(self, scores, *, beta: float):
        return RuntimeReactiveDecisionModelSpec(
            "science-reactive",
            "1",
            ("phase-choice",),
            (phase_cell(),),
            {},
            beta,
            StaticReactiveScoreHook(scores),
        )

    def _story_with_phase_actions(self, story, specs):
        decisions = []
        for decision in story.decisions:
            if decision.id != "d-a1-phase":
                decisions.append(decision)
                continue
            actions = tuple(
                ActionOption(
                    action_id,
                    "actor-phase-action",
                    {"phase": TypedValue("PhaseState", phase)},
                )
                for action_id, phase in specs
            )
            decisions.append(replace(decision, actions=actions))
        return replace(story, decisions=tuple(decisions))

    def test_hidden_state_observation_and_belief_records_are_canonical(self):
        self.require_planning()
        active = PlanningHiddenState(
            "active", {phase_cell(): TypedValue("PhaseState", "active")}
        )
        none = PlanningObservation("none", {})
        belief = PlanningBeliefState({"ready": 0.25, "active": 0.75})
        self.assertEqual(active.state_id, "active")
        self.assertEqual(none.observation_id, "none")
        self.assertEqual(tuple(belief.probabilities), ("active", "ready"))
        self.assertEqual(math.fsum(belief.probabilities.values()), 1.0)
        with self.assertRaises((TypeError, ValueError)):
            PlanningHiddenState("", {phase_cell(): TypedValue("PhaseState", "active")})
        with self.assertRaises((TypeError, ValueError)):
            PlanningObservation("", {})
        for bad in (
            {"active": True, "ready": 0.0},
            {"active": math.nan, "ready": 1.0},
            {"active": -0.1, "ready": 1.1},
            {"active": 0.4, "ready": 0.4},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    PlanningBeliefState(bad)

    def test_model_parameters_and_identity_bind_all_planning_assumptions(self):
        self.require_planning()
        first = self.make_model()
        reordered = self.make_model(
            parameters={
                "nested": {"flags": (True, None)},
                "action_rewards": {"a1-ready": 0.0, "a1-active": 2.0},
            },
            schedule=(("a1-ready", "a1-active"),),
        )
        changed_beta = replace(first, beta=3.0)
        changed_discount = replace(first, discount=0.5)
        self.assertEqual(first.content_hash, reordered.content_hash)
        self.assertNotEqual(first.content_hash, changed_beta.content_hash)
        self.assertNotEqual(first.content_hash, changed_discount.content_hash)
        payload = first.to_dict()
        self.assertEqual(
            payload["softmax_implementation_identity"],
            measure_implementation(finite_softmax).manifest_identity(),
        )
        self.assertEqual(
            payload["runtime_implementation_identity"],
            measure_implementation(RuntimePlanningDecisionModelSpec).manifest_identity(),
        )
        for key in (
            "joint_belief_hook_identity",
            "transition_hook_identity",
            "observation_hook_identity",
            "reward_hook_identity",
        ):
            self.assertIn(key, payload)

    def test_model_parameters_accept_only_canonical_values(self):
        self.require_planning()
        accepted = {
            "none": None,
            "bool": True,
            "int": 3,
            "str": "x",
            "float": 1.25,
            "nested": {"b": [1, 2], "a": (False, None)},
        }
        model = self.make_model(parameters=accepted)
        self.assertEqual(model.parameters["nested"]["a"], (False, None))
        self.assertEqual(model.parameters["nested"]["b"], (1, 2))
        for parameters in (
            {"x": math.nan},
            {"x": math.inf},
            {"x": b"bytes"},
            {"x": {1, 2}},
            {"x": object()},
            {"x": lambda: None},
            {"": 1},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises((TypeError, ValueError)):
                    self.make_model(parameters=parameters)

    def test_public_signature_excludes_world_rng_and_explicit_step(self):
        self.require_planning()
        parameters = inspect.signature(run_runtime_planning_decision).parameters
        self.assertEqual(
            tuple(parameters),
            ("story", "domain", "decision_id", "ledger", "model"),
        )

    def test_context_and_audit_records_are_frozen_and_self_validating(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        model = self.make_model()
        belief = model.belief_model
        self.assertIsNotNone(belief)
        state = model.hidden_states[0]
        action = next(item for item in story.decisions[0].actions if item.id == "a1-active")
        observation = model.observations[0]
        planning_belief = PlanningBeliefState({"active": 0.5, "ready": 0.5})
        RuntimePlanningBeliefContext({}, model.hidden_states, model.parameters)
        PlanningTransitionContext(0, state, action, model.hidden_states, model.parameters)
        PlanningObservationContext(0, state, action, model.observations, model.parameters)
        PlanningRewardContext(0, state, action, state, model.parameters)
        value = PlanningValueRecord(0, planning_belief.content_hash, action.id, 1.0, 0.5, 1.5)
        update = PlanningBeliefUpdate(0, planning_belief.content_hash, action.id, observation.observation_id, 1.0, planning_belief)
        self.assertEqual(value.total_value, 1.5)
        self.assertEqual(update.posterior, planning_belief)
        for call in (
            lambda: PlanningTransitionContext(-1, state, action, model.hidden_states, model.parameters),
            lambda: PlanningValueRecord(-1, planning_belief.content_hash, action.id, 1.0, 0.0, 1.0),
            lambda: PlanningValueRecord(0, planning_belief.content_hash, "", 1.0, 0.0, 1.0),
            lambda: PlanningValueRecord(0, planning_belief.content_hash, action.id, math.nan, 0.0, 1.0),
            lambda: PlanningBeliefUpdate(0, planning_belief.content_hash, action.id, observation.observation_id, 0.0, planning_belief),
        ):
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()

    def test_preflight_rejects_story_ledger_context_cutoff_and_schedule_before_hooks(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        cases = []

        joint = ProductCouplingHook()
        transition = IdentityTransitionHook()
        observation = NoInformationObservationHook()
        reward = ParameterRewardHook()
        cases.append((story, _forge(ledger, source_story_hash=_hash("wrong-story")), self.make_model(joint=joint, transition=transition, observation=observation, reward=reward)))

        joint2 = ProductCouplingHook()
        transition2 = IdentityTransitionHook()
        observation2 = NoInformationObservationHook()
        reward2 = ParameterRewardHook()
        cases.append((story, ledger, self.make_model(joint=joint2, transition=transition2, observation=observation2, reward=reward2, planning_cells=(alert_cell(),), hidden_states=(PlanningHiddenState("false", {alert_cell(): TypedValue("AlertState", False)}), PlanningHiddenState("true", {alert_cell(): TypedValue("AlertState", True)})))))

        cutoff = runtime_evidence_ledger_from_story(story, domain, at_time=7)
        joint3 = ProductCouplingHook()
        transition3 = IdentityTransitionHook()
        observation3 = NoInformationObservationHook()
        reward3 = ParameterRewardHook()
        cases.append((story, cutoff, self.make_model(joint=joint3, transition=transition3, observation=observation3, reward=reward3)))

        joint4 = ProductCouplingHook()
        transition4 = IdentityTransitionHook()
        observation4 = NoInformationObservationHook()
        reward4 = ParameterRewardHook()
        cases.append((story, ledger, self.make_model(joint=joint4, transition=transition4, observation=observation4, reward=reward4, schedule=(("a1-active",),))))

        joint5 = ProductCouplingHook()
        transition5 = IdentityTransitionHook()
        observation5 = NoInformationObservationHook()
        reward5 = ParameterRewardHook()
        cases.append((story, ledger, self.make_model(joint=joint5, transition=transition5, observation=observation5, reward=reward5, schedule=(("a1-active", "a1-ready"), ("a1-active", "missing")))))

        for case_story, case_ledger, model in cases:
            with self.subTest(model=model.action_schedule):
                with self.assertRaises(RuntimePlanningDecisionResolutionError):
                    run_runtime_planning_decision(case_story, domain, "d-a1-phase", case_ledger, model)
                for hook in (model.joint_belief_hook, model.transition_hook, model.observation_hook, model.reward_hook):
                    self.assertEqual(hook.calls, [])

    def test_joint_belief_must_be_a_coupling_of_runtime_cell_marginals(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        decisions = tuple(
            replace(decision, context_cells=(phase_cell(), alert_cell()))
            if decision.id == "d-a1-phase"
            else decision
            for decision in story.decisions
        )
        story = replace(story, decisions=decisions)
        ledger = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        states = tuple(
            PlanningHiddenState(
                f"{phase}-{'on' if alert else 'off'}",
                {
                    phase_cell(): TypedValue("PhaseState", phase),
                    alert_cell(): TypedValue("AlertState", alert),
                },
            )
            for phase in ("active", "ready")
            for alert in (False, True)
        )
        valid_transition = IdentityTransitionHook()
        valid_observation = NoInformationObservationHook()
        valid_reward = ParameterRewardHook()
        valid = self.make_model(
            planning_cells=(phase_cell(), alert_cell()),
            hidden_states=states,
            joint=ProductCouplingHook(),
            transition=valid_transition,
            observation=valid_observation,
            reward=valid_reward,
        )
        result = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, valid)
        self.assertEqual(set(result.planning_belief.probabilities), {item.state_id for item in states})

        transition = IdentityTransitionHook()
        observation = NoInformationObservationHook()
        reward = ParameterRewardHook()
        invalid = self.make_model(
            planning_cells=(phase_cell(), alert_cell()),
            hidden_states=states,
            joint=ConcentratedCouplingHook(),
            transition=transition,
            observation=observation,
            reward=reward,
        )
        with self.assertRaises(RuntimePlanningDecisionResolutionError):
            run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, invalid)
        self.assertEqual(transition.calls, [])
        self.assertEqual(observation.calls, [])
        self.assertEqual(reward.calls, [])

    def test_hook_contexts_are_sanitized_and_module_has_no_world_capability(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        model = self.make_model()
        run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        forbidden = {
            "story", "domain", "ledger", "world", "world_state", "evidence",
            "provenance", "runtime_evidence_history",
        }
        contexts = (
            model.joint_belief_hook.calls
            + model.transition_hook.calls
            + model.observation_hook.calls
            + model.reward_hook.calls
        )
        self.assertTrue(contexts)
        for context in contexts:
            for name in forbidden:
                self.assertFalse(hasattr(context, name), (type(context).__name__, name))
        source = inspect.getsource(runtime_planning_module)
        for banned in (
            "narrative.world",
            "narrative.observation_projection",
            "narrative.simulation",
            "narrative.runtime_intention",
            "narrative.runtime_reactive",
        ):
            self.assertNotIn(banned, source)

    def test_transition_observation_and_reward_hook_schemas_fail_typed(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        for mode in ("nonmapping", "missing", "extra", "bool", "nan", "inf", "negative", "nonunit"):
            with self.subTest(kind="transition", mode=mode):
                model = self.make_model(transition=InvalidDistributionHook(mode))
                with self.assertRaises(RuntimePlanningDecisionResolutionError) as caught:
                    run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
                self.assertIsNotNone(caught.exception.__cause__)
        two_observations = (
            PlanningObservation("active", {phase_cell(): TypedValue("PhaseState", "active")}),
            PlanningObservation("ready", {phase_cell(): TypedValue("PhaseState", "ready")}),
        )
        for mode in ("nonmapping", "missing", "extra", "bool", "nan", "inf", "negative", "nonunit"):
            with self.subTest(kind="observation", mode=mode):
                model = self.make_model(
                    observation_cells=(phase_cell(),),
                    observations=two_observations,
                    schedule=(("a1-active", "a1-ready"), ("a1-active", "a1-ready")),
                    observation=InvalidDistributionHook(mode),
                )
                with self.assertRaises(RuntimePlanningDecisionResolutionError) as caught:
                    run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
                self.assertIsNotNone(caught.exception.__cause__)
        for bad in (True, math.nan, math.inf):
            with self.subTest(kind="reward", bad=bad):
                model = self.make_model(reward=InvalidRewardHook(bad))
                with self.assertRaises(RuntimePlanningDecisionResolutionError) as caught:
                    run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
                self.assertIsNotNone(caught.exception.__cause__)

    def test_zero_evidence_observation_is_skipped_and_no_information_preserves_prediction(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        observations = (
            PlanningObservation("always", {phase_cell(): TypedValue("PhaseState", "active")}),
            PlanningObservation("never", {phase_cell(): TypedValue("PhaseState", "ready")}),
        )
        model = self.make_model(
            observation_cells=(phase_cell(),),
            observations=observations,
            schedule=(("a1-active", "a1-ready"), ("a1-active", "a1-ready")),
            observation=NoInformationObservationHook(),
        )
        result = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        self.assertTrue(result.belief_updates)
        self.assertTrue(all(update.observation_id == "always" for update in result.belief_updates))
        for update in result.belief_updates:
            self.assertEqual(update.posterior.to_dict(), result.planning_belief.to_dict())

    def test_finite_horizon_soft_bellman_matches_hand_computed_values(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        beta = 1.0
        discount = 0.5
        parameters = {
            "depth_rewards": {
                "0": {"a1-active": 1.0, "a1-ready": 0.0},
                "1": {"a1-active": 2.0, "a1-ready": 0.0},
            }
        }
        model = self.make_model(
            beta=beta,
            discount=discount,
            parameters=parameters,
            schedule=(("a1-active", "a1-ready"), ("a1-active", "a1-ready")),
            reward=DepthRewardHook(),
        )
        result = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        terminal_policy = finite_softmax({"a1-active": 2.0, "a1-ready": 0.0}, beta=beta)
        terminal_value = math.fsum(
            terminal_policy[action] * score
            for action, score in {"a1-active": 2.0, "a1-ready": 0.0}.items()
        )
        expected_future = discount * terminal_value
        roots = {
            record.action_id: record
            for record in result.value_records
            if record.depth == 0 and record.belief_hash == result.planning_belief.content_hash
        }
        self.assertAlmostEqual(roots["a1-active"].expected_immediate_reward, 1.0, places=12)
        self.assertAlmostEqual(roots["a1-ready"].expected_immediate_reward, 0.0, places=12)
        for record in roots.values():
            self.assertAlmostEqual(record.expected_future_value, expected_future, places=12)
        self.assertAlmostEqual(result.action_values["a1-active"], 1.0 + expected_future, places=12)
        self.assertAlmostEqual(result.action_values["a1-ready"], expected_future, places=12)
        self.assertNotAlmostEqual(expected_future, discount * 2.0, places=12)

    def test_shared_softmax_produces_complete_root_policy_and_lexical_map(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        model = self.make_model(parameters={"action_rewards": {"a1-active": 1.0, "a1-ready": 1.0}})
        result = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        self.assertEqual(result.action_policy, finite_softmax(result.action_values, beta=model.beta))
        self.assertEqual(set(result.action_policy), {"a1-active", "a1-ready"})
        self.assertEqual(result.selected_action, "a1-active")
        reordered = self.make_model(
            parameters={"action_rewards": {"a1-ready": 1.0, "a1-active": 1.0}},
            schedule=(("a1-ready", "a1-active"),),
        )
        replay = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, reordered)
        self.assertEqual(result.action_policy, replay.action_policy)
        self.assertEqual(result.selected_action, replay.selected_action)

    def test_hook_exceptions_are_wrapped_with_typed_causes(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        two_observations = (
            PlanningObservation("active", {phase_cell(): TypedValue("PhaseState", "active")}),
            PlanningObservation("ready", {phase_cell(): TypedValue("PhaseState", "ready")}),
        )
        cases = (
            ("joint", dict(joint=RaisingHook("joint"))),
            ("transition", dict(transition=RaisingHook("transition"))),
            ("observation", dict(observation=RaisingHook("observation"), observation_cells=(phase_cell(),), observations=two_observations, schedule=(("a1-active", "a1-ready"), ("a1-active", "a1-ready")))),
            ("reward", dict(reward=RaisingHook("reward"))),
        )
        for label, kwargs in cases:
            with self.subTest(label=label):
                model = self.make_model(**kwargs)
                with self.assertRaises(RuntimePlanningDecisionResolutionError) as caught:
                    run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
                self.assertIsInstance(caught.exception.__cause__, RuntimeError)
                self.assertEqual(str(caught.exception.__cause__), f"planning {label} boom")

    def test_result_and_trace_forgery_is_rejected(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        model = self.make_model(schedule=(("a1-active", "a1-ready"), ("a1-active", "a1-ready")))
        result = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        self.assertEqual(result.model_hash, model.content_hash)
        root = next(record for record in result.value_records if record.depth == 0)
        bad_root = _forge(root, total_value=root.total_value + 1.0)
        bad_updates = result.belief_updates
        if bad_updates:
            bad_update = _forge(bad_updates[0], observation_probability=0.0)
            forged_updates = (bad_update,) + bad_updates[1:]
        else:
            forged_updates = bad_updates
        for changes in (
            {"ledger_hash": _hash("wrong-ledger")},
            {"value_records": (bad_root,) + tuple(item for item in result.value_records if item is not root)},
            {"belief_updates": forged_updates} if bad_updates else {"selected_action": "missing"},
            {"selected_action": "a1-ready" if result.selected_action == "a1-active" else "a1-active"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    RuntimePlanningDecisionResult(**{**result.__dict__, **changes})

    def test_fixed_inputs_replay_to_exact_result_and_content_hash(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        model = self.make_model(schedule=(("a1-active", "a1-ready"), ("a1-active", "a1-ready")))
        first = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        second = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)

    def test_horizon_one_policy_is_exactly_equivalent_across_planning_reactive_and_intentional(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        scores = {"a1-active": 2.0, "a1-ready": 0.0}
        beta = 2.0
        planning = run_runtime_planning_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self.make_model(parameters={"action_rewards": scores}, beta=beta),
        )
        reactive = run_runtime_reactive_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self._reactive_model(scores, beta=beta),
        )
        intentional = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self._single_goal_intentional_model(scores, beta=beta),
        )
        self.assertEqual(planning.action_policy, reactive.action_policy)
        self.assertEqual(planning.action_policy, intentional.action_policy)
        self.assertEqual(planning.selected_action, reactive.selected_action)
        self.assertEqual(planning.selected_action, intentional.selected_action)

    def test_state_independent_observation_has_zero_bayesian_information_gain(self):
        self.require_planning()
        domain, story, ledger = empty_runtime_case()
        observations = (
            PlanningObservation("active", {phase_cell(): TypedValue("PhaseState", "active")}),
            PlanningObservation("ready", {phase_cell(): TypedValue("PhaseState", "ready")}),
        )
        model = self.make_model(
            observation_cells=(phase_cell(),),
            observations=observations,
            observation=UniformObservationHook(),
            schedule=(("a1-active", "a1-ready"), ("a1-active", "a1-ready")),
        )
        result = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, model)
        self.assertTrue(result.belief_updates)
        for update in result.belief_updates:
            self.assertEqual(update.posterior.to_dict(), result.planning_belief.to_dict())

    def test_value_of_information_separates_planning_from_reactive_and_intentional(self):
        self.require_planning()
        domain, story, _ = empty_runtime_case()
        story = self._story_with_phase_actions(
            story,
            (("inspect", "ready"), ("act-a", "active"), ("act-b", "ready")),
        )
        ledger = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        observations = (
            PlanningObservation("active", {phase_cell(): TypedValue("PhaseState", "active")}),
            PlanningObservation("ready", {phase_cell(): TypedValue("PhaseState", "ready")}),
        )
        schedule = (("inspect", "act-a", "act-b"), ("act-a", "act-b"))
        common = dict(
            observation_cells=(phase_cell(),),
            observations=observations,
            schedule=schedule,
            beta=8.0,
            discount=1.0,
            parameters={},
            reward=ValueOfInformationRewardHook(),
        )
        informative_model = self.make_model(observation=InformativeObservationHook(), **common)
        informative = run_runtime_planning_decision(story, domain, "d-a1-phase", ledger, informative_model)
        no_info = run_runtime_planning_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self.make_model(observation=UniformObservationHook(), **common),
        )

        myopic_scores = {"inspect": -0.1, "act-a": 0.0, "act-b": 0.0}
        reactive = run_runtime_reactive_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self._reactive_model(myopic_scores, beta=8.0),
        )
        intentional = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self._single_goal_intentional_model(myopic_scores, beta=8.0),
        )
        self.assertNotEqual(informative.action_policy, reactive.action_policy)
        self.assertNotEqual(informative.action_policy, intentional.action_policy)
        self.assertGreater(informative.action_values["inspect"], informative.action_values["act-a"])
        self.assertGreater(informative.action_values["inspect"], informative.action_values["act-b"])
        self.assertLess(no_info.action_values["inspect"], no_info.action_values["act-a"])
        self.assertLess(no_info.action_values["inspect"], no_info.action_values["act-b"])
        self.assertGreater(informative.action_policy["inspect"], no_info.action_policy["inspect"])


if __name__ == "__main__":
    unittest.main()
