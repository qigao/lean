from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.ir import EntityRef, StateCellRef, TypedValue
from narrative_dynamics.narrative.uncertain import UncertainBeliefModelSpec
from tests.narrative_test_support import make_test_domain, make_test_story, target_cell


_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.intention import (
        ChoiceModelSpec,
        ChoiceResolutionError,
        GoalModelSpec,
        GoalResolutionError,
        GoalSpec,
        GoalState,
        IntentionalDecisionModelSpec,
        IntentionalDecisionResolutionError,
        IntentionalDecisionResult,
        run_intentional_decision,
    )
except ImportError as error:
    _IMPORT_ERROR = error


def _value_hash(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def _health_value(value: str) -> TypedValue:
    return TypedValue("HealthState", value)


def _health_hashes() -> dict[str, str]:
    return {
        value: _value_hash(_health_value(value))
        for value in ("failed", "healthy", "recovered")
    }


class _PriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        del agent_id, cell
        configured = parameters["prior"]
        return {_value_hash(h): float(configured[h.value]) for h in hypotheses}


class _LikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        del agent_id
        strength = float(parameters["strength"])
        if evidence.relation == "clear" or evidence.value is None:
            return {_value_hash(h): 1.0 for h in hypotheses}
        if evidence.relation == "equals":
            return {
                _value_hash(h): strength if h == evidence.value else (1.0 - strength)
                for h in hypotheses
            }
        return {
            _value_hash(h): (1.0 - strength) if h == evidence.value else strength
            for h in hypotheses
        }


def _belief_model(
    *,
    prior: dict[str, float] | None = None,
    strength: float = 0.9,
    model_id: str = "health-belief",
) -> UncertainBeliefModelSpec:
    return UncertainBeliefModelSpec(
        model_id=model_id,
        version="1",
        parameters={
            "prior": prior
            or {"failed": 0.50, "healthy": 0.25, "recovered": 0.25},
            "strength": strength,
        },
        prior_hook=_PriorHook(),
        likelihood_hook=_LikelihoodHook(),
    )


def _goal_spec(
    goal_id: str,
    values: dict[str, float],
    *,
    pressure: float = 1.0,
    cost: float = 0.0,
    risk: float = 0.0,
    cell: StateCellRef | None = None,
    cell_weights: dict[StateCellRef, float] | None = None,
    instrumentality: dict[StateCellRef, dict[str, float]] | None = None,
):
    cell = target_cell() if cell is None else cell
    hashes = _health_hashes()
    return GoalSpec(
        goal_id=goal_id,
        pressure=pressure,
        cost=cost,
        risk=risk,
        cell_weights=cell_weights or {cell: 1.0},
        instrumentality=instrumentality
        or {cell: {hashes[name]: score for name, score in values.items()}},
    )


def _goal_model(
    *,
    beta_goal: float = 2.0,
    restore_values: dict[str, float] | None = None,
    avoid_values: dict[str, float] | None = None,
    restore_pressure: float = 1.0,
    avoid_pressure: float = 1.0,
    reverse: bool = False,
):
    restore = _goal_spec(
        "restore-service",
        restore_values or {"failed": 1.0, "healthy": 0.2, "recovered": 0.0},
        pressure=restore_pressure,
    )
    avoid = _goal_spec(
        "avoid-work",
        avoid_values or {"failed": 0.0, "healthy": 0.5, "recovered": 1.0},
        pressure=avoid_pressure,
    )
    goals = (avoid, restore) if reverse else (restore, avoid)
    return GoalModelSpec("health-goals", "1", beta_goal, goals)


def _choice_model(
    *,
    beta_action: float = 2.0,
    values: dict[str, dict[str, float]] | None = None,
    reverse: bool = False,
):
    values = values or {
        "restore-service": {"restart": 2.0, "leave": -1.0},
        "avoid-work": {"restart": -1.0, "leave": 2.0},
    }
    if reverse:
        values = {
            goal: dict(reversed(tuple(actions.items())))
            for goal, actions in reversed(tuple(values.items()))
        }
    return ChoiceModelSpec("health-choice", "1", beta_action, values)


def _model(
    *,
    belief=None,
    goal=None,
    choice=None,
    supported_decision_types=("service-response",),
    model_id="health-intention",
):
    return IntentionalDecisionModelSpec(
        model_id=model_id,
        version="1",
        supported_decision_types=supported_decision_types,
        belief_model=belief or _belief_model(),
        goal_model=goal or _goal_model(),
        choice_model=choice or _choice_model(),
    )


def _run(*, story=None, model=None):
    story = make_test_story() if story is None else story
    return run_intentional_decision(
        story,
        make_test_domain(),
        "d1",
        _model() if model is None else model,
    )


class NarrativeIntentionalDecisionTests(unittest.TestCase):
    def require_intention(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"narrative intentional decision module is missing: {_IMPORT_ERROR}")

    def test_admitted_evidence_changes_belief_goal_and_action(self) -> None:
        self.require_intention()
        model = _model()
        no_receive = _run(story=make_test_story(receive=False), model=model)
        received = _run(story=make_test_story(receive=True), model=model)

        self.assertNotEqual(no_receive.belief_state.to_dict(), received.belief_state.to_dict())
        self.assertNotEqual(dict(no_receive.goal_state.policy), dict(received.goal_state.policy))
        self.assertNotEqual(dict(no_receive.action_policy), dict(received.action_policy))
        self.assertGreater(
            received.goal_state.policy["avoid-work"],
            no_receive.goal_state.policy["avoid-work"],
        )
        self.assertGreater(received.action_policy["leave"], no_receive.action_policy["leave"])

    def test_unobserved_objective_change_does_not_leak_into_intention(self) -> None:
        self.require_intention()
        story = make_test_story(receive=False, bob_observes_failure=False)
        event = story.events[0]
        arguments = dict(event.arguments)
        arguments["health"] = _health_value("healthy")
        changed = replace(
            story,
            events=(replace(event, arguments=arguments),) + story.events[1:],
        )
        model = _model()

        baseline = _run(story=story, model=model)
        counterfactual = _run(story=changed, model=model)

        self.assertEqual(baseline.belief_state.to_dict(), counterfactual.belief_state.to_dict())
        self.assertEqual(baseline.goal_state.to_dict(), counterfactual.goal_state.to_dict())
        self.assertEqual(
            {
                g: dict(p) for g, p in baseline.conditional_action_policies.items()
            },
            {
                g: dict(p) for g, p in counterfactual.conditional_action_policies.items()
            },
        )
        self.assertEqual(dict(baseline.action_scores), dict(counterfactual.action_scores))
        self.assertEqual(dict(baseline.action_policy), dict(counterfactual.action_policy))
        self.assertEqual(baseline.selected_action, counterfactual.selected_action)

    def test_same_belief_different_goal_model_changes_only_downstream_layers(self) -> None:
        self.require_intention()
        story = make_test_story()
        belief = _belief_model()
        choice = _choice_model()
        first = _run(
            story=story,
            model=_model(belief=belief, goal=_goal_model(), choice=choice),
        )
        second = _run(
            story=story,
            model=_model(
                belief=belief,
                goal=_goal_model(restore_pressure=3.0),
                choice=choice,
            ),
        )
        self.assertEqual(first.belief_state.to_dict(), second.belief_state.to_dict())
        self.assertNotEqual(dict(first.goal_state.policy), dict(second.goal_state.policy))
        self.assertNotEqual(dict(first.action_policy), dict(second.action_policy))

    def test_same_belief_and_goal_different_choice_model_preserves_goal_state(self) -> None:
        self.require_intention()
        story = make_test_story()
        belief = _belief_model()
        goal = _goal_model()
        first = _run(
            story=story,
            model=_model(belief=belief, goal=goal, choice=_choice_model(beta_action=1.0)),
        )
        second = _run(
            story=story,
            model=_model(belief=belief, goal=goal, choice=_choice_model(beta_action=5.0)),
        )
        self.assertEqual(first.belief_state.to_dict(), second.belief_state.to_dict())
        self.assertEqual(first.goal_state.to_dict(), second.goal_state.to_dict())
        self.assertNotEqual(
            {g: dict(p) for g, p in first.conditional_action_policies.items()},
            {g: dict(p) for g, p in second.conditional_action_policies.items()},
        )
        self.assertNotEqual(dict(first.action_policy), dict(second.action_policy))

    def test_beta_goal_and_beta_action_control_distinct_layers(self) -> None:
        self.require_intention()
        story = make_test_story()
        belief = _belief_model()
        choice = _choice_model(beta_action=2.0)
        low_goal = _run(
            story=story,
            model=_model(belief=belief, goal=_goal_model(beta_goal=0.5), choice=choice),
        )
        high_goal = _run(
            story=story,
            model=_model(belief=belief, goal=_goal_model(beta_goal=5.0), choice=choice),
        )
        leader = max(low_goal.goal_state.policy, key=low_goal.goal_state.policy.get)
        self.assertGreater(high_goal.goal_state.policy[leader], low_goal.goal_state.policy[leader])
        self.assertEqual(
            {g: dict(p) for g, p in low_goal.conditional_action_policies.items()},
            {g: dict(p) for g, p in high_goal.conditional_action_policies.items()},
        )

        goal = _goal_model(beta_goal=2.0)
        low_action = _run(
            story=story,
            model=_model(belief=belief, goal=goal, choice=_choice_model(beta_action=0.5)),
        )
        high_action = _run(
            story=story,
            model=_model(belief=belief, goal=goal, choice=_choice_model(beta_action=5.0)),
        )
        self.assertEqual(low_action.goal_state.to_dict(), high_action.goal_state.to_dict())
        for goal_id in low_action.conditional_action_policies:
            self.assertGreater(
                max(high_action.conditional_action_policies[goal_id].values()),
                max(low_action.conditional_action_policies[goal_id].values()),
            )

    def test_hypothesis_independent_instrumentality_blocks_belief_effect(self) -> None:
        self.require_intention()
        goal = _goal_model(
            restore_values={"failed": 0.7, "healthy": 0.7, "recovered": 0.7},
            avoid_values={"failed": 0.3, "healthy": 0.3, "recovered": 0.3},
        )
        model = _model(goal=goal)
        no_receive = _run(story=make_test_story(receive=False), model=model)
        received = _run(story=make_test_story(receive=True), model=model)

        self.assertNotEqual(no_receive.belief_state.to_dict(), received.belief_state.to_dict())
        self.assertEqual(dict(no_receive.goal_state.scores), dict(received.goal_state.scores))
        self.assertEqual(dict(no_receive.goal_state.policy), dict(received.goal_state.policy))
        self.assertEqual(
            {g: dict(p) for g, p in no_receive.conditional_action_policies.items()},
            {g: dict(p) for g, p in received.conditional_action_policies.items()},
        )
        self.assertEqual(dict(no_receive.action_policy), dict(received.action_policy))

    def test_action_policy_is_latent_goal_mixture_not_softmax_of_action_scores(self) -> None:
        self.require_intention()
        choice = _choice_model(
            beta_action=1.0,
            values={
                "restore-service": {"restart": 4.0, "leave": -4.0},
                "avoid-work": {"restart": -4.0, "leave": 4.0},
            },
        )
        model = _model(goal=_goal_model(beta_goal=0.7), choice=choice)
        result = _run(model=model)
        expected = {
            action: math.fsum(
                result.goal_state.policy[goal]
                * result.conditional_action_policies[goal][action]
                for goal in sorted(result.goal_state.policy)
            )
            for action in sorted(result.action_policy)
        }
        for action in expected:
            self.assertAlmostEqual(result.action_policy[action], expected[action], places=12)
        wrong = finite_softmax(result.action_scores, beta=model.choice_model.beta_action)
        self.assertTrue(
            any(abs(result.action_policy[action] - wrong[action]) > 1e-6 for action in expected)
        )

    def test_exact_ties_use_lexical_ids_and_ignore_input_order(self) -> None:
        self.require_intention()
        neutral = {"failed": 0.0, "healthy": 0.0, "recovered": 0.0}
        a_goal = _goal_spec("a-goal", neutral)
        z_goal = _goal_spec("z-goal", neutral)
        values = {
            "a-goal": {"restart": 0.0, "leave": 0.0},
            "z-goal": {"restart": 0.0, "leave": 0.0},
        }
        first = _model(
            goal=GoalModelSpec("ties", "1", 2.0, (a_goal, z_goal)),
            choice=ChoiceModelSpec("ties-choice", "1", 2.0, values),
        )
        second_values = {
            goal: dict(reversed(tuple(actions.items())))
            for goal, actions in reversed(tuple(values.items()))
        }
        second = _model(
            goal=GoalModelSpec("ties", "1", 2.0, (z_goal, a_goal)),
            choice=ChoiceModelSpec("ties-choice", "1", 2.0, second_values),
        )
        first_result = _run(model=first)
        second_result = _run(model=second)

        self.assertEqual(first_result.goal_state.selected_goal, "a-goal")
        self.assertEqual(first_result.selected_action, "leave")
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first_result.content_hash, second_result.content_hash)

    def test_goal_context_and_hypothesis_coverage_fail_closed(self) -> None:
        self.require_intention()
        cell = target_cell()
        extra = StateCellRef(EntityRef("ghost", "Service"), "service.health")
        hashes = _health_hashes()
        base_values = {hashes[name]: score for name, score in {
            "failed": 1.0,
            "healthy": 0.2,
            "recovered": 0.0,
        }.items()}

        bad_goals = (
            _goal_spec(
                "restore-service",
                {"failed": 1.0, "healthy": 0.2, "recovered": 0.0},
                cell_weights={extra: 1.0},
            ),
            _goal_spec(
                "restore-service",
                {"failed": 1.0, "healthy": 0.2, "recovered": 0.0},
                cell_weights={cell: 0.5, extra: 0.5},
                instrumentality={cell: base_values, extra: base_values},
            ),
            _goal_spec(
                "restore-service",
                {"failed": 1.0, "healthy": 0.2, "recovered": 0.0},
                instrumentality={cell: dict(tuple(base_values.items())[:-1])},
            ),
            _goal_spec(
                "restore-service",
                {"failed": 1.0, "healthy": 0.2, "recovered": 0.0},
                instrumentality={
                    cell: {**base_values, "sha256:" + "f" * 64: 0.0}
                },
            ),
        )
        avoid = _goal_spec(
            "avoid-work",
            {"failed": 0.0, "healthy": 0.5, "recovered": 1.0},
        )
        for bad in bad_goals:
            with self.subTest(goal=bad.to_dict()):
                model = _model(goal=GoalModelSpec("bad", "1", 2.0, (bad, avoid)))
                with self.assertRaises(GoalResolutionError):
                    _run(model=model)

    def test_choice_goal_and_action_coverage_fail_closed(self) -> None:
        self.require_intention()
        goal = _goal_model()
        with self.assertRaises(ChoiceResolutionError):
            IntentionalDecisionModelSpec(
                "missing-goal",
                "1",
                ("service-response",),
                _belief_model(),
                goal,
                ChoiceModelSpec(
                    "choice",
                    "1",
                    2.0,
                    {"restore-service": {"restart": 1.0, "leave": 0.0}},
                ),
            )
        with self.assertRaises(ChoiceResolutionError):
            IntentionalDecisionModelSpec(
                "extra-goal",
                "1",
                ("service-response",),
                _belief_model(),
                goal,
                ChoiceModelSpec(
                    "choice",
                    "1",
                    2.0,
                    {
                        "restore-service": {"restart": 1.0, "leave": 0.0},
                        "avoid-work": {"restart": 0.0, "leave": 1.0},
                        "invented": {"restart": 0.0, "leave": 0.0},
                    },
                ),
            )

        for values in (
            {
                "restore-service": {"restart": 1.0},
                "avoid-work": {"restart": 0.0, "leave": 1.0},
            },
            {
                "restore-service": {"restart": 1.0, "leave": 0.0, "wait": 0.0},
                "avoid-work": {"restart": 0.0, "leave": 1.0},
            },
        ):
            with self.subTest(values=values):
                model = _model(choice=ChoiceModelSpec("bad-actions", "1", 2.0, values))
                with self.assertRaises(ChoiceResolutionError):
                    _run(model=model)

    def test_invalid_numeric_configuration_fails_closed(self) -> None:
        self.require_intention()
        valid_values = {"failed": 1.0, "healthy": 0.2, "recovered": 0.0}
        for field in ("pressure", "cost", "risk"):
            for bad in (-1.0, float("nan"), float("inf")):
                with self.subTest(field=field, bad=bad):
                    kwargs = {field: bad}
                    with self.assertRaises(ValueError):
                        _goal_spec("bad", valid_values, **kwargs)
        for bad in (-1.0, float("nan"), float("inf")):
            with self.subTest(weight=bad):
                with self.assertRaises(ValueError):
                    _goal_spec(
                        "bad",
                        valid_values,
                        cell_weights={target_cell(): bad},
                    )
        hashes = _health_hashes()
        with self.assertRaises(ValueError):
            _goal_spec(
                "bad",
                valid_values,
                instrumentality={
                    target_cell(): {
                        hashes["failed"]: float("inf"),
                        hashes["healthy"]: 0.2,
                        hashes["recovered"]: 0.0,
                    }
                },
            )
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(beta_goal=bad):
                with self.assertRaises(ValueError):
                    GoalModelSpec(
                        "bad-goal-beta",
                        "1",
                        bad,
                        (
                            _goal_spec("a", valid_values),
                            _goal_spec("b", valid_values),
                        ),
                    )
            with self.subTest(beta_action=bad):
                with self.assertRaises(ValueError):
                    ChoiceModelSpec(
                        "bad-action-beta",
                        "1",
                        bad,
                        {"a": {"restart": 1.0}, "b": {"restart": 0.0}},
                    )
        with self.assertRaises(ValueError):
            ChoiceModelSpec(
                "bad-action-value",
                "1",
                1.0,
                {
                    "restore-service": {"restart": float("inf"), "leave": 0.0},
                    "avoid-work": {"restart": 0.0, "leave": 1.0},
                },
            )

        non_normalized = _goal_spec(
            "restore-service",
            valid_values,
            cell_weights={target_cell(): 0.5},
        )
        model = _model(
            goal=GoalModelSpec(
                "non-normalized",
                "1",
                2.0,
                (
                    non_normalized,
                    _goal_spec(
                        "avoid-work",
                        {"failed": 0.0, "healthy": 0.5, "recovered": 1.0},
                    ),
                ),
            )
        )
        with self.assertRaises(GoalResolutionError):
            _run(model=model)

    def test_invalid_derived_policies_are_typed(self) -> None:
        self.require_intention()
        with patch(
            "narrative_dynamics.narrative.intention.finite_softmax",
            return_value={"avoid-work": 1.1, "restore-service": -0.1},
        ):
            with self.assertRaises(GoalResolutionError):
                _run()

        valid_goal = {"avoid-work": 0.5, "restore-service": 0.5}
        invalid_action = {"leave": 1.1, "restart": -0.1}
        with patch(
            "narrative_dynamics.narrative.intention.finite_softmax",
            side_effect=[valid_goal, invalid_action, invalid_action],
        ):
            with self.assertRaises(ChoiceResolutionError):
                _run()

    def test_model_identity_binds_belief_goal_and_choice_configuration(self) -> None:
        self.require_intention()
        base = _model()
        changed_belief = _model(belief=_belief_model(strength=0.8))
        changed_goal = _model(goal=_goal_model(beta_goal=3.0))
        changed_goal_table = _model(
            goal=_goal_model(
                restore_values={"failed": 0.8, "healthy": 0.2, "recovered": 0.0}
            )
        )
        changed_choice = _model(choice=_choice_model(beta_action=3.0))
        changed_choice_table = _model(
            choice=_choice_model(
                values={
                    "restore-service": {"restart": 3.0, "leave": -1.0},
                    "avoid-work": {"restart": -1.0, "leave": 2.0},
                }
            )
        )

        self.assertNotEqual(base.belief_model.content_hash, changed_belief.belief_model.content_hash)
        self.assertNotEqual(base.content_hash, changed_belief.content_hash)
        self.assertNotEqual(base.goal_model.content_hash, changed_goal.goal_model.content_hash)
        self.assertNotEqual(base.content_hash, changed_goal.content_hash)
        self.assertNotEqual(base.goal_model.content_hash, changed_goal_table.goal_model.content_hash)
        self.assertNotEqual(base.content_hash, changed_goal_table.content_hash)
        self.assertNotEqual(base.choice_model.content_hash, changed_choice.choice_model.content_hash)
        self.assertNotEqual(base.content_hash, changed_choice.content_hash)
        self.assertNotEqual(base.choice_model.content_hash, changed_choice_table.choice_model.content_hash)
        self.assertNotEqual(base.content_hash, changed_choice_table.content_hash)

    def test_result_binds_exact_upstream_belief_payload(self) -> None:
        self.require_intention()
        model = _model()
        result = _run(model=model)
        self.assertIsInstance(result, IntentionalDecisionResult)
        self.assertIsInstance(result.goal_state, GoalState)
        self.assertEqual(
            result.goal_state.belief_state_hash,
            stable_content_hash(result.belief_state.to_dict()),
        )
        self.assertEqual(result.model_hash, model.content_hash)
        self.assertEqual(result.content_hash, stable_content_hash(result.to_dict()))

    def test_invalid_decision_resolution_is_typed(self) -> None:
        self.require_intention()
        with self.assertRaises(IntentionalDecisionResolutionError):
            run_intentional_decision(
                make_test_story(),
                make_test_domain(),
                "missing",
                _model(),
            )
        with self.assertRaises(IntentionalDecisionResolutionError):
            run_intentional_decision(
                make_test_story(),
                make_test_domain(),
                "d1",
                _model(supported_decision_types=("other-decision",)),
            )


if __name__ == "__main__":
    unittest.main()
