from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    ChoiceResolutionError,
    GoalModelSpec,
    GoalResolutionError,
    GoalSpec,
    GoalState,
    IntentionalDecisionModelSpec,
    run_intentional_decision,
)
from narrative_dynamics.narrative.ir import TypedValue
from narrative_dynamics.narrative.runtime_cognition import (
    RuntimeBeliefResolutionError,
    runtime_uncertain_belief_state,
)
from narrative_dynamics.narrative.runtime_perception import (
    runtime_evidence_ledger_from_story,
)
from tests.test_narrative_runtime_cognition import (
    SemanticRuntimeLikelihood,
    ZeroRuntimeLikelihood,
    empty_runtime_case,
    extend_ledger,
    make_runtime_belief_model,
    phase_cell,
)

_RUNTIME_INTENTION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_intention import (
        RuntimeIntentionalDecisionModelSpec,
        RuntimeIntentionalDecisionResolutionError,
        RuntimeIntentionalDecisionResult,
        run_runtime_intentional_decision,
    )
except ImportError as error:
    _RUNTIME_INTENTION_IMPORT_ERROR = error


def _value_hash(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def make_goal_model(*, reverse: bool = False, active_pressure: float = 2.0):
    cell = phase_cell()
    ready = _value_hash(TypedValue("PhaseState", "ready"))
    active = _value_hash(TypedValue("PhaseState", "active"))
    stay = GoalSpec(
        "stay",
        1.0,
        {cell: 1.0},
        {cell: {ready: 1.0, active: 0.0}},
    )
    engage = GoalSpec(
        "engage",
        active_pressure,
        {cell: 1.0},
        {cell: {ready: 0.0, active: 1.0}},
    )
    goals = (engage, stay) if reverse else (stay, engage)
    return GoalModelSpec("runtime-goals", "1", 8.0, goals)


def make_choice_model(*, reverse: bool = False):
    values = {
        "stay": {"a1-ready": 3.0, "a1-active": 0.0},
        "engage": {"a1-ready": 0.0, "a1-active": 3.0},
    }
    if reverse:
        values = {
            key: dict(reversed(tuple(value.items())))
            for key, value in reversed(tuple(values.items()))
        }
    return ChoiceModelSpec("runtime-choice", "1", 8.0, values)


def make_runtime_intentional_model(*, belief=None, goal=None, choice=None):
    return RuntimeIntentionalDecisionModelSpec(
        "runtime-intentional",
        "1",
        ("phase-choice",),
        make_runtime_belief_model() if belief is None else belief,
        make_goal_model() if goal is None else goal,
        make_choice_model() if choice is None else choice,
    )


def make_bad_action_choice_model():
    return ChoiceModelSpec(
        "bad-runtime-choice",
        "1",
        8.0,
        {
            "stay": {"a1-ready": 1.0},
            "engage": {"a1-ready": 1.0},
        },
    )


def make_bad_hypothesis_goal_model():
    cell = phase_cell()
    ready = _value_hash(TypedValue("PhaseState", "ready"))
    active = _value_hash(TypedValue("PhaseState", "active"))
    return GoalModelSpec(
        "bad-runtime-goals",
        "1",
        8.0,
        (
            GoalSpec(
                "stay",
                1.0,
                {cell: 1.0},
                {cell: {ready: 1.0}},
            ),
            GoalSpec(
                "engage",
                1.0,
                {cell: 1.0},
                {cell: {ready: 0.0, active: 1.0}},
            ),
        ),
    )


class NarrativeRuntimeIntentionTests(unittest.TestCase):
    def require_runtime_intention(self) -> None:
        if _RUNTIME_INTENTION_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime intention boundary is missing: "
                f"{_RUNTIME_INTENTION_IMPORT_ERROR}"
            )

    def test_runtime_intentional_model_identity_binds_nested_models_and_both_implementation_identities(self):
        self.require_runtime_intention()
        first = make_runtime_intentional_model()
        same = make_runtime_intentional_model()
        changed_belief = make_runtime_intentional_model(
            belief=make_runtime_belief_model(
                parameters={"match": 0.8, "mismatch": 0.2, "clear": 0.5}
            )
        )
        changed_goal = make_runtime_intentional_model(
            goal=make_goal_model(active_pressure=3.0)
        )
        changed_choice = make_runtime_intentional_model(
            choice=replace(make_choice_model(), beta_action=4.0)
        )
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed_belief.content_hash)
        self.assertNotEqual(first.content_hash, changed_goal.content_hash)
        self.assertNotEqual(first.content_hash, changed_choice.content_hash)
        payload = first.to_dict()
        self.assertEqual(
            payload["runtime_implementation_identity"],
            measure_implementation(
                RuntimeIntentionalDecisionModelSpec
            ).manifest_identity(),
        )
        self.assertEqual(
            payload["authored_intentional_implementation_identity"],
            measure_implementation(IntentionalDecisionModelSpec).manifest_identity(),
        )

    def test_runtime_intentional_result_record_binds_semantic_goal_hash_and_full_audit_belief(self):
        self.require_runtime_intention()
        domain, story, ledger = empty_runtime_case()
        belief_model = make_runtime_belief_model()
        belief = runtime_uncertain_belief_state(
            story,
            domain,
            "a1",
            ledger,
            belief_model,
            (phase_cell(),),
        )
        semantic_hash = stable_content_hash(
            {
                "cells": [
                    {
                        "cell": phase_cell().to_dict(),
                        "posterior": belief.cells[phase_cell()].posterior.to_dict(),
                    }
                ]
            }
        )
        goal = GoalState(
            "runtime-goals",
            make_goal_model().content_hash,
            semantic_hash,
            {"engage": 0.0, "stay": 1.0},
            {"engage": 0.25, "stay": 0.75},
            "stay",
        )
        model = make_runtime_intentional_model(belief=belief_model)
        result = RuntimeIntentionalDecisionResult(
            model_id=model.model_id,
            model_hash=model.content_hash,
            decision_id="d-a1-phase",
            step_index=belief.step_index,
            belief_state=belief,
            goal_state=goal,
            conditional_action_policies={
                "engage": {"a1-active": 0.75, "a1-ready": 0.25},
                "stay": {"a1-active": 0.25, "a1-ready": 0.75},
            },
            action_scores={"a1-active": 0.25, "a1-ready": 0.75},
            action_policy={"a1-active": 0.25, "a1-ready": 0.75},
            selected_action="a1-ready",
        )
        self.assertEqual(result.belief_state, belief)
        self.assertEqual(result.goal_state.belief_state_hash, semantic_hash)
        self.assertEqual(result.content_hash, stable_content_hash(result.to_dict()))
        wrong_goal = replace(goal, belief_state_hash=belief.content_hash)
        with self.assertRaises((TypeError, ValueError)):
            replace(result, goal_state=wrong_goal)

    def test_runtime_intentional_public_signatures_exclude_world_and_step_inputs(self):
        self.require_runtime_intention()
        self.assertEqual(
            tuple(inspect.signature(run_runtime_intentional_decision).parameters),
            ("story", "domain", "decision_id", "ledger", "model"),
        )
        for forbidden in (
            "world_state",
            "world_step",
            "projection_result",
            "step_index",
        ):
            self.assertNotIn(
                forbidden,
                inspect.signature(run_runtime_intentional_decision).parameters,
            )

    def test_empty_ledger_runtime_selection_matches_authored_intentional_math(self):
        self.require_runtime_intention()
        domain, story, ledger = empty_runtime_case()
        runtime_model = make_runtime_intentional_model()
        runtime = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            runtime_model,
        )
        authored_model = IntentionalDecisionModelSpec(
            "authored-intentional-regression",
            "1",
            ("phase-choice",),
            runtime_model.belief_model.seed_model,
            runtime_model.goal_model,
            runtime_model.choice_model,
        )
        authored = run_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            authored_model,
        )
        self.assertEqual(runtime.goal_state.scores, authored.goal_state.scores)
        self.assertEqual(runtime.goal_state.policy, authored.goal_state.policy)
        self.assertEqual(
            runtime.conditional_action_policies,
            authored.conditional_action_policies,
        )
        self.assertEqual(runtime.action_scores, authored.action_scores)
        self.assertEqual(runtime.action_policy, authored.action_policy)
        self.assertEqual(runtime.selected_action, authored.selected_action)

    def test_runtime_percept_changes_posterior_goal_policy_and_selected_action(self):
        self.require_runtime_intention()
        domain, story, initial = empty_runtime_case()
        model = make_runtime_intentional_model()
        before = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            initial,
            model,
        )
        updated = extend_ledger(
            initial,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "active"),
                ),
            ),
            branch="runtime-intention-update",
        )
        after = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            updated,
            model,
        )
        self.assertEqual(before.selected_action, "a1-ready")
        self.assertEqual(after.selected_action, "a1-active")
        self.assertNotEqual(before.goal_state.policy, after.goal_state.policy)
        self.assertNotEqual(before.action_policy, after.action_policy)
        self.assertNotEqual(
            before.belief_state.cells[phase_cell()].posterior,
            after.belief_state.cells[phase_cell()].posterior,
        )

    def test_same_posterior_semantics_ignore_hidden_provenance_for_policy(self):
        self.require_runtime_intention()
        domain, story, initial = empty_runtime_case()
        row = (
            "a1",
            "vision",
            phase_cell(),
            "equals",
            TypedValue("PhaseState", "active"),
        )
        left_ledger = extend_ledger(initial, (row,), branch="intent-left")
        right_ledger = extend_ledger(initial, (row,), branch="intent-right")
        model = make_runtime_intentional_model()
        left = run_runtime_intentional_decision(
            story, domain, "d-a1-phase", left_ledger, model
        )
        right = run_runtime_intentional_decision(
            story, domain, "d-a1-phase", right_ledger, model
        )
        self.assertEqual(
            left.belief_state.cells[phase_cell()].posterior,
            right.belief_state.cells[phase_cell()].posterior,
        )
        self.assertNotEqual(
            left.belief_state.ledger_hash,
            right.belief_state.ledger_hash,
        )
        self.assertEqual(left.goal_state.policy, right.goal_state.policy)
        self.assertEqual(left.goal_state.content_hash, right.goal_state.content_hash)
        self.assertEqual(left.action_policy, right.action_policy)
        self.assertEqual(left.selected_action, right.selected_action)
        self.assertNotEqual(left.content_hash, right.content_hash)

    def test_decision_template_cutoff_actor_type_and_context_fail_closed_before_likelihood(self):
        self.require_runtime_intention()
        domain, story, ledger = empty_runtime_case()
        stage_error = "runtime intentional execution is unavailable in this stage"

        hook = SemanticRuntimeLikelihood()
        supported = make_runtime_intentional_model(
            belief=make_runtime_belief_model(hook=hook)
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                story,
                domain,
                "missing-decision",
                ledger,
                supported,
            )
        self.assertNotEqual(str(caught.exception), stage_error)
        self.assertEqual(hook.inputs, [])

        hook = SemanticRuntimeLikelihood()
        unsupported = RuntimeIntentionalDecisionModelSpec(
            "unsupported-runtime-intentional",
            "1",
            ("noop-choice",),
            make_runtime_belief_model(hook=hook),
            make_goal_model(),
            make_choice_model(),
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                story,
                domain,
                "d-a1-phase",
                ledger,
                unsupported,
            )
        self.assertNotEqual(str(caught.exception), stage_error)
        self.assertEqual(hook.inputs, [])

        hook = SemanticRuntimeLikelihood()
        cutoff_model = make_runtime_intentional_model(
            belief=make_runtime_belief_model(hook=hook)
        )
        cutoff_ledger = runtime_evidence_ledger_from_story(
            story,
            domain,
            at_time=7,
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                story,
                domain,
                "d-a1-phase",
                cutoff_ledger,
                cutoff_model,
            )
        self.assertNotEqual(str(caught.exception), stage_error)
        self.assertEqual(hook.inputs, [])

        decision = next(item for item in story.decisions if item.id == "d-a1-phase")
        malformed = replace(
            story,
            decisions=tuple(
                replace(item, context_cells=()) if item.id == decision.id else item
                for item in story.decisions
            ),
        )
        hook = SemanticRuntimeLikelihood()
        malformed_model = make_runtime_intentional_model(
            belief=make_runtime_belief_model(hook=hook)
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                malformed,
                domain,
                "d-a1-phase",
                ledger,
                malformed_model,
            )
        self.assertNotEqual(str(caught.exception), stage_error)
        self.assertEqual(hook.inputs, [])

        bad_actor = replace(
            story,
            decisions=tuple(
                replace(item, actor_id="svc") if item.id == decision.id else item
                for item in story.decisions
            ),
        )
        hook = SemanticRuntimeLikelihood()
        bad_actor_model = make_runtime_intentional_model(
            belief=make_runtime_belief_model(hook=hook)
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                bad_actor,
                domain,
                "d-a1-phase",
                ledger,
                bad_actor_model,
            )
        self.assertNotEqual(str(caught.exception), stage_error)
        self.assertEqual(hook.inputs, [])

    def test_context_cells_are_exact_runtime_tracked_cells(self):
        self.require_runtime_intention()
        domain, story, ledger = empty_runtime_case()
        model = make_runtime_intentional_model()
        result = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            model,
        )
        decision = next(item for item in story.decisions if item.id == "d-a1-phase")
        self.assertEqual(set(result.belief_state.cells), set(decision.context_cells))
        self.assertEqual(tuple(decision.context_cells), (phase_cell(),))

    def test_action_and_goal_hypothesis_coverage_fail_typed(self):
        self.require_runtime_intention()
        domain, story, ledger = empty_runtime_case()
        bad_choice = make_runtime_intentional_model(
            choice=make_bad_action_choice_model()
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                story, domain, "d-a1-phase", ledger, bad_choice
            )
        self.assertIsInstance(caught.exception.__cause__, ChoiceResolutionError)

        bad_goal = make_runtime_intentional_model(
            goal=make_bad_hypothesis_goal_model()
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                story, domain, "d-a1-phase", ledger, bad_goal
            )
        self.assertIsInstance(caught.exception.__cause__, GoalResolutionError)

    def test_exact_ties_use_lexical_runtime_action_without_input_order_effect(self):
        self.require_runtime_intention()
        domain, story, ledger = empty_runtime_case()
        first_choice = ChoiceModelSpec(
            "tie-choice",
            "1",
            8.0,
            {
                "stay": {"a1-ready": 0.0, "a1-active": 0.0},
                "engage": {"a1-ready": 0.0, "a1-active": 0.0},
            },
        )
        second_choice = ChoiceModelSpec(
            "tie-choice",
            "1",
            8.0,
            {
                "engage": {"a1-active": 0.0, "a1-ready": 0.0},
                "stay": {"a1-active": 0.0, "a1-ready": 0.0},
            },
        )
        first = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            make_runtime_intentional_model(choice=first_choice),
        )
        second = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            make_runtime_intentional_model(choice=second_choice),
        )
        self.assertEqual(first.action_policy, second.action_policy)
        self.assertEqual(first.selected_action, "a1-active")
        self.assertEqual(second.selected_action, "a1-active")

    def test_runtime_selection_wraps_belief_goal_choice_failures_with_typed_causes(self):
        self.require_runtime_intention()
        domain, story, ledger = empty_runtime_case()
        zero_belief = make_runtime_intentional_model(
            belief=make_runtime_belief_model(hook=ZeroRuntimeLikelihood())
        )
        updated = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "active"),
                ),
            ),
            branch="runtime-intention-zero",
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                story,
                domain,
                "d-a1-phase",
                updated,
                zero_belief,
            )
        self.assertIsInstance(caught.exception.__cause__, RuntimeBeliefResolutionError)

        bad_choice = make_runtime_intentional_model(
            choice=make_bad_action_choice_model()
        )
        with self.assertRaises(RuntimeIntentionalDecisionResolutionError) as caught:
            run_runtime_intentional_decision(
                story,
                domain,
                "d-a1-phase",
                ledger,
                bad_choice,
            )
        self.assertIsInstance(caught.exception.__cause__, ChoiceResolutionError)

    def test_authored_intentional_runner_remains_semantically_unchanged(self):
        self.require_runtime_intention()
        domain, story, _ = empty_runtime_case()
        runtime_model = make_runtime_intentional_model()
        authored_model = IntentionalDecisionModelSpec(
            "authored-intentional-regression",
            "1",
            ("phase-choice",),
            runtime_model.belief_model.seed_model,
            runtime_model.goal_model,
            runtime_model.choice_model,
        )
        result = run_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            authored_model,
        )
        self.assertEqual(result.selected_action, "a1-ready")
        identity = measure_implementation(
            IntentionalDecisionModelSpec
        ).manifest_identity()
        self.assertEqual(
            identity["artifacts"][0]["locator"],
            "python-module:narrative_dynamics.narrative.intention",
        )


if __name__ == "__main__":
    unittest.main()
