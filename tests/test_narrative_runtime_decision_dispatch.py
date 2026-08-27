from __future__ import annotations

from dataclasses import fields, replace
import inspect
import unittest

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.narrative.ir import TypedValue
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionResolutionError,
    RuntimeIntentionalDecisionResult,
    run_runtime_intentional_decision,
)
from narrative_dynamics.narrative.runtime_planning import (
    PlanningHiddenState,
    PlanningObservation,
    RuntimePlanningDecisionModelSpec,
    RuntimePlanningDecisionResult,
    run_runtime_planning_decision,
)
from narrative_dynamics.narrative.runtime_reactive import (
    RuntimeReactiveDecisionModelSpec,
    RuntimeReactiveDecisionResolutionError,
    RuntimeReactiveDecisionResult,
    run_runtime_reactive_decision,
)
from tests.test_narrative_runtime_cognition import (
    empty_runtime_case,
    make_runtime_belief_model,
    phase_cell,
)
from tests.test_narrative_runtime_intention import make_runtime_intentional_model
from tests.test_narrative_runtime_planning import (
    IdentityTransitionHook,
    NoInformationObservationHook,
    ParameterRewardHook,
    ProductCouplingHook,
)
from tests.test_narrative_runtime_reactive import CueScoreHook, RaisingScoreHook


_DISPATCH_IMPORT_ERROR: ImportError | None = None
try:
    import narrative_dynamics.narrative.runtime_decision_dispatch as dispatch_module
    from narrative_dynamics.narrative.runtime_decision_dispatch import (
        RuntimeDecisionDispatchError,
        RuntimeDecisionDispatchResult,
        RuntimeDecisionModelSpec,
        run_runtime_decision,
    )
except ImportError as error:
    _DISPATCH_IMPORT_ERROR = error


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


def make_reactive_model(*, hook=None):
    if hook is None:
        hook = CueScoreHook()
    return RuntimeReactiveDecisionModelSpec(
        "dispatch-reactive",
        "1",
        ("phase-choice",),
        (phase_cell(),),
        {},
        2.0,
        hook,
    )


def make_planning_model():
    return RuntimePlanningDecisionModelSpec(
        "dispatch-planning",
        "1",
        ("phase-choice",),
        (phase_cell(),),
        (),
        make_runtime_belief_model(),
        (
            PlanningHiddenState(
                "active",
                {phase_cell(): TypedValue("PhaseState", "active")},
            ),
            PlanningHiddenState(
                "ready",
                {phase_cell(): TypedValue("PhaseState", "ready")},
            ),
        ),
        (PlanningObservation("none", {}),),
        (("a1-active", "a1-ready"),),
        1.0,
        2.0,
        {"action_rewards": {"a1-active": 2.0, "a1-ready": 0.0}},
        ProductCouplingHook(),
        IdentityTransitionHook(),
        NoInformationObservationHook(),
        ParameterRewardHook(),
    )


class NarrativeRuntimeDecisionDispatchTests(unittest.TestCase):
    def require_dispatch(self) -> None:
        if _DISPATCH_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime decision dispatch boundary is missing: "
                f"{_DISPATCH_IMPORT_ERROR}"
            )

    def test_model_wrapper_is_closed_typed_and_binds_nested_and_dispatch_identity(self):
        self.require_dispatch()
        reactive = make_reactive_model()
        intentional = make_runtime_intentional_model()
        planning = make_planning_model()
        wrappers = (
            RuntimeDecisionModelSpec("reactive", reactive),
            RuntimeDecisionModelSpec("intentional", intentional),
            RuntimeDecisionModelSpec("planning", planning),
        )
        self.assertEqual(
            tuple(item.model_kind for item in wrappers),
            ("reactive", "intentional", "planning"),
        )
        self.assertEqual(wrappers[0].model_id, reactive.model_id)
        self.assertEqual(wrappers[1].model_version, intentional.version)
        self.assertEqual(wrappers[2].nested_model_hash, planning.content_hash)
        self.assertEqual(
            wrappers[1].supported_decision_types,
            intentional.supported_decision_types,
        )
        self.assertEqual(
            RuntimeDecisionModelSpec("reactive", make_reactive_model()).content_hash,
            wrappers[0].content_hash,
        )
        self.assertNotEqual(wrappers[0].content_hash, wrappers[1].content_hash)
        self.assertNotEqual(wrappers[1].content_hash, wrappers[2].content_hash)
        payload = wrappers[0].to_dict()
        self.assertEqual(
            payload["wrapper_implementation_identity"],
            measure_implementation(RuntimeDecisionModelSpec).manifest_identity(),
        )
        self.assertEqual(
            payload["runner_implementation_identity"],
            measure_implementation(run_runtime_decision).manifest_identity(),
        )
        for kind, nested in (
            ("planning", reactive),
            ("reactive", intentional),
            ("intentional", planning),
            ("unknown", reactive),
        ):
            with self.subTest(kind=kind):
                with self.assertRaises((TypeError, ValueError)):
                    RuntimeDecisionModelSpec(kind, nested)
        with self.assertRaises((TypeError, ValueError)):
            RuntimeDecisionModelSpec("reactive", object())

    def test_public_signature_and_module_isolation_exclude_world_scheduler_projection_rng(self):
        self.require_dispatch()
        self.assertEqual(
            tuple(inspect.signature(run_runtime_decision).parameters),
            ("story", "domain", "decision_id", "ledger", "model"),
        )
        source = inspect.getsource(dispatch_module)
        for forbidden in (
            "narrative_dynamics.narrative.world",
            "narrative_dynamics.narrative.simulation",
            "narrative_dynamics.narrative.observation_projection",
            "import random",
            "from random",
        ):
            self.assertNotIn(forbidden, source)
        self.assertEqual(
            set(dispatch_module.__all__),
            {
                "RuntimeDecisionModelSpec",
                "RuntimeDecisionDispatchResult",
                "RuntimeDecisionDispatchError",
                "run_runtime_decision",
            },
        )

    def test_direct_family_dispatch_preserves_exact_common_fields_and_nested_result(self):
        self.require_dispatch()
        domain, story, ledger = empty_runtime_case()
        cases = (
            (
                "reactive",
                make_reactive_model(),
                run_runtime_reactive_decision,
                RuntimeReactiveDecisionResult,
            ),
            (
                "intentional",
                make_runtime_intentional_model(),
                run_runtime_intentional_decision,
                RuntimeIntentionalDecisionResult,
            ),
            (
                "planning",
                make_planning_model(),
                run_runtime_planning_decision,
                RuntimePlanningDecisionResult,
            ),
        )
        for kind, nested_model, family_runner, result_type in cases:
            with self.subTest(kind=kind):
                family = family_runner(
                    story,
                    domain,
                    "d-a1-phase",
                    ledger,
                    nested_model,
                )
                wrapper = RuntimeDecisionModelSpec(kind, nested_model)
                result = run_runtime_decision(
                    story,
                    domain,
                    "d-a1-phase",
                    ledger,
                    wrapper,
                )
                self.assertIsInstance(result.model_result, result_type)
                self.assertEqual(result.model_result, family)
                self.assertEqual(result.model_kind, kind)
                self.assertEqual(result.decision_model_hash, wrapper.content_hash)
                self.assertEqual(result.model_id, family.model_id)
                self.assertEqual(result.model_hash, family.model_hash)
                self.assertEqual(result.decision_id, family.decision_id)
                self.assertEqual(result.step_index, family.step_index)
                self.assertEqual(result.action_policy, family.action_policy)
                self.assertEqual(result.selected_action, family.selected_action)
                self.assertEqual(result.model_result_hash, family.content_hash)
                if kind == "intentional":
                    self.assertEqual(result.actor_id, family.belief_state.agent_id)
                    self.assertEqual(
                        result.ledger_hash,
                        family.belief_state.ledger_hash,
                    )
                else:
                    self.assertEqual(result.actor_id, family.actor_id)
                    self.assertEqual(result.ledger_hash, family.ledger_hash)

    def test_dispatch_result_forgery_rejects_common_and_nested_binding_mismatch(self):
        self.require_dispatch()
        domain, story, ledger = empty_runtime_case()
        model = make_reactive_model()
        nested = run_runtime_reactive_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            model,
        )
        wrapper = RuntimeDecisionModelSpec("reactive", model)
        valid = dict(
            model_kind="reactive",
            decision_model_hash=wrapper.content_hash,
            model_id=nested.model_id,
            model_hash=nested.model_hash,
            decision_id=nested.decision_id,
            actor_id=nested.actor_id,
            step_index=nested.step_index,
            ledger_hash=nested.ledger_hash,
            action_policy=nested.action_policy,
            selected_action=nested.selected_action,
            model_result_hash=nested.content_hash,
            model_result=nested,
        )
        RuntimeDecisionDispatchResult(**valid)
        for changes in (
            {"model_kind": "planning"},
            {"model_id": "forged-model"},
            {"decision_id": "forged-decision"},
            {"actor_id": "forged-actor"},
            {"step_index": nested.step_index + 1},
            {"ledger_hash": "sha256:" + "1" * 64},
            {"action_policy": {"a1-active": 1.0, "a1-ready": 0.0}},
            {"selected_action": "a1-ready"},
            {"model_result_hash": "sha256:" + "2" * 64},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    RuntimeDecisionDispatchResult(**{**valid, **changes})

    def test_wrapper_forgery_and_unsupported_decision_type_reject_before_family_hook(self):
        self.require_dispatch()
        domain, story, ledger = empty_runtime_case()
        hook = CueScoreHook()
        model = make_reactive_model(hook=hook)
        wrapper = RuntimeDecisionModelSpec("reactive", model)
        forged = _forge(wrapper, model_kind="planning")
        with self.assertRaises(RuntimeDecisionDispatchError):
            run_runtime_decision(story, domain, "d-a1-phase", ledger, forged)
        self.assertEqual(hook.calls, [])

        unsupported_model = replace(
            model,
            supported_decision_types=("other-choice",),
        )
        unsupported = RuntimeDecisionModelSpec("reactive", unsupported_model)
        with self.assertRaises(RuntimeDecisionDispatchError):
            run_runtime_decision(
                story,
                domain,
                "d-a1-phase",
                ledger,
                unsupported,
            )
        self.assertEqual(hook.calls, [])

    def test_family_failures_are_typed_and_preserve_exact_cause(self):
        self.require_dispatch()
        domain, story, ledger = empty_runtime_case()
        model = make_reactive_model(hook=RaisingScoreHook())
        with self.assertRaises(RuntimeDecisionDispatchError) as caught:
            run_runtime_decision(
                story,
                domain,
                "d-a1-phase",
                ledger,
                RuntimeDecisionModelSpec("reactive", model),
            )
        self.assertIsInstance(
            caught.exception.__cause__,
            RuntimeReactiveDecisionResolutionError,
        )
        self.assertIsInstance(caught.exception.__cause__.__cause__, RuntimeError)

    def test_fixed_inputs_replay_to_exact_dispatch_result_and_hash(self):
        self.require_dispatch()
        domain, story, ledger = empty_runtime_case()
        first_model = RuntimeDecisionModelSpec("planning", make_planning_model())
        second_model = RuntimeDecisionModelSpec("planning", make_planning_model())
        self.assertEqual(first_model.content_hash, second_model.content_hash)
        first = run_runtime_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            first_model,
        )
        second = run_runtime_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            second_model,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)


if __name__ == "__main__":
    unittest.main()
