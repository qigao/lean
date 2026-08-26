from __future__ import annotations

from dataclasses import fields, replace
import inspect
import math
import unittest

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.intention import ChoiceModelSpec
from narrative_dynamics.narrative.ir import TypedValue
from narrative_dynamics.narrative.runtime_intention import run_runtime_intentional_decision
from narrative_dynamics.narrative.runtime_perception import (
    runtime_evidence_ledger_from_story,
)
from tests.test_narrative_runtime_cognition import (
    alert_cell,
    empty_runtime_case,
    extend_ledger,
    phase_cell,
)
from tests.test_narrative_runtime_intention import make_runtime_intentional_model


_REACTIVE_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_reactive import (
        ReactiveCueView,
        RuntimeReactiveCueSnapshot,
        RuntimeReactiveDecisionContext,
        RuntimeReactiveDecisionModelSpec,
        RuntimeReactiveDecisionResolutionError,
        RuntimeReactiveDecisionResult,
        run_runtime_reactive_decision,
    )
except ImportError as error:
    _REACTIVE_IMPORT_ERROR = error


def _hash(label: str) -> str:
    return stable_content_hash({"runtime-reactive-test": label})


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


class CueScoreHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        cue = context.cues[phase_cell()]
        if cue.status == "resolved" and cue.value == TypedValue("PhaseState", "active"):
            return {"a1-ready": 0.0, "a1-active": 2.0}
        if cue.status == "resolved" and cue.value == TypedValue("PhaseState", "ready"):
            return {"a1-ready": 2.0, "a1-active": 0.0}
        return {"a1-ready": 0.0, "a1-active": 0.0}


class CueScoreHookV2(CueScoreHook):
    pass


class ConstantTieHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return {"a1-ready": 0.0, "a1-active": 0.0}


class InvalidScoreHook:
    def __init__(self, mode: str):
        self.mode = mode
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        if self.mode == "nonmapping":
            return []
        if self.mode == "missing":
            return {"a1-active": 1.0}
        if self.mode == "extra":
            return {"a1-active": 1.0, "a1-ready": 0.0, "extra": 0.0}
        bad = {
            "bool": True,
            "nan": math.nan,
            "inf": math.inf,
        }[self.mode]
        return {"a1-active": bad, "a1-ready": 0.0}


class RaisingScoreHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        raise RuntimeError("reactive score boom")


class NarrativeRuntimeReactiveTests(unittest.TestCase):
    def require_reactive(self) -> None:
        if _REACTIVE_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime reactive boundary is missing: "
                f"{_REACTIVE_IMPORT_ERROR}"
            )

    def make_model(
        self,
        *,
        hook=None,
        beta: float = 2.0,
        cue_cells=None,
        parameters=None,
        supported=("phase-choice",),
    ):
        self.require_reactive()
        if hook is None:
            hook = CueScoreHook()
        if cue_cells is None:
            cue_cells = (phase_cell(),)
        if parameters is None:
            parameters = {"mode": "current", "nested": {"scale": [1, 2.0]}}
        return RuntimeReactiveDecisionModelSpec(
            "runtime-reactive",
            "1",
            tuple(supported),
            tuple(cue_cells),
            parameters,
            beta,
            hook,
        )

    def test_model_identity_binds_parameters_hook_and_shared_softmax(self):
        self.require_reactive()
        first = self.make_model(
            hook=CueScoreHook(),
            parameters={"z": [1, 2.0], "a": {"flag": True}},
            supported=("phase-choice", "noop-choice"),
        )
        reordered = self.make_model(
            hook=CueScoreHook(),
            parameters={"a": {"flag": True}, "z": (1, 2.0)},
            supported=("noop-choice", "phase-choice"),
        )
        changed_parameter = self.make_model(
            hook=CueScoreHook(),
            parameters={"a": {"flag": False}, "z": (1, 2.0)},
            supported=("noop-choice", "phase-choice"),
        )
        changed_hook = self.make_model(
            hook=CueScoreHookV2(),
            parameters={"a": {"flag": True}, "z": (1, 2.0)},
            supported=("noop-choice", "phase-choice"),
        )
        changed_beta = replace(first, beta=3.0)

        self.assertEqual(first.content_hash, reordered.content_hash)
        self.assertNotEqual(first.content_hash, changed_parameter.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)
        self.assertNotEqual(first.content_hash, changed_beta.content_hash)
        payload = first.to_dict()
        self.assertEqual(
            payload["softmax_implementation_identity"],
            measure_implementation(finite_softmax).manifest_identity(),
        )
        self.assertEqual(
            payload["runtime_implementation_identity"],
            measure_implementation(
                RuntimeReactiveDecisionModelSpec
            ).manifest_identity(),
        )

    def test_model_parameters_accept_only_canonical_values(self):
        self.require_reactive()
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
        invalid = (
            {"x": math.nan},
            {"x": math.inf},
            {"x": b"bytes"},
            {"x": {1, 2}},
            {"x": object()},
            {"x": lambda: None},
            {"": 1},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises((TypeError, ValueError)):
                    self.make_model(parameters=parameters)

    def test_value_records_are_immutable_canonical_and_self_validating(self):
        self.require_reactive()
        cell = phase_cell()
        ready = TypedValue("PhaseState", "ready")
        resolved = ReactiveCueView(cell, "resolved", ready, 0)
        unknown = ReactiveCueView(cell, "unknown", None, 1)
        self.assertEqual(resolved.value, ready)
        self.assertIsNone(unknown.value)
        with self.assertRaises((TypeError, ValueError)):
            ReactiveCueView(cell, "unknown", ready, 0)
        with self.assertRaises((TypeError, ValueError)):
            ReactiveCueView(cell, "resolved", None, 0)
        with self.assertRaises((TypeError, ValueError)):
            ReactiveCueView(cell, "resolved", ready, True)

        snapshot = RuntimeReactiveCueSnapshot(
            "a1",
            "d-a1-phase",
            0,
            _hash("ledger"),
            {cell: resolved},
        )
        context = RuntimeReactiveDecisionContext(
            "d-a1-phase",
            "a1",
            "phase-choice",
            0,
            ("a1-ready", "a1-active"),
            {cell: resolved},
            {"z": [1], "a": {"flag": True}},
        )
        self.assertEqual(context.actions, ("a1-active", "a1-ready"))
        self.assertEqual(snapshot.content_hash, stable_content_hash(snapshot.to_dict()))
        with self.assertRaises(TypeError):
            context.cues[cell] = unknown
        with self.assertRaises(TypeError):
            context.parameters["new"] = 1

    def test_public_signature_excludes_world_and_step_inputs(self):
        self.require_reactive()
        self.assertEqual(
            tuple(inspect.signature(run_runtime_reactive_decision).parameters),
            ("story", "domain", "decision_id", "ledger", "model"),
        )
        for forbidden in (
            "world_state",
            "world_step",
            "projection_result",
            "step_index",
            "belief_state",
        ):
            self.assertNotIn(
                forbidden,
                inspect.signature(run_runtime_reactive_decision).parameters,
            )

    def test_step_zero_uses_direct_authored_cue_only(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        hook = CueScoreHook()
        result = run_runtime_reactive_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self.make_model(hook=hook),
        )
        cue = result.cue_snapshot.cues[phase_cell()]
        self.assertEqual(result.step_index, 0)
        self.assertEqual(cue.status, "resolved")
        self.assertEqual(cue.value, TypedValue("PhaseState", "ready"))
        self.assertEqual(hook.calls[0].step_index, 0)
        self.assertEqual(result.selected_action, "a1-ready")

    def test_step_positive_uses_only_latest_batch_and_blackout_does_not_fill_history(self):
        self.require_reactive()
        domain, story, initial = empty_runtime_case()
        informative = extend_ledger(
            initial,
            (("a1", "vision", phase_cell(), "equals", TypedValue("PhaseState", "active")),),
            branch="reactive-history",
        )
        blackout = extend_ledger(informative, (), branch="reactive-blackout")
        model = self.make_model()
        seen = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", informative, model
        )
        hidden = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", blackout, model
        )
        self.assertEqual(
            seen.cue_snapshot.cues[phase_cell()].value,
            TypedValue("PhaseState", "active"),
        )
        self.assertEqual(hidden.cue_snapshot.cues[phase_cell()].status, "unknown")
        self.assertIsNone(hidden.cue_snapshot.cues[phase_cell()].value)
        self.assertNotEqual(seen.action_policy, hidden.action_policy)

    def test_agreeing_channels_collapse_to_one_semantic_current_cue(self):
        self.require_reactive()
        domain, story, initial = empty_runtime_case()
        ledger = extend_ledger(
            initial,
            (
                ("a1", "vision", phase_cell(), "equals", TypedValue("PhaseState", "active")),
                ("a1", "status", phase_cell(), "equals", TypedValue("PhaseState", "active")),
            ),
            branch="reactive-agree",
        )
        result = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", ledger, self.make_model()
        )
        cue = result.cue_snapshot.cues[phase_cell()]
        self.assertEqual(cue.status, "resolved")
        self.assertEqual(cue.value, TypedValue("PhaseState", "active"))
        self.assertFalse(hasattr(cue, "channel"))
        self.assertFalse(hasattr(cue, "evidence_refs"))

    def test_same_step_semantic_disagreement_fails_typed(self):
        self.require_reactive()
        domain, story, initial = empty_runtime_case()
        ledger = extend_ledger(
            initial,
            (
                ("a1", "vision", phase_cell(), "equals", TypedValue("PhaseState", "active")),
                ("a1", "status", phase_cell(), "equals", TypedValue("PhaseState", "ready")),
            ),
            branch="reactive-conflict",
        )
        with self.assertRaises(RuntimeReactiveDecisionResolutionError):
            run_runtime_reactive_decision(
                story, domain, "d-a1-phase", ledger, self.make_model()
            )

    def test_hook_context_is_sanitized_and_contains_no_hidden_runtime_objects(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        hook = CueScoreHook()
        run_runtime_reactive_decision(
            story, domain, "d-a1-phase", ledger, self.make_model(hook=hook)
        )
        self.assertEqual(len(hook.calls), 1)
        context = hook.calls[0]
        self.assertIsInstance(context, RuntimeReactiveDecisionContext)
        self.assertEqual(context.actions, ("a1-active", "a1-ready"))
        for forbidden in (
            "story",
            "domain",
            "ledger",
            "world_state",
            "world_step",
            "belief_state",
            "goal_state",
            "evidence_history",
            "projection_result",
        ):
            self.assertFalse(hasattr(context, forbidden), forbidden)
        cue = context.cues[phase_cell()]
        for forbidden in (
            "observer_id",
            "channel",
            "source_world_state_hash",
            "source_world_step_hash",
            "projection_model_hash",
        ):
            self.assertFalse(hasattr(cue, forbidden), forbidden)

    def test_forged_ledger_identity_and_chain_fail_before_hook(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        for forged in (
            _forge(ledger, source_story_hash=_hash("wrong-story")),
            _forge(ledger, domain_spec_hash=_hash("wrong-domain")),
            _forge(ledger, current_world_state_hash=_hash("wrong-current-world")),
        ):
            hook = CueScoreHook()
            with self.subTest(forged=forged):
                with self.assertRaises(RuntimeReactiveDecisionResolutionError):
                    run_runtime_reactive_decision(
                        story,
                        domain,
                        "d-a1-phase",
                        forged,
                        self.make_model(hook=hook),
                    )
                self.assertEqual(hook.calls, [])

    def test_cue_outside_decision_context_and_template_after_cutoff_fail_before_hook(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        hook = CueScoreHook()
        with self.assertRaises(RuntimeReactiveDecisionResolutionError):
            run_runtime_reactive_decision(
                story,
                domain,
                "d-a1-phase",
                ledger,
                self.make_model(hook=hook, cue_cells=(alert_cell(),)),
            )
        self.assertEqual(hook.calls, [])

        cutoff = runtime_evidence_ledger_from_story(story, domain, at_time=7)
        hook = CueScoreHook()
        with self.assertRaises(RuntimeReactiveDecisionResolutionError):
            run_runtime_reactive_decision(
                story,
                domain,
                "d-a1-phase",
                cutoff,
                self.make_model(hook=hook),
            )
        self.assertEqual(hook.calls, [])

    def test_action_scores_require_exact_finite_non_bool_schema(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        for mode in ("nonmapping", "missing", "extra", "bool", "nan", "inf"):
            hook = InvalidScoreHook(mode)
            with self.subTest(mode=mode):
                with self.assertRaises(RuntimeReactiveDecisionResolutionError):
                    run_runtime_reactive_decision(
                        story,
                        domain,
                        "d-a1-phase",
                        ledger,
                        self.make_model(hook=hook),
                    )
                self.assertEqual(len(hook.calls), 1)

    def test_hook_exception_is_wrapped_with_typed_cause(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        hook = RaisingScoreHook()
        with self.assertRaises(RuntimeReactiveDecisionResolutionError) as caught:
            run_runtime_reactive_decision(
                story, domain, "d-a1-phase", ledger, self.make_model(hook=hook)
            )
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(len(hook.calls), 1)

    def test_shared_softmax_produces_complete_simplex_and_lexical_map_tie(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        model = self.make_model(hook=ConstantTieHook(), beta=2.0)
        result = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", ledger, model
        )
        expected = finite_softmax(
            {"a1-active": 0.0, "a1-ready": 0.0}, beta=2.0
        )
        self.assertEqual(dict(result.action_policy), expected)
        self.assertEqual(set(result.action_policy), {"a1-active", "a1-ready"})
        self.assertTrue(
            math.isclose(
                math.fsum(result.action_policy.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(result.selected_action, "a1-active")

    def test_fixed_inputs_replay_to_exact_result_and_content_hash(self):
        self.require_reactive()
        domain, story, initial = empty_runtime_case()
        ledger = extend_ledger(
            initial,
            (("a1", "vision", phase_cell(), "equals", TypedValue("PhaseState", "active")),),
            branch="reactive-replay",
        )
        model = self.make_model(hook=CueScoreHook())
        first = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", ledger, model
        )
        second = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", ledger, model
        )
        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.content_hash, stable_content_hash(first.to_dict()))

    def test_snapshot_and_result_forgery_is_rejected_by_public_record_validation(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        result = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", ledger, self.make_model()
        )
        with self.assertRaises((TypeError, ValueError)):
            replace(result.cue_snapshot, ledger_hash=_hash("wrong-ledger"))
        with self.assertRaises((TypeError, ValueError)):
            replace(result, cue_snapshot_hash=_hash("wrong-snapshot"))
        other = "a1-active" if result.selected_action == "a1-ready" else "a1-ready"
        with self.assertRaises((TypeError, ValueError)):
            replace(result, selected_action=other)

    def test_observational_equivalence_matches_intentional_policy_exactly(self):
        self.require_reactive()
        domain, story, ledger = empty_runtime_case()
        reactive = run_runtime_reactive_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            self.make_model(hook=ConstantTieHook(), beta=2.0),
        )
        identical_values = {
            "stay": {"a1-active": 0.0, "a1-ready": 0.0},
            "engage": {"a1-active": 0.0, "a1-ready": 0.0},
        }
        intentional = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            make_runtime_intentional_model(
                choice=ChoiceModelSpec("equivalent-choice", "1", 2.0, identical_values)
            ),
        )
        self.assertEqual(reactive.action_policy, intentional.action_policy)
        self.assertEqual(reactive.selected_action, intentional.selected_action)

    def test_information_access_intervention_separates_memoryless_reactive_from_intentional(self):
        self.require_reactive()
        domain, story, initial = empty_runtime_case()
        informative = extend_ledger(
            initial,
            (("a1", "vision", phase_cell(), "equals", TypedValue("PhaseState", "active")),),
            branch="reactive-intervention-step1",
        )
        visible = extend_ledger(
            informative,
            (("a1", "vision", phase_cell(), "equals", TypedValue("PhaseState", "active")),),
            branch="reactive-intervention-step2",
        )
        blackout = extend_ledger(
            informative,
            (),
            branch="reactive-intervention-step2",
        )
        self.assertEqual(visible.current_world_state_hash, blackout.current_world_state_hash)

        reactive_model = self.make_model(hook=CueScoreHook())
        reactive_visible = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", visible, reactive_model
        )
        reactive_blackout = run_runtime_reactive_decision(
            story, domain, "d-a1-phase", blackout, reactive_model
        )
        intentional_blackout = run_runtime_intentional_decision(
            story,
            domain,
            "d-a1-phase",
            blackout,
            make_runtime_intentional_model(),
        )
        self.assertEqual(
            reactive_blackout.cue_snapshot.cues[phase_cell()].status,
            "unknown",
        )
        self.assertNotEqual(reactive_visible.action_policy, reactive_blackout.action_policy)
        self.assertNotEqual(reactive_blackout.action_policy, intentional_blackout.action_policy)

    def test_runtime_reactive_module_is_isolated_from_cognition_intention_and_world(self):
        self.require_reactive()
        import narrative_dynamics.narrative.runtime_reactive as reactive

        source = inspect.getsource(reactive)
        for forbidden in (
            "narrative.runtime_cognition",
            "narrative.runtime_intention",
            "narrative.intention",
            "narrative.world",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
