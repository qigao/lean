from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.decision import (
        DecisionResolutionError,
        EpistemicResolutionError,
        EvidenceAccess,
        run_decision_model,
    )
except ImportError as error:
    _IMPORT_ERROR = error

_RESULT_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.decision import DecisionCellView, DecisionResult
except ImportError as error:
    _RESULT_IMPORT_ERROR = error

from narrative_test_support import (
    MissingActionHook,
    RecordingFirstActionHook,
    make_health_model,
    make_model,
    make_test_domain,
    make_test_story,
)


class GenericDecisionTests(unittest.TestCase):
    def require_decision(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"generic narrative decision module is missing: {_IMPORT_ERROR}")

    def test_access_boundary_and_model_identity(self) -> None:
        self.require_decision()
        hook = RecordingFirstActionHook()
        direct = make_model("same", EvidenceAccess.DIRECT_ONLY, hook)
        epistemic = make_model(
            "same", EvidenceAccess.EPISTEMIC, RecordingFirstActionHook()
        )
        self.assertNotEqual(direct.content_hash, epistemic.content_hash)

        run_decision_model(
            make_test_story(), make_test_domain(), "d1", direct
        )
        self.assertIsNotNone(hook.context)
        self.assertFalse(hasattr(hook.context, "story"))
        self.assertFalse(hasattr(hook.context, "objective_state"))
        self.assertTrue(
            all(
                item.evidence_kind == "direct_perception"
                for item in hook.context.evidence_history
            )
        )

    def test_result_and_cell_view_are_typed_one_hot_records(self) -> None:
        self.require_decision()
        if _RESULT_IMPORT_ERROR is not None:
            self.fail(f"decision result surface is missing: {_RESULT_IMPORT_ERROR}")

        hook = RecordingFirstActionHook()
        model = make_model("result", EvidenceAccess.DIRECT_ONLY, hook)
        result = run_decision_model(
            make_test_story(), make_test_domain(), "d1", model
        )

        self.assertIsInstance(result, DecisionResult)
        self.assertEqual(result.model_id, "result")
        self.assertEqual(result.selected_action, "restart")
        self.assertEqual(dict(result.action_scores), {"restart": 1.0, "leave": 0.0})
        self.assertEqual(dict(result.policy), {"restart": 1.0, "leave": 0.0})
        self.assertEqual(result.basis.selected_action, result.selected_action)
        self.assertIsNotNone(hook.context)
        self.assertTrue(
            all(isinstance(view, DecisionCellView) for view in hook.context.cells.values())
        )
        view = next(iter(hook.context.cells.values()))
        with self.assertRaises(FrozenInstanceError):
            view.status = "unknown"

    def test_three_capabilities_produce_expected_reference_actions(self) -> None:
        self.require_decision()
        story = make_test_story()
        domain = make_test_domain()
        selected = tuple(
            run_decision_model(
                story,
                domain,
                "d1",
                make_health_model(name, access),
            ).selected_action
            for name, access in (
                ("direct", EvidenceAccess.DIRECT_ONLY),
                ("epistemic", EvidenceAccess.EPISTEMIC),
                ("omniscient", EvidenceAccess.OMNISCIENT),
            )
        )
        self.assertEqual(selected, ("restart", "leave", "leave"))

    def test_conflicted_epistemic_cell_fails_typed_resolution(self) -> None:
        self.require_decision()
        with self.assertRaises(EpistemicResolutionError):
            run_decision_model(
                make_test_story(second_claim_value="failed"),
                make_test_domain(),
                "d1",
                make_health_model("epistemic", EvidenceAccess.EPISTEMIC),
            )

    def test_undeclared_action_fails_typed_resolution(self) -> None:
        self.require_decision()
        model = make_model(
            "bad",
            EvidenceAccess.DIRECT_ONLY,
            MissingActionHook(),
        )
        with self.assertRaises(DecisionResolutionError):
            run_decision_model(
                make_test_story(), make_test_domain(), "d1", model
            )


if __name__ == "__main__":
    unittest.main()
