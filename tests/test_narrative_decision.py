from __future__ import annotations

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
