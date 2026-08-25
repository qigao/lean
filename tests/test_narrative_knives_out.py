from __future__ import annotations

import unittest

from narrative_dynamics.narrative.analysis import (
    analyze_narrative,
    build_trajectory,
    derive_analysis_scope,
)
from narrative_dynamics.narrative.decision import EpistemicResolutionError
from narrative_dynamics.narrative.interventions import (
    evaluate_counterfactual,
    generate_minimal_interventions,
)
from narrative_dynamics.narrative.ir import EntityRef, StateCellRef, TypedValue
from narrative_dynamics.narrative.replay import direct_state, epistemic_state, objective_state


_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.domains.knives_out import (
        knives_out_confession_story,
        knives_out_direct_model,
        knives_out_domain,
        knives_out_epistemic_model,
        knives_out_omniscient_model,
    )
except ImportError as error:
    _IMPORT_ERROR = error


_CASE_CELL = StateCellRef(EntityRef("harlan-case", "Case"), "case.culprit")
_RANSOM = TypedValue("AgentRef", EntityRef("ransom", "Agent"))
_MARTA = TypedValue("AgentRef", EntityRef("marta", "Agent"))


class KnivesOutConformanceTests(unittest.TestCase):
    def require_knives_out(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"Knives Out conformance domain is missing: {_IMPORT_ERROR}")

    def test_confession_slice_separates_objective_direct_and_epistemic_knowledge(self) -> None:
        self.require_knives_out()
        story = knives_out_confession_story()
        domain = knives_out_domain()

        self.assertEqual(
            tuple((event.id, event.logical_time) for event in story.events),
            (("tamper-medication", 1), ("restore-medication-evidence", 2)),
        )
        self.assertEqual(objective_state(story, domain, at_time=1)[_CASE_CELL], _RANSOM)

        fran_before = direct_state(story, domain, "fran", at_time=1)
        self.assertNotIn(_CASE_CELL, fran_before.cells)
        fran_after = direct_state(story, domain, "fran", at_time=2)
        self.assertEqual(fran_after.cells[_CASE_CELL].resolved_value, _RANSOM)
        self.assertEqual(fran_after.cells[_CASE_CELL].evidence_kind, "direct_perception")
        self.assertEqual(
            fran_after.cells[_CASE_CELL].supporting_id,
            "restore-medication-evidence",
        )

        blanc_direct = direct_state(story, domain, "blanc", at_time=3)
        self.assertNotIn(_CASE_CELL, blanc_direct.cells)

        blanc_epistemic = epistemic_state(story, domain, "blanc", at_time=3)
        view = blanc_epistemic.cells[_CASE_CELL]
        self.assertEqual(view.resolved_value, _RANSOM)
        self.assertEqual(view.evidence_kind, "testimony")
        self.assertEqual(view.supporting_id, "ransom-confession")
        self.assertEqual(view.source_agent, "ransom")
        self.assertEqual(view.evidence_refs, ("ransom-confession", "tamper-medication"))

        with self.assertRaises(EpistemicResolutionError):
            analyze_narrative(
                story,
                domain,
                "blanc-focus",
                knives_out_direct_model(),
            )

        actions = tuple(
            analyze_narrative(story, domain, "blanc-focus", model).baseline.selected_action
            for model in (
                knives_out_epistemic_model(),
                knives_out_omniscient_model(),
            )
        )
        self.assertEqual(actions, ("focus_ransom", "focus_ransom"))

    def test_claim_mutation_flips_focus_and_reception_removal_fails_closed(self) -> None:
        self.require_knives_out()
        story = knives_out_confession_story()
        domain = knives_out_domain()
        model = knives_out_epistemic_model()
        scope = derive_analysis_scope(story, domain, "blanc-focus")
        baseline = build_trajectory(story, domain, scope, model)
        interventions = generate_minimal_interventions(story, domain, scope)

        claim_change = next(
            item
            for item in interventions
            if item.kind == "change_claim_value"
            and item.target_ref == "ransom-confession"
            and item.to_value == _MARTA
        )
        changed = evaluate_counterfactual(
            story,
            domain,
            scope,
            model,
            baseline,
            claim_change,
        )
        self.assertEqual(changed.status, "valid")
        self.assertIsNotNone(changed.trajectory)
        self.assertIsNotNone(changed.first_divergence)
        assert changed.trajectory is not None
        assert changed.first_divergence is not None
        self.assertEqual(changed.first_divergence.logical_time, 3)
        self.assertTrue(changed.first_divergence.action_changed)
        self.assertEqual(changed.trajectory.selected_action, "focus_marta")
        self.assertEqual(changed.trajectory.snapshots[-1].objective_cells[_CASE_CELL], _RANSOM)

        reception_removal = next(
            item
            for item in interventions
            if item.kind == "remove_reception" and item.target_ref == "recv-blanc-confession"
        )
        removed = evaluate_counterfactual(
            story,
            domain,
            scope,
            model,
            baseline,
            reception_removal,
        )
        self.assertEqual(removed.status, "rejected")
        self.assertEqual(removed.rejection_stage, "epistemic_resolution")


if __name__ == "__main__":
    unittest.main()
