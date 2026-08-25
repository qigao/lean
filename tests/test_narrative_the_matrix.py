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
    from narrative_dynamics.narrative.domains.the_matrix import (
        matrix_direct_model,
        matrix_domain,
        matrix_epistemic_model,
        matrix_omniscient_model,
        matrix_red_pill_story,
    )
except ImportError as error:
    _IMPORT_ERROR = error


_WORLD_CELL = StateCellRef(EntityRef("experienced-world", "World"), "world.mode")
_SIMULATED = TypedValue("WorldMode", "simulated")
_REAL = TypedValue("WorldMode", "real")


class TheMatrixConformanceTests(unittest.TestCase):
    def require_matrix(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"The Matrix conformance domain is missing: {_IMPORT_ERROR}")

    def test_red_pill_slice_separates_objective_direct_and_epistemic_world_models(self) -> None:
        self.require_matrix()
        story = matrix_red_pill_story()
        domain = matrix_domain()

        self.assertEqual(
            tuple((event.id, event.logical_time) for event in story.events),
            (("matrix-is-simulation", 1),),
        )
        self.assertEqual(objective_state(story, domain, at_time=1)[_WORLD_CELL], _SIMULATED)

        neo_direct = direct_state(story, domain, "neo", at_time=2)
        self.assertNotIn(_WORLD_CELL, neo_direct.cells)

        neo_epistemic = epistemic_state(story, domain, "neo", at_time=2)
        view = neo_epistemic.cells[_WORLD_CELL]
        self.assertEqual(view.resolved_value, _SIMULATED)
        self.assertEqual(view.evidence_kind, "testimony")
        self.assertEqual(view.supporting_id, "morpheus-matrix-claim")
        self.assertEqual(view.source_agent, "morpheus")
        self.assertEqual(view.evidence_refs, ("morpheus-matrix-claim", "matrix-is-simulation"))

        with self.assertRaises(EpistemicResolutionError):
            analyze_narrative(
                story,
                domain,
                "neo-pill-choice",
                matrix_direct_model(),
            )

        actions = tuple(
            analyze_narrative(story, domain, "neo-pill-choice", model).baseline.selected_action
            for model in (
                matrix_epistemic_model(),
                matrix_omniscient_model(),
            )
        )
        self.assertEqual(actions, ("take_red_pill", "take_red_pill"))

    def test_morpheus_claim_mutation_flips_choice_and_reception_removal_fails_closed(self) -> None:
        self.require_matrix()
        story = matrix_red_pill_story()
        domain = matrix_domain()
        model = matrix_epistemic_model()
        scope = derive_analysis_scope(story, domain, "neo-pill-choice")
        baseline = build_trajectory(story, domain, scope, model)
        interventions = generate_minimal_interventions(story, domain, scope)

        claim_change = next(
            item
            for item in interventions
            if item.kind == "change_claim_value"
            and item.target_ref == "morpheus-matrix-claim"
            and item.to_value == _REAL
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
        self.assertEqual(changed.first_divergence.logical_time, 2)
        self.assertTrue(changed.first_divergence.action_changed)
        self.assertEqual(changed.trajectory.selected_action, "take_blue_pill")
        self.assertEqual(changed.trajectory.snapshots[-1].objective_cells[_WORLD_CELL], _SIMULATED)

        reception_removal = next(
            item
            for item in interventions
            if item.kind == "remove_reception" and item.target_ref == "recv-neo-morpheus-claim"
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
