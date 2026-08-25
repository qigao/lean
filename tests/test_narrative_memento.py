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
from narrative_dynamics.narrative.replay import epistemic_state, objective_state


_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.domains.memento import (
        memento_direct_model,
        memento_domain,
        memento_epistemic_model,
        memento_omniscient_model,
        memento_self_record_story,
    )
except ImportError as error:
    _IMPORT_ERROR = error


_CASE_CELL = StateCellRef(EntityRef("john-g-case", "Case"), "case.directive")
_RECORD_AUTHOR_CELL = StateCellRef(
    EntityRef("teddy-target-record", "MemoryRecord"),
    "record.author",
)
_TARGET_TEDDY = TypedValue("Directive", "target_teddy")
_CONTINUE_SEARCH = TypedValue("Directive", "continue_search")
_BEFORE_RESET = TypedValue(
    "AgentRef",
    EntityRef("leonard-before-reset", "Agent"),
)


class MementoConformanceTests(unittest.TestCase):
    def require_memento(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"Memento conformance domain is missing: {_IMPORT_ERROR}")

    def test_self_authored_external_memory_crosses_reset_without_becoming_objective_truth(self) -> None:
        self.require_memento()
        story = memento_self_record_story()
        domain = memento_domain()

        self.assertEqual(
            tuple((event.id, event.logical_time) for event in story.events),
            (("teddy-explanation-context", 1), ("write-target-record", 3)),
        )

        objective = objective_state(story, domain, at_time=5)
        self.assertNotIn(_CASE_CELL, objective)
        self.assertEqual(objective[_RECORD_AUTHOR_CELL], _BEFORE_RESET)

        before_reset = epistemic_state(
            story,
            domain,
            "leonard-before-reset",
            at_time=2,
        )
        transient = before_reset.cells[_CASE_CELL]
        self.assertEqual(transient.resolved_value, _CONTINUE_SEARCH)
        self.assertEqual(transient.evidence_kind, "testimony")
        self.assertEqual(transient.supporting_id, "teddy-explanation")
        self.assertEqual(transient.source_agent, "teddy")

        after_reset_before_record = epistemic_state(
            story,
            domain,
            "leonard-after-reset",
            at_time=2,
        )
        self.assertNotIn(_CASE_CELL, after_reset_before_record.cells)

        after_reset = epistemic_state(
            story,
            domain,
            "leonard-after-reset",
            at_time=4,
        )
        persisted = after_reset.cells[_CASE_CELL]
        self.assertEqual(persisted.resolved_value, _TARGET_TEDDY)
        self.assertEqual(persisted.evidence_kind, "testimony")
        self.assertEqual(persisted.supporting_id, "leonard-target-record")
        self.assertEqual(persisted.source_agent, "leonard-before-reset")
        self.assertEqual(
            persisted.evidence_refs,
            ("leonard-target-record", "write-target-record"),
        )

        for model in (memento_direct_model(), memento_omniscient_model()):
            with self.assertRaises(EpistemicResolutionError):
                analyze_narrative(story, domain, "post-reset-target-choice", model)

        result = analyze_narrative(
            story,
            domain,
            "post-reset-target-choice",
            memento_epistemic_model(),
        )
        self.assertEqual(result.baseline.selected_action, "pursue_teddy")

    def test_self_record_mutation_flips_post_reset_choice_and_reception_removal_fails_closed(self) -> None:
        self.require_memento()
        story = memento_self_record_story()
        domain = memento_domain()
        model = memento_epistemic_model()
        scope = derive_analysis_scope(story, domain, "post-reset-target-choice")
        baseline = build_trajectory(story, domain, scope, model)
        interventions = generate_minimal_interventions(story, domain, scope)

        claim_change = next(
            item
            for item in interventions
            if item.kind == "change_claim_value"
            and item.target_ref == "leonard-target-record"
            and item.to_value == _CONTINUE_SEARCH
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
        self.assertEqual(changed.first_divergence.logical_time, 4)
        self.assertTrue(changed.first_divergence.action_changed)
        self.assertEqual(changed.trajectory.selected_action, "continue_search")
        self.assertIsNone(
            changed.trajectory.snapshots[-1].objective_cells[_CASE_CELL]
        )
        self.assertEqual(
            changed.trajectory.snapshots[-1].objective_cells[_RECORD_AUTHOR_CELL],
            _BEFORE_RESET,
        )

        reception_removal = next(
            item
            for item in interventions
            if item.kind == "remove_reception"
            and item.target_ref == "recv-after-reset-self-record"
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
