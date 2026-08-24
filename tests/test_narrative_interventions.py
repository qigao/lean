from __future__ import annotations

import unittest

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.analysis import (
        build_trajectory,
        derive_analysis_scope,
    )
    from narrative_dynamics.narrative.interventions import (
        evaluate_counterfactual,
        generate_minimal_interventions,
    )
except ImportError as error:
    _IMPORT_ERROR = error

from narrative_test_support import (
    make_analysis_case,
    make_test_domain,
    make_test_story,
)


class GenericInterventionTests(unittest.TestCase):
    def require_interventions(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"generic narrative interventions module is missing: {_IMPORT_ERROR}")

    def test_kinds_and_reception_divergence(self) -> None:
        self.require_interventions()
        story, domain, direct, epistemic, omniscient = make_analysis_case()
        scope = derive_analysis_scope(story, domain, "d1")
        items = generate_minimal_interventions(story, domain, scope)
        self.assertEqual(
            {item.kind for item in items},
            {
                "remove_event",
                "change_event_argument",
                "remove_observation",
                "change_claim_value",
                "remove_reception",
            },
        )
        baseline = build_trajectory(story, domain, scope, epistemic)
        intervention = next(item for item in items if item.kind == "remove_reception")
        result = evaluate_counterfactual(
            story,
            domain,
            scope,
            epistemic,
            baseline,
            intervention,
        )
        self.assertEqual(result.status, "valid")
        self.assertIsNotNone(result.first_divergence)
        self.assertEqual(result.first_divergence.logical_time, 3)
        self.assertTrue(result.first_divergence.action_changed)

    def test_rejection_stages(self) -> None:
        self.require_interventions()
        story, domain, direct, model, omniscient = make_analysis_case()
        scope = derive_analysis_scope(story, domain, "d1")
        baseline = build_trajectory(story, domain, scope, model)
        items = generate_minimal_interventions(story, domain, scope)

        support = next(
            item
            for item in items
            if item.kind == "remove_observation" and item.target_ref == "o2"
        )
        self.assertEqual(
            evaluate_counterfactual(
                story, domain, scope, model, baseline, support
            ).rejection_stage,
            "narrative_validation",
        )

        healthy = next(
            item
            for item in items
            if item.kind == "change_claim_value"
            and item.to_value is not None
            and item.to_value.value == "healthy"
        )
        self.assertEqual(
            evaluate_counterfactual(
                story, domain, scope, model, baseline, healthy
            ).rejection_stage,
            "decision_resolution",
        )

        conflict_story = make_test_story(
            claim_value="recovered",
            second_claim_value="recovered",
        )
        conflict_domain = make_test_domain()
        conflict_scope = derive_analysis_scope(conflict_story, conflict_domain, "d1")
        conflict_base = build_trajectory(
            conflict_story,
            conflict_domain,
            conflict_scope,
            model,
        )
        conflict = next(
            item
            for item in generate_minimal_interventions(
                conflict_story, conflict_domain, conflict_scope
            )
            if item.kind == "change_claim_value"
            and item.target_ref == "c1"
            and item.to_value is not None
            and item.to_value.value == "failed"
        )
        self.assertEqual(
            evaluate_counterfactual(
                conflict_story,
                conflict_domain,
                conflict_scope,
                model,
                conflict_base,
                conflict,
            ).rejection_stage,
            "epistemic_resolution",
        )


if __name__ == "__main__":
    unittest.main()
