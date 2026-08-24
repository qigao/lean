from __future__ import annotations

import unittest

from narrative_dynamics.narrative.analysis import (
    analyze_narrative,
    build_trajectory,
    derive_analysis_scope,
)
from narrative_dynamics.narrative.interventions import (
    evaluate_counterfactual,
    generate_minimal_interventions,
)
from narrative_dynamics.narrative.ir import EntityRef, StateCellRef
from narrative_dynamics.story import (
    NarrativeScenarioV2,
    analyze_testimony_evolution,
    load_narrative_case_v2,
)

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.compat_story_v2 import (
        adapt_testimony_story_v2,
        legacy_direct_search_model,
        legacy_epistemic_search_model,
        legacy_omniscient_search_model,
        location_testimony_domain,
    )
except ImportError as error:
    _IMPORT_ERROR = error


class StoryV2CompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = NarrativeScenarioV2.from_case(
            load_narrative_case_v2(
                "fixtures/stories/key_location_truthful_testimony_v2.json"
            )
        )
        self.stale = NarrativeScenarioV2.from_case(
            load_narrative_case_v2(
                "fixtures/stories/key_location_stale_testimony_v2.json"
            )
        )

    def require_compat(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"Testimony V2 compatibility module is missing: {_IMPORT_ERROR}")

    @staticmethod
    def value_id(view) -> str | None:
        if view is None or view.resolved_value is None:
            return None
        value = view.resolved_value.value
        return value.entity_id if isinstance(value, EntityRef) else str(value)

    def test_three_model_actions_timeline_and_epistemic_provenance(self) -> None:
        self.require_compat()
        expected = {
            "truthful": ("search_drawer", "search_box", "search_box"),
            "stale": ("search_drawer", "search_drawer", "search_box"),
        }
        epistemic_values = {
            "truthful": ("drawer", "drawer", "box", "box"),
            "stale": ("drawer", "drawer", "drawer", "drawer"),
        }
        for name, legacy_story in (
            ("truthful", self.truthful),
            ("stale", self.stale),
        ):
            generic = adapt_testimony_story_v2(legacy_story)
            domain = location_testimony_domain()
            direct = legacy_direct_search_model()
            epistemic = legacy_epistemic_search_model()
            omniscient = legacy_omniscient_search_model()
            actions = tuple(
                analyze_narrative(generic, domain, "d1", model).baseline.selected_action
                for model in (direct, epistemic, omniscient)
            )
            self.assertEqual(actions, expected[name])

            legacy = analyze_testimony_evolution(legacy_story).baseline
            generic_base = analyze_narrative(
                generic, domain, "d1", epistemic
            ).baseline
            self.assertEqual(
                tuple(snapshot.logical_time for snapshot in generic_base.snapshots),
                tuple(snapshot.logical_time for snapshot in legacy.snapshots),
            )
            self.assertEqual(
                tuple(snapshot.logical_time for snapshot in generic_base.snapshots),
                (1, 2, 3, 4),
            )

            cell = StateCellRef(EntityRef("key", "Object"), "object.location")
            objective = tuple(
                snapshot.objective_cells[cell].value.entity_id
                for snapshot in generic_base.snapshots
            )
            bob_direct = tuple(
                self.value_id(snapshot.agent_views["bob"].direct_cells[cell])
                for snapshot in generic_base.snapshots
            )
            bob_epistemic = tuple(
                self.value_id(snapshot.agent_views["bob"].epistemic_cells[cell])
                for snapshot in generic_base.snapshots
            )
            self.assertEqual(objective, ("drawer", "box", "box", "box"))
            self.assertEqual(bob_direct, ("drawer", "drawer", "drawer", "drawer"))
            self.assertEqual(bob_epistemic, epistemic_values[name])

            report_view = generic_base.snapshots[2].agent_views["bob"].epistemic_cells[cell]
            self.assertEqual(report_view.evidence_kind, "testimony")
            self.assertEqual(report_view.supporting_id, "r1")
            self.assertEqual(report_view.source_agent, "alice")
            self.assertEqual(report_view.evidence_logical_time, 3)

    def test_counterfactual_and_rejection_equivalence_locks(self) -> None:
        self.require_compat()
        generic = adapt_testimony_story_v2(self.truthful)
        domain = location_testimony_domain()
        epistemic = legacy_epistemic_search_model()
        scope = derive_analysis_scope(generic, domain, "d1")
        baseline = build_trajectory(generic, domain, scope, epistemic)
        interventions = generate_minimal_interventions(generic, domain, scope)

        remove_reception = next(
            item for item in interventions if item.kind == "remove_reception"
        )
        reception_result = evaluate_counterfactual(
            generic,
            domain,
            scope,
            epistemic,
            baseline,
            remove_reception,
        )
        self.assertEqual(reception_result.status, "valid")
        self.assertEqual(reception_result.first_divergence.logical_time, 3)
        self.assertTrue(reception_result.first_divergence.action_changed)

        change_claim = next(
            item
            for item in interventions
            if item.kind == "change_claim_value"
            and item.target_ref == "r1"
            and item.to_value is not None
            and item.to_value.value.entity_id == "drawer"
        )
        claim_result = evaluate_counterfactual(
            generic, domain, scope, epistemic, baseline, change_claim
        )
        self.assertEqual(claim_result.status, "valid")
        self.assertEqual(claim_result.first_divergence.logical_time, 3)
        self.assertTrue(claim_result.first_divergence.action_changed)

        direct = legacy_direct_search_model()
        direct_base = build_trajectory(generic, domain, scope, direct)
        remove_bob_direct = next(
            item
            for item in interventions
            if item.kind == "remove_observation"
            and item.target_ref == "obs:e1:bob"
        )
        self.assertEqual(
            evaluate_counterfactual(
                generic, domain, scope, direct, direct_base, remove_bob_direct
            ).rejection_stage,
            "epistemic_resolution",
        )

        remove_alice_support = next(
            item
            for item in interventions
            if item.kind == "remove_observation"
            and item.target_ref == "obs:e2:alice"
        )
        self.assertEqual(
            evaluate_counterfactual(
                generic,
                domain,
                scope,
                epistemic,
                baseline,
                remove_alice_support,
            ).rejection_stage,
            "narrative_validation",
        )


if __name__ == "__main__":
    unittest.main()
