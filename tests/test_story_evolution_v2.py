from __future__ import annotations

import unittest

from narrative_dynamics.story.evolution_v2 import (
    EvolutionCounterfactualV2,
    EvolutionInterventionV2,
    analyze_testimony_evolution,
)
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2
from narrative_dynamics.story.schema_v2 import load_narrative_case_v2

_TRUTHFUL = "fixtures/stories/key_location_truthful_testimony_v2.json"
_STALE = "fixtures/stories/key_location_stale_testimony_v2.json"


class NarrativeEvolutionBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = NarrativeScenarioV2.from_case(load_narrative_case_v2(_TRUTHFUL))
        self.stale = NarrativeScenarioV2.from_case(load_narrative_case_v2(_STALE))

    def test_truthful_baseline_reconstructs_world_direct_testimony_and_action(self):
        analysis = analyze_testimony_evolution(self.truthful)
        baseline = analysis.baseline
        self.assertEqual(baseline.target_object, "key")
        self.assertEqual(baseline.tracked_agents, ("bob", "alice"))
        self.assertEqual(tuple(s.logical_time for s in baseline.snapshots), (1, 2, 3, 4))
        self.assertEqual(
            tuple(s.objective_location for s in baseline.snapshots),
            ("drawer", "box", "box", "box"),
        )
        self.assertEqual(
            tuple(s.agents["bob"].direct_location for s in baseline.snapshots),
            ("drawer", "drawer", "drawer", "drawer"),
        )
        self.assertEqual(
            tuple(s.agents["alice"].direct_location for s in baseline.snapshots),
            (None, "box", "box", "box"),
        )
        self.assertEqual(
            tuple(s.agents["bob"].testimony_location for s in baseline.snapshots),
            ("drawer", "drawer", "box", "box"),
        )
        report_state = baseline.snapshots[2].agents["bob"]
        self.assertEqual(
            (report_state.evidence_kind, report_state.supporting_id, report_state.source_agent),
            ("testimony", "r1", "alice"),
        )
        self.assertEqual(
            tuple((s.trigger_kind, s.trigger_ids) for s in baseline.snapshots),
            (
                ("relocation", ("e1",)),
                ("relocation", ("e2",)),
                ("report", ("r1",)),
                ("decision", ("d1",)),
            ),
        )
        self.assertEqual(baseline.selected_action, "search_box")
        self.assertEqual(baseline.snapshots[-1].selected_action, "search_box")
        self.assertEqual(analysis.counterfactuals, ())
        self.assertFalse(analysis.mechanism_uniqueness_claimed)

    def test_stale_baseline_changes_provenance_at_report_time_without_location_change(self):
        baseline = analyze_testimony_evolution(self.stale).baseline
        bob = tuple(snapshot.agents["bob"] for snapshot in baseline.snapshots)
        self.assertEqual(
            tuple(item.testimony_location for item in bob),
            ("drawer", "drawer", "drawer", "drawer"),
        )
        self.assertEqual(
            (bob[1].evidence_kind, bob[1].supporting_id),
            ("direct_perception", "e1"),
        )
        self.assertEqual(
            (bob[2].evidence_kind, bob[2].supporting_id, bob[2].source_agent),
            ("testimony", "r1", "alice"),
        )
        self.assertEqual(baseline.selected_action, "search_drawer")

    def test_analysis_requires_validated_v2_scenario(self):
        with self.assertRaisesRegex(TypeError, "validated NarrativeScenarioV2"):
            analyze_testimony_evolution(object())

    def test_public_records_freeze_nested_values_and_reject_invalid_enums(self):
        analysis = analyze_testimony_evolution(self.truthful)
        with self.assertRaises(TypeError):
            analysis.baseline.snapshots[0].agents["bob"] = object()
        with self.assertRaisesRegex(ValueError, "intervention kind"):
            EvolutionInterventionV2(
                kind="compound",
                subject_id="x",
                agent=None,
                from_value=None,
                to_value=None,
                logical_time=None,
            )
        with self.assertRaisesRegex(ValueError, "counterfactual status"):
            EvolutionCounterfactualV2(
                intervention=EvolutionInterventionV2(
                    kind="remove_reception",
                    subject_id="r1",
                    agent="bob",
                    from_value="received",
                    to_value=None,
                    logical_time=3,
                ),
                status="unknown",
                trajectory=None,
                first_divergence=None,
                rejection_stage=None,
                rejection_reason=None,
                rejection_logical_time=None,
            )


if __name__ == "__main__":
    unittest.main()
