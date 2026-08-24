from __future__ import annotations

import json
import unittest

from narrative_dynamics.story.evolution_v2 import (
    EvolutionCounterfactualV2,
    EvolutionInterventionV2,
    _first_divergence,
    analyze_testimony_evolution,
)
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2
from narrative_dynamics.story.schema import (
    SearchActionV1,
    SearchDecisionV1,
    StoryEntitiesV1,
)
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
        self.assertIsInstance(analysis.counterfactuals, tuple)
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

    def test_analysis_to_dict_is_deterministic_json_serializable(self):
        analysis = analyze_testimony_evolution(self.truthful)
        first = analysis.to_dict()
        second = analyze_testimony_evolution(self.truthful).to_dict()
        self.assertEqual(first, second)
        encoded = json.dumps(first, sort_keys=True, separators=(",", ":"))
        self.assertEqual(json.loads(encoded), first)
        self.assertFalse(first["mechanism_uniqueness_claimed"])

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


class NarrativeEvolutionValidCounterfactualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = NarrativeScenarioV2.from_case(load_narrative_case_v2(_TRUTHFUL))
        self.stale = NarrativeScenarioV2.from_case(load_narrative_case_v2(_STALE))

    @staticmethod
    def by_kind(analysis, kind: str):
        return tuple(
            item
            for item in analysis.counterfactuals
            if item.intervention.kind == kind
        )

    def test_remove_reception_exposes_provenance_and_action_difference(self):
        item = self.by_kind(
            analyze_testimony_evolution(self.truthful), "remove_reception"
        )[0]
        self.assertEqual(item.status, "valid")
        self.assertEqual(item.trajectory.selected_action, "search_drawer")
        self.assertEqual(item.first_divergence.logical_time, 3)
        self.assertTrue(item.first_divergence.action_changed)
        self.assertIn(
            "agents.bob.evidence_kind", item.first_divergence.changed_fields
        )
        self.assertIn(
            "agents.bob.supporting_id", item.first_divergence.changed_fields
        )
        bob = item.trajectory.snapshots[2].agents["bob"]
        self.assertEqual(
            (bob.evidence_kind, bob.supporting_id),
            ("direct_perception", "e1"),
        )

        stale = self.by_kind(
            analyze_testimony_evolution(self.stale), "remove_reception"
        )[0]
        self.assertEqual(stale.status, "valid")
        self.assertEqual(stale.trajectory.selected_action, "search_drawer")
        self.assertEqual(stale.first_divergence.logical_time, 3)
        self.assertFalse(stale.first_divergence.action_changed)
        self.assertIn(
            "agents.bob.evidence_kind", stale.first_divergence.changed_fields
        )

    def test_change_report_content_diverges_at_report_time_and_flips_action(self):
        item = self.by_kind(
            analyze_testimony_evolution(self.truthful), "change_report_content"
        )[0]
        self.assertEqual(
            (
                item.intervention.subject_id,
                item.intervention.from_value,
                item.intervention.to_value,
            ),
            ("r1", "box", "drawer"),
        )
        self.assertEqual(item.status, "valid")
        self.assertEqual(item.first_divergence.logical_time, 3)
        self.assertTrue(item.first_divergence.action_changed)
        self.assertEqual(item.trajectory.selected_action, "search_drawer")
        self.assertIn(
            "agents.bob.testimony_location",
            item.first_divergence.changed_fields,
        )

        stale = self.by_kind(
            analyze_testimony_evolution(self.stale), "change_report_content"
        )[0]
        self.assertEqual(
            (stale.intervention.from_value, stale.intervention.to_value),
            ("drawer", "box"),
        )
        self.assertEqual(stale.first_divergence.logical_time, 3)
        self.assertEqual(stale.trajectory.selected_action, "search_box")
        self.assertTrue(stale.first_divergence.action_changed)

    def test_identical_trajectory_has_no_first_divergence(self):
        baseline = analyze_testimony_evolution(self.truthful).baseline
        self.assertIsNone(_first_divergence(baseline, baseline))

    def test_counterfactual_order_and_changed_paths_are_deterministic(self):
        analysis = analyze_testimony_evolution(self.truthful)
        keys = tuple(
            (
                item.intervention.kind,
                -1
                if item.intervention.logical_time is None
                else item.intervention.logical_time,
                item.intervention.subject_id,
                ""
                if item.intervention.to_value is None
                else item.intervention.to_value,
                "" if item.intervention.agent is None else item.intervention.agent,
            )
            for item in analysis.counterfactuals
        )
        self.assertEqual(keys, tuple(sorted(keys)))
        for item in analysis.counterfactuals:
            if item.first_divergence is not None:
                self.assertEqual(
                    item.first_divergence.changed_fields,
                    tuple(sorted(item.first_divergence.changed_fields)),
                )


class NarrativeEvolutionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truthful = NarrativeScenarioV2.from_case(load_narrative_case_v2(_TRUTHFUL))
        self.stale = NarrativeScenarioV2.from_case(load_narrative_case_v2(_STALE))

    @staticmethod
    def one(analysis, kind: str):
        items = tuple(
            item
            for item in analysis.counterfactuals
            if item.intervention.kind == kind
        )
        if len(items) != 1:
            raise AssertionError((kind, len(items)))
        return items[0]

    def test_remove_actor_only_observation_is_scenario_validation_rejection(self):
        item = self.one(
            analyze_testimony_evolution(self.truthful), "remove_direct_observation"
        )
        self.assertEqual(
            (
                item.intervention.subject_id,
                item.intervention.agent,
                item.intervention.logical_time,
            ),
            ("e1", "bob", 1),
        )
        self.assertEqual(item.status, "rejected")
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)
        self.assertEqual(item.rejection_stage, "scenario_validation")
        self.assertEqual(item.rejection_logical_time, 1)
        self.assertIn(
            "decision actor must have observed a relocation of the target object",
            item.rejection_reason,
        )

    def test_remove_speaker_support_is_provenance_validation_rejection(self):
        item = self.one(
            analyze_testimony_evolution(self.truthful), "remove_support_observation"
        )
        self.assertEqual(
            (
                item.intervention.subject_id,
                item.intervention.agent,
                item.intervention.from_value,
                item.intervention.logical_time,
            ),
            ("r1", "alice", "e2", 2),
        )
        self.assertEqual(item.status, "rejected")
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)
        self.assertEqual(item.rejection_stage, "scenario_validation")
        self.assertEqual(item.rejection_logical_time, 2)
        self.assertIn(
            "report speaker must have directly observed the support_event",
            item.rejection_reason,
        )

    def test_remove_reception_can_be_action_resolution_rejection(self):
        entities = StoryEntitiesV1(
            agents=self.truthful.entities.agents,
            objects=self.truthful.entities.objects,
            locations=("drawer", "box", "shelf"),
        )
        decision = SearchDecisionV1(
            id="d1",
            time=4,
            actor="bob",
            object="key",
            actions=(
                SearchActionV1(id="search_box", location="box"),
                SearchActionV1(id="search_shelf", location="shelf"),
            ),
        )
        story = NarrativeScenarioV2(
            entities=entities,
            events=self.truthful.events,
            observations=self.truthful.observations,
            reports=self.truthful.reports,
            receptions=self.truthful.receptions,
            decision=decision,
        )
        analysis = analyze_testimony_evolution(story)
        self.assertEqual(analysis.baseline.selected_action, "search_box")
        item = self.one(analysis, "remove_reception")
        self.assertEqual(item.status, "rejected")
        self.assertEqual(item.rejection_stage, "action_resolution")
        self.assertEqual(item.rejection_logical_time, 3)
        self.assertIn("exactly one decision action", item.rejection_reason)
        self.assertIsNone(item.trajectory)
        self.assertIsNone(item.first_divergence)

    def test_same_action_can_have_different_information_mechanisms(self):
        analysis = analyze_testimony_evolution(self.stale)
        reception_removed = self.one(analysis, "remove_reception")
        self.assertEqual(analysis.baseline.selected_action, "search_drawer")
        self.assertEqual(reception_removed.trajectory.selected_action, "search_drawer")
        self.assertEqual(
            analysis.baseline.snapshots[2].agents["bob"].evidence_kind,
            "testimony",
        )
        self.assertEqual(
            reception_removed.trajectory.snapshots[2].agents["bob"].evidence_kind,
            "direct_perception",
        )
        self.assertFalse(reception_removed.first_divergence.action_changed)
        self.assertFalse(analysis.mechanism_uniqueness_claimed)

    def test_full_intervention_kind_set_is_exact(self):
        kinds = {
            item.intervention.kind
            for item in analyze_testimony_evolution(self.truthful).counterfactuals
        }
        self.assertEqual(
            kinds,
            {
                "change_report_content",
                "remove_direct_observation",
                "remove_reception",
                "remove_support_observation",
            },
        )


if __name__ == "__main__":
    unittest.main()
