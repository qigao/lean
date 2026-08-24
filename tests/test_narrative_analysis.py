from __future__ import annotations

import unittest

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.analysis import (
        analyze_narrative,
        build_trajectory,
        derive_analysis_scope,
    )
except ImportError as error:
    _IMPORT_ERROR = error

from narrative_test_support import (
    make_analysis_case,
    make_same_action_different_basis_case,
)


class GenericAnalysisTests(unittest.TestCase):
    def require_analysis(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"generic narrative analysis module is missing: {_IMPORT_ERROR}")

    def test_scope_and_same_action_different_basis(self) -> None:
        self.require_analysis()
        story, domain, direct, epistemic, omniscient = make_analysis_case()
        scope = derive_analysis_scope(story, domain, "d1")
        self.assertEqual(scope.tracked_agents, ("bob", "alice"))
        self.assertEqual(scope.snapshot_times, (1, 2, 3, 4))

        story, domain, direct, epistemic = make_same_action_different_basis_case()
        analysis = analyze_narrative(
            story,
            domain,
            "d1",
            direct,
            comparison_models=(epistemic,),
        )
        pair = analysis.mechanism_comparison.pairwise[0]
        self.assertTrue(pair.action_equal)
        self.assertFalse(pair.basis_equal)
        self.assertFalse(analysis.mechanism_comparison.mechanism_uniqueness_claimed)

    def test_trajectory_uses_frozen_scope_and_decision_only_at_decision_time(self) -> None:
        self.require_analysis()
        story, domain, direct, epistemic, omniscient = make_analysis_case()
        scope = derive_analysis_scope(story, domain, "d1")
        trajectory = build_trajectory(story, domain, scope, direct)
        self.assertEqual(
            tuple(snapshot.logical_time for snapshot in trajectory.snapshots),
            scope.snapshot_times,
        )
        self.assertTrue(
            all(snapshot.decision_result is None for snapshot in trajectory.snapshots[:-1])
        )
        self.assertIsNotNone(trajectory.snapshots[-1].decision_result)
        self.assertEqual(trajectory.selected_action, "restart")


if __name__ == "__main__":
    unittest.main()
