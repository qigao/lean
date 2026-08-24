from __future__ import annotations

import unittest

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from narrative_dynamics.narrative.replay import (
        direct_state,
        epistemic_state,
        objective_state,
    )
except ModuleNotFoundError as error:
    _IMPORT_ERROR = error

from narrative_dynamics.narrative.ir import TypedValue
from narrative_test_support import make_test_domain, make_test_story, target_cell


class GenericReplayTests(unittest.TestCase):
    def require_replay(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"generic narrative replay is missing: {_IMPORT_ERROR}")

    def test_direct_state_never_falls_back_to_objective_state(self) -> None:
        self.require_replay()
        story = make_test_story()
        domain = make_test_domain()
        cell = target_cell()
        self.assertEqual(
            objective_state(story, domain, at_time=2)[cell],
            TypedValue("HealthState", "recovered"),
        )
        direct = direct_state(story, domain, "bob", at_time=2)
        self.assertEqual(
            direct.resolved_values[cell], TypedValue("HealthState", "failed")
        )
        self.assertEqual(
            direct.cells[cell].evidence_kind,
            "direct_perception",
        )
        self.assertEqual(direct.cells[cell].supporting_id, "e1")

    def test_received_claim_updates_epistemic_state_but_unreceived_does_not(self) -> None:
        self.require_replay()
        domain = make_test_domain()
        cell = target_cell()
        received = epistemic_state(make_test_story(), domain, "bob")
        self.assertEqual(received.cells[cell].status, "resolved")
        self.assertEqual(
            received.resolved_values[cell], TypedValue("HealthState", "recovered")
        )
        self.assertEqual(received.cells[cell].evidence_kind, "testimony")
        self.assertEqual(received.cells[cell].supporting_id, "c1")
        self.assertEqual(received.cells[cell].source_agent, "alice")

        unreceived = epistemic_state(
            make_test_story(receive=False), domain, "bob"
        )
        self.assertEqual(
            unreceived.resolved_values[cell], TypedValue("HealthState", "failed")
        )
        self.assertEqual(unreceived.cells[cell].evidence_kind, "direct_perception")
        self.assertEqual(unreceived.cells[cell].supporting_id, "e1")

    def test_not_equals_is_a_constraint_and_conflicting_testimony_stays_explicit(self) -> None:
        self.require_replay()
        domain = make_test_domain()
        cell = target_cell()
        constrained = epistemic_state(
            make_test_story(
                claim_relation="not_equals",
                bob_observes_failure=False,
            ),
            domain,
            "bob",
        )
        self.assertEqual(constrained.cells[cell].status, "unknown")
        self.assertNotIn(cell, constrained.resolved_values)
        self.assertEqual(
            constrained.cells[cell].constraints,
            (TypedValue("HealthState", "recovered"),),
        )

        conflicted = epistemic_state(
            make_test_story(second_claim_value="failed"),
            domain,
            "bob",
        )
        self.assertEqual(conflicted.cells[cell].status, "conflicted")
        self.assertIsNone(conflicted.cells[cell].resolved_value)
        self.assertNotIn(cell, conflicted.resolved_values)
        self.assertEqual(
            tuple(
                (item.evidence_kind, item.supporting_id, item.value.value)
                for item in conflicted.evidence_history
                if item.cell == cell
            ),
            (
                ("direct_perception", "e1", "failed"),
                ("testimony", "c1", "recovered"),
                ("testimony", "c2", "failed"),
            ),
        )

    def test_time_cutoff_excludes_later_claim_evidence(self) -> None:
        self.require_replay()
        domain = make_test_domain()
        cell = target_cell()
        before_claim = epistemic_state(
            make_test_story(), domain, "bob", at_time=2
        )
        after_claim = epistemic_state(
            make_test_story(), domain, "bob", at_time=3
        )
        self.assertEqual(before_claim.cells[cell].supporting_id, "e1")
        self.assertEqual(after_claim.cells[cell].supporting_id, "c1")
        self.assertEqual(after_claim.cells[cell].evidence_logical_time, 3)


if __name__ == "__main__":
    unittest.main()
