import math
import unittest

from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleMemberState,
    LifecycleStatus,
    PopulationLifecycleEvent,
    PopulationLifecycleModel,
    PopulationLifecycleState,
    initialize_lifecycle_population,
)
from tests.lifecycle_fixtures import lifecycle_base_model


def member(state: PopulationLifecycleState, agent_id: str) -> LifecycleMemberState:
    return next(item for item in state.members if item.agent_id == agent_id)


class PopulationLifecycleContractTests(unittest.TestCase):
    def test_initialization_covers_catalog_and_marks_effective_population(self):
        model = PopulationLifecycleModel(
            "population-lifecycle",
            "1",
            lifecycle_base_model(),
            ("a",),
        )

        state = initialize_lifecycle_population(model, beliefs={"a": 1.0, "b": 0.2})

        self.assertEqual(state.model_id, model.model_id)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertEqual(state.round_index, 0)
        self.assertEqual(tuple(item.agent_id for item in state.members), ("a", "b", "c"))
        self.assertEqual(member(state, "a").status, LifecycleStatus.ACTIVE)
        self.assertEqual(member(state, "a").entry_count, 1)
        self.assertTrue(member(state, "a").broadcasting)
        self.assertEqual(member(state, "b").status, LifecycleStatus.INACTIVE)
        self.assertEqual(member(state, "b").belief, 0.2)
        self.assertEqual(member(state, "b").exposure_count, 1)
        self.assertEqual(member(state, "b").entry_count, 0)
        self.assertFalse(member(state, "b").broadcasting)

    def test_model_and_initial_state_are_canonical(self):
        first = PopulationLifecycleModel(
            "population-lifecycle",
            "1",
            lifecycle_base_model(),
            ("b", "a"),
        )
        second = PopulationLifecycleModel(
            "population-lifecycle",
            "1",
            lifecycle_base_model(reverse=True),
            ("a", "b"),
        )

        self.assertEqual(first, second)
        self.assertEqual(first.initial_active_agent_ids, ("a", "b"))
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            initialize_lifecycle_population(first, beliefs={"a": 1.0}).content_hash,
            initialize_lifecycle_population(second, beliefs={"a": 1.0}).content_hash,
        )

    def test_lifecycle_events_validate_kind_and_entry_belief(self):
        enter = PopulationLifecycleEvent("b", LifecycleEventKind.ENTER, 0.8)
        self.assertEqual(enter.to_dict(), {"agent_id": "b", "kind": "enter", "belief": 0.8})

        for bad in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(belief=bad):
                with self.assertRaises((TypeError, ValueError)):
                    PopulationLifecycleEvent("b", LifecycleEventKind.ENTER, bad)
        with self.assertRaisesRegex(ValueError, "only ENTER"):
            PopulationLifecycleEvent("b", LifecycleEventKind.EXIT, 0.4)
        with self.assertRaisesRegex(TypeError, "kind"):
            PopulationLifecycleEvent("b", "enter")

    def test_lifecycle_model_rejects_empty_duplicate_and_unknown_initial_active_ids(self):
        base = lifecycle_base_model()
        with self.assertRaisesRegex(ValueError, "at least one initially active"):
            PopulationLifecycleModel("lifecycle", "1", base, ())
        with self.assertRaisesRegex(ValueError, "unique"):
            PopulationLifecycleModel("lifecycle", "1", base, ("a", "a"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            PopulationLifecycleModel("lifecycle", "1", base, ("missing",))

    def test_member_state_rejects_broadcasting_outside_active_population(self):
        with self.assertRaisesRegex(ValueError, "cannot broadcast"):
            LifecycleMemberState(
                "a",
                1.0,
                1,
                True,
                LifecycleStatus.INACTIVE,
                0,
                0,
            )
        with self.assertRaisesRegex(ValueError, "cannot broadcast"):
            LifecycleMemberState(
                "a",
                1.0,
                1,
                True,
                LifecycleStatus.DEAD,
                1,
                0,
            )

    def test_state_rejects_duplicate_members_and_invalid_parent_chain(self):
        model = PopulationLifecycleModel(
            "population-lifecycle",
            "1",
            lifecycle_base_model(),
            ("a",),
        )
        initial = initialize_lifecycle_population(model)
        with self.assertRaisesRegex(ValueError, "member ids must be unique"):
            PopulationLifecycleState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                (initial.members[0], initial.members[0]),
            )
        with self.assertRaisesRegex(ValueError, "round zero"):
            PopulationLifecycleState(
                initial.model_id,
                initial.model_hash,
                0,
                initial.content_hash,
                initial.members,
            )

    def test_initialization_rejects_unknown_seed(self):
        model = PopulationLifecycleModel(
            "population-lifecycle",
            "1",
            lifecycle_base_model(),
            ("a",),
        )
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            initialize_lifecycle_population(model, beliefs={"missing": 1.0})


if __name__ == "__main__":
    unittest.main()
