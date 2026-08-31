import unittest

from narrative_dynamics.abm import InformationTransmission
from narrative_dynamics.abm.lifecycle import (
    LifecycleRoundResult,
    active_population_view,
    simulate_lifecycle_population,
    simulate_lifecycle_round,
)
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleStatus,
    PopulationLifecycleEvent,
    PopulationLifecycleModel,
    PopulationLifecycleState,
    initialize_lifecycle_population,
)
from tests.lifecycle_fixtures import lifecycle_base_model


def lifecycle_case(*, active: tuple[str, ...] = ("a",)):
    model = PopulationLifecycleModel(
        "population-lifecycle",
        "1",
        lifecycle_base_model(),
        active,
    )
    return model, initialize_lifecycle_population(model, beliefs={"a": 1.0})


def event(agent_id: str, kind: LifecycleEventKind, belief=None):
    return PopulationLifecycleEvent(agent_id, kind, belief)


def member(state: PopulationLifecycleState, agent_id: str):
    return next(item for item in state.members if item.agent_id == agent_id)


class PopulationLifecycleSimulationTests(unittest.TestCase):
    def test_entry_applies_before_one_hop_propagation(self):
        model, initial = lifecycle_case()

        first = simulate_lifecycle_round(
            model,
            initial,
            events=(event("b", LifecycleEventKind.ENTER),),
        )

        self.assertIsInstance(first, LifecycleRoundResult)
        self.assertEqual(member(first.next_state, "b").status, LifecycleStatus.ACTIVE)
        self.assertEqual(member(first.next_state, "b").belief, 1.0)
        self.assertEqual(member(first.next_state, "b").entry_count, 1)
        self.assertEqual(member(first.next_state, "b").exposure_count, 1)
        self.assertEqual(member(first.next_state, "c").belief, 0.0)
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in first.transmissions),
            (("a", "b"),),
        )

        second = simulate_lifecycle_round(
            model,
            first.next_state,
            events=(event("c", LifecycleEventKind.ENTER),),
        )
        self.assertEqual(member(second.next_state, "c").belief, 1.0)
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in second.transmissions),
            (("a", "b"), ("b", "c")),
        )

    def test_exit_before_propagation_removes_bridge(self):
        model, initial = lifecycle_case(active=("a", "b", "c"))

        result = simulate_lifecycle_round(
            model,
            initial,
            events=(event("b", LifecycleEventKind.EXIT),),
        )

        self.assertEqual(member(result.next_state, "b").status, LifecycleStatus.INACTIVE)
        self.assertEqual(member(result.next_state, "b").exit_count, 1)
        self.assertFalse(member(result.next_state, "b").broadcasting)
        self.assertEqual(result.transmissions, ())
        self.assertEqual(member(result.next_state, "c").belief, 0.0)

    def test_death_is_irreversible_and_does_not_count_as_voluntary_exit(self):
        model, initial = lifecycle_case(active=("a", "b"))
        dead = simulate_lifecycle_round(
            model,
            initial,
            events=(event("b", LifecycleEventKind.DEATH),),
        )

        self.assertEqual(member(dead.next_state, "b").status, LifecycleStatus.DEAD)
        self.assertEqual(member(dead.next_state, "b").exit_count, 0)
        with self.assertRaisesRegex(ValueError, "dead agent cannot enter"):
            simulate_lifecycle_round(
                model,
                dead.next_state,
                events=(event("b", LifecycleEventKind.ENTER),),
            )

    def test_zero_active_population_advances_without_transmission(self):
        model, initial = lifecycle_case()
        emptied = simulate_lifecycle_round(
            model,
            initial,
            events=(event("a", LifecycleEventKind.EXIT),),
        )
        advanced = simulate_lifecycle_round(model, emptied.next_state)

        self.assertEqual(emptied.transmissions, ())
        self.assertEqual(advanced.transmissions, ())
        self.assertEqual(advanced.round_index, 2)
        self.assertEqual(advanced.next_state.parent_state_hash, emptied.next_state.content_hash)
        self.assertIsNone(active_population_view(model, advanced.next_state))

    def test_returning_agent_preserves_state_unless_entry_seeds_belief(self):
        model, initial = lifecycle_case(active=("a", "b"))
        exited = simulate_lifecycle_round(
            model,
            initial,
            events=(event("b", LifecycleEventKind.EXIT),),
        )
        returned = simulate_lifecycle_round(
            model,
            exited.next_state,
            events=(event("b", LifecycleEventKind.ENTER),),
        )
        self.assertEqual(member(returned.next_state, "b").belief, 1.0)
        self.assertEqual(member(returned.next_state, "b").entry_count, 2)

        exited_again = simulate_lifecycle_round(
            model,
            returned.next_state,
            events=(event("b", LifecycleEventKind.EXIT),),
        )
        seeded = simulate_lifecycle_round(
            model,
            exited_again.next_state,
            events=(event("b", LifecycleEventKind.ENTER, 0.25),),
        )
        self.assertEqual(member(seeded.next_state, "b").belief, 1.0)
        self.assertEqual(member(seeded.next_state, "b").exposure_count, 3)

    def test_event_input_order_is_canonical_and_replay_stable(self):
        model, initial = lifecycle_case()
        forward = simulate_lifecycle_round(
            model,
            initial,
            events=(
                event("c", LifecycleEventKind.DEATH),
                event("b", LifecycleEventKind.ENTER),
            ),
        )
        reverse = simulate_lifecycle_round(
            model,
            initial,
            events=(
                event("b", LifecycleEventKind.ENTER),
                event("c", LifecycleEventKind.DEATH),
            ),
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.content_hash, reverse.content_hash)
        self.assertEqual(tuple(item.agent_id for item in forward.events), ("b", "c"))

    def test_event_schedule_builds_a_valid_trajectory(self):
        model, initial = lifecycle_case()
        trajectory = simulate_lifecycle_population(
            model,
            initial,
            event_schedule=(
                (event("b", LifecycleEventKind.ENTER),),
                (event("c", LifecycleEventKind.ENTER),),
            ),
        )

        self.assertEqual(trajectory.final_state.round_index, 2)
        self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
        self.assertEqual(trajectory.initial_state, initial)
        self.assertEqual(trajectory.final_state, trajectory.rounds[-1].next_state)

    def test_active_population_view_contains_only_induced_members(self):
        model, initial = lifecycle_case(active=("a", "b"))
        view = active_population_view(model, initial)

        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(tuple(item.agent_id for item in view.agents), ("a", "b"))
        self.assertEqual(view.round_index, initial.round_index)

    def test_round_rejects_invalid_event_targets_transitions_and_duplicates(self):
        model, initial = lifecycle_case()
        with self.assertRaisesRegex(ValueError, "unknown catalog agent"):
            simulate_lifecycle_round(
                model,
                initial,
                events=(event("missing", LifecycleEventKind.ENTER),),
            )
        with self.assertRaisesRegex(ValueError, "event per agent"):
            simulate_lifecycle_round(
                model,
                initial,
                events=(
                    event("b", LifecycleEventKind.ENTER),
                    event("b", LifecycleEventKind.DEATH),
                ),
            )
        with self.assertRaisesRegex(ValueError, "only inactive agents can enter"):
            simulate_lifecycle_round(
                model,
                initial,
                events=(event("a", LifecycleEventKind.ENTER),),
            )
        with self.assertRaisesRegex(ValueError, "only active agents can exit"):
            simulate_lifecycle_round(
                model,
                initial,
                events=(event("b", LifecycleEventKind.EXIT),),
            )

    def test_round_result_exposes_v1_transmission_contract(self):
        model, initial = lifecycle_case()
        result = simulate_lifecycle_round(
            model,
            initial,
            events=(event("b", LifecycleEventKind.ENTER),),
        )
        self.assertIsInstance(result.transmissions[0], InformationTransmission)


if __name__ == "__main__":
    unittest.main()
