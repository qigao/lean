from dataclasses import replace
import unittest

from narrative_dynamics.abm import EdgeSelector, SocialNetwork
from narrative_dynamics.abm.adaptive_contracts import TruthFeedback
from narrative_dynamics.abm.evolving import (
    EvolvingRoundResult,
    evolving_active_population_view,
    simulate_evolving_population,
    simulate_evolving_round,
)
from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel,
    EvolvingPopulationState,
    initialize_evolving_population,
)
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleStatus,
    PopulationLifecycleEvent,
)
from tests.evolving_fixtures import evolving_base_model


def evolving_case(
    *,
    initial_active_agent_ids=("a", "c"),
    beliefs=None,
    base=None,
):
    model = EvolvingNetworkModel(
        "unified-evolution",
        "1",
        evolving_base_model() if base is None else base,
        initial_active_agent_ids,
        0.5,
        0.5,
        0.2,
        0.8,
    )
    return model, initialize_evolving_population(model, beliefs=beliefs)


def enter(agent_id: str, belief=None):
    return PopulationLifecycleEvent(agent_id, LifecycleEventKind.ENTER, belief)


def exit_agent(agent_id: str):
    return PopulationLifecycleEvent(agent_id, LifecycleEventKind.EXIT)


def die(agent_id: str):
    return PopulationLifecycleEvent(agent_id, LifecycleEventKind.DEATH)


def member(state, agent_id: str):
    return next(item for item in state.members if item.agent_id == agent_id)


def trust(state, source: str, target: str):
    return next(
        item
        for item in state.edge_trust
        if item.edge.source_agent_id == source and item.edge.target_agent_id == target
    )


def topology(state, source: str, target: str):
    return next(
        item
        for item in state.edge_topology
        if item.edge.source_agent_id == source and item.edge.target_agent_id == target
    )


def feedback(source: str, target: str, truth: float):
    return TruthFeedback(EdgeSelector(source, target, "peer"), truth)


class EvolvingNetworkSimulationTests(unittest.TestCase):
    def test_entry_propagation_learning_and_rewiring_share_one_next_state(self):
        model, initial = evolving_case(beliefs={"a": 1.0, "c": 0.5})

        result = simulate_evolving_round(
            model,
            initial,
            events=(enter("b"),),
            feedback=(feedback("a", "b", 1.0),),
        )

        self.assertIsInstance(result, EvolvingRoundResult)
        self.assertEqual(member(result.next_state, "b").status, LifecycleStatus.ACTIVE)
        self.assertEqual(member(result.next_state, "b").belief, 0.5)
        self.assertEqual(member(result.next_state, "b").exposure_count, 1)
        self.assertEqual(trust(result.next_state, "a", "b").trust, 0.75)
        self.assertEqual(trust(result.next_state, "a", "b").feedback_count, 1)
        self.assertTrue(topology(result.next_state, "b", "c").active)
        self.assertEqual(topology(result.next_state, "b", "c").rewiring_count, 1)
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in result.transmissions),
            (("a", "b"),),
        )

    def test_learned_trust_and_formed_topology_affect_following_round(self):
        model, initial = evolving_case(beliefs={"a": 1.0, "c": 0.5})
        first = simulate_evolving_round(
            model,
            initial,
            events=(enter("b"),),
            feedback=(feedback("a", "b", 1.0),),
        )

        second = simulate_evolving_round(model, first.next_state)

        by_edge = {
            (item.source_agent_id, item.target_agent_id): item
            for item in second.transmissions
        }
        self.assertEqual(set(by_edge), {("a", "b"), ("b", "c")})
        self.assertEqual(by_edge[("a", "b")].influence, 0.75)
        self.assertEqual(by_edge[("b", "c")].influence, 0.5)
        self.assertEqual(member(second.next_state, "b").belief, 0.875)

    def test_exit_filters_transmission_and_preserves_incident_edge_history(self):
        model, initial = evolving_case(beliefs={"a": 1.0, "c": 0.5})
        first = simulate_evolving_round(model, initial, events=(enter("b"),))
        prior_trust = trust(first.next_state, "a", "b")
        prior_topology = topology(first.next_state, "a", "b")

        exited = simulate_evolving_round(
            model,
            first.next_state,
            events=(exit_agent("a"),),
        )

        self.assertEqual(member(exited.next_state, "a").status, LifecycleStatus.INACTIVE)
        self.assertFalse(
            any(item.source_agent_id == "a" for item in exited.transmissions)
        )
        self.assertEqual(trust(exited.next_state, "a", "b"), prior_trust)
        self.assertEqual(topology(exited.next_state, "a", "b"), prior_topology)
        with self.assertRaisesRegex(ValueError, "did not transmit"):
            simulate_evolving_round(
                model,
                first.next_state,
                events=(exit_agent("a"),),
                feedback=(feedback("a", "b", 1.0),),
            )

    def test_death_is_irreversible_and_zero_active_round_is_valid(self):
        model, initial = evolving_case(beliefs={"a": 1.0, "c": 0.5})
        emptied = simulate_evolving_round(
            model,
            initial,
            events=(exit_agent("a"), die("c")),
        )

        self.assertEqual(emptied.transmissions, ())
        self.assertEqual(emptied.trust_updates, ())
        self.assertEqual(emptied.rewiring_updates, ())
        self.assertIsNone(evolving_active_population_view(model, emptied.next_state))
        with self.assertRaisesRegex(ValueError, "dead agent cannot enter"):
            simulate_evolving_round(
                model,
                emptied.next_state,
                events=(enter("c"),),
            )

    def test_event_and_feedback_input_order_are_canonical(self):
        model, initial = evolving_case(beliefs={"a": 1.0, "c": 0.5})
        forward_events = simulate_evolving_round(
            model,
            initial,
            events=(die("b"), exit_agent("a")),
        )
        reverse_events = simulate_evolving_round(
            model,
            initial,
            events=(exit_agent("a"), die("b")),
        )
        self.assertEqual(forward_events, reverse_events)

        base = evolving_base_model()
        all_active_edges = tuple(replace(item, active=True) for item in base.network.edges)
        active_base = replace(
            base,
            network=SocialNetwork(base.network.agent_ids, all_active_edges),
        )
        feedback_model, feedback_initial = evolving_case(
            initial_active_agent_ids=("a", "b", "c"),
            beliefs={"a": 1.0, "b": 1.0, "c": 0.5},
            base=active_base,
        )
        first = simulate_evolving_round(
            feedback_model,
            feedback_initial,
            feedback=(feedback("a", "b", 1.0), feedback("b", "c", 1.0)),
        )
        second = simulate_evolving_round(
            feedback_model,
            feedback_initial,
            feedback=(feedback("b", "c", 1.0), feedback("a", "b", 1.0)),
        )
        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_paired_schedules_build_exact_unified_trajectory(self):
        model, initial = evolving_case(beliefs={"a": 1.0, "c": 0.5})
        trajectory = simulate_evolving_population(
            model,
            initial,
            event_schedule=((enter("b"),), ()),
            feedback_schedule=((feedback("a", "b", 1.0),), ()),
        )

        self.assertEqual(trajectory.final_state.round_index, 2)
        self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
        self.assertEqual(trajectory.final_state, trajectory.rounds[-1].next_state)

    def test_schedule_validation_requires_equal_nonempty_tuple_lengths(self):
        model, initial = evolving_case()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            simulate_evolving_population(
                model,
                initial,
                event_schedule=(),
                feedback_schedule=(),
            )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            simulate_evolving_population(
                model,
                initial,
                event_schedule=((), ()),
                feedback_schedule=((),),
            )

    def test_runtime_rejects_incomplete_catalog_coverage(self):
        model, initial = evolving_case()
        incomplete = EvolvingPopulationState(
            initial.model_id,
            initial.model_hash,
            initial.round_index,
            initial.parent_state_hash,
            initial.members[:-1],
            initial.edge_trust,
            initial.edge_topology,
        )
        with self.assertRaisesRegex(ValueError, "exact member catalog"):
            simulate_evolving_round(model, incomplete)


if __name__ == "__main__":
    unittest.main()
