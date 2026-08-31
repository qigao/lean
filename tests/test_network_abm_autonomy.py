from dataclasses import replace
import unittest

from narrative_dynamics.abm import EdgeSelector, SocialNetwork
from narrative_dynamics.abm.autonomy import (
    AutonomousRoundResult,
    simulate_autonomous_population,
    simulate_autonomous_round,
)
from narrative_dynamics.abm.autonomy_contracts import (
    AutonomousNetworkModel,
    RoleDecisionPolicy,
    SharingDecision,
    TruthObservation,
    initialize_autonomous_population,
)
from narrative_dynamics.abm.evolving_contracts import EvolvingNetworkModel
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleStatus,
    PopulationLifecycleEvent,
)
from tests.autonomy_fixtures import autonomous_model, role_policies
from tests.evolving_fixtures import evolving_base_model


def enter(agent_id: str, belief=None):
    return PopulationLifecycleEvent(agent_id, LifecycleEventKind.ENTER, belief)


def die(agent_id: str):
    return PopulationLifecycleEvent(agent_id, LifecycleEventKind.DEATH)


def edge(source: str, target: str):
    return EdgeSelector(source, target, "peer")


def truth(source: str, target: str, observed_truth: float):
    return TruthObservation(edge(source, target), observed_truth)


def member(state, agent_id: str):
    return next(
        item for item in state.evolving_state.members if item.agent_id == agent_id
    )


def resource(state, agent_id: str):
    return next(item for item in state.agents if item.agent_id == agent_id)


def edge_trust(state, source: str, target: str):
    return next(
        item.trust
        for item in state.evolving_state.edge_trust
        if item.edge.source_agent_id == source and item.edge.target_agent_id == target
    )


def intent_for(result, agent_id: str):
    return next(item for item in result.intents if item.agent_id == agent_id)


def model_with_policies(policies):
    base = autonomous_model()
    return AutonomousNetworkModel(
        base.model_id,
        base.version,
        base.evolving_model,
        tuple(policies),
    )


class AgentAutonomySimulationTests(unittest.TestCase):
    def test_relay_verifies_low_trust_and_spends_budget_after_receiving(self):
        model = autonomous_model()
        initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        result = simulate_autonomous_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b", 1.0),),
        )

        self.assertIsInstance(result, AutonomousRoundResult)
        intent = intent_for(result, "b")
        self.assertEqual(intent.verification_edge, edge("a", "b"))
        self.assertEqual(intent.verification_budget_before, 2)
        self.assertEqual(intent.verification_budget_after, 1)
        self.assertEqual(intent.sharing_decision, SharingDecision.SHARE)
        self.assertEqual(member(result.next_state, "b").belief, 0.5)
        self.assertEqual(resource(result.next_state, "b").verification_count, 1)
        self.assertEqual(edge_trust(result.next_state, "a", "b"), 0.75)

    def test_exhausted_budget_blocks_verification_of_low_trust_source(self):
        policies = tuple(
            replace(item, initial_verification_budget=1)
            if item.role == "relay"
            else replace(item, initial_verification_budget=0)
            if item.role == "recipient"
            else item
            for item in role_policies()
        )
        model = model_with_policies(policies)
        initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )
        first = simulate_autonomous_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b", 0.0),),
        )
        self.assertEqual(edge_trust(first.next_state, "a", "b"), 0.25)
        self.assertEqual(resource(first.next_state, "b").remaining_verification_budget, 0)

        second = simulate_autonomous_round(model, first.next_state)

        self.assertIsNone(intent_for(second, "b").verification_edge)
        self.assertEqual(resource(second.next_state, "b").verification_count, 1)

    def test_silent_decision_suppresses_outgoing_edges_in_following_round(self):
        policies = tuple(
            replace(item, sharing_enabled=False) if item.role == "relay" else item
            for item in role_policies()
        )
        model = model_with_policies(policies)
        initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )
        first = simulate_autonomous_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b", 1.0),),
        )
        self.assertEqual(intent_for(first, "b").sharing_decision, SharingDecision.SILENT)

        second = simulate_autonomous_round(model, first.next_state)

        self.assertTrue(
            any(item.source_agent_id == "a" for item in second.transmissions)
        )
        self.assertFalse(
            any(item.source_agent_id == "b" for item in second.transmissions)
        )

    def test_low_belief_agent_verifies_then_exits_autonomously(self):
        policies = tuple(
            replace(item, exit_belief_threshold=0.6)
            if item.role == "relay"
            else item
            for item in role_policies()
        )
        model = model_with_policies(policies)
        initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        result = simulate_autonomous_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b", 1.0),),
        )

        intent = intent_for(result, "b")
        self.assertTrue(intent.exit_requested)
        self.assertEqual(intent.sharing_decision, SharingDecision.SILENT)
        self.assertEqual(intent.verification_edge, edge("a", "b"))
        self.assertEqual(member(result.next_state, "b").status, LifecycleStatus.INACTIVE)
        self.assertEqual(member(result.next_state, "b").exit_count, 1)
        self.assertEqual(tuple(item.agent_id for item in result.autonomous_exits), ("b",))
        self.assertEqual(edge_trust(result.next_state, "a", "b"), 0.75)

        later = simulate_autonomous_round(model, result.next_state)
        self.assertFalse(any(item.target_agent_id == "b" for item in later.transmissions))

    def test_environment_exit_and_inexact_truth_coverage_are_rejected(self):
        model = autonomous_model()
        initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )
        with self.assertRaisesRegex(ValueError, "environment EXIT"):
            simulate_autonomous_round(
                model,
                initial,
                environment_events=(
                    PopulationLifecycleEvent("a", LifecycleEventKind.EXIT),
                ),
            )
        with self.assertRaisesRegex(ValueError, "exact selected verification edges"):
            simulate_autonomous_round(
                model,
                initial,
                environment_events=(enter("b"),),
            )
        with self.assertRaisesRegex(ValueError, "exact selected verification edges"):
            simulate_autonomous_round(
                model,
                initial,
                environment_events=(enter("b"),),
                truth_observations=(truth("b", "c", 1.0),),
            )

    def test_agent_selects_lowest_trust_incoming_source_using_only_local_edges(self):
        base = evolving_base_model()
        c_to_b = replace(base.network.edges[0], source_agent_id="c")
        active_edges = (
            replace(base.network.edges[0], active=True),
            replace(c_to_b, active=True),
        )
        two_source_base = replace(
            base,
            network=SocialNetwork(base.network.agent_ids, active_edges),
        )
        evolving = EvolvingNetworkModel(
            "two-source-evolving",
            "1",
            two_source_base,
            ("a", "b", "c"),
            0.5,
            0.5,
            0.2,
            0.8,
        )
        policies = tuple(
            replace(item, sharing_enabled=True)
            if item.role == "recipient"
            else item
            for item in role_policies()
        )
        model = AutonomousNetworkModel("two-source", "1", evolving, policies)
        initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "b": 0.5, "c": 1.0},
        )
        first = simulate_autonomous_round(
            model,
            initial,
            truth_observations=(truth("a", "b", 1.0),),
        )
        self.assertEqual(intent_for(first, "b").verification_edge, edge("a", "b"))

        second = simulate_autonomous_round(
            model,
            first.next_state,
            truth_observations=(truth("c", "b", 1.0),),
        )
        self.assertEqual(intent_for(second, "b").verification_edge, edge("c", "b"))

    def test_truth_observation_input_order_is_canonical(self):
        base = evolving_base_model()
        active_edges = tuple(replace(item, active=True) for item in base.network.edges)
        active_base = replace(
            base,
            network=SocialNetwork(base.network.agent_ids, active_edges),
        )
        evolving = EvolvingNetworkModel(
            "all-active-evolving",
            "1",
            active_base,
            ("a", "b", "c"),
            0.5,
            0.5,
            0.2,
            0.8,
        )
        model = AutonomousNetworkModel("all-active", "1", evolving, role_policies())
        initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "b": 0.5, "c": 0.5},
        )
        observations = (truth("a", "b", 1.0), truth("b", "c", 1.0))

        first = simulate_autonomous_round(model, initial, truth_observations=observations)
        second = simulate_autonomous_round(
            model,
            initial,
            truth_observations=tuple(reversed(observations)),
        )

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_zero_active_population_and_paired_schedules_are_deterministic(self):
        model = autonomous_model()
        initial = initialize_autonomous_population(model)
        emptied = simulate_autonomous_round(
            model,
            initial,
            environment_events=(die("a"), die("c")),
        )
        self.assertEqual(emptied.intents, ())
        self.assertEqual(emptied.transmissions, ())
        self.assertEqual(emptied.trust_updates, ())
        self.assertEqual(emptied.rewiring_updates, ())

        seeded_initial = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )
        trajectory = simulate_autonomous_population(
            model,
            seeded_initial,
            environment_event_schedule=((enter("b"),), ()),
            truth_observation_schedule=((truth("a", "b", 1.0),), (truth("b", "c", 1.0),)),
        )
        self.assertEqual(trajectory.final_state.round_index, 2)
        self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)

    def test_schedule_validation_requires_equal_nonempty_tuple_lengths(self):
        model = autonomous_model()
        initial = initialize_autonomous_population(model)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            simulate_autonomous_population(
                model,
                initial,
                environment_event_schedule=(),
                truth_observation_schedule=(),
            )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            simulate_autonomous_population(
                model,
                initial,
                environment_event_schedule=((), ()),
                truth_observation_schedule=((),),
            )


if __name__ == "__main__":
    unittest.main()
