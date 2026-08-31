from dataclasses import replace
import unittest

from narrative_dynamics.abm import NetworkABMModel, SocialEdge, SocialNetwork
from narrative_dynamics.abm.rewiring import (
    EdgeRewiringUpdate,
    RewiringRoundResult,
    rewiring_population_view,
    simulate_rewiring_population,
    simulate_rewiring_round,
)
from narrative_dynamics.abm.rewiring_contracts import (
    EdgeTopologyState,
    EndogenousRewiringModel,
    RewiringPopulationState,
    initialize_rewiring_population,
)
from tests.rewiring_fixtures import rewiring_base_model


def model_for(base=None) -> EndogenousRewiringModel:
    return EndogenousRewiringModel(
        "endogenous-rewiring",
        "1",
        rewiring_base_model() if base is None else base,
        0.2,
        0.8,
    )


def topology(state, source: str, target: str) -> EdgeTopologyState:
    return next(
        item
        for item in state.edge_topology
        if item.edge.source_agent_id == source and item.edge.target_agent_id == target
    )


class EndogenousRewiringSimulationTests(unittest.TestCase):
    def test_dissolution_affects_following_round_not_current_transmission(self):
        base = rewiring_base_model()
        agents = tuple(
            replace(item, receptivity=0.1) if item.agent_id == "b" else item
            for item in base.agents
        )
        model = model_for(NetworkABMModel(base.model_id, base.version, agents, base.network))
        initial = initialize_rewiring_population(model, beliefs={"a": 1.0, "b": 0.0})

        first = simulate_rewiring_round(model, initial)

        self.assertIsInstance(first, RewiringRoundResult)
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in first.transmissions),
            (("a", "b"),),
        )
        self.assertAlmostEqual(first.next_state.agents[1].belief, 0.1)
        self.assertFalse(topology(first.next_state, "a", "b").active)
        self.assertEqual(topology(first.next_state, "a", "b").rewiring_count, 1)

        second = simulate_rewiring_round(model, first.next_state)
        self.assertEqual(second.transmissions, ())

    def test_formation_uses_post_propagation_beliefs_and_transmits_next_round(self):
        model = model_for()
        initial = initialize_rewiring_population(
            model,
            beliefs={"a": 1.0, "b": 0.0, "c": 1.0},
        )

        first = simulate_rewiring_round(model, initial)

        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in first.transmissions),
            (("a", "b"),),
        )
        self.assertTrue(topology(first.next_state, "b", "c").active)
        self.assertEqual(topology(first.next_state, "b", "c").rewiring_count, 1)

        second = simulate_rewiring_round(model, first.next_state)
        self.assertEqual(
            tuple((item.source_agent_id, item.target_agent_id) for item in second.transmissions),
            (("a", "b"), ("b", "c")),
        )

    def test_similarity_inside_hysteresis_band_preserves_prior_activity(self):
        model = model_for()
        initial = initialize_rewiring_population(
            model,
            beliefs={"b": 0.0, "c": 0.5},
        )

        result = simulate_rewiring_round(model, initial)

        update = next(
            item
            for item in result.edge_updates
            if item.edge.source_agent_id == "b"
        )
        self.assertIsInstance(update, EdgeRewiringUpdate)
        self.assertEqual(update.similarity, 0.5)
        self.assertFalse(update.prior_active)
        self.assertFalse(update.next_active)
        self.assertFalse(update.changed)
        self.assertEqual(topology(result.next_state, "b", "c").rewiring_count, 0)

    def test_zero_active_edges_produce_no_transmissions(self):
        base = rewiring_base_model()
        edges = tuple(replace(item, active=False) for item in base.network.edges)
        inactive_base = NetworkABMModel(
            base.model_id,
            base.version,
            base.agents,
            SocialNetwork(base.network.agent_ids, edges),
        )
        model = model_for(inactive_base)
        initial = initialize_rewiring_population(
            model,
            beliefs={"a": 1.0, "b": 0.0, "c": 0.5},
        )

        result = simulate_rewiring_round(model, initial)

        self.assertEqual(result.transmissions, ())
        self.assertEqual(result.next_state.round_index, 1)

    def test_multi_round_trajectory_has_exact_state_chain(self):
        model = model_for()
        initial = initialize_rewiring_population(
            model,
            beliefs={"a": 1.0, "b": 0.0, "c": 1.0},
        )

        trajectory = simulate_rewiring_population(model, initial, rounds=2)
        replay = simulate_rewiring_population(model_for(), initial, rounds=2)

        self.assertEqual(trajectory.final_state.round_index, 2)
        self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
        self.assertEqual(trajectory.final_state, trajectory.rounds[-1].next_state)
        self.assertEqual(trajectory, replay)
        self.assertEqual(trajectory.content_hash, replay.content_hash)

    def test_population_view_projects_current_topology_identity(self):
        model = model_for()
        initial = initialize_rewiring_population(model, beliefs={"a": 1.0})

        view = rewiring_population_view(model, initial)

        self.assertEqual(view.round_index, 0)
        self.assertEqual(tuple(item.agent_id for item in view.agents), ("a", "b", "c"))
        self.assertNotEqual(view.model_hash, model.base_model.content_hash)

    def test_simulation_rejects_incomplete_or_inconsistent_topology(self):
        model = model_for()
        initial = initialize_rewiring_population(model, beliefs={"a": 1.0})
        incomplete = RewiringPopulationState(
            initial.model_id,
            initial.model_hash,
            initial.round_index,
            initial.parent_state_hash,
            initial.agents,
            initial.edge_topology[:1],
        )
        with self.assertRaisesRegex(ValueError, "exact candidate edge set"):
            simulate_rewiring_round(model, incomplete)

        forged_similarity = RewiringPopulationState(
            initial.model_id,
            initial.model_hash,
            initial.round_index,
            initial.parent_state_hash,
            initial.agents,
            (
                replace(initial.edge_topology[0], last_similarity=0.5),
                initial.edge_topology[1],
            ),
        )
        with self.assertRaisesRegex(ValueError, "similarity is inconsistent"):
            simulate_rewiring_round(model, forged_similarity)

    def test_round_count_validation_rejects_nonpositive_and_boolean_values(self):
        model = model_for()
        initial = initialize_rewiring_population(model)
        for bad in (0, -1, True, 1.5):
            with self.subTest(rounds=bad):
                with self.assertRaises((TypeError, ValueError)):
                    simulate_rewiring_population(model, initial, rounds=bad)


if __name__ == "__main__":
    unittest.main()
