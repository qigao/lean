import math
import unittest

from narrative_dynamics.abm import EdgeSelector
from narrative_dynamics.abm.rewiring_contracts import (
    EdgeTopologyState,
    EndogenousRewiringModel,
    RewiringPopulationState,
    initialize_rewiring_population,
)
from tests.rewiring_fixtures import rewiring_base_model


class EndogenousRewiringContractTests(unittest.TestCase):
    def test_initialization_covers_every_agent_and_candidate_edge(self):
        model = EndogenousRewiringModel(
            "endogenous-rewiring",
            "1",
            rewiring_base_model(),
            0.2,
            0.8,
        )

        state = initialize_rewiring_population(
            model,
            beliefs={"a": 1.0, "b": 0.0, "c": 0.0},
        )

        self.assertEqual(state.model_id, model.model_id)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertEqual(state.round_index, 0)
        self.assertIsNone(state.parent_state_hash)
        self.assertEqual(tuple(item.agent_id for item in state.agents), ("a", "b", "c"))
        self.assertEqual(tuple(item.active for item in state.edge_topology), (True, False))
        self.assertEqual(
            tuple(item.last_similarity for item in state.edge_topology),
            (0.0, 1.0),
        )
        self.assertEqual(tuple(item.rewiring_count for item in state.edge_topology), (0, 0))

    def test_model_and_state_identity_are_canonical(self):
        first = EndogenousRewiringModel(
            "endogenous-rewiring",
            "1",
            rewiring_base_model(),
            0.2,
            0.8,
        )
        second = EndogenousRewiringModel(
            "endogenous-rewiring",
            "1",
            rewiring_base_model(reverse=True),
            0.2,
            0.8,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            initialize_rewiring_population(
                first,
                beliefs={"a": 1.0, "b": 0.0},
            ).content_hash,
            initialize_rewiring_population(
                second,
                beliefs={"b": 0.0, "a": 1.0},
            ).content_hash,
        )

    def test_thresholds_are_probabilities_with_strict_hysteresis(self):
        base = rewiring_base_model()
        for bad in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    EndogenousRewiringModel("rewiring", "1", base, bad, 0.8)
                with self.assertRaises((TypeError, ValueError)):
                    EndogenousRewiringModel("rewiring", "1", base, 0.2, bad)
        with self.assertRaisesRegex(ValueError, "strictly below"):
            EndogenousRewiringModel("rewiring", "1", base, 0.8, 0.8)
        with self.assertRaisesRegex(ValueError, "strictly below"):
            EndogenousRewiringModel("rewiring", "1", base, 0.9, 0.8)

    def test_model_requires_at_least_one_candidate_edge(self):
        base = rewiring_base_model()
        isolated = type(base)(
            "isolated",
            "1",
            base.agents,
            type(base.network)(base.network.agent_ids, ()),
        )
        with self.assertRaisesRegex(ValueError, "candidate edge"):
            EndogenousRewiringModel("rewiring", "1", isolated, 0.2, 0.8)

    def test_edge_topology_validates_activity_similarity_and_count(self):
        selector = EdgeSelector("a", "b", "peer")
        with self.assertRaisesRegex(TypeError, "active"):
            EdgeTopologyState(selector, 1, 0.5, 0)
        with self.assertRaises((TypeError, ValueError)):
            EdgeTopologyState(selector, True, math.nan, 0)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            EdgeTopologyState(selector, True, 0.5, -1)

    def test_state_rejects_duplicate_agent_and_edge_identity(self):
        model = EndogenousRewiringModel(
            "endogenous-rewiring",
            "1",
            rewiring_base_model(),
            0.2,
            0.8,
        )
        initial = initialize_rewiring_population(model)
        with self.assertRaisesRegex(ValueError, "agent ids must be unique"):
            RewiringPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                (initial.agents[0], initial.agents[0]),
                initial.edge_topology,
            )
        with self.assertRaisesRegex(ValueError, "edge identities must be unique"):
            RewiringPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                initial.agents,
                (initial.edge_topology[0], initial.edge_topology[0]),
            )

    def test_initialization_rejects_unknown_belief_seed(self):
        model = EndogenousRewiringModel(
            "endogenous-rewiring",
            "1",
            rewiring_base_model(),
            0.2,
            0.8,
        )
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            initialize_rewiring_population(model, beliefs={"missing": 1.0})


if __name__ == "__main__":
    unittest.main()
