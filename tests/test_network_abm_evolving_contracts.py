import math
import unittest

from narrative_dynamics.abm.evolving_contracts import (
    EvolvingNetworkModel,
    EvolvingPopulationState,
    initialize_evolving_population,
)
from narrative_dynamics.abm.lifecycle_contracts import LifecycleStatus
from tests.evolving_fixtures import evolving_base_model


def evolving_model(*, reverse: bool = False) -> EvolvingNetworkModel:
    return EvolvingNetworkModel(
        "unified-evolution",
        "1",
        evolving_base_model(reverse=reverse),
        ("c", "a") if reverse else ("a", "c"),
        0.5,
        0.5,
        0.2,
        0.8,
    )


class EvolvingNetworkContractTests(unittest.TestCase):
    def test_initialization_covers_all_catalog_dimensions(self):
        model = evolving_model()

        state = initialize_evolving_population(
            model,
            beliefs={"a": 1.0, "b": 0.2, "c": 0.5},
        )

        self.assertEqual(state.model_id, model.model_id)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertEqual(state.round_index, 0)
        self.assertEqual(tuple(item.agent_id for item in state.members), ("a", "b", "c"))
        self.assertEqual(
            tuple(item.status for item in state.members),
            (LifecycleStatus.ACTIVE, LifecycleStatus.INACTIVE, LifecycleStatus.ACTIVE),
        )
        self.assertTrue(state.members[0].broadcasting)
        self.assertFalse(state.members[1].broadcasting)
        self.assertTrue(state.members[2].broadcasting)
        self.assertEqual(tuple(item.entry_count for item in state.members), (1, 0, 1))
        self.assertEqual(tuple(item.trust for item in state.edge_trust), (0.5, 0.5))
        self.assertEqual(tuple(item.feedback_count for item in state.edge_trust), (0, 0))
        self.assertEqual(tuple(item.active for item in state.edge_topology), (True, False))
        self.assertAlmostEqual(state.edge_topology[0].last_similarity, 0.2)
        self.assertAlmostEqual(state.edge_topology[1].last_similarity, 0.7)

    def test_model_and_initial_state_are_canonical(self):
        first = evolving_model()
        second = evolving_model(reverse=True)

        self.assertEqual(first, second)
        self.assertEqual(first.initial_active_agent_ids, ("a", "c"))
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            initialize_evolving_population(
                first,
                beliefs={"a": 1.0, "c": 0.5},
            ).content_hash,
            initialize_evolving_population(
                second,
                beliefs={"c": 0.5, "a": 1.0},
            ).content_hash,
        )

    def test_model_validates_learning_rewiring_and_membership_parameters(self):
        base = evolving_base_model()
        for bad in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    EvolvingNetworkModel("e", "1", base, ("a",), bad, 0.5, 0.2, 0.8)
                with self.assertRaises((TypeError, ValueError)):
                    EvolvingNetworkModel("e", "1", base, ("a",), 0.5, bad, 0.2, 0.8)
        with self.assertRaisesRegex(ValueError, "strictly below"):
            EvolvingNetworkModel("e", "1", base, ("a",), 0.5, 0.5, 0.8, 0.8)
        with self.assertRaisesRegex(ValueError, "initially active"):
            EvolvingNetworkModel("e", "1", base, (), 0.5, 0.5, 0.2, 0.8)
        with self.assertRaisesRegex(ValueError, "unknown"):
            EvolvingNetworkModel("e", "1", base, ("missing",), 0.5, 0.5, 0.2, 0.8)

    def test_state_rejects_duplicate_cross_layer_identities(self):
        model = evolving_model()
        initial = initialize_evolving_population(model)
        with self.assertRaisesRegex(ValueError, "member ids must be unique"):
            EvolvingPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                (initial.members[0], initial.members[0]),
                initial.edge_trust,
                initial.edge_topology,
            )
        with self.assertRaisesRegex(ValueError, "trust edge identities must be unique"):
            EvolvingPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                initial.members,
                (initial.edge_trust[0], initial.edge_trust[0]),
                initial.edge_topology,
            )
        with self.assertRaisesRegex(ValueError, "topology edge identities must be unique"):
            EvolvingPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                initial.members,
                initial.edge_trust,
                (initial.edge_topology[0], initial.edge_topology[0]),
            )

    def test_state_requires_matching_trust_and_topology_edge_sets(self):
        model = evolving_model()
        initial = initialize_evolving_population(model)
        with self.assertRaisesRegex(ValueError, "same edge identities"):
            EvolvingPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                initial.members,
                initial.edge_trust[:1],
                initial.edge_topology,
            )

    def test_initialization_rejects_unknown_seed(self):
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            initialize_evolving_population(evolving_model(), beliefs={"missing": 1.0})


if __name__ == "__main__":
    unittest.main()
